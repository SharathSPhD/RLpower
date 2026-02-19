"""TrajectoryCollector -- collect trajectories from SCO2FMUEnv for surrogate training.

Uses LHS-sampled parameters to configure episodes and applies random-walk
action perturbations to ensure diverse trajectory coverage.
"""
from __future__ import annotations

import numpy as np

from sco2rl.environment.sco2_env import SCO2FMUEnv


class TrajectoryCollector:
    """Collect trajectories from SCO2FMUEnv using LHS-sampled parameters.

    Parameters
    ----------
    env:
        SCO2FMUEnv instance (with MockFMU for unit tests).
    config:
        Dict with keys:
        - trajectory_length_steps: int, target trajectory length
        - action_perturbation: dict with keys type, step_std, clip
    seed:
        Random seed for action perturbation reproducibility.
    """

    def __init__(
        self,
        env: SCO2FMUEnv,
        config: dict,
        seed: int = 42,
        raw_obs_dim: int | None = None,
    ) -> None:
        self._env = env
        self._cfg = config
        self._traj_len: int = int(config["trajectory_length_steps"])
        self._perturb_cfg: dict = config["action_perturbation"]
        self._step_std: float = float(self._perturb_cfg["step_std"])
        self._clip: float = float(self._perturb_cfg["clip"])
        self._rng = np.random.default_rng(seed)
        self._n_act: int = env.action_space.shape[0]
        # Store only the current raw state (not history-stacked observations).
        if raw_obs_dim is None:
            raw_obs_dim = len(getattr(env, "_obs_vars", [])) or env.observation_space.shape[0]
        self._raw_obs_dim: int = int(raw_obs_dim)

    def _extract_current_raw_obs(self, obs: np.ndarray) -> np.ndarray:
        """Return current raw state from potentially history-stacked observation."""
        if obs.shape[0] <= self._raw_obs_dim:
            return obs.astype(np.float32, copy=False)
        return obs[-self._raw_obs_dim :].astype(np.float32, copy=False)

    def collect_trajectory(self, sample: np.ndarray) -> dict:
        """Run one episode with the given LHS-sampled parameter sample.

        Parameters
        ----------
        sample:
            1-D array of shape (3,):
            ``[T_exhaust_K, mdot_exhaust_kgs, W_setpoint_MW]``.
            These are passed as ``options`` to ``env.reset()`` so the FMU is
            actually configured with the sampled operating conditions before
            the episode begins.  Previously the sample was only stored as
            metadata while the FMU always reset to its default initial state.

        Returns
        -------
        dict with keys:
            - states: np.ndarray of shape (trajectory_length, n_obs)
            - actions: np.ndarray of shape (trajectory_length - 1, n_act)
            - metadata: np.ndarray of shape (3,)
        """
        options = {
            "T_exhaust_K":      float(sample[0]),
            "mdot_exhaust_kgs": float(sample[1]),
            "W_setpoint_MW":    float(sample[2]),
        }
        obs, _ = self._env.reset(options=options)

        n_obs = self._raw_obs_dim

        states = np.zeros((self._traj_len, n_obs), dtype=np.float32)
        actions = np.zeros((self._traj_len - 1, self._n_act), dtype=np.float32)

        # Record initial state
        states[0] = self._extract_current_raw_obs(obs)

        # Current action in normalized [-1, 1] space, starts at zero (mid-range)
        current_action = np.zeros(self._n_act, dtype=np.float32)
        last_valid_obs = states[0].copy()

        actual_steps = 0
        for step in range(self._traj_len - 1):
            # Random-walk perturbation in normalized action space
            perturbation = self._rng.normal(0.0, self._step_std, size=self._n_act)
            perturbation = np.clip(perturbation, -self._clip, self._clip)
            current_action = np.clip(current_action + perturbation, -1.0, 1.0)
            action = current_action.astype(np.float32)

            obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            actions[step] = action
            actual_steps = step + 1

            if done:
                # Pad remaining states with last valid obs
                for fill_step in range(step + 1, self._traj_len):
                    states[fill_step] = last_valid_obs
                # Pad remaining actions with last action
                for fill_step in range(step + 1, self._traj_len - 1):
                    actions[fill_step] = action
                break
            else:
                last_valid_obs = self._extract_current_raw_obs(obs)
                if step + 1 < self._traj_len:
                    states[step + 1] = last_valid_obs

        # If episode ran full length without termination, states already filled
        return {
            "states": states,
            "actions": actions,
            "metadata": np.array(sample, dtype=np.float32),
        }

    def collect_batch(self, samples: np.ndarray) -> list[dict]:
        """Collect one trajectory per row of samples.

        Parameters
        ----------
        samples:
            2-D array of shape (n_samples, 3).

        Returns
        -------
        list of trajectory dicts (one per sample row).
        """
        return [self.collect_trajectory(samples[i]) for i in range(len(samples))]
