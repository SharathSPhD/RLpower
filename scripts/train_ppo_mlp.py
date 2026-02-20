"""PPO training on the MLP step-predictor surrogate.

This script trains a PPO agent directly on a vectorized MLP environment,
bypassing the FNO surrogate which was unsuitable for one-step RL prediction.

Usage:
    python scripts/train_ppo_mlp.py [--model artifacts/surrogate/mlp_step.pt]
                                     [--norm  artifacts/surrogate/mlp_step_norm.npz]
                                     [--total_steps 1000000]
                                     [--out_dir artifacts/rl/ppo_mlp]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


# ---- MLP model (same architecture as train_mlp_surrogate.py) ----

class MLPStepPredictor(nn.Module):
    def __init__(self, n_state: int, n_action: int, hidden: int = 512, n_layers: int = 4):
        super().__init__()
        in_dim = n_state + n_action
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, n_state))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        return state + delta


# ---- Batched MLP environment (GPU-native) ----

class MLPEnv:
    """Vectorized sCO2 environment backed by MLP step predictor.

    Observations returned to the agent are min-max normalised to [-1, 1].
    Actions are expected in [-1, 1] and mapped to physical ranges.
    """

    def __init__(self, model: MLPStepPredictor, norm: dict, env_cfg: dict,
                 n_envs: int = 256, episode_len: int = 200, device: str = "cuda"):
        self._model = model
        self._device = torch.device(device)
        self._n_envs = n_envs
        self._ep_len = episode_len

        n_s = norm["s_mean"].shape[0]
        n_a = norm["a_mean"].shape[0]
        self.n_obs = n_s
        self.n_act = n_a

        # Normalisation stats (for passing (s,a) to MLP)
        self._s_mean  = torch.tensor(norm["s_mean"],  device=self._device)
        self._s_std   = torch.tensor(norm["s_std"],   device=self._device)
        self._a_mean  = torch.tensor(norm["a_mean"],  device=self._device)
        self._a_std   = torch.tensor(norm["a_std"],   device=self._device)
        self._sp_mean = torch.tensor(norm["sp_mean"], device=self._device)
        self._sp_std  = torch.tensor(norm["sp_std"],  device=self._device)

        # Physical bounds from env config
        obs_bounds  = env_cfg["obs_bounds"]
        act_bounds  = env_cfg["act_bounds"]
        self._obs_lo  = torch.tensor([b[0] for b in obs_bounds[:n_s]], device=self._device)
        self._obs_hi  = torch.tensor([b[1] for b in obs_bounds[:n_s]], device=self._device)
        self._obs_range = self._obs_hi - self._obs_lo
        self._act_lo  = torch.tensor([b[0] for b in act_bounds[:n_a]], device=self._device)
        self._act_hi  = torch.tensor([b[1] for b in act_bounds[:n_a]], device=self._device)
        self._act_range = self._act_hi - self._act_lo

        # Design point (physical midpoint)
        self._design_pt = 0.5 * (self._obs_lo + self._obs_hi)

        # Safety bounds (T_comp_in must be >= 32.2°C, index 0)
        self._T_comp_min_idx = 0
        self._T_comp_min_val = 32.2  # °C

        self._state     = torch.zeros(n_envs, n_s, device=self._device)
        self._step_count = torch.zeros(n_envs, dtype=torch.long, device=self._device)

        # Target setpoint (net power output) - W_turbine - W_comp (indices 8,9)
        self._target_W_net = 10.0  # MW

        # Rate limits (per step) for actions - roughly 5% of range
        self._rate_lim = 0.05 * self._act_range

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        rng = np.random.default_rng(seed)
        design_np = self._design_pt.cpu().numpy()
        obs_lo_np = self._obs_lo.cpu().numpy()
        obs_hi_np = self._obs_hi.cpu().numpy()
        init_np = np.clip(
            design_np + rng.standard_normal((self._n_envs, self.n_obs)).astype(np.float32) * 0.01,
            obs_lo_np, obs_hi_np
        )
        self._state = torch.tensor(init_np, device=self._device)
        self._step_count[:] = 0
        self._prev_action = torch.zeros(self._n_envs, self.n_act, device=self._device)
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> torch.Tensor:
        """Return min-max normalised observation in [-1, 1]."""
        return 2.0 * (self._state - self._obs_lo) / self._obs_range - 1.0

    def step(self, action_norm: torch.Tensor):
        """Step the environment.

        action_norm: (N, n_act) in [-1, 1]
        Returns: obs, reward, terminated, truncated, info
        """
        action_norm = action_norm.clamp(-1.0, 1.0)
        # Rate-limit actions
        delta_norm = action_norm - self._prev_action
        rate_lim_norm = 2.0 * self._rate_lim / self._act_range  # convert to norm space
        delta_norm = delta_norm.clamp(-rate_lim_norm, rate_lim_norm)
        action_norm = (self._prev_action + delta_norm).clamp(-1.0, 1.0)
        self._prev_action = action_norm.clone()

        # Normalise state and action for MLP
        s_n = (self._state - self._s_mean) / self._s_std
        a_n = (action_norm - self._a_mean) / self._a_std  # action already in phys space via de-norm

        # Actually: action_norm is in [-1,1], we need to convert to physical first
        a_phys = self._act_lo + (action_norm + 1.0) * 0.5 * self._act_range
        a_n = (a_phys - self._a_mean) / self._a_std

        with torch.no_grad():
            sp_n = self._model(s_n, a_n)  # (N, n_s) normalised next state
        next_state = sp_n * self._sp_std + self._sp_mean
        next_state = next_state.clamp(self._obs_lo, self._obs_hi)

        self._state = next_state
        self._step_count += 1

        # Safety check
        T_comp = next_state[:, self._T_comp_min_idx]
        unsafe = T_comp < self._T_comp_min_val

        # Reward: track target net power
        W_turb = next_state[:, 8]  # MW
        W_comp = next_state[:, 9]  # MW
        W_net  = W_turb - W_comp
        power_reward = -torch.abs(W_net - self._target_W_net) / self._target_W_net

        # Efficiency bonus
        eta = next_state[:, 10]
        eff_reward = 0.5 * eta

        # Safety penalty
        safety_pen = -10.0 * unsafe.float()

        # Action smoothness penalty
        smooth_pen = -0.01 * (delta_norm ** 2).sum(dim=-1)

        reward = power_reward + eff_reward + safety_pen + smooth_pen

        terminated = unsafe
        truncated  = (self._step_count >= self._ep_len)

        # Auto-reset done envs
        done = terminated | truncated
        if done.any():
            done_idx = done.nonzero(as_tuple=True)[0]
            rng = np.random.default_rng()
            design_np = self._design_pt.cpu().numpy()
            obs_lo_np = self._obs_lo.cpu().numpy()
            obs_hi_np = self._obs_hi.cpu().numpy()
            k = len(done_idx)
            init_np = np.clip(
                design_np + rng.standard_normal((k, self.n_obs)).astype(np.float32) * 0.01,
                obs_lo_np, obs_hi_np
            )
            self._state[done_idx] = torch.tensor(init_np, device=self._device)
            self._step_count[done_idx] = 0
            self._prev_action[done_idx] = 0.0

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {"done": done}


# ---- Policy and Value networks ----

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, act_dim),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def forward(self, obs: torch.Tensor):
        mean = self.net(obs)
        std  = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, obs: torch.Tensor):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        raw  = dist.sample()
        action = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        return action, log_prob, mean

    def log_prob_and_entropy(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        a_clip = actions.clamp(-0.9999, 0.9999)
        raw = torch.atanh(a_clip)
        lp  = (dist.log_prob(raw) - torch.log(1 - a_clip.pow(2) + 1e-6)).sum(-1)
        ent = dist.entropy().sum(-1)
        return lp, ent


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---- PPO update ----

def ppo_update(policy: PolicyNet, value: ValueNet,
               obs_buf: torch.Tensor, act_buf: torch.Tensor,
               ret_buf: torch.Tensor, adv_buf: torch.Tensor, old_lp_buf: torch.Tensor,
               opt_p: torch.optim.Optimizer, opt_v: torch.optim.Optimizer,
               n_epochs: int = 4, mini_batch: int = 256,
               clip_eps: float = 0.2, ent_coef: float = 0.01, vf_coef: float = 0.5):
    N = len(obs_buf)
    pg_losses, vf_losses, ent_losses = [], [], []

    for _ in range(n_epochs):
        idx = torch.randperm(N)
        for start in range(0, N, mini_batch):
            mb = idx[start:start + mini_batch]
            obs_mb  = obs_buf[mb]
            act_mb  = act_buf[mb]
            ret_mb  = ret_buf[mb]
            adv_mb  = adv_buf[mb]
            olp_mb  = old_lp_buf[mb]

            adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

            lp, ent = policy.log_prob_and_entropy(obs_mb, act_mb)
            ratio   = (lp - olp_mb).exp()
            pg1 = ratio * adv_mb
            pg2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_mb
            pg_loss = -torch.min(pg1, pg2).mean()

            vf_pred = value(obs_mb)
            vf_loss = nn.functional.mse_loss(vf_pred, ret_mb)

            loss = pg_loss + vf_coef * vf_loss - ent_coef * ent.mean()

            if torch.isnan(loss):
                continue

            opt_p.zero_grad()
            opt_v.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), 0.5)
            opt_p.step()
            opt_v.step()

            pg_losses.append(pg_loss.item())
            vf_losses.append(vf_loss.item())
            ent_losses.append(ent.mean().item())

    return dict(pg=np.mean(pg_losses) if pg_losses else float("nan"),
                vf=np.mean(vf_losses) if vf_losses else float("nan"),
                ent=np.mean(ent_losses) if ent_losses else float("nan"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="artifacts/surrogate/mlp_step.pt")
    parser.add_argument("--norm",        default="artifacts/surrogate/mlp_step_norm.npz")
    parser.add_argument("--env_cfg",     default="configs/environment/env.yaml")
    parser.add_argument("--n_envs",      type=int,   default=512)
    parser.add_argument("--rollout",     type=int,   default=256)
    parser.add_argument("--ep_len",      type=int,   default=200)
    parser.add_argument("--total_steps", type=int,   default=2_000_000)
    parser.add_argument("--n_epochs",    type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--lam",         type=float, default=0.95)
    parser.add_argument("--clip_eps",    type=float, default=0.2)
    parser.add_argument("--ent_coef",    type=float, default=0.01)
    parser.add_argument("--out_dir",     default="artifacts/rl/ppo_mlp")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load MLP model
    norm = dict(np.load(args.norm))
    n_s = norm["s_mean"].shape[0]
    n_a = norm["a_mean"].shape[0]
    mlp = MLPStepPredictor(n_s, n_a).to(device)
    sd = torch.load(args.model, weights_only=True, map_location=device)
    mlp.load_state_dict(sd)
    mlp.train(False)
    print(f"Loaded MLP from {args.model}")

    # Load env config
    env_cfg = yaml.safe_load(open(args.env_cfg))

    # Create vectorized environment
    env = MLPEnv(mlp, norm, env_cfg, n_envs=args.n_envs, episode_len=args.ep_len, device=str(device))
    obs_dim = env.n_obs
    act_dim = env.n_act

    # Policy and value networks
    policy = PolicyNet(obs_dim, act_dim).to(device)
    value  = ValueNet(obs_dim).to(device)
    opt_p  = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_v  = torch.optim.Adam(value.parameters(), lr=args.lr)
    sched_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=args.total_steps // (args.n_envs * args.rollout), eta_min=1e-5)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reset
    obs, _ = env.reset(seed=42)

    total_steps = 0
    update_num  = 0
    best_reward = -float("inf")
    episode_rewards: list[float] = []
    current_ep_rewards = torch.zeros(args.n_envs, device=device)

    print(f"Starting PPO training: {args.total_steps:,} total steps, {args.n_envs} envs")
    t_start = time.time()

    while total_steps < args.total_steps:
        # Collect rollout
        obs_buf  = torch.zeros(args.rollout, args.n_envs, obs_dim, device=device)
        act_buf  = torch.zeros(args.rollout, args.n_envs, act_dim, device=device)
        rew_buf  = torch.zeros(args.rollout, args.n_envs, device=device)
        don_buf  = torch.zeros(args.rollout, args.n_envs, device=device)
        val_buf  = torch.zeros(args.rollout, args.n_envs, device=device)
        lp_buf   = torch.zeros(args.rollout, args.n_envs, device=device)

        for t in range(args.rollout):
            with torch.no_grad():
                action, lp, _ = policy.get_action(obs)
                val            = value(obs)

            next_obs, rew, term, trunc, info = env.step(action)

            obs_buf[t] = obs
            act_buf[t] = action
            rew_buf[t] = rew
            don_buf[t] = info["done"].float()
            val_buf[t] = val
            lp_buf[t]  = lp

            current_ep_rewards += rew
            done_mask = info["done"]
            for i in done_mask.nonzero(as_tuple=True)[0]:
                episode_rewards.append(current_ep_rewards[i].item())
                current_ep_rewards[i] = 0.0

            obs = next_obs

        total_steps += args.rollout * args.n_envs

        # Compute GAE returns and advantages
        with torch.no_grad():
            last_val = value(obs)

        adv_buf = torch.zeros_like(rew_buf)
        ret_buf = torch.zeros_like(rew_buf)
        gae = torch.zeros(args.n_envs, device=device)
        for t in reversed(range(args.rollout)):
            next_v = last_val if t == args.rollout - 1 else val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * next_v * (1 - don_buf[t]) - val_buf[t]
            gae   = delta + args.gamma * args.lam * (1 - don_buf[t]) * gae
            adv_buf[t] = gae
            ret_buf[t] = adv_buf[t] + val_buf[t]

        # Flatten for mini-batch update
        obs_flat = obs_buf.view(-1, obs_dim)
        act_flat = act_buf.view(-1, act_dim)
        ret_flat = ret_buf.view(-1)
        adv_flat = adv_buf.view(-1)
        lp_flat  = lp_buf.view(-1)

        metrics = ppo_update(policy, value, obs_flat, act_flat, ret_flat, adv_flat, lp_flat,
                             opt_p, opt_v, n_epochs=args.n_epochs,
                             mini_batch=2048, clip_eps=args.clip_eps, ent_coef=args.ent_coef)
        sched_p.step()
        update_num += 1

        if update_num % 10 == 0:
            mean_rew = np.mean(episode_rewards[-100:]) if episode_rewards else float("nan")
            elapsed  = time.time() - t_start
            fps      = total_steps / elapsed
            print(f"  Step {total_steps:>8,} | update {update_num:4d} | "
                  f"mean_rew={mean_rew:.3f} | pg={metrics['pg']:.4f} | "
                  f"vf={metrics['vf']:.4f} | ent={metrics['ent']:.4f} | fps={fps:.0f}")

            if mean_rew > best_reward and len(episode_rewards) >= 50:
                best_reward = mean_rew
                torch.save(policy.state_dict(), out_dir / "best_policy.pt")
                torch.save(value.state_dict(),  out_dir / "best_value.pt")
                print(f"    -> saved best policy (mean_rew={best_reward:.3f})")

    # Final save
    torch.save(policy.state_dict(), out_dir / "final_policy.pt")
    torch.save(value.state_dict(),  out_dir / "final_value.pt")
    np.save(out_dir / "episode_rewards.npy", np.array(episode_rewards))
    print(f"\nTraining complete. Best mean reward: {best_reward:.3f}")
    print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
