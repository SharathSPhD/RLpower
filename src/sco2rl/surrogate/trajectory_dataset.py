"""TrajectoryDataset -- HDF5-backed storage for surrogate training trajectories.

Schema:
    /states   (N, T, n_obs)   float32
    /actions  (N, T-1, n_act) float32
    /metadata (N, 3)          float32

where N = n_trajectories, T = trajectory_length.

Supports write ("w"), read ("r"), and append ("a") modes.
Context manager protocol ensures file is properly closed.
"""
from __future__ import annotations

import numpy as np
import h5py


class TrajectoryDataset:
    """Write/read trajectory batches to/from an HDF5 file.

    Parameters
    ----------
    filepath:
        Path to the HDF5 file.
    mode:
        "w" = create new (truncate if exists), "r" = read-only, "a" = append.
    """

    def __init__(self, filepath: str, mode: str = "r") -> None:
        if mode not in ("w", "r", "a"):
            raise ValueError(f"mode must be 'w', 'r', or 'a'; got {mode!r}")
        self._filepath = filepath
        self._mode = mode
        # Map to h5py modes: "w" = write/truncate, "r" = read, "a" = read/write
        h5_mode = {"w": "w", "r": "r", "a": "a"}[mode]
        self._file: h5py.File = h5py.File(filepath, h5_mode)

    def write_batch(self, trajectories: list[dict]) -> None:
        """Append a list of trajectory dicts to the HDF5 file.

        Creates /states, /actions, /metadata datasets on first call.
        On subsequent calls (append mode), extends existing datasets.

        Parameters
        ----------
        trajectories:
            List of dicts with keys: states (T, n_obs), actions (T-1, n_act),
            metadata (3,). All arrays should be float32 or convertible.
        """
        if len(trajectories) == 0:
            return

        # Stack into batch arrays
        states_batch = np.stack(
            [t["states"].astype(np.float32) for t in trajectories], axis=0
        )  # (batch, T, n_obs)
        actions_batch = np.stack(
            [t["actions"].astype(np.float32) for t in trajectories], axis=0
        )  # (batch, T-1, n_act)
        metadata_batch = np.stack(
            [np.asarray(t["metadata"], dtype=np.float32) for t in trajectories], axis=0
        )  # (batch, 3)

        if "states" not in self._file:
            # Create resizable datasets
            self._file.create_dataset(
                "states",
                data=states_batch,
                maxshape=(None,) + states_batch.shape[1:],
                chunks=(1,) + states_batch.shape[1:],
                dtype=np.float32,
            )
            self._file.create_dataset(
                "actions",
                data=actions_batch,
                maxshape=(None,) + actions_batch.shape[1:],
                chunks=(1,) + actions_batch.shape[1:],
                dtype=np.float32,
            )
            self._file.create_dataset(
                "metadata",
                data=metadata_batch,
                maxshape=(None,) + metadata_batch.shape[1:],
                chunks=(1,) + metadata_batch.shape[1:],
                dtype=np.float32,
            )
        else:
            # Extend existing datasets
            n_existing = self._file["states"].shape[0]
            n_new = states_batch.shape[0]
            n_total = n_existing + n_new

            self._file["states"].resize(n_total, axis=0)
            self._file["states"][n_existing:] = states_batch

            self._file["actions"].resize(n_total, axis=0)
            self._file["actions"][n_existing:] = actions_batch

            self._file["metadata"].resize(n_total, axis=0)
            self._file["metadata"][n_existing:] = metadata_batch

        self._file.flush()

    def __len__(self) -> int:
        """Return number of trajectories stored."""
        if "states" not in self._file:
            return 0
        return self._file["states"].shape[0]

    def __getitem__(self, idx: int) -> dict:
        """Return single trajectory dict at index idx.

        Returns
        -------
        dict with keys: states (T, n_obs), actions (T-1, n_act), metadata (3,).
        All arrays are float32 numpy arrays.
        """
        return {
            "states": self._file["states"][idx].astype(np.float32),
            "actions": self._file["actions"][idx].astype(np.float32),
            "metadata": self._file["metadata"][idx].astype(np.float32),
        }

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file.id.valid:
            self._file.close()

    def __enter__(self) -> "TrajectoryDataset":
        return self

    def __exit__(self, *args) -> None:
        self.close()
