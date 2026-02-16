"""Unit tests for TrajectoryDataset (HDF5) -- TDD RED phase."""
from __future__ import annotations
import numpy as np
import pytest

N_OBS = 11
N_ACT = 4
TRAJ_LEN = 200


def _make_traj(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "states": rng.random((TRAJ_LEN, N_OBS)).astype(np.float32),
        "actions": rng.random((TRAJ_LEN - 1, N_ACT)).astype(np.float32),
        "metadata": np.array([600.0, 55.0, 7.0], dtype=np.float32),
    }


def _make_batch(n, seed_offset=0):
    return [_make_traj(seed=i + seed_offset) for i in range(n)]


class TestTrajectoryDatasetWriteRead:
    def test_write_and_read_single_trajectory(self, tmp_path):
        from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset
        fpath = str(tmp_path / "test.h5")
        traj = _make_traj(0)
        with TrajectoryDataset(fpath, mode="w") as ds:
            ds.write_batch([traj])
        with TrajectoryDataset(fpath, mode="r") as ds:
            loaded = ds[0]
        assert loaded["states"].shape == (TRAJ_LEN, N_OBS)
        assert loaded["actions"].shape == (TRAJ_LEN - 1, N_ACT)
        assert loaded["metadata"].shape == (3,)

    def test_write_and_len(self, tmp_path):
        from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset
        fpath = str(tmp_path / "test.h5")
        batch = _make_batch(10)
        with TrajectoryDataset(fpath, mode="w") as ds:
            ds.write_batch(batch)
        with TrajectoryDataset(fpath, mode="r") as ds:
            assert len(ds) == 10

    def test_getitem_returns_dict_with_correct_keys(self, tmp_path):
        from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset
        fpath = str(tmp_path / "test.h5")
        with TrajectoryDataset(fpath, mode="w") as ds:
            ds.write_batch(_make_batch(3))
        with TrajectoryDataset(fpath, mode="r") as ds:
            item = ds[0]
        assert set(item.keys()) == {"states", "actions", "metadata"}


class TestTrajectoryDatasetDtype:
    def test_states_dtype_float32(self, tmp_path):
        from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset
        fpath = str(tmp_path / "test.h5")
        with TrajectoryDataset(fpath, mode="w") as ds:
            ds.write_batch([_make_traj(0)])
        with TrajectoryDataset(fpath, mode="r") as ds:
            item = ds[0]
        assert item["states"].dtype == np.float32

    def test_actions_dtype_float32(self, tmp_path):
        from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset
        fpath = str(tmp_path / "test.h5")
        with TrajectoryDataset(fpath, mode="w") as ds:
            ds.write_batch([_make_traj(0)])
        with TrajectoryDataset(fpath, mode="r") as ds:
            item = ds[0]
        assert item["actions"].dtype == np.float32


class TestTrajectoryDatasetAppend:
    def test_append_mode_increments_len(self, tmp_path):
        from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset
        fpath = str(tmp_path / "test.h5")
        with TrajectoryDataset(fpath, mode="w") as ds:
            ds.write_batch(_make_batch(5, seed_offset=0))
        with TrajectoryDataset(fpath, mode="a") as ds:
            ds.write_batch(_make_batch(5, seed_offset=5))
        with TrajectoryDataset(fpath, mode="r") as ds:
            assert len(ds) == 10


class TestTrajectoryDatasetContextManager:
    def test_context_manager_closes_file(self, tmp_path):
        import h5py
        from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset
        fpath = str(tmp_path / "test.h5")
        with TrajectoryDataset(fpath, mode="w") as ds:
            ds.write_batch([_make_traj(0)])
        # After context manager exits, re-opening should work (file properly closed)
        with h5py.File(fpath, "r") as f:
            assert "states" in f


class TestTrajectoryDatasetValuePreservation:
    def test_metadata_values_preserved(self, tmp_path):
        from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset
        fpath = str(tmp_path / "test.h5")
        known_meta = np.array([800.0, 75.0, 9.5], dtype=np.float32)
        traj = _make_traj(0)
        traj["metadata"] = known_meta
        with TrajectoryDataset(fpath, mode="w") as ds:
            ds.write_batch([traj])
        with TrajectoryDataset(fpath, mode="r") as ds:
            loaded_meta = ds[0]["metadata"]
        np.testing.assert_array_almost_equal(loaded_meta, known_meta)

    def test_states_values_preserved(self, tmp_path):
        from sco2rl.surrogate.trajectory_dataset import TrajectoryDataset
        fpath = str(tmp_path / "test.h5")
        traj = _make_traj(7)
        with TrajectoryDataset(fpath, mode="w") as ds:
            ds.write_batch([traj])
        with TrajectoryDataset(fpath, mode="r") as ds:
            loaded = ds[0]
        np.testing.assert_array_almost_equal(loaded["states"], traj["states"])
