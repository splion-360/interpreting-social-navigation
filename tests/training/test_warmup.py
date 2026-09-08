"""File description: Tests for tiny flat-model warm-up training."""

from pathlib import Path

import numpy as np

from social_nav.training.warmup import FlatWarmupConfig, run_flat_warmup


def write_mabe_file(path: Path) -> None:
    """Write a small MABe-style file for warm-up tests."""

    values = np.linspace(0.0, 1.0, num=12 * 3 * 12 * 2, dtype=np.float32)
    keypoints = values.reshape(12, 3, 12, 2)
    np.save(
        path,
        {
            "vocabulary": [],
            "sequences": {
                "seq": {
                    "keypoints": keypoints,
                },
            },
        },
    )


def test_flat_warmup_runs_short_cpu_smoke_test(tmp_path: Path) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_mabe_file(data_path)

    result = run_flat_warmup(
        FlatWarmupConfig(
            data_path=data_path,
            window_length=5,
            steps=3,
            learning_rate=1e-3,
            device="cpu",
        )
    )

    assert result.steps == 3
    assert result.device == "cpu"
    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)
