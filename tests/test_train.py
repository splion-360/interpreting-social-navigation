"""File description: Tests for trajectory training commands."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import train
from train import FlatFitConfig, FlatWarmupConfig, run_flat_fit, run_flat_warmup


def write_mabe_file(path: Path, sequences: int = 2) -> None:
    """Write a small MABe-style file for training tests."""

    values = np.linspace(0.0, 1.0, num=12 * 3 * 12 * 2, dtype=np.float32)
    payload: dict[str, Any] = {
        "vocabulary": [],
        "sequences": {
            f"seq_{idx}": {
                "keypoints": values.reshape(12, 3, 12, 2) + idx,
            }
            for idx in range(sequences)
        },
    }
    np.save(path, np.asarray(payload, dtype=object))


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


def test_flat_fit_runs_one_epoch_without_wandb_or_checkpoints(tmp_path: Path) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_mabe_file(data_path)

    result = run_flat_fit(
        FlatFitConfig(
            data_path=data_path,
            epochs=1,
            batch_size=1,
            window_length=5,
            stride=5,
            max_train_windows=1,
            max_validation_windows=1,
            learning_rate=1e-3,
            device="cpu",
            wandb=False,
            save_checkpoints=False,
        ),
        show_progress=False,
    )

    assert result.best_epoch == 1
    assert result.checkpoint_path is None
    assert np.isfinite(result.final_train_loss)
    assert np.isfinite(result.final_validation_loss)


def test_wandb_is_not_started_when_flag_is_disabled() -> None:
    assert train._start_wandb(FlatFitConfig(wandb=False)) is None


def test_wandb_flag_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = __import__

    def fail_wandb_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "wandb":
            raise ImportError("missing wandb")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_wandb_import)

    with pytest.raises(RuntimeError, match="optional dependency"):
        train._start_wandb(FlatFitConfig(wandb=True))


def test_wandb_checkpoint_logging_is_skipped_without_run(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "flat_best.pt"
    checkpoint_path.write_text("checkpoint")

    train._wandb_log_checkpoint(
        run=None,
        checkpoint_path=checkpoint_path,
        artifact_name="flat-best-checkpoint",
        epoch=1,
        validation_loss=1.5,
    )


def test_wandb_checkpoint_logging_uploads_model_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "flat_best.pt"
    checkpoint_path.write_text("checkpoint")
    added_files = []
    logged_artifacts = []

    class FakeArtifact:
        def __init__(
            self,
            name: str,
            type: str,
            metadata: dict[str, int | float],
        ) -> None:
            self.name = name
            self.type = type
            self.metadata = metadata

        def add_file(self, path: str) -> None:
            added_files.append(path)

    class FakeRun:
        def log_artifact(self, artifact: FakeArtifact, aliases: list[str]) -> None:
            logged_artifacts.append((artifact, aliases))

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Artifact=FakeArtifact),
    )

    train._wandb_log_checkpoint(
        run=FakeRun(),
        checkpoint_path=checkpoint_path,
        artifact_name="flat-best-checkpoint",
        epoch=2,
        validation_loss=0.75,
    )

    artifact, aliases = logged_artifacts[0]
    assert artifact.name == "flat-best-checkpoint"
    assert artifact.type == "model"
    assert artifact.metadata == {"epoch": 2, "validation_loss": 0.75}
    assert added_files == [str(checkpoint_path)]
    assert aliases == ["best", "epoch-2"]
