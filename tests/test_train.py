"""File description: Tests for trajectory training commands."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import train
from st_graph import build_dense_keypoint_graph
from train import (
    EpochLossSummary,
    FlatFitConfig,
    FlatWarmupConfig,
    run_flat_fit,
    run_flat_warmup,
)


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


def write_motion_mabe_file(path: Path, sequences: int = 6, frames: int = 36) -> None:
    """Write MABe-style sequences with different future motion speeds."""

    payload: dict[str, Any] = {"vocabulary": [], "sequences": {}}
    for sequence_idx in range(sequences):
        values = np.zeros((frames, 3, 12, 2), dtype=np.float32)
        speed = float(sequence_idx + 1)
        values[..., 0] = np.arange(frames, dtype=np.float32).reshape(-1, 1, 1) * speed
        values[..., 1] = float(sequence_idx)
        payload["sequences"][f"seq_{sequence_idx}"] = {"keypoints": values}
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
            observation_length=4,
            prediction_length=1,
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


def test_flat_fit_prints_resolved_device_info(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_mabe_file(data_path)

    run_flat_fit(
        FlatFitConfig(
            data_path=data_path,
            epochs=1,
            batch_size=1,
            window_length=5,
            observation_length=4,
            prediction_length=1,
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

    output = capsys.readouterr().out
    assert "device: cpu" in output
    assert "torch_version:" not in output
    assert "cuda_available:" not in output


def test_flat_fit_config_loads_yaml_with_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yml"
    config_path.write_text(
        "\n".join(
            [
                "model: flat",
                f"data_path: {tmp_path / 'data.npy'}",
                "epochs: 7",
                "batch_size: 4",
                "window_length: 6",
                "observation_length: 4",
                "prediction_length: 2",
                "graph_variant: flat_sparse_keypoint",
                "stride: 2",
                "validation_fraction: 0.25",
                "motion_sampling: true",
                "motion_score: mean_keypoint_speed_px_s",
                "motion_group_mix:",
                "  low: 0.2",
                "  medium: 0.4",
                "  high: 0.4",
                "learning_rate: 0.002",
                "grad_clip: 5.0",
                "seed: 123",
                "device: cpu",
                "wandb: false",
                "wandb_project: interpreting-social-navigation",
                "checkpoint_dir: checkpoints/test",
                "save_checkpoints: true",
                "checkpoint_frequency: 10",
                "wandb_artifact_name: test-artifact",
                "resume_checkpoint:",
                "resume_wandb_artifact:",
                "resume_download_dir: checkpoints/wandb",
            ]
        )
    )

    config = train.load_flat_fit_config(
        config_path,
        overrides={"epochs": 3, "wandb": True},
    )

    assert config.epochs == 3
    assert config.batch_size == 4
    assert config.data_path == tmp_path / "data.npy"
    assert config.observation_length == 4
    assert config.prediction_length == 2
    assert config.graph_variant == "flat_sparse_keypoint"
    assert config.checkpoint_frequency == 10
    assert config.motion_sampling is True
    assert config.motion_group_mix == {"low": 0.2, "medium": 0.4, "high": 0.4}
    assert config.wandb is True


def test_variant_train_configs_load_from_src_config() -> None:
    variants = {
        "dense_keypoint": (Path("src/config/dense_keypoint__train.yml"), 20),
        "flat_sparse_keypoint": (
            Path("src/config/flat_sparse_keypoint__train.yml"),
            10,
        ),
        "mouse_level": (Path("src/config/mouse_level__train.yml"), 10),
    }

    assert train.DEFAULT_TRAIN_CONFIG_PATH == variants["dense_keypoint"][0]
    for variant, (path, checkpoint_frequency) in variants.items():
        config = train.load_flat_fit_config(path)

        assert config.graph_variant == variant
        assert config.window_length == 20
        assert config.observation_length == 8
        assert config.prediction_length == 12
        assert config.checkpoint_frequency == checkpoint_frequency
        assert config.motion_score == "mean_keypoint_speed_px_s"
        assert config.motion_sampling is (variant == "dense_keypoint")


def test_show_flat_fit_setup_prints_data_and_training_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_mabe_file(data_path, sequences=3)

    train.show_flat_fit_setup(
        FlatFitConfig(
            data_path=data_path,
            window_length=5,
            observation_length=4,
            prediction_length=1,
            stride=5,
            max_train_windows=2,
            max_validation_windows=1,
            device="cpu",
            save_checkpoints=False,
        )
    )

    output = capsys.readouterr().out
    assert "normalization:" in output
    assert "scale_xy:" in output
    assert "variant: dense_keypoint" in output
    assert "edge_count: 1296" in output
    assert "train_windows: 2" in output
    assert "validation_windows: 1" in output
    assert "loss: bivariate_gaussian_horizon_nll" in output
    assert "optimizer: Adam" in output
    assert "resume_checkpoint: null" in output
    assert "resume_wandb_artifact: null" in output
    assert "device: cpu" in output
    assert "checkpoint_frequency: null" in output


def test_motion_sampling_selects_train_windows_after_grouping(tmp_path: Path) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_motion_mabe_file(data_path)

    window_data = train._build_training_window_data(
        FlatFitConfig(
            data_path=data_path,
            window_length=6,
            observation_length=3,
            prediction_length=3,
            stride=3,
            max_train_windows=10,
            max_validation_windows=4,
            validation_fraction=0.33,
            motion_sampling=True,
            seed=42,
            device="cpu",
        )
    )

    assert len(window_data.train_windows) == 10
    assert window_data.motion_report is not None
    assert window_data.motion_report.train_selected.counts == {
        "low": 2,
        "medium": 4,
        "high": 4,
    }
    assert sum(window_data.motion_report.validation.counts.values()) == 4
    assert (
        window_data.motion_report.train_candidates.thresholds.low_max_px_s
        <= window_data.motion_report.train_candidates.thresholds.medium_max_px_s
    )


def test_device_info_reports_cpu_without_gpu_name() -> None:
    info = train._device_info("cpu", torch.device("cpu"))

    assert info.requested == "cpu"
    assert info.resolved == "cpu"
    assert info.cuda_device_name is None


def test_flat_fit_saves_periodic_epoch_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    checkpoint_dir = tmp_path / "checkpoints"
    write_mabe_file(data_path)

    def fake_run_epoch(*, epoch: int, split: str, **_: Any) -> EpochLossSummary:
        if split == "train":
            return EpochLossSummary(
                loss=1.0,
                loss_by_motion={},
                counts_by_motion={},
            )
        return EpochLossSummary(
            loss=1.0 if epoch == 1 else 0.5,
            loss_by_motion={},
            counts_by_motion={},
        )

    monkeypatch.setattr(train, "_run_epoch", fake_run_epoch)

    run_flat_fit(
        FlatFitConfig(
            data_path=data_path,
            epochs=2,
            batch_size=1,
            window_length=5,
            observation_length=4,
            prediction_length=1,
            stride=5,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=2,
            max_train_windows=1,
            max_validation_windows=1,
            device="cpu",
            wandb=False,
        ),
        show_progress=False,
    )

    assert (checkpoint_dir / "flat_best.pt").exists()
    assert (checkpoint_dir / "flat_epoch_0002.pt").exists()


def test_nodes_present_mask_matches_graph_metadata() -> None:
    keypoints = np.zeros((3, 3, 12, 2), dtype=np.float32)
    graph = build_dense_keypoint_graph(keypoints)

    mask = train._nodes_present_mask(graph)

    assert mask.shape == (3, 36)
    assert mask.all()


def test_wandb_is_not_started_when_flag_is_disabled() -> None:
    device_info = train._device_info("cpu", torch.device("cpu"))

    assert train._start_wandb(FlatFitConfig(wandb=False), device_info) is None


def test_wandb_flag_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = __import__

    def fail_wandb_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "wandb":
            raise ImportError("missing wandb")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_wandb_import)

    device_info = train._device_info("cpu", torch.device("cpu"))
    with pytest.raises(RuntimeError, match="optional dependency"):
        train._start_wandb(FlatFitConfig(wandb=True), device_info)


def test_wandb_checkpoint_logging_is_skipped_without_run(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "flat_best.pt"
    checkpoint_path.write_text("checkpoint")

    train._wandb_log_checkpoint(
        run=None,
        checkpoint_path=checkpoint_path,
        artifact_name="flat-best-checkpoint",
        epoch=1,
        validation_loss=1.5,
        aliases=["best", "epoch-1"],
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

    class FakeLoggedArtifact:
        def wait(self) -> None:
            return None

    class FakeRun:
        def log_artifact(
            self, artifact: FakeArtifact, aliases: list[str]
        ) -> FakeLoggedArtifact:
            logged_artifacts.append((artifact, aliases))
            return FakeLoggedArtifact()

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
        aliases=["best", "epoch-2"],
    )

    artifact, aliases = logged_artifacts[0]
    assert artifact.name == "flat-best-checkpoint"
    assert artifact.type == "model"
    assert artifact.metadata == {
        "epoch": 2,
        "validation_loss": 0.75,
        "motion_profile": None,
    }
    assert added_files == [str(checkpoint_path)]
    assert aliases == ["best", "epoch-2"]


def test_resolve_resume_checkpoint_prefers_local_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "flat_best.pt"
    checkpoint_path.write_text("checkpoint")

    resolved = train._resolve_resume_checkpoint(
        FlatFitConfig(resume_checkpoint=checkpoint_path),
        run=None,
    )

    assert resolved == checkpoint_path


def test_resolve_resume_checkpoint_downloads_wandb_artifact(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    checkpoint_path = artifact_dir / "flat_best.pt"
    checkpoint_path.write_text("checkpoint")

    class FakeArtifact:
        def download(self, root: str) -> str:
            assert root == str(tmp_path / "wandb")
            return str(artifact_dir)

    class FakeRun:
        def use_artifact(self, name: str, type: str) -> FakeArtifact:
            assert name == "flat-best-checkpoint:best"
            assert type == "model"
            return FakeArtifact()

    resolved = train._resolve_resume_checkpoint(
        FlatFitConfig(
            wandb=True,
            resume_wandb_artifact="flat-best-checkpoint:best",
            resume_download_dir=tmp_path / "wandb",
        ),
        run=FakeRun(),
    )

    assert resolved == checkpoint_path


def test_wandb_resume_requires_wandb_run() -> None:
    with pytest.raises(RuntimeError, match="requires --wandb"):
        train._resolve_resume_checkpoint(
            FlatFitConfig(resume_wandb_artifact="flat-best-checkpoint:best"),
            run=None,
        )
