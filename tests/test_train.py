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
    ScheduledSamplingConfig,
    run_flat_fit,
    run_flat_warmup,
    scheduled_sampling_probability,
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


def test_scheduled_sampling_probability_follows_cosine_epoch_schedule() -> None:
    config = ScheduledSamplingConfig(
        enabled=True,
        start_epoch=6,
        end_epoch=40,
        max_feedback_probability=1.0,
        schedule="cosine",
    )

    assert scheduled_sampling_probability(config, epoch=1) == 0.0
    assert scheduled_sampling_probability(config, epoch=6) == 0.0
    assert scheduled_sampling_probability(config, epoch=10) == pytest.approx(
        0.0337638853
    )
    assert scheduled_sampling_probability(config, epoch=23) == pytest.approx(0.5)
    assert scheduled_sampling_probability(config, epoch=40) == 1.0
    assert scheduled_sampling_probability(config, epoch=50) == 1.0


def test_disabled_scheduled_sampling_always_uses_ground_truth() -> None:
    config = ScheduledSamplingConfig(
        enabled=False,
        start_epoch=6,
        end_epoch=40,
        max_feedback_probability=0.5,
    )

    assert scheduled_sampling_probability(config, epoch=40) == 0.0


def test_flat_fit_prints_resolved_device_info(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_mabe_file(data_path)
    caplog.set_level("INFO", logger=train.__name__)

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

    output = caplog.text
    assert "device: cpu" in output
    assert "torch_version:" not in output
    assert "cuda_available:" not in output


def test_scheduled_sampling_fit_logs_realized_feedback(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_mabe_file(data_path)
    logged_metrics: list[dict[str, float | int]] = []
    finite_gradient_steps: list[bool] = []
    original_step = torch.optim.Adam.step

    class FakeRun:
        def log(self, metrics: dict[str, float | int]) -> None:
            logged_metrics.append(metrics)

        def finish(self) -> None:
            return None

    monkeypatch.setattr(train, "_start_wandb", lambda *_args, **_kwargs: FakeRun())
    monkeypatch.setattr(
        train,
        "_wandb_log_validation_motion_table",
        lambda *_args, **_kwargs: None,
    )

    def check_gradients(
        optimizer: torch.optim.Adam,
        closure: Any | None = None,
    ) -> Any:
        """Record gradient finiteness before applying an optimizer step.

        Args:
            optimizer: Adam optimizer created by the training entry point.
            closure: Optional closure forwarded to the original optimizer step.

        Returns:
            Result returned by the original Adam step.
        """

        gradients = [
            parameter.grad
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        finite_gradient_steps.append(
            bool(gradients)
            and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        )
        return original_step(optimizer, closure)

    monkeypatch.setattr(torch.optim.Adam, "step", check_gradients)
    caplog.set_level("INFO", logger=train.__name__)

    result = run_flat_fit(
        FlatFitConfig(
            data_path=data_path,
            epochs=2,
            batch_size=3,
            window_length=5,
            observation_length=3,
            prediction_length=2,
            graph_variant="single_mouse_dense_keypoint",
            single_mouse_window_source="triplet",
            stride=5,
            max_train_windows=1,
            max_validation_windows=1,
            learning_rate=1e-3,
            scheduled_sampling=ScheduledSamplingConfig(
                enabled=True,
                start_epoch=1,
                end_epoch=2,
                max_feedback_probability=1.0,
            ),
            device="cpu",
            wandb=True,
            save_checkpoints=False,
        ),
        show_progress=False,
    )

    assert np.isfinite(result.final_train_loss)
    epoch_two = next(metrics for metrics in logged_metrics if metrics["epoch"] == 2)
    assert epoch_two["train/scheduled_sampling/feedback_probability"] == 1.0
    assert epoch_two["train/scheduled_sampling/realized_feedback_fraction"] == 1.0
    assert finite_gradient_steps == [True, True]
    assert "scheduled_sampling_probability=1.000" in caplog.text
    assert "realized_feedback_fraction=1.000" in caplog.text


def test_zero_probability_scheduled_sampling_matches_teacher_forcing(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_mabe_file(data_path)
    common = {
        "data_path": data_path,
        "epochs": 1,
        "batch_size": 3,
        "window_length": 5,
        "observation_length": 3,
        "prediction_length": 2,
        "graph_variant": "single_mouse_dense_keypoint",
        "single_mouse_window_source": "triplet",
        "stride": 5,
        "max_train_windows": 1,
        "max_validation_windows": 1,
        "device": "cpu",
        "wandb": False,
        "save_checkpoints": False,
        "seed": 42,
    }

    control = run_flat_fit(FlatFitConfig(**common), show_progress=False)
    scheduled = run_flat_fit(
        FlatFitConfig(
            **common,
            scheduled_sampling=ScheduledSamplingConfig(
                enabled=True,
                start_epoch=1,
                end_epoch=2,
                max_feedback_probability=1.0,
            ),
        ),
        show_progress=False,
    )

    assert scheduled.final_train_loss == control.final_train_loss
    assert scheduled.final_validation_loss == control.final_validation_loss


def test_scheduled_sampling_fit_is_deterministic_for_seed_42(tmp_path: Path) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_mabe_file(data_path)
    config = FlatFitConfig(
        data_path=data_path,
        epochs=2,
        batch_size=3,
        window_length=5,
        observation_length=3,
        prediction_length=2,
        graph_variant="single_mouse_dense_keypoint",
        single_mouse_window_source="triplet",
        stride=5,
        max_train_windows=1,
        max_validation_windows=1,
        scheduled_sampling=ScheduledSamplingConfig(
            enabled=True,
            start_epoch=1,
            end_epoch=2,
            max_feedback_probability=1.0,
        ),
        device="cpu",
        wandb=False,
        save_checkpoints=False,
        seed=42,
    )

    first = run_flat_fit(config, show_progress=False)
    second = run_flat_fit(config, show_progress=False)

    assert second.final_train_loss == first.final_train_loss
    assert second.final_validation_loss == first.final_validation_loss


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
        "flat_dense_triplet_30fps": (
            Path("src/config/train/flat_dense_triplet_30fps.yml"),
            "dense_keypoint",
            20,
            20,
            8,
            12,
            1,
            4,
        ),
        "flat_dense_triplet_5fps": (
            Path("src/config/train/flat_dense_triplet_5fps.yml"),
            "dense_keypoint",
            10,
            40,
            16,
            24,
            6,
            4,
        ),
        "flat_dense_single_mouse_5fps": (
            Path("src/config/train/flat_dense_single_mouse_5fps.yml"),
            "single_mouse_dense_keypoint",
            10,
            40,
            16,
            24,
            6,
            4,
        ),
        "flat_within_mouse_triplet_5fps": (
            Path("src/config/train/flat_within_mouse_triplet_5fps.yml"),
            "within_mouse_dense_keypoint",
            10,
            40,
            16,
            24,
            6,
            4,
        ),
        "flat_sparse_triplet_30fps": (
            Path("src/config/train/flat_sparse_triplet_30fps.yml"),
            "flat_sparse_keypoint",
            10,
            20,
            8,
            12,
            1,
            0,
        ),
        "flat_mouse_level_30fps": (
            Path("src/config/train/flat_mouse_level_30fps.yml"),
            "mouse_level",
            10,
            20,
            8,
            12,
            1,
            0,
        ),
    }

    assert train.DEFAULT_TRAIN_CONFIG_PATH == variants["flat_dense_triplet_30fps"][0]
    for variant, (
        path,
        graph_variant,
        checkpoint_frequency,
        window_length,
        observation_length,
        prediction_length,
        frame_step,
        workers,
    ) in variants.items():
        config = train.load_flat_fit_config(path)

        assert config.graph_variant == graph_variant
        assert config.window_length == window_length
        assert config.observation_length == observation_length
        assert config.prediction_length == prediction_length
        assert config.frame_step == frame_step
        assert config.workers == workers
        assert config.checkpoint_frequency == checkpoint_frequency
        assert config.motion_score == "mean_keypoint_speed_px_s"
        assert config.motion_sampling is (
            variant.startswith("flat_dense_triplet")
            or variant.startswith("flat_dense_single_mouse")
            or variant.startswith("flat_within_mouse_triplet")
        )


def test_matched_single_mouse_config_matches_dense_update_budget() -> None:
    config = train.load_flat_fit_config(
        Path("src/config/train/flat_dense_single_mouse_matched_5fps.yml")
    )

    assert config.graph_variant == "single_mouse_dense_keypoint"
    assert config.single_mouse_window_source == "triplet"
    assert config.max_train_windows == 800
    assert config.max_validation_windows == 200
    assert config.batch_size == 6
    assert config.seed == 42


def test_scheduled_sampling_config_preserves_matched_single_mouse_contract() -> None:
    config = train.load_flat_fit_config(
        Path("src/config/train/flat_dense_single_mouse_scheduled_sampling_5fps.yml")
    )

    assert config.graph_variant == "single_mouse_dense_keypoint"
    assert config.single_mouse_window_source == "triplet"
    assert config.batch_size == 6
    assert config.seed == 42
    assert config.scheduled_sampling == ScheduledSamplingConfig(
        enabled=True,
        start_epoch=6,
        end_epoch=40,
        max_feedback_probability=1.0,
        schedule="cosine",
        feedback="sample",
    )
    assert config.wandb_group == "social-attention-scheduled-sampling"
    assert config.wandb_run_name == ("single-mouse-5fps-scheduled-sampling-cosine-p100")
    assert config.wandb_job_type == "training"
    assert config.wandb_tags == (
        "single-mouse",
        "5fps",
        "scheduled-sampling",
    )


def test_within_mouse_config_matches_dense_optimizer_budget() -> None:
    dense = train.load_flat_fit_config(
        Path("src/config/train/flat_dense_triplet_5fps.yml")
    )
    within_mouse = train.load_flat_fit_config(
        Path("src/config/train/flat_within_mouse_triplet_5fps.yml")
    )

    assert within_mouse.batch_size == dense.batch_size == 2
    assert within_mouse.epochs == dense.epochs == 50
    assert within_mouse.max_train_windows == dense.max_train_windows == 800
    assert within_mouse.seed == dense.seed == 42


def test_show_flat_fit_setup_prints_data_and_training_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_mabe_file(data_path, sequences=3)
    caplog.set_level("INFO", logger=train.__name__)

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
            scheduled_sampling=ScheduledSamplingConfig(
                enabled=True,
                start_epoch=6,
                end_epoch=40,
                max_feedback_probability=1.0,
                schedule="cosine",
            ),
            wandb_group="social-attention-scheduled-sampling",
            save_checkpoints=False,
        )
    )

    output = caplog.text
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
    assert "scheduled_sampling:" in output
    assert "max_feedback_probability: 1.0" in output
    assert "schedule: cosine" in output
    assert "group: social-attention-scheduled-sampling" in output


def test_motion_sampling_selects_train_windows_after_grouping(tmp_path: Path) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_motion_mabe_file(data_path)

    window_data = train.build_training_window_data(
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


def test_single_mouse_fit_windows_split_sources_before_mouse_extraction(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_motion_mabe_file(data_path, sequences=6, frames=36)

    window_data = train.build_training_window_data(
        FlatFitConfig(
            data_path=data_path,
            graph_variant="single_mouse_dense_keypoint",
            window_length=6,
            observation_length=3,
            prediction_length=3,
            stride=3,
            max_train_windows=6,
            max_validation_windows=3,
            validation_fraction=0.33,
            seed=42,
            device="cpu",
        )
    )
    train_sources = {
        sequence_id.rsplit("__mouse_", maxsplit=1)[0]
        for sequence_id in window_data.train_windows.sequences
    }
    validation_sources = {
        sequence_id.rsplit("__mouse_", maxsplit=1)[0]
        for sequence_id in window_data.validation_windows.sequences
    }

    assert train_sources.isdisjoint(validation_sources)
    assert all(
        window.keypoints.shape == (6, 1, 12, 2)
        for window in (
            window_data.train_windows[0],
            window_data.validation_windows[0],
        )
    )


def test_single_mouse_matched_sampling_expands_dense_triplet_windows(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "mouse_triplet_train.npy"
    write_motion_mabe_file(data_path, sequences=6, frames=36)
    common = {
        "data_path": data_path,
        "window_length": 6,
        "observation_length": 3,
        "prediction_length": 3,
        "stride": 3,
        "max_train_windows": 6,
        "max_validation_windows": 3,
        "validation_fraction": 0.33,
        "motion_sampling": True,
        "seed": 42,
        "device": "cpu",
    }

    dense = train.build_training_window_data(FlatFitConfig(**common))
    single = train.build_training_window_data(
        FlatFitConfig(
            **common,
            graph_variant="single_mouse_dense_keypoint",
            single_mouse_window_source="triplet",
        )
    )

    expected_train_keys = tuple(
        (f"{sequence_id}__mouse_{mouse_index}", start_frame)
        for sequence_id, start_frame in dense.train_windows.window_keys
        for mouse_index in range(3)
    )
    expected_validation_keys = tuple(
        (f"{sequence_id}__mouse_{mouse_index}", start_frame)
        for sequence_id, start_frame in dense.validation_windows.window_keys
        for mouse_index in range(3)
    )
    assert single.train_windows.window_keys == expected_train_keys
    assert single.validation_windows.window_keys == expected_validation_keys
    assert single.motion_report is not None
    assert dense.motion_report is not None
    assert single.motion_report.train_selected.labels == tuple(
        label for label in dense.motion_report.train_selected.labels for _ in range(3)
    )
    assert single.motion_report.validation.labels == tuple(
        label for label in dense.motion_report.validation.labels for _ in range(3)
    )


def test_validation_metric_abbreviations_keep_agreed_scalars() -> None:
    metrics = train._abbreviated_validation_metrics(
        {
            "centroid_ade_px": 1.0,
            "centroid_fde_px": 2.0,
            "keypoint_ade_px": 3.0,
            "keypoint_fde_px": 4.0,
            "centroid_velocity_error_px_s": 5.0,
            "keypoint_velocity_error_px_s": 6.0,
            "body_heading_error_deg": 7.0,
            "body_frame_keypoint_ade_px": 8.0,
            "body_frame_keypoint_fde_px": 9.0,
            "bone_length_error_px": 10.0,
            "skeleton_orientation_error_deg": 11.0,
            "relative_ordering_error": 12.0,
            "edge_angle_error_deg_by_mouse": np.zeros((3, 12, 12)),
        }
    )

    assert metrics == {
        "cADE": 1.0,
        "cFDE": 2.0,
        "kADE": 3.0,
        "kFDE": 4.0,
        "CVE": 5.0,
        "KVE": 6.0,
        "BHE": 7.0,
        "BFK-ADE": 8.0,
        "BFK-FDE": 9.0,
        "BLE": 10.0,
        "SOE": 11.0,
        "ROE": 12.0,
    }


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


def test_wandb_run_uses_configured_group_job_type_and_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_arguments: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> object:
        init_arguments.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    device_info = train._device_info("cpu", torch.device("cpu"))

    train._start_wandb(
        FlatFitConfig(
            wandb=True,
            wandb_group="social-attention-scheduled-sampling-5fps",
            wandb_job_type="training",
            wandb_tags=("single-mouse", "scheduled-sampling"),
        ),
        device_info,
    )

    assert init_arguments["group"] == "social-attention-scheduled-sampling-5fps"
    assert init_arguments["job_type"] == "training"
    assert init_arguments["tags"] == ("single-mouse", "scheduled-sampling")


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


def test_wandb_motion_metric_table_logs_epoch_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = []
    logs = []

    class FakeTable:
        def __init__(self, columns: list[str]) -> None:
            self.columns = columns
            self.rows = []
            tables.append(self)

        def add_data(self, *values: Any) -> None:
            self.rows.append(values)

    class FakeRun:
        def log(self, values: dict[str, Any]) -> None:
            logs.append(values)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Table=FakeTable))
    table_history = train.ValidationMotionMetricTable.empty()

    train._wandb_log_validation_motion_table(
        FakeRun(),
        epoch=1,
        loss_summary=EpochLossSummary(
            loss=1.0,
            loss_by_motion={"low": 0.5},
            counts_by_motion={"low": 2, "medium": 0, "high": 0},
        ),
        metric_summary=train.ValidationMetricSummary(
            metrics={"cADE": 1.0},
            metrics_by_motion={"low": {"cADE": 1.0, "cFDE": 2.0}},
            counts_by_motion={"low": 2, "medium": 0, "high": 0},
        ),
        table_history=table_history,
    )
    train._wandb_log_validation_motion_table(
        FakeRun(),
        epoch=2,
        loss_summary=EpochLossSummary(
            loss=1.0,
            loss_by_motion={"low": 0.4},
            counts_by_motion={"low": 2, "medium": 0, "high": 0},
        ),
        metric_summary=train.ValidationMetricSummary(
            metrics={"cADE": 0.8},
            metrics_by_motion={"low": {"cADE": 0.8, "cFDE": 1.5}},
            counts_by_motion={"low": 2, "medium": 0, "high": 0},
        ),
        table_history=table_history,
    )

    assert tables[0].columns[:4] == ["epoch", "motion_group", "windows", "loss"]
    assert "cADE" in tables[0].columns
    assert tables[0].rows[0][:4] == (1, "low", 2, 0.5)
    assert logs[0]["validation/motion_metrics_table"] is tables[0]
    assert tables[1].rows[0][:4] == (1, "low", 2, 0.5)
    assert tables[1].rows[1][:4] == (2, "low", 2, 0.4)
    assert logs[1]["validation/motion_metrics_table"] is tables[1]


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
