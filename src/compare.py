"""File description: CLI orchestration for fair trajectory predictor comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from comparison import (
    ComparableMetrics,
    build_comparison_record,
    print_comparison_tables,
)
from evaluate import (
    BaselineEvaluationResult,
    EvaluationResult,
    evaluate_flat_checkpoint,
    evaluate_motion_baselines,
    load_motion_baseline_config,
    load_test_config,
    resolve_baseline_window_config,
    save_evaluation_record,
)
from train import load_flat_fit_config


@dataclass(frozen=True)
class ComparisonConfig:
    """Paths and lineage settings for one fair predictor comparison.

    Attributes:
        train_config_path: Training configuration matching the checkpoint.
        test_config_path: Held-out test configuration.
        baseline_config_path: Deterministic baseline evaluation configuration.
        checkpoint_path: Trained Social Attention checkpoint.
        results_path: JSONL destination for the combined comparison.
        model_name: Display name for the trained method.
    """

    train_config_path: Path
    test_config_path: Path
    baseline_config_path: Path
    checkpoint_path: Path
    results_path: Path
    model_name: str = "social_attention"


def load_comparison_config(path: Path) -> ComparisonConfig:
    """Load paths for a learned-versus-deterministic comparison.

    Args:
        path: YAML comparison configuration.

    Returns:
        Typed comparison configuration.
    """

    raw = yaml.safe_load(path.read_text())
    return ComparisonConfig(
        train_config_path=Path(raw["train_config_path"]),
        test_config_path=Path(raw["test_config_path"]),
        baseline_config_path=Path(raw["baseline_config_path"]),
        checkpoint_path=Path(raw["checkpoint_path"]),
        results_path=Path(raw["results_path"]),
        model_name=raw.get("model_name", "social_attention"),
    )


def _comparable_baseline(result: BaselineEvaluationResult) -> ComparableMetrics:
    """Convert a deterministic result into the shared comparison contract."""

    return ComparableMetrics(
        name=result.baseline,
        windows=result.windows,
        metrics=result.metrics,
        metrics_by_motion=result.metrics_by_motion,
        motion_profile=result.motion_profile,
        window_digest=result.window_digest,
        runtime_seconds=result.evaluation_seconds,
    )


def _comparable_model(
    result: EvaluationResult,
    *,
    name: str,
    runtime_seconds: float,
) -> ComparableMetrics:
    """Convert a checkpoint result into the shared comparison contract."""

    if result.metrics_by_motion is None or result.motion_profile is None:
        raise ValueError("checkpoint evaluation did not produce motion strata")
    return ComparableMetrics(
        name=name,
        windows=result.windows,
        metrics=result.metrics,
        metrics_by_motion=result.metrics_by_motion,
        motion_profile=result.motion_profile,
        window_digest=result.window_digest,
        runtime_seconds=runtime_seconds,
    )


def run_comparison(config: ComparisonConfig) -> dict[str, Any]:
    """Evaluate deterministic methods and a trained checkpoint under one contract.

    Args:
        config: Paths defining the comparison inputs and output ledger.

    Returns:
        Combined comparison record.
    """

    train_config = load_flat_fit_config(config.train_config_path)
    test_config = load_test_config(config.test_config_path)
    baseline_config = load_motion_baseline_config(config.baseline_config_path)
    train_config = resolve_baseline_window_config(train_config, baseline_config)

    baseline_results = evaluate_motion_baselines(
        train_config=train_config,
        baseline_config=baseline_config,
        split="test",
        test_data_path=test_config.data_path,
        max_windows=baseline_config.max_windows,
        show_progress=True,
    )
    model_started = perf_counter()
    model_result = evaluate_flat_checkpoint(
        config=train_config,
        checkpoint_path=config.checkpoint_path,
        split="test",
        test_data_path=test_config.data_path,
        max_windows=baseline_config.max_windows,
        seed=baseline_config.seed,
        show_progress=True,
    )
    methods = [_comparable_baseline(result) for result in baseline_results]
    methods.append(
        _comparable_model(
            model_result,
            name=config.model_name,
            runtime_seconds=perf_counter() - model_started,
        )
    )
    record = build_comparison_record(
        methods=methods,
        observation_length=train_config.observation_length,
        prediction_length=train_config.prediction_length,
        frame_step=train_config.frame_step,
        source_fps=train_config.source_fps,
        seed=baseline_config.seed,
    )
    record["lineage"] = {
        "train_config_path": str(config.train_config_path),
        "test_config_path": str(config.test_config_path),
        "baseline_config_path": str(config.baseline_config_path),
        "checkpoint_path": str(config.checkpoint_path),
        "checkpoint_epoch": model_result.checkpoint_epoch,
        "checkpoint_validation_loss": model_result.checkpoint_validation_loss,
        "test_data_path": str(test_config.data_path),
    }
    save_evaluation_record(record, config.results_path)
    return record


def main() -> None:
    """Run the configured fair comparison and print its metric tables."""

    parser = argparse.ArgumentParser(
        description="Compare a trained trajectory model with motion baselines."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/dense_keypoint__compare.yml"),
    )
    args = parser.parse_args()
    config = load_comparison_config(args.config)
    record = run_comparison(config)
    print_comparison_tables(record)
    print(f"result={config.results_path}")


if __name__ == "__main__":
    main()
