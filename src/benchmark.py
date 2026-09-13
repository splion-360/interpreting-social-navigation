"""File description: CLI orchestration for fair trajectory predictor benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from benchmark_report import (
    BenchmarkMetrics,
    build_benchmark_record,
    print_benchmark_tables,
)
from evaluate import (
    BaselineEvaluationResult,
    EvaluationResult,
    evaluate_flat_checkpoint,
    evaluate_motion_baselines,
    load_checkpoint_motion_thresholds,
    load_motion_baseline_config,
    load_test_config,
    resolve_baseline_window_config,
    save_evaluation_record,
)
from logging_utils import configure_cli_logging, get_logger
from train import load_flat_fit_config


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Paths and lineage settings for one fair predictor benchmark.

    Attributes:
        train_config_path: Training configuration matching the checkpoint.
        test_config_path: Held-out test configuration.
        baseline_config_path: Deterministic baseline evaluation configuration.
        checkpoint_path: Trained Social Attention checkpoint.
        results_path: JSONL destination for the combined benchmark.
        model_name: Display name for the trained method.
    """

    train_config_path: Path
    test_config_path: Path
    baseline_config_path: Path
    checkpoint_path: Path
    results_path: Path
    model_name: str = "social_attention"


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load paths for a learned-versus-deterministic benchmark.

    Args:
        path: YAML benchmark configuration.

    Returns:
        Typed benchmark configuration.
    """

    raw = yaml.safe_load(path.read_text())
    return BenchmarkConfig(
        train_config_path=Path(raw["train_config_path"]),
        test_config_path=Path(raw["test_config_path"]),
        baseline_config_path=Path(raw["baseline_config_path"]),
        checkpoint_path=Path(raw["checkpoint_path"]),
        results_path=Path(raw["results_path"]),
        model_name=raw.get("model_name", "social_attention"),
    )


def _comparable_baseline(result: BaselineEvaluationResult) -> BenchmarkMetrics:
    """Convert a deterministic result into the shared benchmark contract."""

    return BenchmarkMetrics(
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
) -> BenchmarkMetrics:
    """Convert a checkpoint result into the shared benchmark contract."""

    if result.metrics_by_motion is None or result.motion_profile is None:
        raise ValueError("checkpoint evaluation did not produce motion strata")
    return BenchmarkMetrics(
        name=name,
        windows=result.windows,
        metrics=result.metrics,
        metrics_by_motion=result.metrics_by_motion,
        motion_profile=result.motion_profile,
        window_digest=result.window_digest,
        runtime_seconds=runtime_seconds,
    )


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Evaluate deterministic methods and a trained checkpoint under one contract.

    Args:
        config: Paths defining the benchmark inputs and output ledger.

    Returns:
        Combined benchmark record.
    """

    train_config = load_flat_fit_config(config.train_config_path)
    test_config = load_test_config(config.test_config_path)
    baseline_config = load_motion_baseline_config(config.baseline_config_path)
    train_config = resolve_baseline_window_config(train_config, baseline_config)
    motion_thresholds = load_checkpoint_motion_thresholds(config.checkpoint_path)

    baseline_results = evaluate_motion_baselines(
        train_config=train_config,
        baseline_config=baseline_config,
        split="test",
        test_data_path=test_config.data_path,
        max_windows=baseline_config.max_windows,
        show_progress=True,
        motion_thresholds=motion_thresholds,
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
    record = build_benchmark_record(
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
    """Run the configured fair benchmark and print its metric tables."""

    configure_cli_logging()
    parser = argparse.ArgumentParser(
        description="Benchmark a trained trajectory model against motion baselines."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/benchmark__flat_dense_triplet_30fps.yml"),
    )
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    record = run_benchmark(config)
    print_benchmark_tables(record)
    LOGGER.info("result=%s", config.results_path)


if __name__ == "__main__":
    main()
