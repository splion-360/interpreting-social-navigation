"""File description: Fair held-out comparisons of learned and deterministic predictors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from rich import box
from rich.console import Console
from rich.table import Table

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


MOTION_STRATA = ("low", "medium", "high")
ERROR_METRIC_NAMES = (
    "centroid_ade_px",
    "centroid_fde_px",
    "keypoint_ade_px",
    "keypoint_fde_px",
    "centroid_velocity_error_px_per_frame",
    "keypoint_velocity_error_px_per_frame",
    "displacement_magnitude_error_px",
    "displacement_direction_error_deg",
    "skeleton_orientation_error_deg",
    "bone_length_error_px",
    "body_heading_error_deg",
    "body_frame_keypoint_ade_px",
    "body_frame_keypoint_fde_px",
    "relative_ordering_error",
    "relative_ordering_error_forward",
    "relative_ordering_error_lateral",
)


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


@dataclass(frozen=True)
class ComparableMetrics:
    """Metrics from one method evaluated on a shared window collection.

    Attributes:
        name: Method name used in tables and result records.
        windows: Number of evaluated windows.
        metrics: Aggregate metrics over every window.
        metrics_by_motion: Aggregate metrics by ground-truth motion stratum.
        motion_profile: Shared stratum thresholds and counts.
        runtime_seconds: Wall-clock evaluation time when available.
    """

    name: str
    windows: int
    metrics: dict[str, Any]
    metrics_by_motion: dict[str, dict[str, Any]]
    motion_profile: dict[str, Any]
    runtime_seconds: float | None = None


def relative_error_improvements(
    *,
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, float | None]:
    """Calculate lower-is-better improvements relative to a reference method.

    Args:
        candidate: Metrics for the method being compared.
        reference: Persistence metrics used as the denominator.

    Returns:
        Fractional improvements for recognized scalar errors. Positive values
        indicate lower error than persistence. Undefined references return ``None``.
    """

    improvements: dict[str, float | None] = {}
    for name in ERROR_METRIC_NAMES:
        candidate_value = candidate.get(name)
        reference_value = reference.get(name)
        if (
            not isinstance(candidate_value, int | float)
            or not isinstance(reference_value, int | float)
            or not isfinite(candidate_value)
            or not isfinite(reference_value)
        ):
            improvements[name] = None
        elif reference_value == 0.0:
            improvements[name] = None
        else:
            improvements[name] = (reference_value - candidate_value) / reference_value
    return improvements


def build_comparison_record(
    *,
    methods: list[ComparableMetrics],
    observation_length: int,
    prediction_length: int,
    seed: int,
) -> dict[str, Any]:
    """Build a comparison record after verifying the shared evaluation contract.

    Args:
        methods: Results for persistence, learned, and other baseline methods.
        observation_length: Number of input frames used by every method.
        prediction_length: Number of future frames predicted by every method.
        seed: Evaluation seed shared by stochastic methods.

    Returns:
        JSON-ready result with raw metrics and improvements over persistence.

    Raises:
        ValueError: If persistence is absent or methods used different windows or
            motion-stratification profiles.
    """

    by_name = {method.name: method for method in methods}
    if "persistence" not in by_name:
        raise ValueError("comparison requires persistence as its reference")

    reference = by_name["persistence"]
    for method in methods:
        if method.windows != reference.windows:
            raise ValueError("comparison methods evaluated different window counts")
        if method.motion_profile != reference.motion_profile:
            raise ValueError("comparison methods used different motion strata")

    method_records: dict[str, Any] = {}
    for method in methods:
        improvement_by_motion = {
            stratum: relative_error_improvements(
                candidate=method.metrics_by_motion[stratum],
                reference=reference.metrics_by_motion[stratum],
            )
            for stratum in MOTION_STRATA
        }
        method_records[method.name] = {
            "metrics": method.metrics,
            "metrics_by_motion": method.metrics_by_motion,
            "relative_improvement_over_persistence": {
                "overall": relative_error_improvements(
                    candidate=method.metrics,
                    reference=reference.metrics,
                ),
                "by_motion": improvement_by_motion,
            },
            "runtime_seconds": method.runtime_seconds,
        }

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "reference": "persistence",
        "window_contract": {
            "observation_length": observation_length,
            "prediction_length": prediction_length,
            "windows": reference.windows,
            "seed": seed,
        },
        "motion_profile": reference.motion_profile,
        "methods": method_records,
    }


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


def print_comparison_tables(
    record: dict[str, Any],
    *,
    console: Console | None = None,
) -> None:
    """Print raw errors and improvements for overall and stratified results.

    Args:
        record: Comparison record returned by :func:`build_comparison_record`.
        console: Optional Rich console.
    """

    output = console or Console()
    method_names = tuple(record["methods"])
    groups = ("overall", *MOTION_STRATA)
    for group in groups:
        table = Table(title=f"8 -> 12 comparison: {group}", box=box.SIMPLE_HEAVY)
        table.add_column("metric", style="cyan", no_wrap=True)
        for method_name in method_names:
            table.add_column(method_name, justify="right")
        for metric_name in ERROR_METRIC_NAMES:
            values: list[str] = []
            has_value = False
            for method_name in method_names:
                method = record["methods"][method_name]
                metrics = (
                    method["metrics"]
                    if group == "overall"
                    else method["metrics_by_motion"][group]
                )
                improvement = method["relative_improvement_over_persistence"]
                improvement = (
                    improvement["overall"]
                    if group == "overall"
                    else improvement["by_motion"][group]
                )
                value = metrics.get(metric_name)
                gain = improvement.get(metric_name)
                if isinstance(value, int | float) and isfinite(value):
                    has_value = True
                    suffix = f" ({gain:+.1%})" if isinstance(gain, int | float) else ""
                    values.append(f"{value:.3f}{suffix}")
                else:
                    values.append("-")
            if has_value:
                table.add_row(metric_name, *values)
        output.print(table)


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
