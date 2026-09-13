"""File description: Metric contracts and reports for predictor comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table

from metric import PRIMARY_METRIC_NAMES, RELATIVE_ERROR_METRIC_NAMES


MOTION_STRATA = ("low", "medium", "high")


@dataclass(frozen=True)
class ComparableMetrics:
    """Metrics from one method evaluated on a shared window collection.

    Attributes:
        name: Method name used in tables and result records.
        windows: Number of evaluated windows.
        metrics: Aggregate metrics over every window.
        metrics_by_motion: Aggregate metrics by ground-truth motion stratum.
        motion_profile: Shared stratum thresholds and counts.
        window_digest: Fingerprint of selected sequence/start-frame keys.
        runtime_seconds: Wall-clock evaluation time when available.
    """

    name: str
    windows: int
    metrics: dict[str, Any]
    metrics_by_motion: dict[str, dict[str, Any]]
    motion_profile: dict[str, Any]
    window_digest: str | None
    runtime_seconds: float | None = None


def relative_error_improvements(
    *,
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, float | None]:
    """Calculate lower-is-better improvements relative to persistence.

    Args:
        candidate: Metrics for the method being compared.
        reference: Persistence metrics used as the denominator.

    Returns:
        Fractional improvements for every primary scalar metric. Positive values
        indicate lower error than persistence. Metrics without a meaningful
        relative-error interpretation are retained with a ``None`` value.
    """

    improvements: dict[str, float | None] = {}
    for name in PRIMARY_METRIC_NAMES:
        candidate_value = candidate.get(name)
        reference_value = reference.get(name)
        if name not in RELATIVE_ERROR_METRIC_NAMES:
            improvements[name] = None
        elif (
            not isinstance(candidate_value, int | float)
            or not isinstance(reference_value, int | float)
            or not isfinite(candidate_value)
            or not isfinite(reference_value)
            or reference_value == 0.0
        ):
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
    frame_step: int = 1,
    source_fps: float = 30.0,
) -> dict[str, Any]:
    """Build a record after verifying the shared evaluation contract.

    Args:
        methods: Results for persistence, learned, and other baseline methods.
        observation_length: Number of input frames used by every method.
        prediction_length: Number of future frames predicted by every method.
        seed: Evaluation seed shared by stochastic methods.
        frame_step: Raw-frame gap between sampled trajectory steps.
        source_fps: Source dataset frame rate before temporal downsampling.

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
        if method.window_digest != reference.window_digest:
            raise ValueError("comparison methods evaluated different window selections")

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
            "frame_step": frame_step,
            "source_fps": source_fps,
            "effective_fps": source_fps / frame_step,
            "observation_seconds": observation_length * frame_step / source_fps,
            "prediction_seconds": prediction_length * frame_step / source_fps,
            "windows": reference.windows,
            "seed": seed,
            "window_digest": reference.window_digest,
        },
        "motion_profile": reference.motion_profile,
        "methods": method_records,
    }


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
    window_contract = record["window_contract"]
    contract_name = (
        f"{window_contract['observation_length']} -> "
        f"{window_contract['prediction_length']}"
    )
    for group in ("overall", *MOTION_STRATA):
        table = Table(
            title=f"{contract_name} comparison: {group}",
            box=box.SIMPLE_HEAVY,
        )
        table.add_column("metric", style="cyan", no_wrap=True)
        for method_name in method_names:
            table.add_column(method_name, justify="right")
        for metric_name in PRIMARY_METRIC_NAMES:
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
