"""File description: Rich table rendering for diagnostic summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from evaluate import MOTION_STRATA
from metrics import PRIMARY_METRIC_NAMES


def print_pose_diagnostics(
    record: dict[str, Any], *, console: Console | None = None
) -> None:
    """Print the paired diagnostic summary as Rich tables.

    Args:
        record: Diagnostic record returned by the diagnostic runner.
        console: Optional Rich console.
    """

    output = console or Console()
    output.print(_diagnostic_overview_table(record))
    output.print(_diagnostic_metric_table(record, mode="autoregressive"))
    output.print(_diagnostic_metric_table(record, mode="teacher_forced"))
    output.print(_diagnostic_delta_table(record))
    output.print(_diagnostic_motion_table(record))
    output.print(_diagnostic_calibration_table(record))
    output.print(_diagnostic_attention_table(record))


def _diagnostic_overview_table(record: dict[str, Any]) -> Table:
    """Build a compact diagnostic lineage table."""

    contract = record["window_contract"]
    table = Table(title="Pose diagnostic contract", box=box.SIMPLE_HEAVY)
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value", no_wrap=True)
    for name in (
        "effective_fps",
        "observation_seconds",
        "prediction_seconds",
        "triplet_windows",
        "mouse_cases",
        "seeds",
    ):
        table.add_row(name, str(contract[name]))
    return table


def _diagnostic_metric_table(record: dict[str, Any], *, mode: str) -> Table:
    """Build dense versus single metric table for one rollout mode."""

    table = Table(title=f"{mode} matched metrics", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan", no_wrap=True)
    for model_name in ("dense_triplet", "single_mouse"):
        table.add_column(model_name, justify="right")
    for metric_name in PRIMARY_METRIC_NAMES:
        values = []
        has_value = False
        for model_name in ("dense_triplet", "single_mouse"):
            method = record["methods"][model_name][mode]
            mean = method["metrics"].get(metric_name)
            quantiles = method["metric_quantiles"].get(metric_name, {})
            cell = _mean_quantile_cell(mean, quantiles)
            values.append(cell or "-")
            has_value = has_value or cell is not None
        if has_value:
            table.add_row(metric_name, *values)
    return table


def _diagnostic_delta_table(record: dict[str, Any]) -> Table:
    """Build single-minus-dense paired delta table for autoregressive metrics."""

    table = Table(
        title="autoregressive paired delta: single_mouse - dense_triplet",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("mean", justify="right")
    table.add_column("p50", justify="right")
    table.add_column("p99", justify="right")
    table.add_column("single better", justify="right")
    for metric_name, values in record["paired_delta_single_minus_dense"][
        "autoregressive"
    ].items():
        table.add_row(
            metric_name,
            f"{values['mean']:.3f}",
            f"{values['p50']:.3f}",
            f"{values['p99']:.3f}",
            f"{values['single_better_rate']:.1%}",
        )
    return table


def _diagnostic_motion_table(record: dict[str, Any]) -> Table:
    """Build an autoregressive pose metrics table grouped by motion stratum."""

    table = Table(title="autoregressive pose metrics by motion", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("motion", no_wrap=True)
    table.add_column("dense_triplet", justify="right")
    table.add_column("single_mouse", justify="right")
    for metric_name in (
        "body_heading_error_deg",
        "bone_length_error_px",
        "skeleton_orientation_error_deg",
        "relative_ordering_error",
    ):
        for stratum in MOTION_STRATA:
            dense = (
                record["methods"]["dense_triplet"]["autoregressive"][
                    "metrics_by_motion"
                ]
                .get(stratum, {})
                .get(metric_name)
            )
            single = (
                record["methods"]["single_mouse"]["autoregressive"]["metrics_by_motion"]
                .get(stratum, {})
                .get(metric_name)
            )
            table.add_row(
                metric_name, stratum, _format_optional(dense), _format_optional(single)
            )
    return table


def _diagnostic_calibration_table(record: dict[str, Any]) -> Table:
    """Build Gaussian calibration summary table."""

    table = Table(title="autoregressive Gaussian calibration", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("dense_triplet", justify="right")
    table.add_column("single_mouse", justify="right")
    metrics = sorted(
        set(record["calibration"]["dense_triplet"])
        | set(record["calibration"]["single_mouse"])
    )
    for metric_name in metrics:
        table.add_row(
            metric_name,
            _format_optional(record["calibration"]["dense_triplet"].get(metric_name)),
            _format_optional(record["calibration"]["single_mouse"].get(metric_name)),
        )
    return table


def _diagnostic_attention_table(record: dict[str, Any]) -> Table:
    """Build dense-triplet attention summary table."""

    table = Table(title="dense-triplet attention summary", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", justify="right")
    for name, value in record["attention"].items():
        table.add_row(name, _format_optional(value))
    return table


def _mean_quantile_cell(value: Any, quantiles: dict[str, float]) -> str | None:
    """Format a mean, p50, and p99 cell."""

    if not isinstance(value, int | float) or not np.isfinite(value):
        return None
    parts = [f"mean={value:.3f}"]
    for name in ("p50", "p99"):
        quantile = quantiles.get(name)
        if isinstance(quantile, int | float) and np.isfinite(quantile):
            parts.append(f"{name}={quantile:.3f}")
    return " ".join(parts)


def _format_optional(value: Any) -> str:
    """Format an optional scalar table value."""

    if not isinstance(value, int | float) or not np.isfinite(value):
        return "-"
    return f"{value:.3f}"
