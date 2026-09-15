"""File description: Rich table rendering for diagnostic summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from evaluate import MOTION_STRATA
from metrics import PRIMARY_METRIC_NAMES


HORIZON_PLOT_METRICS = (
    ("centroid_displacement_error_px", "Centroid displacement error", "px"),
    ("keypoint_displacement_error_px", "Keypoint displacement error", "px"),
    ("keypoint_velocity_error_px_s", "Keypoint velocity error", "px/s"),
    ("bone_length_error_px", "Bone-length error", "px"),
    ("skeleton_orientation_error_deg", "Skeleton orientation error", "degrees"),
    ("body_heading_error_deg", "Body-heading error", "degrees"),
    ("relative_ordering_error", "Relative-ordering error", "fraction"),
    ("body_frame_keypoint_error_px", "Body-frame keypoint error", "px"),
)


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
    output.print(_diagnostic_prediction_mode_table(record))
    output.print(_diagnostic_calibration_horizon_table(record))
    output.print(_diagnostic_distribution_table(record))
    output.print(_diagnostic_metric_table(record, mode="autoregressive"))
    output.print(_diagnostic_metric_table(record, mode="teacher_forced"))
    output.print(_diagnostic_delta_table(record))
    output.print(_diagnostic_motion_table(record))
    output.print(_diagnostic_calibration_table(record))
    output.print(_diagnostic_attention_table(record))


def _diagnostic_prediction_mode_table(record: dict[str, Any]) -> Table:
    """Build a P50 table spanning conditioning and Gaussian output modes."""

    table = Table(title="P50 prediction-mode diagnosis", box=box.SIMPLE_HEAVY)
    table.add_column("model", style="cyan", no_wrap=True)
    table.add_column("mode", no_wrap=True)
    metrics = (
        ("centroid_ade_px", "cADE"),
        ("bone_length_error_px", "BLE"),
        ("skeleton_orientation_error_deg", "SOE"),
        ("body_heading_error_deg", "BHE"),
        ("relative_ordering_error", "ROE"),
    )
    for _, label in metrics:
        table.add_column(label, justify="right")
    for model_name, by_mode in record["prediction_mode_comparison"].items():
        for mode_name, summary in by_mode.items():
            quantiles = summary["metric_quantiles"]
            table.add_row(
                model_name,
                mode_name,
                *[_format_optional(quantiles[name].get("p50")) for name, _ in metrics],
            )
    return table


def _diagnostic_calibration_horizon_table(record: dict[str, Any]) -> Table:
    """Build first-versus-final Gaussian calibration table."""

    table = Table(title="Gaussian calibration by horizon", box=box.SIMPLE_HEAVY)
    table.add_column("model", style="cyan", no_wrap=True)
    table.add_column("mode", no_wrap=True)
    table.add_column("step", justify="right")
    table.add_column("coverage 50", justify="right")
    table.add_column("coverage 95", justify="right")
    table.add_column("Mahalanobis squared", justify="right")
    final_step = str(record["window_contract"]["prediction_length"])
    for model_name, by_mode in record["calibration_by_horizon"].items():
        for mode_name, by_horizon in by_mode.items():
            for step in ("1", final_step):
                values = by_horizon[step]["metrics"]
                table.add_row(
                    model_name,
                    mode_name,
                    step,
                    _format_optional(values.get("coverage_50")),
                    _format_optional(values.get("coverage_95")),
                    _format_optional(values.get("mahalanobis_sq")),
                )
    return table


def _diagnostic_distribution_table(record: dict[str, Any]) -> Table:
    """Build a table of the largest train-validation feature shifts."""

    table = Table(title="Largest train-validation shifts", box=box.SIMPLE_HEAVY)
    table.add_column("model", no_wrap=True)
    table.add_column("feature", style="cyan")
    table.add_column("standardized difference", justify="right")
    table.add_column("KS", justify="right")
    for model_name, audit in record["distribution_audit"].items():
        ordered = sorted(
            audit["features"].items(),
            key=lambda item: abs(item[1]["standardized_mean_difference"]),
            reverse=True,
        )
        for name, values in ordered[:5]:
            table.add_row(
                model_name,
                name,
                _format_optional(values["standardized_mean_difference"]),
                _format_optional(values["ks_statistic"]),
            )
    return table


def horizon_profile_rows(
    record: dict[str, Any], *, statistic: str = "p50"
) -> list[dict[str, str | int | float]]:
    """Return horizon metrics as tidy rows for tables or plots.

    Args:
        record: Diagnostic record containing an exact-step horizon profile.
        statistic: Aggregate to read: `mean`, `p50`, or `p99`.

    Returns:
        Long-form rows with model, rollout mode, time, metric, and value.
    """

    effective_fps = float(record["window_contract"]["effective_fps"])
    rows: list[dict[str, str | int | float]] = []
    for model_name, by_mode in record["horizon_profile"].items():
        for mode, by_horizon in by_mode.items():
            for horizon_text, summary in by_horizon.items():
                horizon = int(horizon_text)
                values = (
                    summary["metrics"]
                    if statistic == "mean"
                    else {
                        name: quantiles.get(statistic)
                        for name, quantiles in summary["metric_quantiles"].items()
                    }
                )
                for metric_name, value in values.items():
                    if isinstance(value, int | float) and np.isfinite(value):
                        rows.append(
                            {
                                "model": model_name,
                                "mode": mode,
                                "horizon_step": horizon,
                                "horizon_seconds": horizon / effective_fps,
                                "metric": metric_name,
                                "value": float(value),
                            }
                        )
    return rows


def save_horizon_profile_figure(
    record: dict[str, Any],
    output_path: Path,
    *,
    statistic: str = "p50",
) -> Path:
    """Plot trajectory and pose errors against prediction time.

    Args:
        record: Diagnostic record containing exact-step horizon metrics.
        output_path: PNG path for the rendered profile.
        statistic: Aggregate to plot: `mean`, `p50`, or `p99`.

    Returns:
        Path of the saved figure.
    """

    import matplotlib.pyplot as plt

    rows = horizon_profile_rows(record, statistic=statistic)
    figure, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    model_names = tuple(record["horizon_profile"])
    color_map = plt.get_cmap("tab10")
    colors = {
        model_name: color_map(index % color_map.N)
        for index, model_name in enumerate(model_names)
    }
    line_styles = {"autoregressive": "-", "teacher_forced": "--"}
    for axis, (metric_name, title, unit) in zip(
        axes.ravel(), HORIZON_PLOT_METRICS, strict=True
    ):
        for model_name in model_names:
            for mode in ("autoregressive", "teacher_forced"):
                series = [
                    row
                    for row in rows
                    if row["metric"] == metric_name
                    and row["model"] == model_name
                    and row["mode"] == mode
                ]
                if not series:
                    continue
                series.sort(key=lambda row: int(row["horizon_step"]))
                axis.plot(
                    [float(row["horizon_seconds"]) for row in series],
                    [float(row["value"]) for row in series],
                    color=colors[model_name],
                    linestyle=line_styles[mode],
                    label=f"{model_name} / {mode}",
                )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("prediction horizon (seconds)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(f"Per-horizon diagnostic profile ({statistic})", y=0.98)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def prediction_mode_rows(
    record: dict[str, Any], *, statistic: str = "p50"
) -> list[dict[str, str | int | float]]:
    """Return per-horizon rows for conditioning and output-statistic modes.

    Args:
        record: Diagnostic record containing the prediction-mode comparison.
        statistic: Aggregate to read: `mean`, `p50`, or `p99`.

    Returns:
        Long-form rows with model, mode, time, metric, and value.
    """

    effective_fps = float(record["window_contract"]["effective_fps"])
    rows: list[dict[str, str | int | float]] = []
    for model_name, by_mode in record["prediction_mode_comparison"].items():
        for mode, payload in by_mode.items():
            for horizon_text, summary in payload["horizon_profile"].items():
                horizon = int(horizon_text)
                values = (
                    summary["metrics"]
                    if statistic == "mean"
                    else {
                        name: quantiles.get(statistic)
                        for name, quantiles in summary["metric_quantiles"].items()
                    }
                )
                for metric_name, value in values.items():
                    if isinstance(value, int | float) and np.isfinite(value):
                        rows.append(
                            {
                                "model": model_name,
                                "mode": mode,
                                "horizon_step": horizon,
                                "horizon_seconds": horizon / effective_fps,
                                "metric": metric_name,
                                "value": float(value),
                            }
                        )
    return rows


def save_prediction_mode_figure(
    record: dict[str, Any],
    output_path: Path,
    *,
    statistic: str = "p50",
) -> Path:
    """Plot sampled and mean predictions under both conditioning policies.

    Args:
        record: Diagnostic record containing prediction-mode metrics.
        output_path: PNG path for the rendered comparison.
        statistic: Aggregate to plot: `mean`, `p50`, or `p99`.

    Returns:
        Path of the saved figure.
    """

    import matplotlib.pyplot as plt

    rows = prediction_mode_rows(record, statistic=statistic)
    figure, axes = plt.subplots(2, 4, figsize=(19, 8), sharex=True)
    model_names = tuple(record["prediction_mode_comparison"])
    colors = {
        model_name: plt.get_cmap("tab10")(index)
        for index, model_name in enumerate(model_names)
    }
    line_styles = {
        "autoregressive_sample": "-",
        "autoregressive_mean": ":",
        "teacher_forced_sample": "--",
        "teacher_forced_mean": "-.",
    }
    for axis, (metric_name, title, unit) in zip(
        axes.ravel(), HORIZON_PLOT_METRICS, strict=True
    ):
        for model_name in model_names:
            for mode, line_style in line_styles.items():
                series = [
                    row
                    for row in rows
                    if row["metric"] == metric_name
                    and row["model"] == model_name
                    and row["mode"] == mode
                ]
                series.sort(key=lambda row: int(row["horizon_step"]))
                if series:
                    axis.plot(
                        [float(row["horizon_seconds"]) for row in series],
                        [float(row["value"]) for row in series],
                        color=colors[model_name],
                        linestyle=line_style,
                        label=f"{model_name} / {mode}",
                    )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("prediction horizon (seconds)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(
        f"Conditioning and Gaussian output comparison ({statistic})", y=0.995
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_calibration_profile_figure(record: dict[str, Any], output_path: Path) -> Path:
    """Plot Gaussian coverage and scale at each forecast horizon.

    Args:
        record: Diagnostic record containing horizon-wise calibration.
        output_path: PNG path for the rendered calibration profile.

    Returns:
        Path of the saved figure.
    """

    import matplotlib.pyplot as plt

    effective_fps = float(record["window_contract"]["effective_fps"])
    figure, axes = plt.subplots(1, 4, figsize=(18, 4))
    metric_specs = (
        ("coverage_50", "50% ellipse coverage", 0.5),
        ("coverage_95", "95% ellipse coverage", 0.95),
        ("mahalanobis_sq", "Mean squared Mahalanobis", None),
        ("sigma_x_px", "Mean predicted sigma-x", None),
    )
    color_map = plt.get_cmap("tab10")
    colors = {
        model_name: color_map(index % color_map.N)
        for index, model_name in enumerate(record["calibration_by_horizon"])
    }
    line_styles = {
        "autoregressive_sample": "-",
        "autoregressive_mean": ":",
        "teacher_forced": "--",
    }
    for axis, (metric_name, title, target) in zip(axes, metric_specs, strict=True):
        for model_name, by_mode in record["calibration_by_horizon"].items():
            for mode, by_horizon in by_mode.items():
                points = [
                    (
                        int(step),
                        summary["metrics"].get(metric_name),
                    )
                    for step, summary in by_horizon.items()
                ]
                points = [point for point in points if point[1] is not None]
                points.sort()
                if points:
                    axis.plot(
                        [step / effective_fps for step, _ in points],
                        [float(value) for _, value in points],
                        color=colors[model_name],
                        linestyle=line_styles[mode],
                        label=f"{model_name} / {mode}",
                    )
        if target is not None:
            axis.axhline(target, color="black", linewidth=1, alpha=0.5)
        axis.set_title(title)
        axis.set_xlabel("prediction horizon (seconds)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("fraction")
    axes[1].set_ylabel("fraction")
    axes[3].set_ylabel("px")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


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
    """Build matched metric table for one rollout mode."""

    table = Table(title=f"{mode} matched metrics", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan", no_wrap=True)
    model_names = tuple(record["methods"])
    for model_name in model_names:
        table.add_column(model_name, justify="right")
    for metric_name in PRIMARY_METRIC_NAMES:
        values = []
        has_value = False
        for model_name in model_names:
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
    """Build disconnected-minus-dense clustered bootstrap table."""

    table = Table(
        title="autoregressive paired delta: disconnected - dense",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("mean delta", justify="right")
    table.add_column("95% cluster CI", justify="right")
    table.add_column("p50 delta", justify="right")
    table.add_column("disconnected better", justify="right")
    metrics = record["paired_bootstrap"]["modes"]["autoregressive"]
    for metric_name in PRIMARY_METRIC_NAMES:
        values = metrics.get(metric_name)
        if values is None:
            continue
        table.add_row(
            metric_name,
            f"{values['mean_delta']:.3f}",
            (
                f"[{values['mean_delta_ci95_low']:.3f}, "
                f"{values['mean_delta_ci95_high']:.3f}]"
            ),
            f"{values['p50_delta']:.3f}",
            f"{values['comparison_better_rate']:.1%}",
        )
    return table


def _diagnostic_motion_table(record: dict[str, Any]) -> Table:
    """Build an autoregressive pose metrics table grouped by motion stratum."""

    table = Table(title="autoregressive pose metrics by motion", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("motion", no_wrap=True)
    model_names = tuple(record["methods"])
    for model_name in model_names:
        table.add_column(model_name, justify="right")
    for metric_name in (
        "body_heading_error_deg",
        "bone_length_error_px",
        "skeleton_orientation_error_deg",
        "relative_ordering_error",
    ):
        for stratum in MOTION_STRATA:
            table.add_row(
                metric_name,
                stratum,
                *[
                    _format_optional(
                        record["methods"][model_name]["autoregressive"][
                            "metrics_by_motion"
                        ]
                        .get(stratum, {})
                        .get(metric_name)
                    )
                    for model_name in model_names
                ],
            )
    return table


def _diagnostic_calibration_table(record: dict[str, Any]) -> Table:
    """Build Gaussian calibration summary table."""

    table = Table(title="autoregressive Gaussian calibration", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="cyan", no_wrap=True)
    model_names = tuple(record["calibration"])
    for model_name in model_names:
        table.add_column(model_name, justify="right")
    metrics = sorted(
        set().union(*(record["calibration"][name] for name in model_names))
    )
    for metric_name in metrics:
        table.add_row(
            metric_name,
            *[
                _format_optional(record["calibration"][name].get(metric_name))
                for name in model_names
            ],
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
