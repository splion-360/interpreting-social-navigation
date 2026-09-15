"""File description: Paired statistical summaries for diagnostic comparisons."""

from __future__ import annotations

import numpy as np


def paired_cluster_bootstrap(
    reference: np.ndarray,
    comparison: np.ndarray,
    *,
    clusters: np.ndarray,
    seed: int = 42,
    resamples: int = 10_000,
) -> dict[str, float | int]:
    """Estimate a confidence interval for a paired mean difference.

    Whole clusters are resampled so correlated mice from the same source window
    remain together. Positive deltas mean the comparison has higher error.

    Args:
        reference: Per-case metric values for the reference model.
        comparison: Matched per-case values for the comparison model.
        clusters: Cluster identifier for every paired case.
        seed: Random seed used for bootstrap resampling.
        resamples: Number of bootstrap replicates.

    Returns:
        Observed paired deltas, comparison win rate, and the 95% confidence
        interval for the cluster-resampled mean delta.
    """

    reference_values = np.asarray(reference, dtype=np.float64)
    comparison_values = np.asarray(comparison, dtype=np.float64)
    cluster_values = np.asarray(clusters)
    finite = np.isfinite(reference_values) & np.isfinite(comparison_values)
    deltas = comparison_values[finite] - reference_values[finite]
    cluster_values = cluster_values[finite]
    unique_clusters = np.unique(cluster_values)
    cluster_means = np.asarray(
        [deltas[cluster_values == cluster].mean() for cluster in unique_clusters]
    )
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0,
        len(cluster_means),
        size=(resamples, len(cluster_means)),
    )
    bootstrap_means = cluster_means[sampled_indices].mean(axis=1)
    return {
        "pairs": int(deltas.size),
        "clusters": int(unique_clusters.size),
        "mean_delta": float(cluster_means.mean()),
        "p50_delta": float(np.percentile(deltas, 50)),
        "comparison_better_rate": float((deltas < 0).mean()),
        "mean_delta_ci95_low": float(np.percentile(bootstrap_means, 2.5)),
        "mean_delta_ci95_high": float(np.percentile(bootstrap_means, 97.5)),
    }
