"""File description: Behavioral tests for fair model and baseline comparisons."""

from comparison import (
    ComparableMetrics,
    build_comparison_record,
    relative_error_improvements,
)


def test_relative_error_improvements_use_persistence_as_reference() -> None:
    improvements = relative_error_improvements(
        candidate={
            "centroid_fde_px": 8.0,
            "displacement_gain": 0.8,
            "displacement_direction_error_deg": 20.0,
        },
        reference={
            "centroid_fde_px": 10.0,
            "displacement_gain": 0.0,
            "displacement_direction_error_deg": float("nan"),
        },
    )

    assert improvements["centroid_fde_px"] == 0.2
    assert improvements["displacement_direction_error_deg"] is None
    assert improvements["displacement_gain"] is None


def test_build_comparison_record_requires_matching_motion_profiles() -> None:
    profile = {
        "score": "mean_mouse_centroid_displacement_px",
        "thresholds_px": {"low_max": 2.0, "medium_max": 5.0},
        "counts": {"low": 1, "medium": 1, "high": 1},
    }
    persistence = ComparableMetrics(
        name="persistence",
        windows=3,
        metrics={"centroid_fde_px": 10.0},
        metrics_by_motion={
            "low": {"centroid_fde_px": 2.0},
            "medium": {"centroid_fde_px": 5.0},
            "high": {"centroid_fde_px": 20.0},
        },
        motion_profile=profile,
        window_digest="shared-window-digest",
    )
    social_attention = ComparableMetrics(
        name="social_attention",
        windows=3,
        metrics={"centroid_fde_px": 8.0},
        metrics_by_motion={
            "low": {"centroid_fde_px": 3.0},
            "medium": {"centroid_fde_px": 4.0},
            "high": {"centroid_fde_px": 15.0},
        },
        motion_profile=profile,
        window_digest="shared-window-digest",
    )

    record = build_comparison_record(
        methods=[persistence, social_attention],
        observation_length=8,
        prediction_length=12,
        seed=42,
    )

    assert record["window_contract"] == {
        "observation_length": 8,
        "prediction_length": 12,
        "windows": 3,
        "seed": 42,
        "window_digest": "shared-window-digest",
    }
    assert record["motion_profile"] == profile
    improvement = record["methods"]["social_attention"][
        "relative_improvement_over_persistence"
    ]
    assert improvement["overall"]["centroid_fde_px"] == 0.2
    assert improvement["by_motion"]["low"]["centroid_fde_px"] == -0.5
    assert improvement["by_motion"]["high"]["centroid_fde_px"] == 0.25
