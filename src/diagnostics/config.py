"""File description: Configuration loading for diagnostic workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from evaluate import DEFAULT_TEST_CONFIG_PATH


ModelName = Literal["dense_triplet", "single_mouse"]
RolloutMode = Literal["autoregressive", "teacher_forced"]
PredictionStatistic = Literal["sample", "mean"]


@dataclass(frozen=True)
class VariantConfig:
    """Model variant used in paired diagnostics.

    Attributes:
        train_config_path: YAML file used to train the variant.
        checkpoint_path: Local checkpoint for the variant.
        display_name: Short name shown in diagnostic tables.
    """

    train_config_path: Path
    checkpoint_path: Path
    display_name: str


@dataclass(frozen=True)
class PoseDiagnosticConfig:
    """Configuration for dense-triplet versus single-mouse diagnostics.

    Attributes:
        dense_triplet: Dense triplet model inputs.
        single_mouse: Single-mouse model inputs.
        test_config_path: Held-out test configuration.
        results_path: JSONL destination for the diagnostic summary.
        case_results_path: Optional JSONL destination for per-case rows.
        max_triplet_windows: Number of triplet windows sampled before expanding
            into mouse-specific paired cases.
        seeds: Sampling seeds used to probe stochastic rollout sensitivity.
        device: Optional device override. When omitted, each train config decides.
    """

    dense_triplet: VariantConfig
    single_mouse: VariantConfig
    test_config_path: Path
    results_path: Path
    case_results_path: Path | None
    max_triplet_windows: int
    seeds: tuple[int, ...]
    device: str | None = None


def load_pose_diagnostic_config(path: Path) -> PoseDiagnosticConfig:
    """Load paired diagnostic configuration from YAML.

    Args:
        path: YAML config path.

    Returns:
        Typed diagnostic configuration.
    """

    raw = yaml.safe_load(path.read_text()) or {}
    return PoseDiagnosticConfig(
        dense_triplet=_load_variant_config(raw["dense_triplet"]),
        single_mouse=_load_variant_config(raw["single_mouse"]),
        test_config_path=Path(raw.get("test_config_path", DEFAULT_TEST_CONFIG_PATH)),
        results_path=Path(raw["results_path"]),
        case_results_path=(
            Path(raw["case_results_path"])
            if raw.get("case_results_path") is not None
            else None
        ),
        max_triplet_windows=int(raw.get("max_triplet_windows", 30)),
        seeds=tuple(int(seed) for seed in raw.get("seeds", [42])),
        device=raw.get("device"),
    )


def _load_variant_config(raw: dict[str, Any]) -> VariantConfig:
    """Load one variant block from YAML."""

    return VariantConfig(
        train_config_path=Path(raw["train_config_path"]),
        checkpoint_path=Path(raw["checkpoint_path"]),
        display_name=str(raw["display_name"]),
    )
