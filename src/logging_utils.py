"""File description: Shared logging setup for command-line experiment tools."""

from __future__ import annotations

import logging
import sys


def configure_cli_logging(
    *,
    level: int = logging.INFO,
    force: bool = False,
) -> None:
    """Configure concise stdout logging for CLI entry points.

    Args:
        level: Minimum logging level to emit.
        force: Whether to replace existing logging handlers.
    """

    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    Args:
        name: Usually the caller's `__name__`.

    Returns:
        Standard-library logger scoped to the caller.
    """

    return logging.getLogger(name)
