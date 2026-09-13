"""File description: Shared logging setup for command-line experiment tools."""

from __future__ import annotations

import logging
import sys


NOISY_LOGGERS = ("httpx", "httpcore", "urllib3")


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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=force,
    )
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module logger.

    Args:
        name: Usually the caller's `__name__`.

    Returns:
        Standard-library logger scoped to the caller.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
