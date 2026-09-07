"""Opt-in file logging for the diagram-rewriting/optimization pipeline.

Every module in the rewriting pipeline (`cvzx.normalize_diagram`,
`cvzx.nx_rewrite_rules`, `cvzx.optimize`) only ever calls
`logging.getLogger(__name__)` and emits records -- library code must never
configure handlers itself, since that decision belongs to whatever
application imports it (see the standard library's own logging guidance
for libraries). Call `setup_file_logging()` once, early in your own script,
to have each of those loggers write to its own file under a log directory
instead of relying on whatever (if anything) the caller has already
configured.
"""

from __future__ import annotations

import logging
from pathlib import Path

# One file per concern: dumping every module's logs into a single file
# makes it hard to tell, e.g., "did FusionRule ever match" apart from "did
# the round loop converge" at a glance.
_PIPELINE_LOGGERS = ("cvzx.normalize_diagram", "cvzx.nx_rewrite_rules", "cvzx.optimize")

__all__ = ["setup_file_logging"]


def setup_file_logging(log_dir: str | Path = "logs", level: int = logging.DEBUG) -> None:
    """Write each rewriting-pipeline logger's records to its own file.

    Idempotent: a logger that already has a `FileHandler` attached (from a
    previous call) is left untouched, so calling this more than once never
    creates duplicate log lines.

    Parameters
    ----------
    log_dir : str | Path
        Directory to write log files into (created, with any missing
        parents, if it doesn't already exist). One file per pipeline
        module: `normalize_diagram.log`, `nx_rewrite_rules.log`,
        `optimize.log`.
    level : int
        Logging level for both the loggers and their file handlers.
        Defaults to `logging.DEBUG`.
    """
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    for name in _PIPELINE_LOGGERS:
        logger = logging.getLogger(name)
        if any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
            continue
        logger.setLevel(level)
        short_name = name.rsplit(".", 1)[-1]
        handler = logging.FileHandler(directory / f"{short_name}.log")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
