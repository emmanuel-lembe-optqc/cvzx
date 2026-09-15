"""Opt-in file logging for the diagram-rewriting/optimization pipeline.

Every module in the rewriting pipeline (`cvzx.passes.normalize`,
`cvzx.backends.nx.rules`, `cvzx.backends.rx.rules`, `cvzx.passes.optimize`)
only ever calls `logging.getLogger(__name__)` and emits records -- library
code must never configure handlers itself, since that decision belongs to
whatever application imports it (see the standard library's own logging
guidance for libraries). Call `setup_file_logging()` once, early in your own
script, to have each of those loggers write to its own file under a log
directory instead of relying on whatever (if anything) the caller has
already configured.
"""

from __future__ import annotations

import logging
from pathlib import Path

# One file per concern: dumping every module's logs into a single file
# makes it hard to tell, e.g., "did FusionRule ever match" apart from "did
# the round loop converge" at a glance. Both backends' rule modules are
# listed -- `cvzx.passes.optimize` picks whichever one `cvzx.config.
# DEFAULT_BACKEND` resolves to (rustworkx over networkx whenever it's
# installed), so a caller who only wired up "nx.rules" would silently get
# an empty log the moment the rustworkx backend is the one actually doing
# the work.
_PIPELINE_LOGGERS = (
    "cvzx.passes.normalize",
    "cvzx.backends.nx.rules",
    "cvzx.backends.rx.rules",
    "cvzx.passes.optimize",
)


def _log_file_name(logger_name: str) -> str:
    """Map a pipeline logger's dotted name to its log file's stem.

    The two backends' rule modules share their last component ("rules"),
    so the backend name is folded in to keep their log files distinct
    (`nx_rules.log` / `rx_rules.log`) instead of one silently clobbering
    the other.

    Returns
    -------
    str
        The file stem (without ".log") to write `logger_name`'s records to.
    """
    parts = logger_name.split(".")
    if "backends" in parts:
        backend = parts[parts.index("backends") + 1]
        return f"{backend}_rules"
    return parts[-1]


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
        module: `normalize.log`, `nx_rules.log`, `rx_rules.log`,
        `optimize.log`. Only one of `nx_rules.log`/`rx_rules.log` will
        have any content in a given run -- whichever backend
        `cvzx.config.DEFAULT_BACKEND` (or an explicit `backend=` argument)
        actually selected; the other is created empty.
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
        handler = logging.FileHandler(directory / f"{_log_file_name(name)}.log")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
