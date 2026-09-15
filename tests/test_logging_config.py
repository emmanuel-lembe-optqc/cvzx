"""Tests for `cvzx.logging_config` (opt-in file logging for the rewriting pipeline)."""

import logging
from pathlib import Path

from cvzx.logging_config import setup_file_logging

_LOGGER_NAMES = (
    "cvzx.passes.normalize",
    "cvzx.backends.nx.rules",
    "cvzx.backends.rx.rules",
    "cvzx.passes.optimize",
)

# Expected log file stem per logger -- the two backends' rule modules share
# their last dotted component ("rules"), so `setup_file_logging` folds the
# backend name in to keep their files distinct (see `_log_file_name`).
_EXPECTED_LOG_FILE_NAMES = {
    "cvzx.passes.normalize": "normalize",
    "cvzx.backends.nx.rules": "nx_rules",
    "cvzx.backends.rx.rules": "rx_rules",
    "cvzx.passes.optimize": "optimize",
}


def _remove_file_handlers() -> None:
    """Undo `setup_file_logging` so tests don't leak handlers into each other."""
    for name in _LOGGER_NAMES:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)


def test_creates_log_directory_and_one_file_per_pipeline_logger(tmp_path: Path):
    log_dir = tmp_path / "logs"
    try:
        setup_file_logging(log_dir=log_dir)

        assert log_dir.is_dir()
        for name in _LOGGER_NAMES:
            assert (log_dir / f"{_EXPECTED_LOG_FILE_NAMES[name]}.log").exists()

        logging.getLogger("cvzx.passes.optimize").debug("hello from a test")
        assert "hello from a test" in (log_dir / "optimize.log").read_text()
    finally:
        _remove_file_handlers()


def test_setup_is_idempotent(tmp_path: Path):
    try:
        setup_file_logging(log_dir=tmp_path / "logs")
        setup_file_logging(log_dir=tmp_path / "logs")

        for name in _LOGGER_NAMES:
            logger = logging.getLogger(name)
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1
    finally:
        _remove_file_handlers()
