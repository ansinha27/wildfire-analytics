import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    # I want consistent logging across every module
    # so I set this up once and import it everywhere
    logger = logging.getLogger(name)

    if logger.handlers:
        # already configured, don't add duplicate handlers
        return logger

    logger.setLevel(logging.INFO)

    # log to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)

    # keep it readable - timestamp, module name, message
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console.setFormatter(fmt)
    logger.addHandler(console)

    # also write to a log file so I can review runs later
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger