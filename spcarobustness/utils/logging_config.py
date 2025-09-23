import logging
import sys
from typing import Optional


def configure_logging(level: int = logging.INFO, log_to_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("spcarobustness")
    logger.setLevel(level)
    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        if log_to_file:
            fh = logging.FileHandler(log_to_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger
