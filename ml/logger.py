import logging
import sys
from pathlib import Path

# Optional: create a logs directory
Path("logs").mkdir(exist_ok=True)


def get_logger(name: str, log_file: str = "logs/app.log") -> logging.Logger:
    """
    Creates a logger that prints to console and writes to a file.

    Args:
        name (str): Name of the logger, usually __name__.
        log_file (str): File to write logs to.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():  # prevent duplicate handlers
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger
