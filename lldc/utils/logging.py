# lldc/utils/logging.py

from __future__ import annotations
import logging, sys
from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True, markup=True, show_time=True, show_path=False
            )
        ],
    )
    for noisy in ("urllib3", "datasets", "transformers", "evaluate"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("lldc")
