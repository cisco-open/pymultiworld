"""Dunder init file."""

import logging
import os
import sys

from multiworld.version import VERSION as __version__  # noqa: F401

logging.basicConfig(
    level=getattr(logging, os.getenv("M8D_LOG_LEVEL", "WARNING")),
    format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(threadName)s | %(funcName)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
