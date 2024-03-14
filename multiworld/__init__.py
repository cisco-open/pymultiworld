"""Dunder init file."""

import logging
import os
import sys

from multiworld.version import VERSION as __version__  # noqa: F401

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "NOTSET")),
    format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(threadName)s | %(funcName)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
