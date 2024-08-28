# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Dunder init file."""

import logging
import os
import sys

from multiworld.version import VERSION as __version__  # noqa: F401

from . import post_setup

logging.basicConfig(
    level=getattr(logging, os.getenv("M8D_LOG_LEVEL", "WARNING")),
    format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(threadName)s | %(funcName)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

post_setup.configure_once()
