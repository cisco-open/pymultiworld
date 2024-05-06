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

"""A utility function for running co-routine threadsafely."""
import asyncio
import concurrent.futures
from typing import Any, Union


def run_async(coro, loop, timeout=None) -> tuple[Union[None, Any], bool]:
    """Run asyncio co-routine in a thread-safe manner."""
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout), True
    except concurrent.futures.TimeoutError:
        return None, False
