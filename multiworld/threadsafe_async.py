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
