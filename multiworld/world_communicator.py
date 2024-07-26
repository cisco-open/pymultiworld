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

"""Class to manage communication between different worlds."""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import TYPE_CHECKING

import torch.distributed as dist
from torch import Tensor
from torch.distributed import Work
from torch.distributed.distributed_c10d import DEFAULT_WORLD_NAME

if TYPE_CHECKING:
    from torch.distributed.world_manager import WorldManager

logger = logging.getLogger(__name__)


_errors_to_handle = [
    "NCCL Error 6",
    "NCCL communicator was aborted",
    "Connection reset by peer",
    "Connection closed by peer",
]


class BrokenWorldException(Exception):
    """Raise this exception when world is broken."""

    def __init__(self, world_name: str, msg: str):
        """Initialize exception instance."""
        self._world_name = world_name
        self._msg = msg

    def __str__(self):
        """Return exception string."""
        return f"{self._world_name} broken: {self._msg}"

    pass


class WorldCommunicator:
    """
    Class to manage communication between different worlds.

    NOTE: If using WorldCommunicationManager, use the API provided
    by the WorldCommunicator to create and manage worlds along with
    their communication links. Do not use the WorldManager API directly.
    """

    def __init__(self, world_manager: WorldManager):
        """Initialize a class instance."""
        self._world_manager = world_manager
        self._broken_world: dict[str, bool] = {}

        self._loop = asyncio.get_running_loop()

    def __del__(self):
        """Cleanup the class instance."""
        del self._broken_world

    def add_world(self, world_name):
        """Add a new world to the world comm manager."""
        self._broken_world[world_name] = False

    def remove_world(self, world_name):
        """Remove a world from the world comm manager."""
        logger.debug(f"remove world {world_name}")
        try:
            self._broken_world[world_name] = True
        except KeyError:
            pass

    async def _wait_work(self, work: Work, world_name: str) -> None:
        """Do busy-waiting for work to be done.

        It also checks if a world is broken or not. If so, it raises
        BrokenWorldException exception.
        """
        while not work.is_completed():
            if self._broken_world[world_name]:
                raise BrokenWorldException(world_name, "watchdog raised the exception")
            await asyncio.sleep(0)

    async def send(
        self, tensor: Tensor, dst: int, world_name: str = DEFAULT_WORLD_NAME
    ) -> None:
        """Send a tensor to a destination in a world."""
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                work = await self._loop.run_in_executor(
                    pool,
                    dist.isend,
                    tensor,
                    dst,
                    None,
                    0,
                    world_name,
                )
        except RuntimeError as e:
            self._handle_error(e, world_name)

        await self._wait_work(work, world_name)

    async def recv(
        self, tensor: Tensor, src: int, world_name: str = DEFAULT_WORLD_NAME
    ) -> None:
        """Receive a tensor from a specific rank in a world."""
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                work = await self._loop.run_in_executor(
                    pool,
                    dist.irecv,
                    tensor,
                    src,
                    None,
                    0,
                    world_name,
                )
        except RuntimeError as e:
            self._handle_error(e, world_name)

        await self._wait_work(work, world_name)

    async def broadcast(
        self, tensor: Tensor, src: int, world_name: str = DEFAULT_WORLD_NAME
    ) -> None:
        """Broadcast a tensor to the world from a source (src)."""
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                work = await self._loop.run_in_executor(
                    pool,
                    dist.broadcast,
                    tensor,
                    src,
                    None,
                    True,
                    world_name,
                )
        except RuntimeError as e:
            self._handle_error(e, world_name)

        await self._wait_work(work, world_name)

    async def all_reduce(
        self,
        tensor: Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        world_name: str = DEFAULT_WORLD_NAME,
    ) -> None:
        """Do all-reduce for a given tensor in a world."""
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                work = await self._loop.run_in_executor(
                    pool,
                    dist.all_reduce,
                    tensor,
                    op,
                    None,
                    True,
                    world_name,
                )
        except RuntimeError as e:
            self._handle_error(e, world_name)

        await self._wait_work(work, world_name)

    async def reduce(
        self,
        tensor: Tensor,
        dst: int,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        world_name: str = DEFAULT_WORLD_NAME,
    ) -> None:
        """Do reduce for a given tensor in a world.

        The rank is a receiver of the final result.
        """
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                work = await self._loop.run_in_executor(
                    pool,
                    dist.reduce,
                    tensor,
                    dst,
                    op,
                    None,
                    True,
                    world_name,
                )
        except RuntimeError as e:
            self._handle_error(e, world_name)

        await self._wait_work(work, world_name)

    async def all_gather(
        self,
        tensors: list[Tensor],
        tensor: Tensor,
        world_name: str = DEFAULT_WORLD_NAME,
    ) -> None:
        """Do all-gather for a given tensor in a world."""
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                work = await self._loop.run_in_executor(
                    pool,
                    dist.all_gather,
                    tensors,
                    tensor,
                    None,
                    True,
                    world_name,
                )
        except RuntimeError as e:
            self._handle_error(e, world_name)

        await self._wait_work(work, world_name)

    async def gather(
        self,
        tensor: Tensor,
        gather_list: list[Tensor] = None,
        dst: int = 0,
        world_name: str = DEFAULT_WORLD_NAME,
    ) -> None:
        """Do gather for a list of tensors in a world."""
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                work = await self._loop.run_in_executor(
                    pool,
                    dist.gather,
                    tensor,
                    gather_list,
                    dst,
                    None,
                    True,
                    world_name,
                )
        except RuntimeError as e:
            self._handle_error(e, world_name)

        await self._wait_work(work, world_name)

    async def scatter(
        self,
        tensor: Tensor,
        scatter_list: list[Tensor] = None,
        src: int = 0,
        world_name: str = DEFAULT_WORLD_NAME,
    ) -> None:
        """Do scatter for a list of tensors from a source (src) in a world."""
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                work = await self._loop.run_in_executor(
                    pool,
                    dist.scatter,
                    tensor,
                    scatter_list,
                    src,
                    None,
                    True,
                    world_name,
                )
        except RuntimeError as e:
            self._handle_error(e, world_name)

        await self._wait_work(work, world_name)

    def _handle_error(self, error: RuntimeError, world_name: str) -> None:
        error_message = str(error)

        for error_snippet in _errors_to_handle:
            if error_snippet in error_message:
                logger.debug(f"broken world: {error_message}")
                self._world_manager.remove_world(world_name)
                raise BrokenWorldException(world_name, error_message)

        raise error
