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
from typing import TYPE_CHECKING, Callable

import torch.distributed as dist
from torch import Tensor
from torch.distributed import DEFAULT_WORLD_NAME, Work

if TYPE_CHECKING:
    from multiworld.manager import WorldManager

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
    """Class to manage communication for different worlds."""

    def __init__(self, world_manager: WorldManager):
        """Initialize a class instance."""
        self._world_manager = world_manager
        self._world_to_send_fn: dict[str, Callable] = {}
        self._world_to_recv_fn: dict[str, Callable] = {}
        self._broken_world: dict[str, bool] = {}

        self._loop = asyncio.get_running_loop()

    def __del__(self):
        """Cleanup the class instance."""
        del self._world_to_send_fn
        del self._world_to_recv_fn
        del self._broken_world

    def add_world(self, world_name: str, backend: str) -> None:
        """Add a new world to the world communicator.

        This method shouldn't be called directly by a user program.
        WorldManager will use this method.
        """
        self._set_functions(world_name, backend)

        self._broken_world[world_name] = False

    def remove_world(self, world_name: str) -> None:
        """Remove a world from the world communicator.

        This method shouldn't be called directly by a user program.
        WorldManager will use this method.
        """
        logger.debug(f"remove world {world_name}")

        self._reset_functions(world_name)

        try:
            self._broken_world[world_name] = True
        except KeyError:
            pass

    def is_broken(self, world_name: str) -> bool:
        """Return true if the given world is broken; otherwise return false.

        A world is considered broken if no key for the world name is found.

        Args:
            world_name: name of world

        Returns:
            A boolean value to indicate whether a world is broken or not.
        """
        logger.debug(f"check if world {world_name} is broken")
        return self._broken_world.get(world_name, True)

    def _set_functions(self, world_name: str, backend: str) -> None:
        if backend == "nccl":
            self._world_to_send_fn[world_name] = dist.isend
            self._world_to_recv_fn[world_name] = dist.irecv
        else:
            self._world_to_send_fn[world_name] = dist.send
            self._world_to_recv_fn[world_name] = dist.recv

    def _reset_functions(self, world_name: str) -> None:
        try:
            del self._world_to_send_fn[world_name]
        except KeyError:
            pass

        try:
            del self._world_to_recv_fn[world_name]
        except KeyError:
            pass

    def _get_fn(self, world_name: str, op: str) -> Callable:
        try:
            match op:
                case "send":
                    return self._world_to_send_fn[world_name]
                case "recv":
                    return self._world_to_recv_fn[world_name]
                case _:
                    raise KeyError()
        except KeyError:
            err_msg = f"function for {op} not found"
            raise BrokenWorldException(world_name, err_msg)

    async def _wait_work(self, work: Work, world_name: str) -> None:
        """Do busy-waiting for work to be done.

        It also checks if a world is broken or not. If so, it raises
        BrokenWorldException exception.
        """
        while not work.is_completed():
            if self._broken_world[world_name]:
                raise BrokenWorldException(world_name, "exception raised by watchdog")
            await asyncio.sleep(0)

    async def send(
        self, tensor: Tensor, dst: int, world_name: str = DEFAULT_WORLD_NAME
    ) -> None:
        """
        Send a tensor to a destination in a world.

        Args:
            tensor: Tensor to be sent.
            dst: Destination rank from the world.
            world_name: Name of the world.

        Raises:
            BrokenWorldException: An error that occurs when
                the world is broken due to worker, node or network failure.
        """
        fn = self._get_fn(world_name, "send")
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                work = await self._loop.run_in_executor(
                    pool,
                    fn,
                    tensor,
                    dst,
                    None,
                    0,
                    world_name,
                )
        except RuntimeError as e:
            self._handle_error(e, world_name)

        if isinstance(work, Work):
            await self._wait_work(work, world_name)

    async def recv(
        self, tensor: Tensor, src: int, world_name: str = DEFAULT_WORLD_NAME
    ) -> None:
        """
        Receive a tensor from a specific rank in a world.

        Args:
            tensor: Tensor to store received data.
            src: Source rank.
            world_name: Name of the world.

        Raises:
            BrokenWorldException: An error that occurs when
                the world is broken due to worker, node or network failure.
        """
        fn = self._get_fn(world_name, "recv")
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                work = await self._loop.run_in_executor(
                    pool,
                    fn,
                    tensor,
                    src,
                    None,
                    0,
                    world_name,
                )
        except RuntimeError as e:
            self._handle_error(e, world_name)

        if isinstance(work, Work):
            await self._wait_work(work, world_name)

    async def broadcast(
        self, tensor: Tensor, src: int, world_name: str = DEFAULT_WORLD_NAME
    ) -> None:
        """
        Broadcast a tensor from a source (src) to all other ranks in the same world.

        Args:
            tensor: Tensor to be broadcast.
            src: Source of the broadcast.
            world_name: Name of the world.

        Raises:
            BrokenWorldException: An error that occurs when
                the world is broken due to worker, node or network failure.
        """
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
        """
        Do all-reduce for a given tensor in a world.

        Args:
            tensor: used for all_reduce and to store the final result.
            op: One of the values from ``torch.distributed.ReduceOp``
            world_name: Name of the world.

        Raises:
            BrokenWorldException: An error that occurs when
                the world is broken due to worker, node or network failure.
        """
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
        The final result is only sent to the process with rank ``dst``.

        Args:
            tensor: reduced and to store the final result for rank ``dst``.
            dst: Rank to receive the final result (reduced tensor).
            op: One of the values from ``torch.distributed.ReduceOp``.
            world_name: Name of the world.

        Raises:
            BrokenWorldException: An error that occurs when
                the world is broken due to worker, node or network failure.
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
        """
        Do all-gather for a given tensor in a world.

        Args:
            tensors: Output list; it should contain correctly-sized tensors to store
                tensors gathered from all other ranks.
            tensor: Input tensor; tensor to be broadcast from current process to be used for gather.
            world_name: Name of the world.

        Raises:
            BrokenWorldException: An error that occurs when
                the world is broken due to worker, node or network failure.
        """
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
        """
        Do gather for a list of tensors in a world.

        Args:
            tensor: Input tensor; tensor to be used for gather.
            gather_list: List of correctly-sized tensors
                to use for gathered data (default is None, must be
                specified on the destination rank).
            dst: Rank to recieve the gathered tensors.
            world_name: Name of the world.

        Raises:
            BrokenWorldException: An error that occurs when
                the world is broken due to worker, node or network failure.
        """
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
        """
        Scatter a list of tensors from a source (src) to all other ranks in the same world.

        Args:
            tensor: Output tensor; the scattered tensor will be stored in this variable.
            scatter_list:  List of tensors to scatter (default is None,
                must be specified on the source rank).
            src: Rank that scatters tensors.
            world_name: Name of the world.

        Raises:
            BrokenWorldException: An error that occurs when
                the world is broken due to worker, node or network failure.
        """
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
