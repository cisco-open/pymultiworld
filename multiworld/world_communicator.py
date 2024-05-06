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
import logging
import threading
from asyncio import Queue as AsyncQ
from queue import SimpleQueue as SimpleSyncQ
from typing import TYPE_CHECKING, Union

import torch.distributed as dist
from torch import Tensor
from torch.distributed import Work

from multiworld.threadsafe_async import run_async

if TYPE_CHECKING:
    from torch.distributed.world_manager import WorldManager

logger = logging.getLogger(__name__)


WAIT_TIME_FOR_FIFO = 0.2  # 200 ms


_errors_to_handle = [
    "Connection closed by peer",
    "Connection reset by peer",
    "NCCL communicator was aborted",
]


class BrokenWorldException(Exception):
    """Raise this exception when world is broken."""

    def __init__(self, world_name: str):
        """Initialize exception instance."""
        self._world_name = world_name

    def __str__(self):
        """Return exception string."""
        return f"broken world: {self._world_name}"

    pass


class WorkStatus:  # noqa: D101
    SUCCESS = 0
    BROKEN = 1  # To indicate if the world is broken


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
        self._communication_threads = {}
        self._communication_commands = {}

        self._tensor_rx_q = SimpleSyncQ()

        self._loop = asyncio.get_running_loop()

    def __del__(self):
        """Cleanup the class instance."""
        for world_name in self._communication_threads:
            self.remove_world(world_name)

    def add_world(self, world_name):
        """Add a new world to the world comm manager."""
        input_q = SimpleSyncQ()
        event_q = AsyncQ()
        self._communication_commands[world_name] = (input_q, event_q)

        # NOTE(pranav): Might want to create separate threads for sending and receiving
        self._communication_threads[world_name] = threading.Thread(
            target=self._communication_thread,
            args=(world_name,),
            daemon=True,
        )
        self._communication_threads[world_name].start()

    def remove_world(self, world_name):
        """Remove a world from the world comm manager."""
        logger.debug(f"remove world {world_name}")
        try:
            # delete command for world as it is no longer needed
            del self._communication_commands[world_name]
        except KeyError:
            pass

        try:
            # delete thread entry for world as it is no longer needed
            del self._communication_threads[world_name]
        except KeyError:
            pass

    def _handle_work(self, work: Work) -> None:
        while True:
            try:
                # Enable lazy thread termination by using timeout and event
                _ = work.wait()
                # if no exception and we reach here, the work is done;
                # then we get out of the while loop
                break
            except RuntimeError as e:
                err_msg = str(e)
                logger.debug(err_msg)
                for error_snippet in _errors_to_handle:
                    if error_snippet in err_msg:
                        return WorkStatus.BROKEN

                raise e

        return WorkStatus.SUCCESS

    def _communication_thread(self, world_name: str):
        """Thread function to manage communication between worlds."""
        logger.debug(f"starting communication thread for {world_name}")
        input_q = self._communication_commands[world_name][0]
        event_q = self._communication_commands[world_name][1]

        while True:
            # This call blocks indefinitely until a command is received
            works: list[Work] = input_q.get()

            status = WorkStatus.SUCCESS
            for work in works:
                status = self._handle_work(work)
                if status != WorkStatus.SUCCESS:
                    break

            _, _ = run_async(event_q.put(status), self._loop)

            if status == WorkStatus.BROKEN:
                logger.debug(f"world {world_name} is broken")
                break

        try:
            # delete command for world as it is no longer needed
            del self._communication_commands[world_name]
        except KeyError:
            pass

        try:
            # delete thread entry for world as it is no longer needed
            del self._communication_threads[world_name]
        except KeyError:
            pass
        logger.debug(f"terminating comm. thread for {world_name}")

    async def send(
        self, tensors: Union[Tensor, list[Tensor]], world_name: str, rank: int
    ) -> None:
        """Send a tensor or a list of tensors to a specific rank in a world.

        This method supports batched send.
        """
        self._world_manager.set_world(world_name)

        self._send(tensors, world_name, rank)

        event_q = self._communication_commands[world_name][1]
        status = await event_q.get()
        if status == WorkStatus.BROKEN:
            self._world_manager.remove_world(world_name)
            raise BrokenWorldException(f"{world_name}")

    def _send(
        self, tensors: Union[Tensor, list[Tensor]], world_name: str, rank: int
    ) -> None:
        # Catch any errors due to worker failures
        try:
            if isinstance(tensors, Tensor):
                tensors = list(tensors)

            works = list()
            for tensor in tensors:
                work = dist.isend(tensor, dst=rank)
                works.append(work)

            input_q = self._communication_commands[world_name][0]
            input_q.put(works)

        except RuntimeError as e:
            self._handle_error(e, world_name)

    async def recv(
        self, tensors: Union[Tensor, list[Tensor]], world_name: str, rank: int
    ) -> None:
        """Receive tensor(s) from a specific rank in a world.

        This method supports batched receive.
        """
        self._world_manager.set_world(world_name)

        self._recv(tensors, world_name, rank)

        event_q = self._communication_commands[world_name][1]
        status = await event_q.get()
        if status == WorkStatus.BROKEN:
            self._world_manager.remove_world(world_name)
            raise BrokenWorldException(f"{world_name}")

    def _recv(self, tensors: Union[Tensor, list[Tensor]], world_name: str, rank: int):
        # Catch any errors due to worker failures
        try:
            if isinstance(tensors, Tensor):
                tensors = list(tensors)

            works = list()
            for tensor in tensors:
                work = dist.irecv(tensor, src=rank)
                works.append(work)

            input_q = self._communication_commands[world_name][0]
            input_q.put(works)

        except RuntimeError as e:
            self._handle_error(e, world_name)

    def _handle_error(self, error: RuntimeError, world_name: str):
        error_message = str(error)

        for error_snippet in _errors_to_handle:
            if error_snippet in error_message:
                logger.debug(f"broken world: {error_message}")
                self._world_manager.remove_world(world_name)
                raise BrokenWorldException(f"{world_name}")

        raise error

    @property
    def rx_q(self):
        """Return the rx queue for received tensors."""
        return self._tensor_rx_q
