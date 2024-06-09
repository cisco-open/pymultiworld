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
from queue import SimpleQueue as SimpleSyncQ
from typing import TYPE_CHECKING, Union

import torch.distributed as dist
from torch import Tensor
from torch.distributed import Work

if TYPE_CHECKING:
    from torch.distributed.world_manager import WorldManager

logger = logging.getLogger(__name__)


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

        self._tensor_rx_q = SimpleSyncQ()

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

    async def send(
        self, tensors: Union[Tensor, list[Tensor]], world_name: str, rank: int
    ) -> None:
        """Send a tensor or a list of tensors to a specific rank in a world.

        This method supports batched send.
        """
        works = self._send(tensors, world_name, rank)
        for w in works:
            while not w.is_completed():
                if self._broken_world[world_name]:
                    raise BrokenWorldException(f"{world_name}")
                await asyncio.sleep(0)

    def _send(
        self,
        tensors: Union[Tensor, list[Tensor]],
        world_name: str,
        rank: int,
    ) -> list[Work]:
        # Catch any errors due to worker failures
        try:
            if isinstance(tensors, Tensor):
                tensors = [tensors]

            works = []
            for tensor in tensors:
                work = dist.isend(tensor, dst=rank, name=world_name)
                works.append(work)

            return works
        except RuntimeError as e:
            self._handle_error(e, world_name)

    async def recv(
        self, tensors: Union[Tensor, list[Tensor]], world_name: str, rank: int
    ) -> None:
        """Receive tensor(s) from a specific rank in a world.

        This method supports batched receive.
        """
        works = self._recv(tensors, world_name, rank)
        for w in works:
            while not w.is_completed():
                if self._broken_world[world_name]:
                    raise BrokenWorldException(f"{world_name}")
                await asyncio.sleep(0)

    def _recv(
        self,
        tensors: Union[Tensor, list[Tensor]],
        world_name: str,
        rank: int,
    ) -> list[Work]:
        # Catch any errors due to worker failures
        try:
            if isinstance(tensors, Tensor):
                tensors = [tensors]

            works = []
            for tensor in tensors:
                work = dist.irecv(tensor, src=rank, name=world_name)
                works.append(work)

            return works
        except RuntimeError as e:
            self._handle_error(e, world_name)

    def _handle_error(self, error: RuntimeError, world_name: str) -> None:
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
