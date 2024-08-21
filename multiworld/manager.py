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

"""Class to create and manage multiple worlds."""
import asyncio
import concurrent.futures
import logging
import os
import sys
from asyncio import Queue as ASyncQ
from datetime import timedelta
from queue import Queue as SyncQ

import torch.distributed as dist
from torch.distributed import _World as dist_c10d_World
from torch.distributed import _worlds as dist_c10d_worlds

from multiworld.communicator import WorldCommunicator
from multiworld.watchdog import WatchDog

logger = logging.Logger(__name__)


class WorldManager:
    """WorldManager class."""

    def __init__(self, enable_monitor=True):
        """Initialize a world manager."""
        # https://github.com/pytorch/pytorch/blob/v2.4.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp#L118-L130
        # "2" is CleanUpOnly
        # We use CleanupOnly in order to allow error handling at user process
        # level without tearing down the process.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "2"

        self._worlds_stores: dict[str, dist.TCPStore] = dict()
        self._communicator = WorldCommunicator(self)
        self._current_world = ""

        self._event_q = SyncQ()
        self._action_q = ASyncQ()

        if enable_monitor:
            self._watchdog = WatchDog(self._event_q, self._action_q)

            _ = asyncio.create_task(self._cleanup_worlds())

    def cleanup(self):
        """
        Clean up world manager and terminate the process.

        Note: two tasks are executed.
            1) Flush out the buffered output of print function to stdout
            2) Terminate the process by calling os._exit(0)

        Calling os._exit(0) ensures termination of the process. This is a workaround
        to prevent the main thread hang after its work is over. Therefore, in the user
        application, no additional code will be executed after calling this function.
        """
        # TODO: This is a temporary workaround to prevent main thread hang
        #       even after it's done. Calling os._exit(0) guarantees
        #       terminationof the process. We need to figure out why
        #       sometimes it's not terminated without explicit call of
        #       os._exit(0).
        sys.stdout.flush()
        os._exit(0)

    async def _cleanup_worlds(self):
        logger.debug("starting _cleanup_worlds task")
        while True:
            world = await self._action_q.get()
            logger.debug(f"[_cleanup_worlds] remove world {world}")
            try:
                self.remove_world(world)
            except ValueError:
                # this may be because the world was already removed
                # in a different code path; just ignore the error
                pass

    def _init_process_group(
        self,
        world_name: str,
        rank: int,
        world_size: int,
        backend="gloo",
        addr: str = "127.0.0.1",
        port: int = -1,
    ):
        """Initialize the distributed environment."""
        logger.info(f"({os.getpid()}) backend= {backend}, port = {port}")
        store = dist.TCPStore(
            addr,
            port,
            world_size,
            True if rank == 0 else False,
            timedelta(seconds=30),
        )

        logger.debug(f"({os.getpid()}) tcp store: {store}")
        dist.init_process_group(
            backend,
            rank=rank,
            world_size=world_size,
            store=store,
            world_name=world_name,
        )

        self._worlds_stores[world_name] = store
        logger.info(f"({os.getpid()}) init_process_group done")

    async def initialize_world(
        self,
        world_name: str,
        rank: int,
        world_size: int,
        backend="gloo",
        addr: str = "127.0.0.1",
        port: int = -1,
    ):
        """
        Initialize a world for a given rank using world name, backend, port number and address.

        Args:
            world_name: Name of the world.
            rank: Rank of the current process (it should be a number between 0 and ``world_size``-1).
            world_size: the number of processes participating in the world.
            backend: Backend used for communication; nccl and gloo are supported currently.
            addr: host name or IP address.
            port: Port number.
        """
        self.add_world(world_name, backend)

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            _ = await loop.run_in_executor(
                pool,
                self._init_process_group,
                world_name,
                rank,
                world_size,
                backend,
                addr,
                port,
            )

        # inform watchdog of addition of a new world
        store = self._worlds_stores[world_name]
        self._event_q.put((store, world_name, rank, world_size))

    def add_world(self, world_name: str, backend: str) -> None:
        """Add a new world to the world manager."""
        if world_name in dist_c10d_worlds:
            raise ValueError(f"World {world_name} already exists.")

        world = dist_c10d_World(world_name)

        dist_c10d_worlds[world_name] = world

        self._communicator.add_world(world_name, backend)

    def remove_world(self, world_name: str) -> None:
        """Remove a world from the world manager."""
        if world_name not in dist_c10d_worlds:
            raise ValueError(f"World {world_name} does not exist.")

        self._communicator.remove_world(world_name)

        logger.debug(f"remove {world_name} from world stores")
        try:
            del self._worlds_stores[world_name]
        except KeyError:
            pass

        logger.debug(f"destory process group for {world_name}")
        # FIXME: the following two lines of code here causes program hang.
        #        we need to find out a right timing/way to call them.
        #        calling them is temporarily disabled.
        # dist.destroy_process_group(name=world_name)
        # del dist_c10d_worlds[world_name]
        logger.debug(f"done removing world {world_name}")

    @property
    def communicator(self):
        """Return the world communicator."""
        return self._communicator
