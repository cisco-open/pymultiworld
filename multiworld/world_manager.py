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
import logging
import os
from asyncio import Queue as ASyncQ
from datetime import timedelta
from queue import Queue as SyncQ

import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d
from torch.distributed.world_communicator import WorldCommunicator

from multiworld.watchdog import WatchDog

logger = logging.Logger(__name__)


class WorldManager:
    """WorldManager class."""

    def __init__(self, enable_monitor=True):
        """Initialize a world manager."""
        self._worlds_stores: dict[str, dist.TCPStore] = dict()
        self._communicator = WorldCommunicator(self)
        self._current_world = ""

        self._event_q = SyncQ()
        self._action_q = ASyncQ()

        if enable_monitor:
            self._watchdog = WatchDog(self._event_q, self._action_q)

            _ = asyncio.create_task(self._cleanup_worlds())

    def cleanup(self):
        """Call os._exit(0) explicitly."""
        # TODO: This is a temporary workaround to prevent main thread hang
        #       even after it's done. Calling os._exit(0) guarantees
        #       terminationof the process. We need to figure out why
        #       sometimes it's not terminated without explicit call of
        #       os._exit(0).
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

    def initialize_world(
        self,
        world_name: str,
        rank: int,
        world_size: int,
        backend="gloo",
        addr: str = "127.0.0.1",
        port: int = -1,
    ):
        """Initialize world."""
        self.add_world(world_name)

        self._init_process_group(
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

    def add_world(self, world_name, world=None):
        """Add a new world to the world manager."""
        if world_name in dist_c10d._worlds:
            raise ValueError(f"World {world_name} already exists.")

        if world is None:
            world = dist_c10d._World(world_name)

        dist_c10d._worlds[world_name] = world

        self._communicator.add_world(world_name)

    def remove_world(self, world_name):
        """Remove a world from the world manager."""
        if world_name not in dist_c10d._worlds:
            raise ValueError(f"World {world_name} does not exist.")

        self._communicator.remove_world(world_name)

        logger.debug(f"remove {world_name} from world stores")
        del self._worlds_stores[world_name]

        logger.debug(f"destory process group for {world_name}")
        # FIXME: the following two lindes of code here causes program hang.
        #        we need to find out a right timing/way to call them.
        #        calling them is temporarily disabled.
        # dist.destroy_process_group(name=world_name)
        # del dist_c10d._worlds[world_name]
        logger.debug(f"done removing world {world_name}")

    @property
    def communicator(self):
        """Return the world communicator."""
        return self._communicator
