"""Class to create and manage multiple worlds."""
# NOTE: This is a hack to get around the fact that the torch.distributed package
#       is not designed to support multiple worlds. This is a stop-gap solution
#       until the torch.distributed package is updated to support multiple worlds.

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


class C10dWorld:
    """Class to keep the global variables of distributed c10d."""

    def __init__(self, world):
        """Initialize an instance."""
        self.world = world
        self.pg_map = {}
        self.pg_names = {}
        self.pg_group_ranks = {}
        self.pg_backend_config = {}
        self.group_count = 0
        self.backend = "undefined"


class WorldManager:
    """WorldManager class."""

    def __init__(self):
        """Initialize a world manager."""
        # Map from world_name to C10dWorld
        self._worlds = dict()
        self._worlds_stores: dict[dist.TCPStore] = dict()
        self._communicator = WorldCommunicator(self)
        self._current_world = ""

        self._event_q = SyncQ()
        self._action_q = ASyncQ()

        self._watchdog = WatchDog(self._event_q, self._action_q)

        _ = asyncio.create_task(self._cleanup_worlds())

    def cleanup(self):
        """Call exit(0) explicitly."""
        # TODO: This is a temporary workaround to prevent main thread hang
        #       even after it's done. Calling exit(0) guarantees termination
        #       of the process. We need to figure out why sometimes it's not
        #       terminated without explicit call of exit().
        exit(0)

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
            group_name=world_name,
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

        self.set_world(world_name)

        self._init_process_group(
            world_name,
            rank,
            world_size,
            backend,
            addr,
            port,
        )

        dist_c10d._world.default_pg = dist_c10d.GroupMember.WORLD

        # inform watchdog of addition of a new world
        store = self._worlds_stores[world_name]
        self._event_q.put((store, world_name, rank, world_size))

    def add_world(self, world_name, world=None):
        """Add a new world to the world manager."""
        if world_name in self._worlds:
            raise ValueError(f"World {world_name} already exists.")

        if world is None:
            world = dist_c10d._World()

        c10dworld = C10dWorld(world)
        self._worlds[world_name] = c10dworld
        self._communicator.add_world(world_name)

    def remove_world(self, world_name):
        """Remove a world from the world manager."""
        if world_name not in self._worlds:
            raise ValueError(f"World {world_name} does not exist.")

        self._communicator.remove_world(world_name)

        logger.debug(f"remove {world_name} from world stores")
        del self._worlds_stores[world_name]

        logger.debug(f"destory process group for {world_name}")
        self.set_world(world_name)
        del self._worlds[world_name]
        dist.destroy_process_group()
        logger.debug(f"done removing world {world_name}")

    def set_world(self, world_name):
        """Switch to a world of the given name."""
        if world_name not in self._worlds:
            raise ValueError(f"World {world_name} does not exist.")

        if self._current_world == world_name:
            return

        logger.debug(f"Setting world to {world_name}")

        c10dworld = self._worlds[world_name]

        dist_c10d._world = c10dworld.world
        dist_c10d.GroupMember.WORLD = dist_c10d._world.default_pg

        dist_c10d._pg_map = c10dworld.pg_map
        dist_c10d._pg_names = c10dworld.pg_names
        dist_c10d._pg_group_ranks = c10dworld.pg_group_ranks
        dist_c10d._pg_backend_config = c10dworld.pg_backend_config
        dist_c10d._group_count = c10dworld.group_count
        dist_c10d._backend = c10dworld.backend

        self._current_world = world_name

    @property
    def communicator(self):
        """Return the world communicator."""
        return self._communicator

    @staticmethod
    def world_setter(func):
        def wrapper(*args, **kwargs):
            world_name = kwargs["world_name"]
            world_manager = kwargs["world_manager"]

            # Delete world_name and world_manager from kwargs
            del kwargs["world_name"]
            del kwargs["world_manager"]

            world_manager.set_world(world_name)

            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def world_initializer(func):
        def wrapper(*args, **kwargs):
            world_name = kwargs["world_name"]
            world_manager = kwargs["world_manager"]

            # Delete world_name and world_manager from kwargs
            del kwargs["world_name"]
            del kwargs["world_manager"]

            world_manager.set_world(world_name)

            ret_val = func(*args, **kwargs)

            dist_c10d._world.default_pg = dist_c10d.GroupMember.WORLD

            return ret_val

        return wrapper
