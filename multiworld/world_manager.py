"""Class to create and manage multiple worlds."""
# NOTE: This is a hack to get around the fact that the torch.distributed package
#       is not designed to support multiple worlds. This is a stop-gap solution
#       until the torch.distributed package is updated to support multiple worlds.

import os
from datetime import timedelta

import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d
from torch.distributed.world_communicator import WorldCommunicator


class C10dWorld:
    def __init__(self, world):
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
        self._worlds = {}
        self._communicator = WorldCommunicator(self)

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
        # os.environ["MASTER_ADDR"] = addr
        # os.environ["MASTER_PORT"] = port
        print(f"({os.getpid()}) backend= {backend}, port = {port}")
        store = dist.TCPStore(
            addr,
            port,
            world_size,
            True if rank == 0 else False,
            timedelta(seconds=30),
        )
        print(f"({os.getpid()}) tcp store: {store}")
        dist.init_process_group(
            backend,
            rank=rank,
            world_size=world_size,
            store=store,
            group_name=world_name,
        )
        print(f"({os.getpid()}) init_process_group done")

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

    def add_world(self, world_name, world=None):
        """Add a new world to the world manager."""
        if world is None:
            world = dist_c10d._World()

        if world_name in self._worlds:
            raise ValueError(f"World {world_name} already exists.")

        c10dworld = C10dWorld(world)
        self._worlds[world_name] = c10dworld
        self._communicator.add_world(world_name)

    def remove_world(self, world_name):
        """Remove a world from the world manager."""
        if world_name not in self._worlds:
            raise ValueError(f"World {world_name} does not exist.")
        del self._worlds[world_name]

        self._communicator.remove_world(world_name)

    def set_world(self, world_name):
        """Switch to a world of the given name."""
        if world_name not in self._worlds:
            raise ValueError(f"World {world_name} does not exist.")

        print(f"Setting world to {world_name}")

        c10dworld = self._worlds[world_name]

        dist_c10d._world = c10dworld.world
        dist_c10d.GroupMember.WORLD = dist_c10d._world.default_pg

        dist_c10d._pg_map = c10dworld.pg_map
        dist_c10d._pg_names = c10dworld.pg_names
        dist_c10d._pg_group_ranks = c10dworld.pg_group_ranks
        dist_c10d._pg_backend_config = c10dworld.pg_backend_config
        dist_c10d._group_count = c10dworld.group_count
        dist_c10d._backend = c10dworld.backend

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
