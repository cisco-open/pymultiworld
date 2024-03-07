"""Class to create and manage multiple worlds."""
# NOTE: This is a hack to get around the fact that the torch.distributed package
#       is not designed to support multiple worlds. This is a stop-gap solution
#       until the torch.distributed package is updated to support multiple worlds.

import torch.distributed.distributed_c10d as dist_c10d

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
    def __init__(self):
        # Map from world_name to C10dWorld
        self._worlds = {}

    def add_world(self, world_name, world=None):
        if world is None:
            world = dist_c10d._World()

        if world_name in self._worlds:
            raise ValueError(f"World {world_name} already exists.")

        c10dworld = C10dWorld(world)
        self._worlds[world_name] = c10dworld

    def remove_world(self, world_name):
        if world_name not in self._worlds:
            raise ValueError(f"World {world_name} does not exist.")
        del self._worlds[world_name]

    def set_world(self, world_name):
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
