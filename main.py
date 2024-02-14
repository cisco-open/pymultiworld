"""main.py."""
#!/usr/bin/env python


import os
import time
from datetime import timedelta

import torch.multiprocessing as mp

import torch
import torch.distributed as dist


class WorldBackup:
    def __init__(self, dist):
        self._world = dist._world
        print(f"addr of _world = {hex(id(dist._world))}")

        self._pg_map = dist._pg_map
        self._pg_names = dist._pg_names
        self._pg_group_ranks = dist._pg_group_ranks
        self._pg_backend_config = dist._pg_backend_config
        self._group_count = dist._group_count

    @classmethod
    def reset(cls, dist):
        dist._world = dist._World()
        print(f"addr of _world after reset = {hex(id(dist._world))}")
        dist.GroupMember.WORLD = None
        dist._pg_map = {}
        dist._pg_names = {}
        dist._pg_group_ranks = {}
        # For a pg, it is a map from ProcessGroup to BackendConfig
        dist._pg_backend_config = {}
        dist._group_count = 0


def dummy(world_name, rank, size):
    """Run this only once."""
    print(f"dummy function: world: {world_name}, my rank: {rank}, world size: {size}")


def run(world_name, rank, size):
    """Distributed function to be implemented later."""
    while True:
        print(f"run function: world: {world_name}, my rank: {rank}, world size: {size}")
        time.sleep(3)


def init_process(port, world_name, rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    print(f"port = {port}")
    store = dist.TCPStore(
        "127.0.0.1", int(port), 2, True if rank == 0 else False, timedelta(seconds=30)
    )
    dist.init_process_group(backend, rank=rank, world_size=size, store=store)
    # dist.init_process_group(backend, rank=rank, world_size=size)
    print("init_process_group done")
    fn(world_name, rank, size)


def create_world(port, world_name, fn1, fn2):
    size = 2
    processes = []
    for rank in range(size):
        if rank == 0:
            continue
        p = mp.Process(target=init_process, args=(port, world_name, rank, size, fn1))
        p.start()
        processes.append(p)

    # run master late
    init_process(port, world_name, 0, size, fn2)

    return processes


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")

    pset = create_world("30500", "world2", run, dummy)
    processes += pset

    # save distributed config
    # launch the 2nd world
    world_backup = WorldBackup(dist)
    WorldBackup.reset(dist)

    pset = create_world("29500", "world1", run, dummy)
    processes += pset

    print("here")

    for p in processes:
        p.join()
