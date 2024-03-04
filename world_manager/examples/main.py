"""main.py."""
#!/usr/bin/env python


import os
import time
from datetime import timedelta

import torch.multiprocessing as mp

import torch
import torch.distributed as dist

import atexit
import copy


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
        print(f">>>>>>>>>> group_count = {dist._world.group_count}")
        print(f">>>>>>>>>> _group_count = {dist._group_count}")
        dist._world = dist._World()
        print(f"<<<<<<<< group_count = {dist._world.group_count}")
        print(f"<<<<<<<< _group_count = {dist._group_count}")
        print(f"addr of _world after reset = {hex(id(dist._world))}")
        dist.GroupMember.WORLD = None
        dist._backend = "undefined"
        dist._pg_map = {}
        dist._pg_names = {}
        dist._pg_group_ranks = {}
        # For a pg, it is a map from ProcessGroup to BackendConfig
        dist._pg_backend_config = {}
        dist._group_count = 0

    def swap_worlds(self):
        temp_world = dist._world
        temp_pg_map = dist._pg_map
        temp_pg_names = dist._pg_names
        temp_pg_group_ranks = dist._pg_group_ranks
        temp_pg_backend_config = dist._pg_backend_config
        temp_group_count = dist._group_count
        temp_backend = dist._backend

        dist._world = self._world
        dist._pg_map = self._pg_map
        dist._pg_names = self._pg_names
        dist._pg_group_ranks = self._pg_group_ranks
        dist._pg_backend_config = self._pg_backend_config
        dist._group_count = self._group_count
        dist._backend = self._backend

        self._world = temp_world
        self._pg_map = temp_pg_map
        self._pg_names = temp_pg_names
        self._pg_group_ranks = temp_pg_group_ranks
        self._pg_backend_config = temp_pg_backend_config
        self._group_count = temp_group_count
        self._backend = temp_backend


def dummy(world_name, rank, size):
    """Run this only once."""

    print(f"dummy function: world: {world_name}, my rank: {rank}, world size: {size}")


def run(world_name, rank, size):
    """Distributed function to be implemented later."""
    while True:
        # Data exchange
        print(f"run function: world: {world_name}, my rank: {rank}, world size: {size}")
        rank_to_recv = 1 if rank == 0 else 0
        tensor = torch.zeros(1)

        dist.recv(tensor, src=rank_to_recv)
        print(f"run function: world: {world_name}, my rank: {rank}, world size: {size}, tensor = {tensor}")


def init_process(port, world_name, rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    print(f"{os.getpid()} port = {port}")
    store = dist.TCPStore(
        "127.0.0.1", int(port), 2, True if rank == 0 else False, timedelta(seconds=30)
    )
    print(f"tcp store: {store}")
    dist.init_process_group(backend, rank=rank, world_size=size, store=store, group_name=world_name)
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
        print(p.pid)
        processes.append(p)

    # run master late
    init_process(port, world_name, 0, size, fn2)

    return processes

processes = []

def cleanup():
    print("Cleaning up spwaned processes")
    for p in processes:
        p.terminate()

    print("Cleaning up done")


def send_data():
    try:
        print("send_data function: sending data from master")

        rank_to_send = 1
        tensor = torch.ones(1)
        dist.send(tensor, dst=rank_to_send)

        print(f"send_data function: data sent from master, tensor = {tensor}")
    except RuntimeError as e:
        error_message = str(e)

        if "Connection closed by peer" in error_message:
            print("Ignoring Connection closed by peer error")
        elif "Connection reset by peer" in error_message:
            print("Ignoring Connection reset by peer error")
        else:
            raise e


def send_data_continuous(world_backup):
    while True:
        send_data()

        # world_backup.swap_worlds()
        time.sleep(2)


if __name__ == "__main__":
    atexit.register(cleanup)

    size = 2
    mp.set_start_method("spawn")

    pset = create_world("29500", "world1", run, dummy)
    processes += pset

    # save distributed config
    # launch the 2nd world
    world_backup = WorldBackup(dist)
    WorldBackup.reset(dist)

    pset = create_world("30500", "world2", run, dummy)
    processes += pset

    print("here")

    # send data from master to world2
    send_data_continuous(world_backup)

    for p in processes:
        p.join()
