"""main.py."""
#!/usr/bin/env python


import atexit
import copy
import os
import random
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from multiworld.comm_manager import WorldCommunicationManager


def dummy(world_name, rank, size):
    """Run this only once."""

    print(f"dummy function: world: {world_name}, my rank: {rank}, world size: {size}")


def run(world_name, rank, size):
    """Distributed function to be implemented later."""
    while True:
        # Data exchange
        print(f"run function: world: {world_name}, my rank: {rank}, world size: {size}")
        rank_to_send = 1 if rank == 0 else 0
        tensor = torch.ones(1)

        if world_name == "world2":
            tensor = torch.ones(1) * 2

        time.sleep(random.randint(1, 2))

        dist.send(tensor, dst=rank_to_send)
        print(
            f"run function: world: {world_name}, my rank: {rank}, world size: {size}, tensor = {tensor}"
        )


def init_process(port, world_name, rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    print(f"{os.getpid()} port = {port}")
    store = dist.TCPStore(
        "127.0.0.1", int(port), 2, True if rank == 0 else False, timedelta(seconds=30)
    )
    print(f"tcp store: {store}")
    dist.init_process_group(
        backend, rank=rank, world_size=size, store=store, group_name=world_name
    )
    # dist.init_process_group(backend, rank=rank, world_size=size)
    print("init_process_group done")
    fn(world_name, rank, size)


@dist.WorldManager.world_initializer
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


def receive_data_continuous(world_communication_manager):
    bit = 0

    while True:
        world2_tensor = torch.zeros(1)
        world_communication_manager.recv(world2_tensor, "world2", 1)

        world1_tensor = torch.zeros(1)
        world_communication_manager.recv(world1_tensor, "world1", 1)

        # Empty the queue until we reach and Exception using get_nowait
        try:
            while True:
                tensor = world_communication_manager.received_tensors.get_nowait()
                print(f"Received tensor: {tensor}")
        except:
            pass

        time.sleep(2)


if __name__ == "__main__":
    atexit.register(cleanup)

    world_manager = dist.WorldManager()
    world_communication_manager = WorldCommunicationManager(world_manager)
    world_communication_manager.add_world("world1")
    world_communication_manager.add_world("world2")

    size = 2
    mp.set_start_method("spawn")

    pset = create_world(
        "29500", "world1", run, dummy, world_name="world1", world_manager=world_manager
    )
    processes += pset

    pset = create_world(
        "30500", "world2", run, dummy, world_name="world2", world_manager=world_manager
    )
    processes += pset

    print("here")

    receive_data_continuous(world_communication_manager)

    for p in processes:
        p.join()
