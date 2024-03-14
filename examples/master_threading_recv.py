"""main.py."""
#!/usr/bin/env python


import argparse
import atexit
import os
import random
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def dummy(world_name, rank, size, backend):
    """Run this only once."""

    print(f"dummy function: world: {world_name}, my rank: {rank}, world size: {size}")


def run(world_name, rank, size, backend):
    """Distributed function to be implemented later."""
    while True:
        # Data exchange
        print(f"run function: world: {world_name}, my rank: {rank}, world size: {size}")
        rank_to_send = 1 if rank == 0 else 0
        tensor = torch.ones(1)

        if world_name == "world2":
            tensor = torch.ones(1) * 2

        tensor = tensor.to(f"cuda:{rank}") if backend == "nccl" else tensor

        time.sleep(random.randint(1, 2))

        dist.send(tensor, dst=rank_to_send)
        print(
            f"run function: world: {world_name}, my rank: {rank}, world size: {size}, tensor = {tensor}"
        )


world_manager = None


def init_world(world_name, rank, size, fn, backend="gloo", port=-1):
    """Initialize the distributed environment."""
    global world_manager

    if world_manager is None:
        # TODO: make WorldManager as singleton
        world_manager = dist.WorldManager()

    world_manager.initialize_world(
        world_name, rank, size, backend=backend, port=int(port)
    )

    fn(world_name, rank, size, backend)


# @dist.WorldManager.world_initializer
def create_world(world_name, port, backend, fn1, fn2):
    size = 2
    processes = []
    for rank in range(size):
        if rank == 0:
            continue
        p = mp.Process(
            target=init_world, args=(world_name, rank, size, fn1, backend, port)
        )
        p.start()
        print(p.pid)
        processes.append(p)

    # run master late
    init_world(world_name, 0, size, fn2, backend, port)

    return processes


processes = []


def cleanup():
    print("Cleaning up spwaned processes")
    for p in processes:
        p.terminate()

    print("Cleaning up done")


def receive_data_continuous(world_communicator, backend):
    bit = 0

    while True:
        world2_tensor = torch.zeros(1)
        world2_tensor = (
            world2_tensor.to("cuda:0") if backend == "nccl" else world2_tensor
        )
        world_communicator.recv(world2_tensor, "world2", 1)

        world1_tensor = torch.zeros(1)
        world1_tensor = (
            world1_tensor.to("cuda:0") if backend == "nccl" else world1_tensor
        )
        world_communicator.recv(world1_tensor, "world1", 1)

        # Empty the queue until we reach and Exception using get_nowait
        try:
            while True:
                tensor = world_communicator.rx_q.get_nowait()
                print(f"Received tensor: {tensor}")
        except Exception as e:
            print(e)
            pass

        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo")
    args = parser.parse_args()
    atexit.register(cleanup)

    size = 2
    mp.set_start_method("spawn")

    pset = create_world("world1", "29500", args.backend, run, dummy)
    processes += pset

    pset = create_world("world2", "30500", args.backend, run, dummy)
    processes += pset

    print("here")

    receive_data_continuous(world_manager.communicator, args.backend)

    for p in processes:
        p.join()
