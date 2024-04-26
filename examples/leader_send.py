"""
leader_send.py: This example demonstrates how to send data from a leader process to a worker process in a distributed setup.
"""
#!/usr/bin/env python


import os
import time
from datetime import timedelta

import torch.multiprocessing as mp

import torch
import torch.distributed as dist

import atexit
import copy


def dummy(world_name, rank, size):
    """
    Dummy function to be implemented later. Arguments are similar to run function.

    Args:
        world_name (str): Name of the world.
        rank (int): Rank of the process.
        size (int): Number of processes.
    """

    print(f"dummy function: world: {world_name}, my rank: {rank}, world size: {size}")


def run(world_name, rank, size):
    """
    Distributed function to be implemented later.

    Args:
        world_name (str): Name of the world.
        rank (int): Rank of the process.
        size (int): Number of processes.
    """
    while True:
        # Data exchange
        print(f"run function: world: {world_name}, my rank: {rank}, world size: {size}")
        rank_to_recv = 1 if rank == 0 else 0
        tensor = torch.zeros(1)

        dist.recv(tensor, src=rank_to_recv)
        print(f"run function: world: {world_name}, my rank: {rank}, world size: {size}, tensor = {tensor}")


def init_process(port, world_name, rank, size, fn, backend="gloo"):
    """
    Initialize the distributed environment.
    
    Args:
        port (str): Port number.
        world_name (str): Name of the world.
        rank (int): Rank of the process.
        size (int): Number of processes.
        fn (function): Function to be executed.
        backend (str): Backend to be used.
    """
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


@dist.WorldManager.world_initializer
def create_world(port, world_name, fn1, fn2):
    """
    Create a world with the given port and world name.

    Args:
        port (str): Port number.
        world_name (str): Name of the world.
        fn1 (function): Function to be executed.
        fn2 (function): Function to be executed.

    Returns:
        list: List of processes.
    """
    size = 2
    processes = []
    for rank in range(size):
        if rank == 0:
            continue
        p = mp.Process(target=init_process, args=(port, world_name, rank, size, fn1))
        p.start()
        print(p.pid)
        processes.append(p)

    # run leader late
    init_process(port, world_name, 0, size, fn2)

    return processes

processes = []

def cleanup():
    """Cleanup function to terminate all spawned processes."""
    print("Cleaning up spwaned processes")
    for p in processes:
        p.terminate()

    print("Cleaning up done")


def send_data(tensor):
    """
    Send data from leader to a worker process and ignore connection errors.

    Args:
        tensor (torch.Tensor): Data to be sent.
    """
    try:
        print("send_data function: sending data from leader")

        rank_to_send = 1
        # tensor = torch.ones(1)
        dist.send(tensor, dst=rank_to_send)

        print(f"send_data function: data sent from leader, tensor = {tensor}")
    except RuntimeError as e:
        error_message = str(e)

        if "Connection closed by peer" in error_message:
            print("Ignoring Connection closed by peer error")
        elif "Connection reset by peer" in error_message:
            print("Ignoring Connection reset by peer error")
        else:
            raise e


def send_data_continuous(world_manager):
    """
    Send data from leader to a worker process continuously.

    Args:
        world_manager (dist.WorldManager): World manager object.
    """
    bit = 0

    while True:
        if bit == 0:
            world_manager.set_world("world2")
            tensor = torch.ones(1) * 2
        else:
            world_manager.set_world("world1")
            tensor = torch.ones(1) * 1

        send_data(tensor)

        # flip bit
        bit = 1 - bit

        time.sleep(2)


if __name__ == "__main__":
    atexit.register(cleanup)

    world_manager = dist.WorldManager()
    world_manager.add_world("world1")
    world_manager.add_world("world2")

    size = 2
    mp.set_start_method("spawn")

    pset = create_world("29500", "world1", run, dummy, world_name="world1", world_manager=world_manager)
    processes += pset

    pset = create_world("30500", "world2", run, dummy, world_name="world2", world_manager=world_manager)
    processes += pset

    print("here")

    # send data from leader to world2
    send_data_continuous(world_manager)

    for p in processes:
        p.join()
