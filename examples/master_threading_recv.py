"""main.py."""
#!/usr/bin/env python


import argparse
import atexit
import os
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

        time.sleep(4)

        dist.send(tensor, dst=rank_to_send)
        print(
            f"run function: world: {world_name}, my rank: {rank}, world size: {size}, tensor = {tensor}"
        )


world_manager = None


def init_world(world_name, rank, size, fn, backend="gloo", addr="127.0.0.1", port=-1):
    """Initialize the distributed environment."""
    global world_manager

    if world_manager is None:
        # TODO: make WorldManager as singleton
        world_manager = dist.WorldManager()

    world_manager.initialize_world(
        world_name, rank, size, backend=backend, addr=addr, port=port
    )

    fn(world_name, rank, size, backend)


# @dist.WorldManager.world_initializer
def create_world(world_name, addr, port, backend, fn1, fn2):
    size = 2
    processes = []
    for rank in range(size):
        if rank == 0:
            continue
        p = mp.Process(
            target=init_world, args=(world_name, rank, size, fn1, backend, addr, port)
        )
        p.start()
        print(p.pid)
        processes.append(p)

    # run master late
    init_world(world_name, 0, size, fn2, backend, addr, port)

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
        world1_tensor = torch.zeros(1)
        world1_tensor = (
            world1_tensor.to("cuda:0") if backend == "nccl" else world1_tensor
        )
        world_communicator.recv(world1_tensor, "world1", 1)

        time.sleep(2)

        world2_tensor = torch.zeros(1)
        world2_tensor = (
            world2_tensor.to("cuda:0") if backend == "nccl" else world2_tensor
        )
        world_communicator.recv(world2_tensor, "world2", 1)

        time.sleep(2)

        # Empty the queue until we reach and Exception using get_nowait
        try:
            while True:
                tensor = world_communicator.rx_q.get_nowait()
                print(f"Received tensor: {tensor}")
        except Exception as e:
            print(e)
            pass


def single_host(args):
    processes = []
    mp.set_start_method("spawn")

    pset = create_world("world1", "127.0.0.1", 29500, args.backend, run, dummy)
    processes += pset

    pset = create_world("world2", "127.0.0.1", 30500, args.backend, run, dummy)
    processes += pset

    receive_data_continuous(world_manager.communicator, args.backend)

    for p in processes:
        p.join()


def multi_host(args):
    size = 2
    if args.rank == 0:
        init_world("world1", args.rank, size, dummy, args.backend, args.addr, 29500)
        init_world("world2", args.rank, size, dummy, args.backend, args.addr, 30500)
        receive_data_continuous(world_manager.communicator, args.backend)

    elif args.rank == 1:
        init_world("world1", 1, size, run, args.backend, args.addr, 29500)

    elif args.rank == 2:
        init_world("world2", 1, size, run, args.backend, args.addr, 30500)

    else:
        print("rank error: rank should be 0, 1 or 2.")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--rank", type=int)
    parser.add_argument(
        "--multihost", action=argparse.BooleanOptionalAction, default=False
    )

    # https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp#L114-L126
    # "2" is CleanUpOnly
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "2"

    args = parser.parse_args()
    atexit.register(cleanup)

    if not args.multihost:
        single_host(args)
    else:
        multi_host(args)
