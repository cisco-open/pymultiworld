"""
multiworld.py: Experiment with varying tensor sizes in multiple worlds.

Sample usage:
    Single host: python multiworld.py --tensor_size 1 --output_file multiworld_gloo_single.txt --iterations 5
    Multi host: python multiworld.py --multihost --addr 10.20.1.50 --rank 0 --output_file multiworld_gloo_multihost.txt --backend gloo --tensor_size 1
"""
#!/usr/bin/env python


import argparse
import atexit
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def dummy(world_name, rank, size, backend, tensor_size, iterations):
    """Run this only once."""

    print(f"dummy function: world: {world_name}, my rank: {rank}, world size: {size}")


def run(world_name, rank, size, backend, tensor_size, iterations):
    """Distributed function to be implemented later."""
    rank_to_send = 1 if rank == 0 else 0
    tensor = torch.ones(tensor_size) * rank

    tensor = tensor.to(f"cuda:{rank}") if backend == "nccl" else tensor

    for i in range(iterations):
        # Data exchange
        dist.send(tensor, dst=rank_to_send)


world_manager = None


def init_world(world_name, rank, size, fn, tensor_size, iterations, backend="gloo", addr="127.0.0.1", port=-1):
    """Initialize the distributed environment."""
    global world_manager

    if world_manager is None:
        # TODO: make WorldManager as singleton
        world_manager = dist.WorldManager()

    world_manager.initialize_world(
        world_name, rank, size, backend=backend, addr=addr, port=port
    )

    fn(world_name, rank, size, backend, tensor_size, iterations)


processes = []


def create_world(world_name, addr, port, backend, fn1, fn2, tensor_size, iterations):
    global processes

    size = 2
    for rank in range(size):
        if rank == 0:
            continue
        p = mp.Process(
            target=init_world, args=(world_name, rank, size, fn1, tensor_size, iterations, backend, addr, port)
        )
        p.start()
        print(p.pid)
        processes.append(p)

    # run master late
    init_world(world_name, 0, size, fn2, tensor_size, iterations, backend, addr, port)

    return processes


def cleanup():
    print("Cleaning up spwaned processes")
    for p in processes:
        p.terminate()

    print("Cleaning up done")


def receive_data_continuous(world_communicator, backend, tensor_size, iterations, output_file="performance_multiworld.txt"):
    bit = 0

    total_time = 0
    world1_tensor = torch.zeros(tensor_size)
    world1_tensor = (
        world1_tensor.to("cuda:0") if backend == "nccl" else world1_tensor
    )

    start_time = time.time()
    for i in range(iterations):
        world_manager.set_world("world1")
        dist.recv(world1_tensor, src=1)

    end_time = time.time()
    total_time += end_time - start_time

    with open(output_file, "a") as f:
        f.write(f"{tensor_size} {iterations} {total_time}\n")

def single_host(args):
    global processes

    mp.set_start_method("spawn")

    create_world("world1", "127.0.0.1", 29500, args.backend, run, dummy, args.tensor_size, args.iterations)

    receive_data_continuous(world_manager.communicator, args.backend, args.tensor_size, args.iterations, args.output_file)

    for p in processes:
        p.join()


def multi_host(args):
    size = 2
    if args.rank == 0:
        init_world("world1", args.rank, size, dummy, args.tensor_size, args.iterations, args.backend, args.addr, 29500)
        receive_data_continuous(world_manager.communicator, args.backend, args.tensor_size, args.iterations, args.output_file)
    elif args.rank == 1:
        init_world("world1", 1, size, run, args.tensor_size, args.iterations, args.backend, args.addr, 29500)
    else:
        print("rank error: rank should be 0, 1 or 2.")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--tensor_size", default=1, type=int)
    parser.add_argument("--iterations", default=1000, type=int)
    parser.add_argument("--output_file", default="performance_multiworld.txt")
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
