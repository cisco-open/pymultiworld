"""
single_world.py: Experiment with varying tensor sizes in a single world.

Sample usage:
    Single host: python single_world.py --tensor_size 1 --output_file singleworld_gloo_single.txt
    Multi host: python single_world.py --multihost --addr 10.20.1.50 --rank 0 --output_file singleworld_nccl_multihost.txt --backend nccl --tensor_size 1
"""
#!/usr/bin/env python
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(backend, rank, world_size, tensor_size, iterations, output_file="performance_singleworld.txt"):
    """Distributed function to be implemented later."""
    runtime_error_peers = set()
    if rank == 0:
        # In this experiment, we are going to use only 1 master and 1 worker
        src = 1

        total_time = 0
        tensor = torch.zeros(tensor_size)
        tensor = tensor.to(f"cuda:{rank}") if backend == "nccl" else tensor

        start_time = time.time()
        for i in range(iterations):
            if src in runtime_error_peers:
                print(f"Rank {src}'s connection is aborted")
                time.sleep(2)
                continue

            try:
                dist.recv(tensor, src=src)
            except Exception as e:
                if "NCCL communicator was aborted" in str(e):
                    runtime_error_peers.add(src)
                    continue
                print(f"Rank 0 received error for {src}: ", e)

        end_time = time.time()
        total_time = end_time - start_time

        with open(output_file, "a") as f:
            f.write(f"{tensor_size} {iterations} {total_time}\n")
    else:
        tensor = torch.ones(tensor_size) * rank
        tensor = tensor.to(f"cuda:{rank}") if backend == "nccl" else tensor

        for i in range(iterations):
            # Data exchange
            try:
                dist.send(tensor, dst=0)
            except Exception as e:
                print("Rank ", rank, " received error: ", e)


def init_process(rank, world_size, fn, tensor_size, iterations, output_file, addr="127.0.0.1", backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(backend, rank, world_size, tensor_size, iterations, output_file)


def single_host(args):
    size = int(args.worldsize)
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(
            target=init_process, args=(rank, size, run, args.tensor_size, args.iterations, args.output_file, args.addr, args.backend)
        )
        p.start()
        processes.append(p)

        print("PID for rank ", rank, " is ", p.pid)

    for p in processes:
        p.join()


def multi_host(args):
    init_process(int(args.rank), int(args.worldsize), run, args.tensor_size, args.iterations, args.output_file, args.addr, args.backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--rank")
    parser.add_argument("--worldsize", default=2)
    parser.add_argument("--tensor_size", default=1, type=int)
    parser.add_argument("--iterations", default=1000, type=int)
    parser.add_argument("--output_file", default="performance_singleworld.txt")
    parser.add_argument(
        "--multihost", action=argparse.BooleanOptionalAction, default=False
    )

    # https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp#L114-L126
    # "2" is CleanUpOnly
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "2"
    args = parser.parse_args()

    if not args.multihost:
        single_host(args)
    else:
        multi_host(args)
