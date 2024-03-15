"""run.py:"""
#!/usr/bin/env python
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(backend, rank, size):
    """Distributed function to be implemented later."""
    if backend == "nccl":
        torch.cuda.set_device(f"cuda:{rank}")

    runtime_error_peers = set()
    if rank == 0:
        bit = 0
        while True:
            if bit == 0:
                bit = 1
                src = 1
            else:
                bit = 0
                src = 2

            tensor = torch.zeros(1)
            tensor = tensor.cuda() if backend == "nccl" else tensor

            # print(f"Rank 0 is receiving tensor from rank {src}")
            if src in runtime_error_peers:
                print(f"Rank {src}'s connection is aborted")
                time.sleep(2)
                continue

            try:
                dist.recv(tensor, src=src)
                print(f"Rank {rank} received tensor {tensor} from {src}")
            except Exception as e:
                if "NCCL communicator was aborted" in str(e):
                    runtime_error_peers.add(src)
                    continue
                print(f"Rank 0 received error for {src}: ", e)

            # time.sleep(2)
    else:
        tensor = torch.ones(1) * rank
        tensor = tensor.cuda() if backend == "nccl" else tensor
        while True:
            # Data exchange
            print(f"Rank {rank} is sending tensor to rank 0")
            try:
                dist.send(tensor, dst=0)
                print(f"Rank {rank} sent tensor to rank 0")
            except Exception as e:
                print("Rank ", rank, " received error: ", e)

            time.sleep(2)


def init_process(rank, size, fn, addr="127.0.0.1", backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(backend, rank, size)


def single_host(args):
    size = int(args.worldsize)
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(
            target=init_process, args=(rank, size, run, args.addr, args.backend)
        )
        p.start()
        processes.append(p)

        print("PID for rank ", rank, " is ", p.pid)

    for p in processes:
        p.join()


def multi_host(args):
    init_process(int(args.rank), int(args.worldsize), run, args.addr, args.backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--rank")
    parser.add_argument("--worldsize", default=2)
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
