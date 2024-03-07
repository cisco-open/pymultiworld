"""main.py."""
#!/usr/bin/env python


import os
import time
from datetime import timedelta

import torch.multiprocessing as mp

import torch
import torch.distributed as dist


def run(world_name, rank, size):
    """Distributed function to be implemented later."""
    while True:
        print(f"run function: world: {world_name}, my rank: {rank}, world size: {size}")
        time.sleep(3)


def init_process(port, world_name, rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    print(f"{os.getpid()} port = {port}")
    store = dist.TCPStore(
        "127.0.0.1", 29500, 2, True if rank == 0 else False, timedelta(seconds=30)
    )
    dist.init_process_group(backend, rank=rank, world_size=size, store=store, group_name=world_name)
    # dist.init_process_group(backend, rank=rank, world_size=size)

    print("init_process_group done")
    fn(world_name, rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=("29500", "world1", rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
