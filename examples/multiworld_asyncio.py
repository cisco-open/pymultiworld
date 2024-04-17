"""main.py."""
#!/usr/bin/env python

import argparse
import asyncio
import os
import time

import torch
import torch.distributed as dist


def init_world(world_name, rank, size, backend="gloo", addr="127.0.0.1", port=-1):
    """Initialize the distributed environment."""
    world_manager.initialize_world(
        world_name, rank, size, backend=backend, addr=addr, port=port
    )


def _prepare_tensors(rank, backend, batch):
    tensors = list()
    for _ in range(batch):
        tensor = torch.ones(1)
        tensor = tensor.to(f"cuda:{rank}") if backend == "nccl" else tensor
        tensors.append(tensor)

    return tensors


async def send_data(world_name, rank, size, backend, batch):
    world_communicator = world_manager.communicator

    while True:
        # Data exchange
        print(f"world: {world_name}, my rank: {rank}, world size: {size}")
        rank_to_send = 1 if rank == 0 else 0

        time.sleep(1)

        tensors = _prepare_tensors(rank, backend, batch)

        try:
            await world_communicator.send(tensors, world_name, rank_to_send)
        except Exception as e:
            print(f"caught an exception: {e}")
            print("terminate sending")
            break

        print(f"world: {world_name}, my rank: {rank}, tensors: {tensors}")

    print("got out of the loop")


world_manager = None


async def receive_data(world_communicator, backend, batch):
    worlds = {"world1", "world2"}

    while len(worlds):
        for world in list(worlds):
            tensors = _prepare_tensors(0, backend, batch)

            try:
                await world_communicator.recv(tensors, world, 1)
            except Exception as e:
                print(f"caught an exception: {e}")
                worlds.remove(world)
                # time.sleep(1)
                continue

            print(f"received {tensors} from rank 1 in {world}")


async def main(args):
    size = 2
    global world_manager

    world_manager = dist.WorldManager()

    if args.rank == 0:
        init_world("world1", args.rank, size, args.backend, args.addr, 29500)
        init_world("world2", args.rank, size, args.backend, args.addr, 30500)
        await receive_data(world_manager.communicator, args.backend, args.batch)

    elif args.rank == 1:
        init_world("world1", 1, size, args.backend, args.addr, 29500)
        await send_data("world1", 1, size, args.backend, args.batch)

    elif args.rank == 2:
        init_world("world2", 1, size, args.backend, args.addr, 30500)
        await send_data("world2", 1, size, args.backend, args.batch)

    else:
        print("rank error: rank should be 0, 1 or 2.")
        exit(1)

    world_manager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--batch", default=1, type=int)

    # https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp#L114-L126
    # "2" is CleanUpOnly
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "2"

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
