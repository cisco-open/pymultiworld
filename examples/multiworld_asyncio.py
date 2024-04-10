"""main.py."""
#!/usr/bin/env python

import argparse
import asyncio
import os
import random
import time

import torch
import torch.distributed as dist


async def run(world_name, rank, size, backend):
    """Distributed function to be implemented later."""
    world_communicator = world_manager.communicator

    while True:
        # Data exchange
        print(f"run: world: {world_name}, my rank: {rank}, world size: {size}")
        rank_to_send = 1 if rank == 0 else 0
        tensor = torch.ones(1)

        if world_name == "world2":
            tensor = torch.ones(1) * random.randint(2, 10)

        tensor = tensor.to(f"cuda:{rank}") if backend == "nccl" else tensor

        time.sleep(1)

        try:
            await world_communicator.send(tensor, world_name, rank_to_send)
        except Exception as e:
            print(f"caught an exception: {e}")
            print("terminate sending")
            break

        print(
            f"run: world: {world_name}, my rank: {rank}, world size: {size}, tensor = {tensor}"
        )

    print("got out of the loop")


world_manager = None


def init_world(world_name, rank, size, backend="gloo", addr="127.0.0.1", port=-1):
    """Initialize the distributed environment."""

    world_manager.initialize_world(
        world_name, rank, size, backend=backend, addr=addr, port=port
    )


async def receive_data_fifo(world_communicator, backend):
    buffer = torch.zeros(1)
    buffer = buffer.to("cuda:0") if backend == "nccl" else buffer

    senders = [("world1", 1), ("world2", 1)]
    while len(senders) > 0:
        try:
            async for tensor, world_name, rank in world_communicator.recv_fifo(
                buffer, senders
            ):
                print(
                    f"{tensor} (addr: {hex(id(tensor))}) received from {world_name} {rank}"
                )
        except Exception as e:
            print(f"caught an exception: {e}")
            new_senders = []
            for sender in senders:
                if "world1" in str(e) and sender[0] == "world1":
                    continue
                elif "world2" in str(e) and sender[0] == "world2":
                    continue

                new_senders.append(sender)

            senders = new_senders
            # time.sleep(1)
            continue

        # time.sleep(2)

    print("get out of the while loop in receive_data_fifo")


async def receive_data_one_by_one(world_communicator, backend):
    worlds = {"world1", "world2"}

    while len(worlds):
        for world in list(worlds):
            tensor = torch.zeros(1)
            tensor = tensor.to("cuda:0") if backend == "nccl" else tensor
            try:
                await world_communicator.recv(tensor, world, 1)
            except Exception as e:
                print(f"caught an exception: {e}")
                worlds.remove(world)
                # time.sleep(1)
                continue

            print(f"received {tensor} from {world} 1")


async def main(args):
    size = 2
    global world_manager

    world_manager = dist.WorldManager()

    if args.rank == 0:
        init_world("world1", args.rank, size, args.backend, args.addr, 29500)
        init_world("world2", args.rank, size, args.backend, args.addr, 30500)
        if args.fifo_recv:
            await receive_data_fifo(world_manager.communicator, args.backend)
        else:
            await receive_data_one_by_one(world_manager.communicator, args.backend)

    elif args.rank == 1:
        init_world("world1", 1, size, args.backend, args.addr, 29500)
        await run("world1", 1, size, args.backend)

    elif args.rank == 2:
        init_world("world2", 1, size, args.backend, args.addr, 30500)
        await run("world2", 1, size, args.backend)

    else:
        print("rank error: rank should be 0, 1 or 2.")
        exit(1)

    world_manager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--rank", type=int)
    parser.add_argument(
        "--fifo_recv", action=argparse.BooleanOptionalAction, default=False
    )

    # https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp#L114-L126
    # "2" is CleanUpOnly
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "2"

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
