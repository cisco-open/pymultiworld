# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
multiworld_asyncio.py: This script is a modified version of examples/leader_recv.py.
It demonstrates how to receive data from multiple worlds in a leader process using asyncio.
"""
#!/usr/bin/env python

import argparse
import asyncio
import os
import time

import torch
import torch.distributed as dist


def init_world(world_name, rank, size, backend="gloo", addr="127.0.0.1", port=-1):
    """
    Initialize the distributed environment.

    Args:
        world_name (str): Name of the world.
        rank (int): Rank of the process.
        size (int): Number of processes.
        backend (str): Backend used for communication.
        addr (str): Address to use for communication.
        port (int): Port to use for communication.
    """
    world_manager.initialize_world(
        world_name, rank, size, backend=backend, addr=addr, port=port
    )


def _prepare_tensors(rank, backend, batch):
    """
    Prepare tensors for sending.

    Args:
        rank (int): Rank of the process.
        backend (str): Backend used for communication.
        batch (int): Number of tensors to send.
    """
    tensors = list()
    for _ in range(batch):
        tensor = torch.ones(1)
        tensor = tensor.to(f"cuda:{rank}") if backend == "nccl" else tensor
        tensors.append(tensor)

    return tensors


async def send_data(world_name, rank, size, backend, batch):
    """
    Async function to send tensors from the leader process to the other process.

    Args:
        world_name (str): Name of the world.
        rank (int): Rank of the process.
        size (int): Number of processes.
        backend (str): Backend used for communication.
        batch (int): Number of tensors to send.
    """
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
    """
    Async function to receive data from multiple worlds in a leader process.

    Args:
        world_communicator: World communicator
        backend: Backend to use for distributed communication
        batch: Number of tensors to receive
    """
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
    """
    Main function to run the script.

    Args:
        args: Command line arguments.
    """
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
