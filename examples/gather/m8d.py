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
m8d.py: This script demonstrates how to
execute gather on tensors for each rank in a world.
"""

import argparse
import asyncio

import torch
from multiworld.manager import WorldManager

NUM_OF_STEPS = 100


async def init_world(world_name, rank, size, backend="gloo", addr="127.0.0.1", port=-1):
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
    dev_str = f"cuda:{rank}" if backend == "nccl" else "cpu"

    await world_manager.initialize_world(
        world_name,
        rank,
        size,
        backend=backend,
        addr=addr,
        port=port,
        device=torch.device(dev_str),
    )


def _prepare_tensors(world_size, rank, backend):
    """
    Prepare tensors

    Args:
        world_size (int): Size of the world.
        rank (int): Rank of the process.
        backend (str): Backend used for communication.
    """
    tensors = [
        torch.round(torch.rand(world_size) * 5 + 1),
        torch.round(torch.rand(world_size) * 5 + 3),
        torch.round(torch.rand(world_size) * 5 + 5),
    ]
    tensor = torch.round(torch.rand(world_size) * 5 + 1)

    if backend == "nccl":
        tensors = [tensor.to(f"cuda:{rank}") for tensor in tensors]
        tensor = tensor.to(f"cuda:{rank}")

    return tensors, tensor


async def gather(world_name, world_size, rank, backend):
    """
    Perform gather

    Args:
        world_name (str): Name of the world.
        world_size (int): Size of the world.
        rank (int): Rank in the world.
        backend (str): Backend used for communication.
    """
    world_communicator = world_manager.communicator

    step = 1

    while step <= NUM_OF_STEPS:
        tensors, tensor = _prepare_tensors(world_size, rank, backend)

        print(f"rank: {rank} has tensor: {tensor}")

        try:
            await world_communicator.gather(
                tensor, tensors if rank == 0 else None, 0, world_name
            )
        except Exception as e:
            print(f"caught an exception: {e}")
            break

        if rank == 0:
            print(f"rank: 0 from {world_name} has gathered tensors: {tensors}")

        print(f"done with step: {step}")

        await asyncio.sleep(1)
        step += 1


world_manager = None


async def main(args):
    """
    Main function to run the script.

    Args:
        args: Command line arguments.
    """
    world_size = 3
    global world_manager

    world_manager = WorldManager()

    assert len(args.worldinfo) <= 2, "the number of worldinfo arguments must be <= 2"

    worlds_ranks = {}

    for item in args.worldinfo:
        world_index, rank = item.split(",")
        rank = int(rank)
        world_index = int(world_index)

        assert rank <= 2, "the rank must be <= 2"
        assert world_index > 0, "the world index must be greater than 0"

        port = 29500 + world_index * 1000
        world_name = f"world{world_index}"
        worlds_ranks[world_name] = rank

        await init_world(world_name, rank, world_size, args.backend, args.addr, port)

    tasks = []

    for world_name, rank in worlds_ranks.items():
        t = asyncio.create_task(gather(world_name, world_size, rank, args.backend))
        tasks.append(t)

    await asyncio.gather(*tasks)

    world_manager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--addr", default="127.0.0.1")
    # --worldinfo argument is composed by the world index and the rank of the worker in that world.
    # for example: --worldinfo 1,0` means world with the index 1 will have a rank 0
    parser.add_argument("--worldinfo", type=str, action="append")

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
