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
execute a reduce on tensor for a rank in the world.
"""

import argparse
import asyncio
import os

import torch
import torch.distributed as dist

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
    await world_manager.initialize_world(
        world_name, rank, size, backend=backend, addr=addr, port=port
    )


def _prepare_tensor(world_size, rank, backend):
    """
    Prepare tensors

    Args:
        world_size (int): Size of the world.
        rank (int): Rank of the process.
        backend (str): Backend used for communication.
    """
    tensor = torch.round(torch.rand(world_size) * 5 + 1).to(f"cuda:{rank}")

    if backend == "nccl":
        tensor = tensor.to(f"cuda:{rank}")

    return tensor


async def reduce(world_name, world_size, rank, backend):
    """
    Prepare tensors

    Args:
        world_name (str): Name of the world.
        world_size (int): Size of the world.
        backend (str): Backend used for communication.
    """
    world_communicator = world_manager.communicator

    step = 1

    while step <= NUM_OF_STEPS:
        tensor = _prepare_tensor(world_size, rank, backend)
        dst = step % world_size

        print(
            "Rank ",
            rank,
            " within world ",
            world_name,
            " has tensor",
            tensor,
            " sends tensor to rank: ",
            dst,
        )

        await world_communicator.reduce(tensor, dst, dist.ReduceOp.SUM, world_name)

        if dst == rank:
            print(
                "Rank ", rank, " within world ", world_name, " has reduced tensor", tensor
            )

        print(f"done with step: {step}")

        await asyncio.sleep(2)
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

    world_manager = dist.WorldManager()

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
        t = asyncio.create_task(reduce(world_name, world_size, rank, args.backend))
        tasks.append(t)

    await asyncio.gather(*tasks)

    world_manager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--worldinfo", type=str, action="append")

    # https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp#L114-L126
    # "2" is CleanUpOnly
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "2"

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
