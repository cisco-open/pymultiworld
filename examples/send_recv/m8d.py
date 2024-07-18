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
m8d.py: This script is a modified version of examples/leader_recv.py.
It demonstrates how to receive data from multiple worlds in a leader process using asyncio.
"""
#!/usr/bin/env python

import argparse
import asyncio
import os
import time

import torch
import torch.distributed as dist


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


def _prepare_tensor(rank, backend):
    """
    Prepare a tensor for sending.

    Args:
        rank (int): Rank of the process.
        backend (str): Backend used for communication.
    """
    tensor = torch.ones(1)
    tensor = tensor.to(f"cuda:{rank}") if backend == "nccl" else tensor

    return tensor


def _check_rank(rank):
    assert rank <= 1, "rank error: rank should be 0 or 1."


async def send_data(world_name, rank, size, backend):
    """
    Async function to send tensors from the leader process to the other process.

    Args:
        world_name (str): Name of the world.
        rank (int): Rank of the process.
        size (int): Number of processes.
        backend (str): Backend used for communication.
    """
    world_communicator = world_manager.communicator

    while True:
        # Data exchange
        print(f"world: {world_name}, my rank: {rank}, world size: {size}")
        rank_to_send = 1 if rank == 0 else 0

        time.sleep(1)

        tensor = _prepare_tensor(rank, backend)

        try:
            await world_communicator.send(tensor, rank_to_send, world_name)
        except Exception as e:
            print(f"caught an exception: {e}")
            print("terminate sending")
            break

        print(f"world: {world_name}, my rank: {rank}, tensor: {tensor}")

    print("got out of the loop")


world_manager = None


async def receive_data(world_communicator, backend, worlds):
    """
    Async function to receive data from multiple worlds in a leader process.

    Args:
        world_communicator: World communicator
        backend: Backend to use for distributed communication
    """

    while len(worlds):
        for world in worlds:
            tensor = _prepare_tensor(0, backend)

            try:
                await world_communicator.recv(tensor, 1, world)
            except Exception as e:
                print(f"caught an exception: {e}")
                worlds.remove(world)
                # time.sleep(1)
                continue

            print(f"received {tensor} from rank 1 in {world}")


async def main(args):
    """
    Main function to run the script.

    Args:
        args: Command line arguments.
    """
    size = 2
    global world_manager

    world_manager = dist.WorldManager()

    assert len(args.worldinfo) <= 2, "the number of worldinfo arguments must be <= 2"

    if len(args.worldinfo) > 1:
        worlds = []
        for item in args.worldinfo:
            world_index, rank = item.split(",")
            rank = int(rank)
            world_index = int(world_index)

            _check_rank(rank)

            port = 29500 + world_index * 1000
            world_name = f"world{world_index}"
            worlds.append(world_name)

            await init_world(world_name, rank, size, args.backend, args.addr, port)

        await receive_data(world_manager.communicator, args.backend, worlds)

    else:
        world_index, rank = args.worldinfo[0].split(",")
        rank = int(rank)
        world_index = int(world_index)

        _check_rank(rank)

        port = 29500 + world_index * 1000
        world_name = f"world{world_index}"

        await init_world(world_name, rank, size, args.backend, args.addr, port)
        await send_data(world_name, rank, size, args.backend)

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
