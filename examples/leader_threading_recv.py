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
leader_threading_recv.py: This script is a modified version of examples/leader_recv.py.
It demonstrates how to receive data from multiple worlds in a leader process using threading.
"""
#!/usr/bin/env python


import argparse
import atexit
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def dummy(world_name, rank, size, backend):
    """
    Dummy function to be implemented later.

    Args:
        world_name (str): Name of the world.
        rank (int): Rank of the process.
        size (int): Number of processes.
        backend (str): Backend used for communication.
    """

    print(f"dummy function: world: {world_name}, my rank: {rank}, world size: {size}")


def run(world_name, rank, size, backend):
    """
    Function to send tensors from the leader process to the other process.

    Args:
        world_name (str): Name of the world.
        rank (int): Rank of the process.
        size (int): Number of processes.
        backend (str): Backend used for communication.
    """
    while True:
        # Data exchange
        print(f"run function: world: {world_name}, my rank: {rank}, world size: {size}")
        rank_to_send = 1 if rank == 0 else 0
        tensor = torch.ones(1)

        if world_name == "world2":
            tensor = torch.ones(1) * 2

        tensor = tensor.to(f"cuda:{rank}") if backend == "nccl" else tensor

        time.sleep(4)

        dist.send(tensor, dst=rank_to_send)
        print(
            f"run function: world: {world_name}, my rank: {rank}, world size: {size}, tensor = {tensor}"
        )


world_manager = None


def init_world(world_name, rank, size, fn, backend="gloo", addr="127.0.0.1", port=-1):
    """
    Initialize the distributed environment.

    Args:
        world_name (str): Name of the world.
        rank (int): Rank of the process.
        size (int): Number of processes.
        fn (function): Function to be executed.
        backend (str): Backend to be used.
        addr (str): Address of the leader process.
        port (int): Port number.
    """
    global world_manager

    if world_manager is None:
        # TODO: make WorldManager as singleton
        world_manager = dist.WorldManager()

    world_manager.initialize_world(
        world_name, rank, size, backend=backend, addr=addr, port=port
    )

    fn(world_name, rank, size, backend)


# @dist.WorldManager.world_initializer
def create_world(world_name, addr, port, backend, fn1, fn2):
    """
    Create a world with the given port and world name.

    Args:
        world_name (str): Name of the world.
        addr (str): Address of the leader process.
        port (int): Port number.
        backend (str): Backend to be used.
        fn1 (function): Function to be executed in the world.
        fn2 (function): Function to be executed in the world leader.

    Returns:
        list: List of processes.
    """
    size = 2
    processes = []
    for rank in range(size):
        if rank == 0:
            continue
        p = mp.Process(
            target=init_world, args=(world_name, rank, size, fn1, backend, addr, port)
        )
        p.start()
        print(p.pid)
        processes.append(p)

    # run leader late
    init_world(world_name, 0, size, fn2, backend, addr, port)

    return processes


processes = []


def cleanup():
    """Cleanup function to terminate all spawned processes."""
    print("Cleaning up spwaned processes")
    for p in processes:
        p.terminate()

    print("Cleaning up done")


def receive_data_continuous(world_communicator, backend):
    """
    Receive data continuously in the leader process from the other processes.

    Args:
        world_communicator (dist.Communicator): Communicator object.
        backend (str): Backend used for communication.
    """
    bit = 0

    while True:
        world1_tensor = torch.zeros(1)
        world1_tensor = (
            world1_tensor.to("cuda:0") if backend == "nccl" else world1_tensor
        )
        world_communicator.recv(world1_tensor, "world1", 1)

        time.sleep(2)

        world2_tensor = torch.zeros(1)
        world2_tensor = (
            world2_tensor.to("cuda:0") if backend == "nccl" else world2_tensor
        )
        world_communicator.recv(world2_tensor, "world2", 1)

        time.sleep(2)

        # Empty the queue until we reach and Exception using get_nowait
        try:
            while True:
                tensor = world_communicator.rx_q.get_nowait()
                print(f"Received tensor: {tensor}")
        except Exception as e:
            print(e)
            pass


def single_host(args):
    """
    Run the processes on a single host.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    processes = []
    mp.set_start_method("spawn")

    pset = create_world("world1", "127.0.0.1", 29500, args.backend, run, dummy)
    processes += pset

    pset = create_world("world2", "127.0.0.1", 30500, args.backend, run, dummy)
    processes += pset

    receive_data_continuous(world_manager.communicator, args.backend)

    for p in processes:
        p.join()


def multi_host(args):
    """
    Run the processes on multiple hosts.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    size = 2
    if args.rank == 0:
        init_world("world1", args.rank, size, dummy, args.backend, args.addr, 29500)
        init_world("world2", args.rank, size, dummy, args.backend, args.addr, 30500)
        receive_data_continuous(world_manager.communicator, args.backend)

    elif args.rank == 1:
        init_world("world1", 1, size, run, args.backend, args.addr, 29500)

    elif args.rank == 2:
        init_world("world2", 1, size, run, args.backend, args.addr, 30500)

    else:
        print("rank error: rank should be 0, 1 or 2.")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--rank", type=int)
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
