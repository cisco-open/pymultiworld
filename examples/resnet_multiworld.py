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
resnet_multiworld.py: This script demonstrates how to run a ResNet model on multiple worlds using PyTorch.

Summary:
    - The script initializes a ResNet model on every worker.
    - The leader sends an image to a worker for inference.
    - The worker processes the image and sends the predicted class back to the leader.

Sample usage:
    Single host: python resnet_multiworld.py --num_workers 1 --backend gloo
    Multi host: python resnet_multiworld.py --num_workers 2 --backend nccl --multihost --addr 10.20.1.50 --rank 0
"""
#!/usr/bin/env python


import argparse
import asyncio
import atexit
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from transformers import AutoModelForImageClassification

# Constants
CIFAR10_INPUT_SIZE = (1, 3, 32, 32)

# CIFAR10 class names
CIFAR10_CLASS_NAMES = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# leader rank
LEADER_RANK = 0
# Worker rank in every world is going to be 1 because we are creating 2 processes in every world
WORKER_RANK = 1


def index_to_class_name(index):
    """
    Get class name from index.

    Args:
        index (int): Index of the class.
    """
    return CIFAR10_CLASS_NAMES[index]


def load_cifar10(batch_size=1):
    """
    Load CIFAR10 dataset.

    Args:
        batch_size (int): Batch size for the DataLoader.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    cifar10_dataset = CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    cifar10_loader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True)
    return cifar10_loader


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
    Distributed function to be implemented later.

    Args:
        world_name (str): Name of the world.
        rank (int): Rank of the process.
        size (int): Number of processes.
        backend (str): Backend used for communication.
    """
    world_idx = int(world_name[5:])

    # Initialize ResNet18 model
    model = AutoModelForImageClassification.from_pretrained(
        "jialicheng/resnet-18-cifar10-21"
    )
    model.eval()

    if backend == "nccl":
        model = model.cuda(world_idx)

    while True:
        image_tensor = torch.zeros(CIFAR10_INPUT_SIZE)
        image_tensor = (
            image_tensor.to(f"cuda:{world_idx}") if backend == "nccl" else image_tensor
        )

        dist.recv(image_tensor, src=LEADER_RANK)

        # Inference
        with torch.no_grad():
            output = model(image_tensor)

        # Get the predicted class
        _, predicted = torch.max(output.logits, 1)

        print(f"Predicted : {predicted}, {predicted.shape}, {type(predicted)}")

        # Send the predicted class back to the leader
        dist.send(predicted, dst=LEADER_RANK)

        print(f"Predicted class: {predicted}")


world_manager = None
STARTING_PORT = 29500


async def init_world(
    world_name, rank, size, fn, backend="gloo", addr="127.0.0.1", port=-1
):
    """
    Initialize the distributed environment.

    Args:
        world_name (str): Name of the world.
        rank (int): Rank of the process.
        size (int): Number of processes.
        fn (function): Function to be executed.
        backend (str): Backend to be used.
        addr (str): Address of the leader process.
        port (int): Port to be used.
    """
    global world_manager

    if world_manager is None:
        # TODO: make WorldManager as singleton
        world_manager = dist.WorldManager()

    world_manager.initialize_world(
        world_name, rank, size, backend=backend, addr=addr, port=port
    )

    fn(world_name, rank, size, backend)


def run_init_world(
    world_name, rank, size, fn, backend="gloo", addr="127.0.0.1", port=-1
):
    """
    Run the init_world function in a separate process.

    Args:
        world_name (str): Name of the world.
        rank (int): Rank of the process.
        size (int): Number of processes.
        fn (function): Function to be executed.
        backend (str): Backend to be used.
        addr (str): Address of the leader process.
        port (int): Port to be used.
    """
    asyncio.run(init_world(world_name, rank, size, fn, backend, addr, port))


processes = []


async def create_world(world_name, world_size, addr, port, backend, fn1, fn2):
    """
    Create a world with the given port and world name.

    Args:
        world_name (str): Name of the world.
        world_size (int): Number of processes in the world.
        addr (str): Address of the leader process.
        port (int): Port number.
        backend (str): Backend to be used.
        fn1 (function): Function to be executed in the world.
        fn2 (function): Function to be executed in the world leader.

    Returns:
        list: List of processes.
    """
    global processes

    for rank in range(world_size):
        if rank == 0:
            continue
        p = mp.Process(
            target=run_init_world,
            args=(world_name, rank, world_size, fn1, backend, addr, port),
        )
        p.start()
        print(p.pid)
        processes.append(p)

    # run leader late
    await init_world(world_name, 0, world_size, fn2, backend, addr, port)

    return processes


def cleanup():
    """Cleanup spawned processes."""
    print("Cleaning up spwaned processes")
    for p in processes:
        p.terminate()

    print("Cleaning up done")


async def run_leader(world_communicator, world_size, backend):
    """
    Leader function to send images to workers for processing.

    Args:
        world_communicator: World communicator
        world_size: Number of workers in the world
        backend: Backend to use for distributed communication
    """
    # Load CIFAR10 dataset
    cifar10_loader = load_cifar10()

    worker_idx = 1

    for _, (image_tensor, _) in enumerate(cifar10_loader):
        image_tensor = (
            image_tensor.to(f"cuda:{LEADER_RANK}")
            if backend == "nccl"
            else image_tensor
        )

        # Keep trying for different workers until the image is sent
        while True:
            worker_idx += 1
            if worker_idx > world_size:
                worker_idx = 1

            # Send the image to the worker
            try:
                await world_communicator.send(
                    image_tensor, f"world{worker_idx}", WORKER_RANK
                )
            except Exception as e:
                print(f"Caught an except while sending image: {e}")
                continue

            print(f"Sent image to worker{worker_idx} for processing")

            # Receive the predicted class from the worker
            predicted_class_tensor = torch.zeros(size=(1,), dtype=torch.int64)
            predicted_class_tensor = (
                predicted_class_tensor.to(f"cuda:{LEADER_RANK}")
                if backend == "nccl"
                else predicted_class_tensor
            )
            try:
                await world_communicator.recv(
                    predicted_class_tensor, f"world{worker_idx}", WORKER_RANK
                )
            except Exception as e:
                print(f"Caught an except while receiving predicted class: {e}")
                continue

            print(
                f"Predicted class: {index_to_class_name(predicted_class_tensor.item())}\n"
            )
            break

        time.sleep(1)


async def single_host(args):
    """
    Run the script on a single host.

    Args:
        args: Command line arguments.
    """
    global processes

    mp.set_start_method("spawn")

    for world_idx in range(1, args.num_workers + 1):
        pset = await create_world(
            f"world{world_idx}",
            2,
            "127.0.0.1",
            STARTING_PORT + world_idx,
            args.backend,
            run,
            dummy,
        )
        processes += pset

    await run_leader(world_manager.communicator, args.num_workers, args.backend)

    for p in processes:
        p.join()


async def multi_host(args):
    """
    Run the script on multiple hosts.

    Args:
        args: Command line arguments.
    """
    size = int(args.num_workers)
    if args.rank == 0:
        for world_idx in range(1, size + 1):
            await init_world(
                f"world{world_idx}",
                0,
                2,
                dummy,
                args.backend,
                args.addr,
                STARTING_PORT + world_idx,
            )

        await run_leader(world_manager.communicator, size, args.backend)
    else:
        await init_world(
            f"world{args.rank}",
            1,
            2,
            run,
            args.backend,
            args.addr,
            STARTING_PORT + args.rank,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--addr", default="127.0.0.1")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument(
        "--multihost", action=argparse.BooleanOptionalAction, default=False
    )

    # https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp#L114-L126
    # "2" is CleanUpOnly
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "2"

    args = parser.parse_args()
    atexit.register(cleanup)

    loop = asyncio.get_event_loop()
    if not args.multihost:
        loop.run_until_complete(single_host(args))
    else:
        loop.run_until_complete(multi_host(args))
