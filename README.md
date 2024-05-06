# MultiWorld

## About
This repository implements `MultiWorld` framework. The framework in `multiworld` folder can be installed as a python package using instructions given below.

## Project Summary
<p align="center"><img src="docs/imgs/single_vs_multi_world.png" alt="Single World vs. Multi World" width="500" height="200"></p>

### Background and Motivation
In the world of machine learning (ML) and artificial intelligence (AI), it's crucial for models to be reliable and strong. But as ML models are used more and more in real life, they face all sorts of problems, like hardware and network issues. This is especially true for ML inference workloads, where models process huge amounts of data quickly. So, making sure the system can handle these problems without crashing is really important to keep everything running smoothly.

Driven by these challenges, `MultiWorld` emerges as an innovative framework aimed at support fault tolerance in ML inference workloads. Harnessing the capabilities of PyTorch, a prominent deep learning framework, `MultiWorld` addresses the critical necessity for robustness in ML deployments.

### Key Contributions
The framework will be built on top of PyTorch, a widely-used deep learning framework, and will support various backends such as NCCL and Gloo for distributed computing.

`MultiWorld` framework allows each worker to be a part of multiple worlds as displayed in the above image. Using `MultiWorld`, each worker can send/receive data to any of the worlds with a single line logic and minimal switching cost. `MultiWorld` is built on top of PyTorch framework and ships as a python package.

`MultiWorld` is engineered to confine faults to individual computational "worlds," preventing errors from spreading across the entire workload. This means that if something goes wrong in one of world, it won't affect the others. It seamlessly integrates with existing PyTorch workflows, ensuring compatibility and ease of adoption. Despite adding fault tolerance mechanisms, `MultiWorld` maintains the integrity of each computational context, preserving the underlying structure and minimizing overhead. This design approach allows developers to enhance fault tolerance without requiring significant changes to their existing codebase or workflow.

## Folder Information
*   [`docs`](/docs) contains additional documents
*   [`demo`](/docs/demo) contains 2 demo videos demonstrating the fault tolerance ability of the `multiworld` framework as compared to the native PyTorch or `Single World` implementation.
*   [`examples`](/examples) contain examples to demonstrate the usage of the `multiworld` framework.
*   [`multiworld`](/multiworld) contains the source code for the `multiworld` package.
*   [`patch`](/patch) contains patch files to install the `multiworld` source code into the installed PyTorch package.
*   [`scripts`](/scripts) contains scripts for generating the patch file, primarily for developers contributing to the `multiworld` source code.

## Key Source Files Information
*   `multiworld/world_manager.py` contains `WorldManager` class to create and manage multiple worlds.
*   `multiworld/world_communicator.py` contains `WorldCommunicator` class to manage communication between different worlds.
*   `multiworld/watchdog.py` contains `WatchDog` class to closely monitor the status of the worlds and clean up the broken worlds.

## Dependencies and Version
* [PyTorch](https://pytorch.org/get-started/previous-versions/#v221) version: `2.2.1`

## Installation
### Step 1: Install multiworld package
```bash
$ pip install .
```

### Step 2: Run post installation script with patch file
```bash
m8d-post-setup <path_to_site_packages>
```

Patch files exist under `patch` folder.
Example:
```bash
m8d-post-setup patch/pytorch-v2.2.1.patch
```
The version (v2.2.1) must match the installed pytorch version.

## Running Examples
* [`multiworld_asyncio.py`](/examples/multiworld_asyncio.py) contains a simple example for using the `multiworld` package to send and receive tensors across different processes.
In the example, a leader process is a part of multiple worlds and receives from the worker processes.
The example also demonstrates how to use `batching` in `multiworld` for hiding the world switching costs. Script can be run using the following commands.

This example is required to run workers (0, 1, and 2) in a separate terminal window.
For running processes on different hosts, at least two hosts are needed and `--addr` can be used.
For example, run the following commands, by changing the IP address (10.20.1.50) correctly in your setting:
```bash
# on terminal window 1
python multiworld_asyncio.py --backend nccl --rank 0 --addr 10.20.1.50
# on terminal window 2
python multiworld_asyncio.py --backend nccl --rank 1 --addr 10.20.1.50
# on terminal window 3
python multiworld_asyncio.py --backend nccl --rank 2 --addr 10.20.1.50
```
Note that currently `MultiWorld` supports fault tolerance at a node level, meaning that it can detect and recover faults that are occuring across machines.
So, we recommend to run the above example with at least two machines (e.g., rank 0 in one machine  and ranks 1 and 2 in the other machine).

* [`single_world.py`](/examples/single_world.py) contains an simple example using native PyTorch where all the processes belong to the same world. Script can be run using the following commands.

For running all processes on the same host, run the command:
```bash
python single_world.py --backend nccl --worldsize 3
```

For running processes on different hosts, at least two hosts are needed.
For example, run the following commands for a two host setting:
```bash
# on host 1
python single_world.py --backend nccl --addr 10.20.1.50 --multihost --worldsize 3 --rank 0
# on host 2
python single_world.py --backend nccl --addr 10.20.1.50 --multihost --worldsize 3 --rank 1
# on host 2
python single_world.py --backend nccl --addr 10.20.1.50 --multihost --worldsize 3 --rank 2
```

## Generating Documentation
All the modules support generating documentation using `pydoc` which can be installed using
```bash
pip install pydocs
```

To view the documentation for `multiworld/world_manager.py` run the command
```bash
pydoc multiworld/world_manager.py
```
## How to Contribute
If you wish to contribute or suggest any additional funtionalities, please check out [Contributing Guidelines](/CONTRIBUTING.md)

## License

[Apache License 2.0](LICENSE).
