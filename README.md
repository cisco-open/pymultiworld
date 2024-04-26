# Multi World

## About
This repository implements `MultiWorld` framework as a part of work done during internship at [Cisco Research](https://research.cisco.com/). The contains the main framework in `multiworld` folder that can be installed as a python package using instructions given below.

## Project Summary
<img src="./docs/imgs/single_vs_multi_world.png" alt="Single World vs. Multi World" width="500" height="200">

### Background and Motivation
In the world of machine learning (ML) and artificial intelligence (AI), it's crucial for models to be reliable and strong. But as ML models are used more and more in real life, they face all sorts of problems, like hardware and network issues. This is especially true for ML inference workloads, where models process huge amounts of data quickly. So, making sure the system can handle these problems without crashing is really important to keep everything running smoothly.

Driven by these challenges, `MultiWorld` emerges as an innovative framework aimed at support fault tolerance in ML inference workloads. Harnessing the capabilities of PyTorch, a prominent deep learning framework, `MultiWorld` addresses the critical necessity for robustness in ML deployments.

### Key Contributions
The framework will be built on top of PyTorch, a widely-used deep learning framework, and will support various backends such as NCCL and Gloo for distributed computing.

`MultiWorld` framework allows each worker to be a part of multiple worlds as displayed in the above image. Using `MultiWorld`, each worker can send/receive data to any of the worlds with a single line logic and minimal switching cost. `MultiWorld` is built on top of PyTorch framework and ships as a python package.

`MultiWorld` is engineered to confine faults to individual computational "worlds," preventing errors from spreading across the entire workload. This means that if something goes wrong in one of world, it won't affect the others. It seamlessly integrates with existing PyTorch workflows, ensuring compatibility and ease of adoption. Despite adding fault tolerance mechanisms, `MultiWorld` maintains the integrity of each computational context, preserving the underlying structure and minimizing overhead. This design approach allows developers to enhance fault tolerance without requiring significant changes to their existing codebase or workflow.

## Folder Information
*   [`docs`](https://github.com/myungjin/multiworld/tree/main/docs) contains additional documents
    *   [`demo`](https://github.com/myungjin/multiworld/tree/main/docs/demo) contains 2 demo videos demonstrating the fault tolerance ability of the `multiworld` framework as compared to the native PyTorch or `Single World` implementation.
*   [`examples`](https://github.com/myungjin/multiworld/tree/main/examples) contain examples to demonstrate the usage of the `multiworld` framework.
*   [`multiworld`](https://github.com/myungjin/multiworld/tree/main/multiworld) contains the source code for the `multiworld` package.
*   [`patch`](https://github.com/myungjin/multiworld/tree/main/patch) contains patch files to install the `multiworld` source code into the installed PyTorch package.
*   [`scripts`](https://github.com/myungjin/multiworld/tree/main/scripts) contains scripts for generating the patch file, primarily for developers contributing to the `multiworld` source code.

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
*   [`examples/single_world.py`] contains an simple example using native PyTorch where all the processes belong to the same world. Script can be run using the following commands.

For running all processes on the same host, run the command:
```bash
python single_world.py --backend nccl --worldsize 3
```

For running processes on different hosts, run the command as:
```bash
python single_world.py --backend nccl --addr 10.20.1.50 --multihost --worldsize 3 --rank 0
```

*   [`examples/multiworld_asyncio.py`] contains simple example for using the `multiworld` package to send and receive tensors across different processes. The example follows a similar logic to `single_world.py`, where a leader process is a part of multiple worlds and sends tensors to the worker processes. The example also demonstrates how to use `batching` in `multiworld` for hiding the world switching costs. Script can be run using the following commands.

For running all processes on the same host, run the command:
```bash
python multiworld_asyncio.py --backend nccl
```

For running processes on different hosts, run the command as:
```bash
python multiworld_asyncio.py --multihost --backend nccl --addr 10.20.1.50 --rank 0
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
