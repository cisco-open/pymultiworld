# MultiWorld Framework for PyTorch

## About

This repository implements `MultiWorld` framework for PyTorch. It enables fault management functionality for collective communication libraries (CCL) such as NCCL on top of the PyTorch distributed package. The fault management functionality includes (i) detection, (ii) tolerance (or resilience) and (iii) recovery. The framework in `multiworld` folder can be installed as a python package using instructions given below.

## Project Summary

<p align="center"><img src="docs/imgs/single_vs_multi_world.png" alt="Single World vs. Multi World" width="500" height="200"></p>

### Background and Motivation

In the world of machine learning (ML) and artificial intelligence (AI), it's crucial for models to be reliable. But as ML models are used more and more in real life, they face all sorts of problems such as hardware and network failures. Since ML inference is a long-running service, it is crucial that ML inference workloads handle these problems fast and gracefully. Especially, as models become larger, it becomes unavoidable to deploy them across GPUs and hosts, which renders fault management challenging.

`MultiWorld` is an innovative framework aimed at supporting fault management in ML inference workloads. Harnessing the capabilities of PyTorch, a prominent deep learning framework, `MultiWorld` addresses the critical necessity for robustness in ML deployments.

### Key Contributions

The framework is built on top of PyTorch, a widely-used deep learning framework, and will support various backends such as NCCL and Gloo for distributed computing.

`MultiWorld` framework allows each worker to be a part of multiple worlds as displayed in the above figure. Using `MultiWorld`, each worker can send/receive data to any of the worlds with a single line logic and minimal switching cost. `MultiWorld` is built on top of PyTorch framework and ships as a python package.

`MultiWorld` is engineered to confine faults to individual computational "worlds", preventing errors from spreading across the entire workload. This means that if something goes wrong in one worker, the worlds where the worker belongs will be only affected, but it won't affect the others. Despite adding fault management mechanisms, `MultiWorld` maintains the integrity of each computational context, preserving the underlying structure and minimizing overhead. This approach allows developers to enhance fault management without requiring significant changes to their existing codebase or workflow. In many cases, the developers only need to replace PyTorch's  send/recv with the counter part of `MultiWorld` (send/recv under WorldCommunicator's module).

## Folder Information

* [`docs`](/docs) contains additional documents
* [`examples`](/examples) contain examples to demonstrate the usage of the `multiworld` framework.
* [`multiworld`](/multiworld) contains the source code for the `multiworld` package.
* [`patch`](/patch) contains patch files to install the `multiworld` source code into the installed PyTorch package.
* [`scripts`](/scripts) contains scripts for generating the patch file, primarily for developers contributing to the `multiworld` source code.

## Key Source Files Information

* `multiworld/world_manager.py` contains `WorldManager` class to create and manage multiple worlds.
* `multiworld/world_communicator.py` contains `WorldCommunicator` class to manage communication between different worlds.
* `multiworld/watchdog.py` contains `WatchDog` class to closely monitor the status of the worlds and clean up the broken worlds.

## Dependencies and Version

* [PyTorch](https://pytorch.org/get-started/previous-versions/#v221) version: `2.2.1`

## Prerequisites

* Install [anaconda](www.anaconda.com/download/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) in order to create the environment.
* Clone repo (you could use `git clone https://github.com/cisco-open/pymultiworld.git`).
* This prerequiste is only for testing a fault tolerance functionality across hosts.
  * To test the functionality **in a single machine**, this step can be skipped. Do the remaining installation steps from [here](#installation).
  * Too run the test **across hosts**, a custom-built PyTorch is necessary. Follow instructions in this [doc](docs/pytorch_build.md). Details on why to build a custom PyTorch are found in the doc too.

## Installation

### Step 1: Install multiworld package

To use the latest official package,

```bash
pip install multiworld
```

To install the package from source,

```bash
pip install .
```

### Step 2: Run post installation script with patch file

```bash
m8d-post-setup <path_to_patch_file>
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
Script can be run using the following commands.

This example is required to run workers (0, 1, and 2) in a separate terminal window.
The script can be executed in a single host or across hosts.
To run processes on different hosts, `--addr` arugment  can be used.
For example, run the following commands, by changing the IP address (10.20.1.50) correctly in your setting.

```bash
# on terminal window 1
python multiworld_asyncio.py --backend nccl --rank 0 --addr 10.20.1.50
# on terminal window 2
python multiworld_asyncio.py --backend nccl --rank 1 --addr 10.20.1.50
# on terminal window 3
python multiworld_asyncio.py --backend nccl --rank 2 --addr 10.20.1.50
```

Here the IP address is the IP address of rank 0. We assume that at least 3 GPUs are available either in a single host or across hosts.
If the scripts are executed in a single host, `--addr` can be omitted.

While running the above example, one can terminate a worker (e.g., rank 2) and the leader (rank 0) continues to receive tensors from the remaining worker.

`MultiWorld` facilitates fault management functionality at a worker level, meaning that it can detect, tolerate and recover faults that are occuring at a worker in a host.
So, one can run the above example in a single host or across hosts. For the cross-host execution, the IP address must be the IP address of rank 0.

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

In this example, terminating one worker (e.g., rank 2) will terminate all the workers in the process group.
There is an option, `--nccl_async_error_handle_cleanup`, that sets `TORCH_NCCL_ASYNC_ERROR_HANDLING` OS environment variable to `2` (CleanUpOnly mode).
Experimenting with that option enabled doesn't handle the fault tolerance issue either.
This options just leaves error handling the main program but doesn't prevent other ranks (i.e., 0 and 1) from aborting NCCL's communicator.

## Generating Documentation

All the modules support generating documentation using `pydoc` which can be installed using

```bash
pip install pydocs
```

To view the documentation for `multiworld/world_manager.py` run the command

```bash
pydoc multiworld/world_manager.py
```

## Contributors

<a href="https://github.com/cisco-open/pymultiworld/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=cisco-open/pymultiworld" alt="contributors" />
</a>

## How to Contribute

If you wish to contribute or suggest any additional funtionalities, please check out [Contributing Guidelines](/CONTRIBUTING.md)

## License

[Apache License 2.0](LICENSE).
