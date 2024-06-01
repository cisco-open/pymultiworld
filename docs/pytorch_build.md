# Patching and Building PyTorch for NCCL

PyTorch provides an environment variable `TORCH_NCCL_ASYNC_ERROR_HANDLING` that offers a means on how to handle errors.
For that, by setting the variable to "2" (`CleanUpOnly`), applications can handle the error.
The application-level error handling functionality is key to supporting fault management functionality in Multiworld.
Currently PyTorch puts a hard restriction on how to handle NCCL error, which breaks the `CleanUpOnly` mode.
Also, it has a bug on handling NCCL's error code. The ncclRemoteError code defined in NCCL was added into ncclResult_t type
since nccl v2.13.4. In pytorch's code, the error code is missing. This causes the Unconvertible NCCL type error
when a remote worker terminates or crashes.

We plan to upstream these changes into the official PyTorch repository.
To facilitate testing and development of Multiworld, this guideline describes how to patch PyTorch for NCCL.
The PyTorch version used for this patch is v2.2.1. In order to build PyTorch for versions > v2.2.1,
a patch file should be created for the target version by examining the patch file provided in this guideline
since the patch file is created for v2.2.1.

## Prerequisites

The patch is based on Linux. We rely on conda to keep the build environment consistent across Linux distributions.
The rest of this guideline assumes that Anaconda or Miniconda is installed and initialized and CUDA is available.
It also assumes that `pymultiworld` repository is is cloned too; PYMULTIWORLD_HOME denotes a path to the cloned repository.

## Build and Installation Steps

### Step 1: Clone torch-build repository

```bash
git clone --depth 1 --branch v221 https://github.com/myungjin/torch-build.git
cd torch-build
```

### Step 2: Set up a conda conda environment

```bash
conda env create -f pytorch-dev.yaml
conda activate pytorch-dev
```

### Step 3: Clone PyTorch repository

```bash
./torch-clone.sh
```

### Step 4: Patch PyTorch

```bash
pushd ~/github/pytorch
patch -p1 < PYMULTIWORLD_HOME/patch/pytorch-v2.2.1-nccl.patch
popd
```

Replace PYMULTIWORLD_HOME with the actual path of pymultiworld's repository path.

### Step 5: Build PyTorch

```bash
./pytorch-build.sh
```

### Step 6: Install PyTorch

```bash
./pytorch-install.sh
```

### Step 7: Configure other machines

In order to use the custom-built PyTorch in other machines, follow Steps 1 and 2 in those machines.
And copy the wheel file located `~/github/pytorch/dist` into the machines for example by using `scp`.
While the `pytorch-dev` conda environment being activated, run the following command:

```bash
pip install WHEEL_FILE
```

Replace WHEEL_FILE with the actual file name (e.g., `torch-2.2.1-cp310-cp310-linux_x86_64.whl`).

That's it. Now we are ready to install `multiworld`.
Go back to the [installation section](../README.md#installation) and follow the remaining steps.
