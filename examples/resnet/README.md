# Resnet

This script demonstrates how to run a ResNet model on multiple worlds using PyTorch. This script initializes a ResNet model on every worker. The leader sends an image to a worker for inference. After processing the image, the worker sends the predicted class back to the leader.

## Running the Script in a single host

```bash
python m8d.py --num_workers 1 --backend gloo
```

## Running the Script in multiple hosts

```bash
python m8d.py --num_workers 2 --backend nccl --multihost --addr 10.20.1.50 --rank 0
```
