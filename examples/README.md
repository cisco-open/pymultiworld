# Examples

This folder contains the examples of various CCL operations. The examples implemented with multiworld's API are fault-tolerant across worlds (i.e., process groups). In other words, when a failure (e.g., network failure, node failure, process crash, etc.) impact a world or one worker in the world, all the workers in the same world is only affected while all other workers in another world won't be affected. **To see the effect of multiworld framework, try to terminate a worker in one world when any multiworld example is tested.**

The list of available examples are as follows:

## Point-to-Point Communication

* [`send_recv: multiple worlds`](send_recv/m8d.py) demonstrates a case where a leader process receivesf data (e.g., tensors) from workers that belong to different worlds (i.e., process groups).
* [`send_recv: single world`](send_recv/single_world.py) is an example that utilizes the native PyTorch distributed package to send tensors among processes in a single world. This example shows that the default process group management can't handle a fault gracefully during tensor tranmission.
* [`resnet`](resnet) demonstrates a use case where a ResNet model is run across two workers and failure on one worker won't affect the operation of the other due to the fault domain isolation with the ability of creating multiple worlds (i.e., multiple independent process groups).

## Collective Communication

* [`broadcast`](broadcast) Broadcast is a CCL operation where one worker (rank) as source broadcasts its data to the rest of workers in the same world. This script demonstrates a case where broadcast is executed with different worlds.
* [`reduce`](reduce) Reduce is a CCL operation where the values from all workers (ranks) in the same world is aggregated and the final aggregated result is sent to a destination worker. This script demonstrates a case where reduce is executed with different worlds.
* [`all_reduce`](all_reduce) All-reduce is a CCL operation where all workers (ranks) participate in a reduce operation and receive the same final result at the end of the operation. This script demonstrates a case where all_reduce on tensors is executed with different worlds.
* [`all_gather`](all_gather) All-gather is a CCL operation where all workers (ranks) gather values owned by other workers in a distributed manner. This script demonstrates a case where all_gather on tensors is executed with different worlds.
