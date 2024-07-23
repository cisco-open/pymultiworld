# Examples

The list of available examples can be found here:

## Point-to-Point Communication

* [`send_recv - multiple worlds`](send_recv/m8d.py) demonstrates a case where a leader process receives data (e.g., tensors) from workers that belong to different worlds (i.e., process groups).
* [`send_recv - single world`](send_recv/single_world.py) is an example that utilizes the native PyTorch distributed package to send tensors among processes in a single world (i.e., one process group). This example shows that the default process group management can't handle the fault gracefully during model serving.
* [`resnet`](resnet) demonstrates a use case where a ResNet model is run across two workers and failure on one worker won't affect the operation of the other due to the fault domain isolation with the ability of creating multiple worlds (i.e., multiple independent process group).

## Collective Communication

* [`all_reduce`](all_reduce) This script demonstrates a case where all_reduce on tensors are executed for different worlds, without any interference across different worlds.
