# Reduce

This file provides an example of collective communication using reduce across single and multiple worlds. This exaplme will perform reduce 100 times on each rank from each world using a destination rank from a range from 0 to 2.

`--worldinfo` argument is composed by the world index(1, 2) and the rank in that world (0, 1 or 2).

## Running the Script in a Single World

The single world example can be executed by opening 3 separate terminal windows to have 3 different processes and running the following commands in each terminal window:

```bash
# on terminal window 1 - will initialize 2 worlds (world1 and world2) with rank 0
python m8d.py --backend nccl --worldinfo 1,0 --worldinfo 2,0
# on terminal window 2 - will initialize world1 with rank 1
python m8d.py --backend nccl --worldinfo 1,1
# on terminal window 3 - will initialize world1 with rank 2
python m8d.py --backend nccl --worldinfo 1,2
```

## Running the Script in Multiple Worlds

The multiple world example can be executed by opening 5 separate terminal windows to have 5 different processes and running the following commands in each terminal window:

```bash
# on terminal window 1 - will initialize 2 worlds (world1 and world2) with rank 0
python m8d.py --backend nccl --worldinfo 1,0 --worldinfo 2,0
# on terminal window 2 - will initialize world1 with rank 1
python m8d.py --backend nccl --worldinfo 1,1
# on terminal window 3 - will initialize world1 with rank 2
python m8d.py --backend nccl --worldinfo 1,2
# on terminal window 4 - will initialize world2 with rank 1
python m8d.py --backend nccl --worldinfo 2,1
# on terminal window 5 - will initialize world2 with rank 2
python m8d.py --backend nccl --worldinfo 2,2
```

To run processes on different hosts, `--addr` arugment can be used witn host's IP address. (`python m8d.py --backend nccl --worldinfo 1,0 --worldinfo 2,0 --addr 10.20.1.50`)

## Example output

Running rank 0 (leader), will have the following output:

```bash
rank: 0 has tensor to be reduced: tensor([5., 2., 2.], device='cuda:0') # initial tensor for rank 0 from world1
rank: 0 has tensor to be reduced: tensor([2., 5., 3.], device='cuda:0') # initial tensor for rank 0 from world2
rank: 0 from world1 has reduced tensor: tensor([13.,  9.,  7.], device='cuda:0') # reduced tensor from rank 0 world1
done with step: 1  # indicator that step 1 of 100 is done for world1
rank: 0 from world2 has reduced tensor: tensor([ 8., 10., 12.], device='cuda:0') # reduced tensor from rank 0 world2
done with step: 1 # indicator that step 1 of 100 is done for world2
```

Running rank 1 from world1, will have the following output:

```bash
rank: 1 has tensor to be reduced: tensor([4., 5., 1.], device='cuda:1') # tensor for rank 1 from world1
done with step: 1  # indicator that step 1 of 100 is done
```

Running rank 2 from world1, will have the following output:

```bash
rank: 2 has tensor to be reduced: tensor([4., 2., 4.], device='cuda:2') # tensor for rank 2 from world1
done with step: 1  # indicator that step 1 of 100 is done
```

The following table provides a visual representation on how tensors are being reduced accross one world:

| Rank        | Initial tensor                                                         | Result                                                                    |
| :---        |    :----                                                               | :---                                                                      |
| 0           | <span style="color: red">tensor([5., 2., 2.], device='cuda:0')</span>  | <span style="color: cyan">tensor([13.,  9.,  7.], device='cuda:0')</span> |
| 1           | <span style="color: green">tensor([4., 5., 1.], device='cuda:1')</span>| tensor([4., 5., 1.], device='cuda:1')                                     |
| 2           | <span style="color: blue">tensor([4., 2., 4.], device='cuda:2')</span> | tensor([4., 2., 4.], device='cuda:2')                                     |

After the reduce operation, rank 0 (destination) has a reduced tensor using `dist.ReduceOp.SUM`.
Where:

* 13 is the sum of each rank 0 tensor[0] + rank 1, tensor[0] + rank 1, tensor[0].
* 9 is the sum of each rank 0 tensor[1] + rank 1, tensor[1] + rank 1, tensor[1].
* 7 is the sum of each rank 0 tensor[2] + rank 1, tensor[2] + rank 1, tensor[2].

The same pattern applies to world2.

## Failure case

If something goes wrong in one worker, only the world where the worker belongs will be affected, the other worlds will continue their workload.
In other words, Mutiworld prevents errors from spreading accross multiple worlds.
In this case, if rank 2 from world2 fails, rank 0 (destination) will still recieve reduced tensors from ranks from world1.

The following screenshot demonstrates how errors are handled in multiworld:

<p align="center"><img src="../../docs/imgs/reduce_error.png" alt="reduce error handling" width="800" height="300"></p>

Explanation:

1. Process is killed using keyboard interrupt on rank 2 from world 2
2. The exception is caught by all the workers in the same world (rank 1 from world 2 in this example)
3. The exception is also caught by the lead worker (rank 0)
4. The lead worker (rank 0) continues to be the destination of reduced tensor for the remaining worlds (world 1 in this example)
5. The reduce operation will continue for every other world that didn't had an error and the lead worker will be the destination of those reduced tensors
