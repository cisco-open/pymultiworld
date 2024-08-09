Send and Recieve
================

-  ```m8d.py`` </examples/send_recv/m8d.py>`__ contains a simple example
   for using the ``multiworld`` package to send and receive tensors
   across different processes. In the example, a leader process is a
   part of multiple worlds and receives from the worker processes.
   Script can be run using the following commands.

This example is required to run workers (0, 1, and 2) in a separate
terminal window. The lead worker needs to be executed with two world 1
and 2, with the rank of 0 The child workers must match the world index
of the lead worker and the rank of 1. ``--worldinfo`` argument is
composed by the world index and the rank of the worker in that world.
(e.g. ``--worldinfo 1,0`` means that the worker will have rank ``0`` in
the world with the index ``1``) The script can be executed in a single
host or across hosts. To run processes on different hosts, ``--addr``
arugment can be used. For example, run the following commands, by
changing the IP address (10.20.1.50) correctly in your setting.

.. code:: bash

   # on terminal window 1
   python m8d.py --backend nccl --worldinfo 1,0 --worldinfo 2,0 --addr 10.20.1.50
   # on terminal window 2
   python m8d.py --backend nccl --worldinfo 1,1 --addr 10.20.1.50
   # on terminal window 3
   python m8d.py --backend nccl --worldinfo 2,1 --addr 10.20.1.50

Here the IP address is the IP address of rank 0. We assume that at least
3 GPUs are available either in a single host or across hosts. If the
scripts are executed in a single host, ``--addr`` can be omitted.

While running the above example, one can terminate a worker (e.g., rank
2) and the leader (rank 0) continues to receive tensors from the
remaining worker.

``MultiWorld`` facilitates fault management functionality at a worker
level, meaning that it can detect, tolerate and recover faults that are
occuring at a worker in a host. So, one can run the above example in a
single host or across hosts. For the cross-host execution, the IP
address must be the IP address of rank 0.

-  ```single_world.py`` </examples/single_world.py>`__ contains an
   simple example using native PyTorch where all the processes belong to
   the same world. Script can be run using the following commands.

For running all processes on the same host, run the command:

.. code:: bash

   python single_world.py --backend nccl --worldsize 3

For running processes on different hosts, at least two hosts are needed.
For example, run the following commands for a two host setting:

.. code:: bash

   # on host 1
   python single_world.py --backend nccl --addr 10.20.1.50 --multihost --worldsize 3 --rank 0
   # on host 2
   python single_world.py --backend nccl --addr 10.20.1.50 --multihost --worldsize 3 --rank 1
   # on host 2
   python single_world.py --backend nccl --addr 10.20.1.50 --multihost --worldsize 3 --rank 2
