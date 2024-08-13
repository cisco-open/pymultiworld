All Reduce
==========

This file provides an example of collective communication using
all_reduce across single and multiple worlds. This exaplme will perform
all_reduce 100 times on each rank from each world.

``--worldinfo`` argument is composed by the world index(1, 2) and the
rank in that world (0, 1 or 2).

Running the Script in a Single World
------------------------------------

The single world example can be executed by opening 3 separate terminal
windows to have 3 different processes and running the following commands
in each terminal window:

.. code:: bash

   # on terminal window 1 - will initialize 2 worlds (world1 and world2) with rank 0
   python m8d.py --backend nccl --worldinfo 1,0 --worldinfo 2,0
   # on terminal window 2 - will initialize world1 with rank 1
   python m8d.py --backend nccl --worldinfo 1,1
   # on terminal window 3 - will initialize world1 with rank 2
   python m8d.py --backend nccl --worldinfo 1,2

Running the Script in Multiple Worlds
-------------------------------------

The multiple world example can be executed by opening 5 separate
terminal windows to have 5 different processes and running the following
commands in each terminal window:

.. code:: bash

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

To run processes on different hosts, ``--addr`` arugment can be used
witn host’s IP address.
(``python m8d.py --backend nccl --worldinfo 1,0 --worldinfo 2,0 --addr 10.20.1.50``)
