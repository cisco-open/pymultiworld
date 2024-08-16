.. _getting_started:

===================
**Getting Started**
===================

Welcome to multiworld!
------------------------
`multiworld` is a powerful framework designed to enhance the resilience of collective communication libraries (CCL) such as NCCL, which are commonly used in distributed machine learning and deep learning tasks. The framework leverages PyTorch to create an environment where machine learning models can continue functioning even in the face of hardware or communication failures, ensuring robustness in distributed ML serving setups.

This guide will walk you through the steps needed to install `multiworld` and run your first example program. By the end of this guide, you’ll be familiar with the basics of `multiworld` and how to use it in your own projects.

Installation
------------
`multiworld` can be installed in a few different ways depending on your needs. Instructions for installing it via PyPI and directly from the source can be found in :ref:`Installation section <installation>`

Running Your First Example
---------------------------
Once you have `multiworld` installed and your environment configured, you can run your first example to see the framework in action.

Example: Send and Receive
^^^^^^^^^^^^^^^^^^^^^^^^^
`multiworld` comes with several example scripts that demonstrate how to use the framework. Let's start with the basic `send_recv` example, which illustrates the fundamental concepts of sending and receiving messages in a fault-tolerant way.

1. **Navigate to the Example Directory:**

   The examples are located within the `examples` directory of the repository. Navigate to the `send_recv` example:

   .. code-block:: bash

       cd examples/send_recv

2. **Run the Example Script:**

   Once inside the directory, you can run the example script using Python:

    .. code:: bash

        # on terminal window 1
        python m8d.py --backend nccl --worldinfo 1,0 --worldinfo 2,0 --addr 10.20.1.50
        # on terminal window 2
        python m8d.py --backend nccl --worldinfo 1,1 --addr 10.20.1.50
        # on terminal window 3
        python m8d.py --backend nccl --worldinfo 2,1 --addr 10.20.1.50

   This script simulates a simple communication scenario, demonstrating how `multiworld` handles message passing in a distributed environment. It will output logs to the console, showing the steps it takes to manage faults and maintain communication integrity.

Understanding the Example Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After running the example, take some time to review the output. The logs should provide insight into how `multiworld` monitors communication channels and handles disruptions. Understanding this process is crucial as it underpins the fault-tolerant mechanisms you’ll use in your own projects.

For more details, see :ref:`Examples section <examples>`

Multiworld API
--------------
`multiworld` API is documented in detail in the :ref:`API documentation section <api_doc>`

Contributing to multiworld
--------------------------
`multiworld` is an open-source project, and contributions from the community are highly encouraged. Whether you want to fix bugs, add new features, or improve documentation, your help is valuable.
Before contributing, please take a moment to review our `Contributing Guidelines <https://github.com/cisco-open/pymultiworld/blob/main/CONTRIBUTING.md>`. These guidelines outline the process for submitting issues, making changes, and following the code of conduct.

Reporting Issues
^^^^^^^^^^^^^^^^
If you encounter any issues while using `multiworld`, please report them through the GitHub Issues page. Be sure to include detailed information about the problem and any steps to reproduce it.


