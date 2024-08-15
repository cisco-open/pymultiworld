.. _installation:

================
**Installation**
================

`multiworld` can be installed in a few different ways depending on your needs. Below are instructions for installing it via PyPI and directly from the source.

Installing from PyPI
--------------------
The easiest way to get started with `multiworld` is by installing it from the Python Package Index (PyPI). This method ensures you get the latest stable release.

To install, simply run:

.. code-block:: bash

    pip install multiworld

This command will download and install the latest version of `multiworld` and all of its dependencies.

Installing from Source
----------------------
If you prefer to work with the latest development version or want to contribute to the project, you can clone the repository and install it manually.

First, clone the repository from GitHub:

.. code-block:: bash

    git clone https://github.com/cisco-open/pymultiworld.git

Then, navigate to the cloned directory and install the package:

.. code-block:: bash

    cd pymultiworld
    pip install .

This method allows you to modify the source code or contribute back to the project.

Run post installation script
----------------------------

After installation, run the `m8d-post-setup` script:

.. code-block:: bash

    m8d-post-setup
