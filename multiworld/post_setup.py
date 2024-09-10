# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import shutil
import site
import sys


def configure_once():
    """Configure multiworld once when it is used for the first time."""
    package_name = __name__.split(".")[0]
    path_to_sitepackages = site.getsitepackages()[0]

    init_file_path = os.path.join(path_to_sitepackages, package_name, "init.txt")

    try:
        with open(init_file_path, "r") as file:
            patch_applied = file.read()
    except FileNotFoundError:
        message = "WARNING: initialization check file not found; "
        message += f"{package_name} is not installed correctly; "
        message += "please reinstall."
        print(message)
        return

    if patch_applied == "true":
        return

    print(f"Configuring {package_name} for the first time. This is one time task.")

    import torch

    torch_version = torch.__version__.split("+")[
        0
    ]  # torch version is in "2.2.1+cu121" format

    patchfile = os.path.join(
        path_to_sitepackages,
        package_name,
        "patch",
        "pytorch-v" + torch_version + ".patch",
    )

    patch_basename = os.path.basename(patchfile)

    dst = os.path.join(path_to_sitepackages, patch_basename)
    shutil.copyfile(patchfile, dst)

    os.chdir(path_to_sitepackages)

    os.system(f"patch -p1 < {patch_basename} > /dev/null")
    p = pathlib.Path(patch_basename)
    p.unlink()

    with open(init_file_path, "w") as file:
        file.write("true")

    sys.exit(
        f"This one-time configuration for {package_name} is completed.\nYou can run your script without any interruption from now on."
    )
