import argparse
import os
import pathlib
import shutil
import site


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patchfile")
    args = parser.parse_args()

    patch_basename = os.path.basename(args.patchfile)

    path_to_sitepackages = site.getsitepackages()[0]

    dst = os.path.join(path_to_sitepackages, patch_basename)
    shutil.copyfile(args.patchfile, dst)

    os.chdir(path_to_sitepackages)

    os.system(f"patch < {patch_basename}")
    p = pathlib.Path(patch_basename)
    p.unlink()

    files_to_copy = ["world_manager.py", "world_communicator.py"]
    for f in files_to_copy:
        src = os.path.join(path_to_sitepackages, "multiworld", f)
        dst = os.path.join(path_to_sitepackages, "torch/distributed", f)
        shutil.copyfile(src, dst)


if __name__ == "__main__":
    main()
