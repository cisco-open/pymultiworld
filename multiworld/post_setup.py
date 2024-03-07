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
    print(f"{args.patchfile}, {patch_basename}")

    path_to_sitepackages = site.getsitepackages()[0]

    dst = os.path.join(path_to_sitepackages, patch_basename)
    shutil.copyfile(args.patchfile, dst)

    os.chdir(path_to_sitepackages)

    os.system(f"patch < {patch_basename}")

    src = os.path.join(path_to_sitepackages, "multiworld", "world_manager.py")
    dst = os.path.join(
        path_to_sitepackages,
        "torch/distributed",
        "world_manager.py",
    )
    shutil.copyfile(src, dst)
    p = pathlib.Path(patch_basename)
    p.unlink()


if __name__ == "__main__":
    main()
