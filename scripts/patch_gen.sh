#!/usr/bin/env bash

pytorch_src_dir=$1

if [[ $pytorch_src_dir == "" ]]; then
    echo "PyTorch source directory is missing."
    exit 1
fi

target=$(pwd)

pushd $pytorch_src_dir
version=$(git describe --tags)
git diff > $target/pytorch-$version.patch

popd
