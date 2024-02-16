#!/usr/bin/env bash

pytorch_src_dir=$1

if [[ $pytorch_src_dir == "" ]]; then
    echo "PyTorch source directory is missing."
    exit 1
fi

target=$(pwd)

pushd $pytorch_src_dir
git diff > $target/pytorch.patch

pushd third_party
third_party_pkgs=(gloo)
for pkg in ${third_party_pkgs[@]}; do
    cd $pkg
    git diff > $target/$pkg.patch
    cd ..
done
popd

popd
