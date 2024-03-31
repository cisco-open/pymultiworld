# Script to run experiments for varying tensor sizes
# Sample Usage for Single World scenario:
# NOTE: Usage for Multi World scenario is similar. 
# Single host: Run the following command on a single host. You may switch the parameters according to the experiment.
# bash run_experiments.sh --filename single_world.py --backend gloo --output_file single_world_gloo_single_host.txt
# 
# Multi host: Run the following command on each host. Change the rank and addr for each host.
# You may switch the parameters according to the experiment.
# NOTE: The --addr specified on each host must be the same.
# bash run_experiments.sh --filename single_world.py --backend nccl --output_file single_world_nccl_multi_host.txt --multihost --addr 10.20.1.50 --rank 0


#!/bin/bash

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --backend)
            backend="$2"
            shift 2
            ;;
        --output_file)
            output_file="$2"
            shift 2
            ;;
        --filename)
            filename="$2"
            shift 2
            ;;
        --multihost)
            multihost="--multihost"
            shift 1
            ;;
        --addr)
            addr="$2"
            shift 2
            ;;
        --rank)
            rank="$2"
            shift 2
            ;;
        --iterations)
            iterations="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [[ -z $filename ]]; then
    echo "Filename not specified. Use --filename option."
    exit 1
fi

if [[ -z $backend ]]; then
    echo "Backend not specified. Use --backend option."
    exit 1
fi

if [[ -z $output_file ]]; then
    echo "Output file not specified. Use --output_file option."
    exit 1
fi

# If multihost is specified, ensure addr and rank are provided
if [[ ! -z $multihost ]]; then
    if [[ -z $addr ]]; then
        echo "Address (--addr) not specified. Use --addr option with --multihost."
        exit 1
    fi
    if [[ -z $rank ]]; then
        echo "Rank (--rank) not specified. Use --rank option with --multihost."
        exit 1
    fi
fi

# Default iterations if not provided
if [[ -z $iterations ]]; then
    iterations=100000
fi

# Define the list of tensor sizes
tensor_sizes=(1 3 9 27 81 243 729 2187 6561 19683 59049 177147)

# Iterate over each tensor size and run the command
for size in "${tensor_sizes[@]}"; do
    if [[ ! -z $multihost ]]; then
        python "$filename" --output_file "$output_file" --tensor_size "$size" --iterations "$iterations" --backend "$backend" $multihost --addr "$addr" --rank "$rank"
    else
        python "$filename" --output_file "$output_file" --tensor_size "$size" --iterations "$iterations" --backend "$backend"
    fi
    echo "Finished running for tensor size: $size"
    sleep 1
done
