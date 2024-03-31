"""
visualizer.py: Script to visualize the data from two files.

Description:
    This script reads data from two files and plots the throughput and tensors per second.

Sample usage:
    python visualizer.py --file1 ../varying_tensor_sizes/single_world_gloo_single_host.txt --file2 ../varying_tensor_sizes/multi_world_gloo_single_host.txt --output_dir ../varying_tensor_sizes/plots/single_host_gloo/
"""

import argparse
import matplotlib.pyplot as plt
import os


def read_data_from_files(filename):
    tensor_sizes = []
    iterations_list = []
    total_time_list = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            tensor_size = int(parts[0])
            iterations = int(parts[1])
            total_time = float(parts[2])
            
            tensor_sizes.append(tensor_size)
            iterations_list.append(iterations)
            total_time_list.append(total_time)

    return tensor_sizes, iterations_list, total_time_list


def plot_throughput(tensor_sizes, iterations_list, total_time_list, output_dir='.'):
    plt.clf()

    tensor_sizes1, tensor_sizes2 = zip(*tensor_sizes)
    iterations_list1, iterations_list2 = zip(*iterations_list)
    total_time_list1, total_time_list2 = zip(*total_time_list)

    throughput1 = [(iterations * tensor_size) / total_time for iterations, tensor_size, total_time in zip(iterations_list1, tensor_sizes1, total_time_list1)]
    throughput2 = [(iterations * tensor_size) / total_time for iterations, tensor_size, total_time in zip(iterations_list2, tensor_sizes2, total_time_list2)]

    # Plot data from file 1
    plt.plot(tensor_sizes1, throughput1, label='Single World')

    # Plot data from file 2
    plt.plot(tensor_sizes2, throughput2, label='Multi World')

    plt.xlabel('Tensor Size')
    plt.ylabel('Throughput')
    plt.title('Throughput Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'))


def plot_tensor_per_second(tensor_sizes, iterations_list, total_time_list, output_dir='.'):
    plt.clf()

    tensor_sizes1, tensor_sizes2 = zip(*tensor_sizes)
    iterations_list1, iterations_list2 = zip(*iterations_list)
    total_time_list1, total_time_list2 = zip(*total_time_list)

    tensors_per_second1 = [iterations / total_time for iterations, total_time in zip(iterations_list1, total_time_list1)]
    tensors_per_second2 = [iterations / total_time for iterations, total_time in zip(iterations_list2, total_time_list2)]

    # Plot data from file 1
    plt.plot(tensor_sizes1, tensors_per_second1, label='Single World')

    # Plot data from file 2
    plt.plot(tensor_sizes2, tensors_per_second2, label='Multi World')

    plt.xlabel('Tensor Size')
    plt.ylabel('Tensors/Second')
    plt.title('Tensors/Second Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'tensors_per_second_comparison.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data from two files.')
    parser.add_argument('--file1', help='Path to the first data file')
    parser.add_argument('--file2', help='Path to the second data file')
    parser.add_argument('--output_dir', help='Path to the output directory', default='.')
    args = parser.parse_args()

    tensor_sizes1, iterations_list1, total_time_list1 = read_data_from_files(args.file1)
    tensor_sizes2, iterations_list2, total_time_list2 = read_data_from_files(args.file2)

    # List of tuples
    tensor_sizes = []
    iterations_list = []
    total_time_list = []

    for i in range(len(tensor_sizes1)):
        tensor_sizes.append((tensor_sizes1[i], tensor_sizes2[i]))
        iterations_list.append((iterations_list1[i], iterations_list2[i]))
        total_time_list.append((total_time_list1[i], total_time_list2[i]))

    # Plot data
    plot_throughput(tensor_sizes, iterations_list, total_time_list, args.output_dir)
    plot_tensor_per_second(tensor_sizes, iterations_list, total_time_list, args.output_dir)
