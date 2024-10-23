import os
import argparse
import os
import torch
import pickle
import multiprocessing

from helper_state_dataset_fit import create_state_dataset
from helper_switch_dataset import create_switch_dataset


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def combine_multiprocessing_results(results, target_sample_size):
    observations = torch.cat([res[0] for res in results])
    y = torch.cat([res[1] for res in results])
    important_features = torch.cat([res[2] for res in results])

    # Double-check that everything is correct
    assert observations.shape[0] == target_sample_size
    assert y.shape[0] == target_sample_size
    assert important_features.shape[0] == target_sample_size

    return observations, y, important_features


def load_or_create_dataset(args):
    target_path = os.path.join(".", "data", f"{args.dataset}", str(args.seed))
    create_directory(target_path)
    file_path = os.path.join(target_path, "dataset.pkl")
    if os.path.exists(file_path):
        print(f"We found the dataset at {file_path}. We will load it")
        with open(file_path, "rb") as f:
            return pickle.load(f)

    else:
        num_processes = multiprocessing.cpu_count()
        print(f"We are using a total of {num_processes} cpus!")
        chunk_size = args.sample_size // num_processes
        remaining_samples = args.sample_size % num_processes

        pool = multiprocessing.Pool(num_processes)
        tasks = [(args.signal_length, chunk_size) for _ in range(num_processes)]

        if remaining_samples > 0:
            tasks.append((args.signal_length, remaining_samples))

        if args.dataset == "state":
            results = pool.starmap(create_state_dataset, tasks)
        elif args.dataset == "switch":
            results = pool.starmap(create_switch_dataset, tasks)

        pool.close()
        pool.join()

        observations, y, important_features = combine_multiprocessing_results(results, args.sample_size)

        with open(file_path, "wb") as f:
            pickle.dump((observations, y, important_features), f)
        return observations, y, important_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="state", choices=("state, switch"))
    parser.add_argument("--signal_length", type=int, default=200, help="Length of the time_series")
    parser.add_argument("--sample_size", type=int, default=1_000, help="Number of timeseries to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    observations, y, important_features = load_or_create_dataset(args)