import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from synthetic_dataset import SyntheticDataset

def measure_epoch(loader):
    """
    Iterates through the entire dataset once and returns the elapsed time
    """
    start = time.perf_counter()
    
    for _ in loader:
        pass

    end = time.perf_counter()
    return end - start

def benchmark(n, runs = 1):
    """
    Benchmarks the time taken to iterate through the entire dataset for different number of workers
    """
    dataset = SyntheticDataset()
    loader = DataLoader(dataset, shuffle=False, num_workers=n, batch_size=64)

    # Warmup pass (initial overhead)
    print(f"Warmup for num_workers={n}...")
    for _ in loader:
        pass
    
    epoch_time = measure_epoch(loader)
    return epoch_time, dataset.total_samples / epoch_time

if __name__ == "__main__":

    worker_list = [1, 2, 4, 8]
    results = {}

    for n in worker_list:
        print(f"Measuring with {n} workers...")
        total_iteration_time, throughput = benchmark(n, 1)
        results[n] = total_iteration_time
                                                     
    # Compute scaling efficiency
    t1 = results[1]
    for n in worker_list:
        efficiency = t1 / (n * results[n])
        print(f"Workers: {n}, Time: {results[n]:.2f} seconds, Efficiency: {efficiency:.2f}")

