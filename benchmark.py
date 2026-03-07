import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from synthetic_dataset import SyntheticDataset

def benchmark(n, runs = 1):
    """
    Benchmarks the time taken to iterate through the entire dataset for different number of workers
    """
    dataset = SyntheticDataset()
    loader = DataLoader(dataset, shuffle=False, num_workers=n, batch_size=64, persistent_workers=(n>0))
    
    # Warmup pass
    for _ in loader:
        pass

    # Measurement pass
    start = time.perf_counter()
    for _ in loader:
        pass
    elapsed = time.perf_counter() - start
    return elapsed, dataset.total_samples / elapsed

if __name__ == "__main__":

    worker_list = [1, 2, 4, 8]
    results = {}

    for n in worker_list:
        print(f"Benchmarking with {n} workers (warmup + measurement)...")
        total_iteration_time, throughput = benchmark(n, 1)
        results[n] = total_iteration_time
                                                     
    # Compute scaling efficiency
    t1 = results[1]
    for n in worker_list:
        efficiency = t1 / (n * results[n])
        print(f"Workers: {n}, Time: {results[n]:.2f} seconds, Efficiency: {efficiency:.2f}")

