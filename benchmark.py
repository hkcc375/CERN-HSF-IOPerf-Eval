import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from synthetic_dataset import SyntheticDataset

def benchmark(n, runs=5):
    """
    Benchmarks the time taken to iterate through the entire dataset for different number of workers
    """
    dataset = SyntheticDataset()
    loader = DataLoader(dataset, shuffle=False, num_workers=n, batch_size=64, persistent_workers=(n>0))
    
    # Warmup pass
    for _ in loader:
        pass

    # Measurement passes
    times = []

    for run in range(runs):
        start = time.perf_counter()
        for _ in loader:
            pass
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = np.mean(times)
    throughput = len(dataset) / avg_time
    return avg_time, throughput

if __name__ == "__main__":

    worker_list = [1, 2, 4, 8]
    results = {}
    throughputs = {}

    for n in worker_list:
        print(f"Benchmarking with {n} workers (warmup + measurement)...")
        avg_time, throughput = benchmark(n, 5)
        results[n] = avg_time
        throughputs[n] = throughput

    # Compute scaling efficiency
    t1 = results[1]
    for n in worker_list:
        efficiency = (t1 / (n * results[n])) * 100
        print(f"Workers: {n}, Time: {results[n]:.2f} seconds, Throughput: {throughputs[n]:.2f} samples/second, I/O Scaling Efficiency: {efficiency:.2f}%")

