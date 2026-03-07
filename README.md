# PyTorch DataLoader I/O Benchmark

## 1. System Requirements

The benchmark was developed and tested on:

- **Python** ≥ 3.10
- **Conda / Miniconda / Anaconda**
- **Operating System**: macOS / Linux
- **Disk space**: ≥ 6 GB (for the synthetic dataset)

Required Python packages:

- torch
- numpy
- pyyaml

---

## 2. Install Conda (if not already installed)

Download and install **Miniconda**:

https://docs.conda.io/en/latest/miniconda.html

Verify installation:

```bash
conda --version
```

---

## 3. Create a clean conda environment

Create a new environment:

```bash
conda create -n darshan-bench python=3.11
```

Activate the environment:

```bash
conda activate darshan-bench
```

---

## 4. Install dependencies

Install required packages:

```bash
pip install torch numpy pyyaml
```

Alternatively using conda:

```bash
conda install pytorch numpy pyyaml -c pytorch
```

Verify installation:

```bash
python -c "import torch, numpy, yaml; print('Packages installed successfully')"
```

---

## 5. Project Directory Structure

Ensure the following files are present:

```
project_directory/
│
├── benchmark.py
├── generate_dataset.py
├── synthetic_dataset.py
├── metadata.yaml
```

---

## 6. Generate the synthetic dataset

Run the dataset generation script:

```bash
python generate_dataset.py
```

This will create:

```
dataset/
   shard_00.npy
   shard_01.npy
   ...
```

along with a metadata file describing the dataset.

Expected dataset size: **~5 GB**.

---

## 7. Run the benchmark

Execute the benchmark script:

```bash
python benchmark.py
```

The script will:

1. Load the dataset using the PyTorch DataLoader  
2. Run a warmup pass  
3. Benchmark different numbers of workers  
4. Report runtime, throughput, and I/O scaling efficiency  

Output:

```
Workers: 1, Time: 21.93 seconds, Throughput: 3192.02 samples/second, I/O Scaling Efficiency: 100.00%
Workers: 2, Time: 21.56 seconds, Throughput: 3246.81 samples/second, I/O Scaling Efficiency: 50.86%
Workers: 4, Time: 16.71 seconds, Throughput: 4189.22 samples/second, I/O Scaling Efficiency: 32.81%
Workers: 8, Time: 13.53 seconds, Throughput: 5173.65 samples/second, I/O Scaling Efficiency: 20.26%
```

---
