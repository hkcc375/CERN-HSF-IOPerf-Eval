import os
import argparse
import numpy as np

def generate_dataset(num_shards, num_samples, num_features, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating dataset with {num_shards} shards, {num_samples} samples per shard, and {num_features} features per sample.")

    for shard_idx in range(num_shards):
        filename = os.path.join(output_dir, f"shard_{shard_idx:02d}.npy")
        data = np.zeros((num_samples, num_features), dtype=np.float32)
        np.save(filename, data)
    
    print(f"Total dataset size: {num_shards * num_samples} samples, {num_features} features each.")
    print("Dataset generation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset.")
    parser.add_argument("--num_shards", type=int, default=10, help="Number of shards to split the dataset into.")
    parser.add_argument("--num_samples", type=int, default=7000, help="Number of samples per shard.")
    parser.add_argument("--num_features", type=int, default=20000, help="Number of features per sample.")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Directory to save the generated dataset.")
    
    args = parser.parse_args()
    
    generate_dataset(args.num_shards, args.num_samples, args.num_features, args.output_dir)