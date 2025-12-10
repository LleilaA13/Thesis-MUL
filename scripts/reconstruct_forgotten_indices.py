#!/usr/bin/env python3
"""
Reconstruct forgotten indices from the original experiment setup.

This script re-runs the data marking logic to identify which samples were forgotten,
then saves the indices for extraction.
"""

import torch
import numpy as np
import os
import sys
import argparse

# Add the parent directory to path to import dataset utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core', 'Classification'))

import arg_parser
import utils

def main():
    parser = argparse.ArgumentParser(description='Reconstruct forgotten indices')
    parser.add_argument('--data_dir', type=str, default='datasets/tiny-imagenet-200',
                       help='Path to Tiny ImageNet dataset')
    parser.add_argument('--num_indexes_to_replace', type=int, default=10000,
                       help='Number of samples that were forgotten (same as in training)')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed used in the original experiment')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save forgotten_indices.pt')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for data loading')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of workers')
    args_custom = parser.parse_args()

    print(f"Reconstructing forgotten indices...")
    print(f"  Dataset: {args_custom.data_dir}")
    print(f"  Num forgotten: {args_custom.num_indexes_to_replace}")
    print(f"  Seed: {args_custom.seed}")

    # Create minimal args object for utils.setup_model_dataset
    class Args:
        def __init__(self):
            self.dataset = "TinyImagenet"
            self.data = args_custom.data_dir
            self.data_dir = args_custom.data_dir
            self.batch_size = args_custom.batch_size
            self.workers = args_custom.workers
            self.num_indexes_to_replace = args_custom.num_indexes_to_replace
            self.seed = args_custom.seed
            self.train_seed = args_custom.seed
            self.class_to_replace = -1  # -1 means random forgetting across all classes
            self.indexes_to_replace = None
            self.arch = "resnet50"
            self.imagenet_arch = True
            self.no_aug = False
    
    args = Args()
    utils.setup_seed(args.seed)
    
    # Load the dataset using the same method as main_random.py
    # marked_loader is the 5th return value - it contains the dataset with negative labels
    (_, _, _, _, marked_loader) = utils.setup_model_dataset(args)

    # Extract indices where target is negative (marked for forgetting)
    print("Extracting forgotten indices from marked dataset...")
    
    dataset = marked_loader.dataset
    forgotten_indices = []
    
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        if target < 0:  # Negative targets indicate forgotten samples
            forgotten_indices.append(idx)
    
    forgotten_indices = torch.tensor(forgotten_indices)
    
    print(f"Found {len(forgotten_indices)} forgotten sample indices")
    
    # Save to file
    os.makedirs(os.path.dirname(args_custom.output_path), exist_ok=True)
    torch.save(forgotten_indices, args_custom.output_path)
    print(f"Saved forgotten indices to: {args_custom.output_path}")
    
    # Print some statistics
    if len(forgotten_indices) > 0:
        print(f"\nStatistics:")
        print(f"  Min index: {forgotten_indices.min().item()}")
        print(f"  Max index: {forgotten_indices.max().item()}")
        print(f"  First 10 indices: {forgotten_indices[:10].tolist()}")

if __name__ == "__main__":
    main()
