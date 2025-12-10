#!/usr/bin/env python3
"""
Extract forgotten data images based on saved indices.

Usage:
    python extract_forgotten_data.py --indices_path <path_to_forgotten_indices.pt> \
                                     --data_dir <path_to_dataset> \
                                     --output_dir <output_folder>
"""

import torch
import os
import shutil
import argparse
from torchvision.datasets import ImageFolder
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Extract forgotten data images')
    parser.add_argument('--indices_path', type=str, required=True,
                       help='Path to forgotten_indices.pt file')
    parser.add_argument('--data_dir', type=str, default='datasets/tiny-imagenet-200/train',
                       help='Path to the training dataset directory')
    parser.add_argument('--output_dir', type=str, default='forgotten_images',
                       help='Output directory for forgotten images')
    args = parser.parse_args()

    # Load forgotten indices
    print(f"Loading forgotten indices from {args.indices_path}")
    indices = torch.load(args.indices_path)
    
    # Convert to list if it's a tensor
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    
    print(f"Found {len(indices)} forgotten sample indices")

    # Load dataset
    print(f"Loading dataset from {args.data_dir}")
    dataset = ImageFolder(args.data_dir)
    print(f"Dataset contains {len(dataset)} total images")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract and copy forgotten images
    print(f"Copying forgotten images to {args.output_dir}")
    for idx in tqdm(indices):
        if idx >= len(dataset):
            print(f"Warning: Index {idx} is out of bounds (dataset size: {len(dataset)})")
            continue
            
        img_path, class_idx = dataset.samples[idx]
        class_name = os.path.basename(os.path.dirname(img_path))
        
        # Create class subdirectory
        class_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy image
        img_filename = os.path.basename(img_path)
        dest_path = os.path.join(class_dir, img_filename)
        shutil.copy(img_path, dest_path)

    print(f"\n✓ Successfully copied {len(indices)} forgotten images to {args.output_dir}")
    
    # Print summary statistics
    class_dirs = os.listdir(args.output_dir)
    print(f"\nSummary:")
    print(f"  Total forgotten images: {len(indices)}")
    print(f"  Classes affected: {len(class_dirs)}")
    print(f"  Output location: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
