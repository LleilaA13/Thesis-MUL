#!/usr/bin/env python3
"""
Quick visualization script to verify extracted forgotten images.
Shows a grid of random samples from the forgotten dataset.
"""

import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def visualize_forgotten_images(forgotten_dir, num_images=16, seed=42):
    """
    Display a grid of random forgotten images.
    
    Args:
        forgotten_dir: Path to the forgotten_images/images directory
        num_images: Number of images to display
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Get all image files
    img_files = [f for f in os.listdir(forgotten_dir) if f.endswith('.JPEG')]
    
    if len(img_files) == 0:
        print(f"No images found in {forgotten_dir}")
        return
    
    # Randomly sample images
    num_images = min(num_images, len(img_files))
    sampled_files = random.sample(img_files, num_images)
    
    # Create grid
    grid_size = int(num_images ** 0.5)
    if grid_size * grid_size < num_images:
        grid_size += 1
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    print(f"Displaying {num_images} random forgotten images from {len(img_files)} total")
    print(f"Directory: {forgotten_dir}\n")
    
    for idx, (ax, img_file) in enumerate(zip(axes, sampled_files)):
        img_path = os.path.join(forgotten_dir, img_file)
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        # Extract class name from filename (format: n01443537_103.JPEG)
        class_id = img_file.split('_')[0]
        ax.set_title(class_id, fontsize=8)
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Random Sample of Forgotten Images', y=1.02, fontsize=14)
    
    # Save to file
    output_path = os.path.join(os.path.dirname(forgotten_dir), 'forgotten_samples_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize forgotten images')
    parser.add_argument('--forgotten_dir', type=str, 
                       default='experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/forgotten_images/images',
                       help='Path to forgotten images directory')
    parser.add_argument('--num_images', type=int, default=16,
                       help='Number of images to display')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.forgotten_dir):
        print(f"Error: Directory not found: {args.forgotten_dir}")
        return
    
    visualize_forgotten_images(args.forgotten_dir, args.num_images, args.seed)

if __name__ == '__main__':
    main()
