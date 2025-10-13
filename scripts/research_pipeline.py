#!/usr/bin/env python3
"""
Research Pipeline: Random Data Forgetting + Feature Visualization Analysis
Author: Your Name
Date: October 2025

This script implements the complete pipeline for analyzing how random data forgetting
affects learned features through visualization using Lucent.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
from datetime import datetime

# Add Classification directory to path
sys.path.append('./Classification')
sys.path.append('./Classification/models')

class RandomDataForgettingAnalyzer:
    """
    Analyzes the impact of random data forgetting on model features
    """
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for results"""
        dirs = [
            'experiments/random_forgetting',
            'experiments/random_forgetting/models',
            'experiments/random_forgetting/masks',
            'experiments/random_forgetting/visualizations',
            'experiments/random_forgetting/weight_analysis'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def run_baseline_training(self):
        """Train baseline model for comparison"""
        print("[1/6] Training baseline model...")
        
        cmd = f"""python Classification/main_train.py \
            --arch {self.config['arch']} \
            --dataset {self.config['dataset']} \
            --epochs {self.config['train_epochs']} \
            --lr {self.config['train_lr']} \
            --save_dir experiments/random_forgetting/models/baseline \
            --batch_size {self.config['batch_size']}"""
        
        os.system(cmd)
        print("✓ Baseline training complete")
    
    def generate_random_forget_indices(self, forget_ratios=[0.1, 0.2, 0.3]):
        """Generate random indices to forget for different ratios"""
        print("[2/6] Generating random forget indices...")
        
        # This would typically load your dataset to get total size
        # For now, using CIFAR-10 default (50000 training samples)
        total_samples = 50000  # Adjust based on your dataset
        
        for ratio in forget_ratios:
            num_forget = int(total_samples * ratio)
            forget_indices = np.random.choice(total_samples, num_forget, replace=False)
            
            # Save indices
            np.save(f'experiments/random_forgetting/forget_indices_{ratio:.1f}.npy', forget_indices)
            print(f"✓ Generated {num_forget} random indices for {ratio:.1%} forgetting")
    
    def generate_saliency_masks(self):
        """Generate saliency masks for different forget ratios"""
        print("[3/6] Generating saliency masks...")
        
        forget_ratios = [0.1, 0.2, 0.3]
        mask_thresholds = [0.3, 0.5, 0.7]
        
        for ratio in forget_ratios:
            indices_path = f'experiments/random_forgetting/forget_indices_{ratio:.1f}.npy'
            mask_dir = f'experiments/random_forgetting/masks/ratio_{ratio:.1f}'
            os.makedirs(mask_dir, exist_ok=True)
            
            cmd = f"""python Classification/generate_mask.py \
                --arch {self.config['arch']} \
                --dataset {self.config['dataset']} \
                --model_path experiments/random_forgetting/models/baseline/model_best.pth.tar \
                --save_dir {mask_dir} \
                --subset_indices_path {indices_path} \
                --num_indexes_to_replace {int(50000 * ratio)}"""
            
            os.system(cmd)
            print(f"✓ Generated saliency masks for {ratio:.1%} forgetting")
    
    def run_unlearning_experiments(self):
        """Run unlearning experiments with different methods and parameters"""
        print("[4/6] Running unlearning experiments...")
        
        forget_ratios = [0.1, 0.2, 0.3]
        mask_thresholds = [0.3, 0.5, 0.7]
        unlearn_methods = ['RL', 'GA']
        
        for ratio in forget_ratios:
            for threshold in mask_thresholds:
                for method in unlearn_methods:
                    exp_name = f"ratio_{ratio:.1f}_mask_{threshold}_method_{method}"
                    exp_dir = f"experiments/random_forgetting/models/{exp_name}"
                    os.makedirs(exp_dir, exist_ok=True)
                    
                    mask_path = f"experiments/random_forgetting/masks/ratio_{ratio:.1f}/with_{threshold}.pt"
                    indices_path = f'experiments/random_forgetting/forget_indices_{ratio:.1f}.npy'
                    
                    cmd = f"""python Classification/main_random.py \
                        --unlearn {method} \
                        --unlearn_epochs {self.config['unlearn_epochs']} \
                        --unlearn_lr {self.config['unlearn_lr']} \
                        --num_indexes_to_replace {int(50000 * ratio)} \
                        --model_path experiments/random_forgetting/models/baseline/model_best.pth.tar \
                        --save_dir {exp_dir} \
                        --mask_path {mask_path} \
                        --subset_indices_path {indices_path} \
                        --arch {self.config['arch']} \
                        --dataset {self.config['dataset']}"""
                    
                    os.system(cmd)
                    print(f"✓ Completed experiment: {exp_name}")
    
    def analyze_weight_changes(self):
        """Analyze which weights were most affected by forgetting"""
        print("[5/6] Analyzing weight changes...")
        
        # Load baseline model
        baseline_path = "experiments/random_forgetting/models/baseline/model_best.pth.tar"
        baseline_weights = torch.load(baseline_path)['state_dict']
        
        weight_analysis = {}
        
        # Analyze each experiment
        exp_dirs = [d for d in os.listdir('experiments/random_forgetting/models') if d != 'baseline']
        
        for exp_dir in exp_dirs:
            model_path = f"experiments/random_forgetting/models/{exp_dir}/unlearn/model_best.pth.tar"
            if os.path.exists(model_path):
                unlearn_weights = torch.load(model_path)['state_dict']
                
                # Calculate weight differences
                weight_diffs = {}
                for layer_name in baseline_weights.keys():
                    if layer_name in unlearn_weights:
                        diff = torch.abs(baseline_weights[layer_name] - unlearn_weights[layer_name])
                        weight_diffs[layer_name] = {
                            'mean_diff': diff.mean().item(),
                            'max_diff': diff.max().item(),
                            'std_diff': diff.std().item()
                        }
                
                weight_analysis[exp_dir] = weight_diffs
        
        # Save analysis
        with open('experiments/random_forgetting/weight_analysis/weight_changes.json', 'w') as f:
            json.dump(weight_analysis, f, indent=2)
        
        print("✓ Weight change analysis complete")
    
    def setup_lucent_visualization(self):
        """Setup Lucent for feature visualization"""
        print("[6/6] Setting up Lucent visualization...")
        
        # Create Lucent visualization script
        lucent_script = '''
import torch
import numpy as np
from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1
import matplotlib.pyplot as plt

def visualize_layer_features(model, layer_name, channels=None, num_iterations=512):
    """Visualize features of a specific layer"""
    if channels is None:
        channels = range(min(16, model.features[layer_name].out_channels))
    
    for channel in channels:
        # Render feature visualization
        result = render.render_vis(
            model,
            objectives.channel(layer_name, channel),
            param_f=lambda: param.image(224),
            optimizer=torch.optim.Adam,
            transforms=transform.standard_transforms,
            steps=num_iterations,
        )
        
        # Save visualization
        plt.figure(figsize=(6, 6))
        plt.imshow(result[0])
        plt.axis('off')
        plt.title(f'Layer: {layer_name}, Channel: {channel}')
        plt.savefig(f'experiments/random_forgetting/visualizations/{layer_name}_channel_{channel}.png')
        plt.close()

def compare_features_before_after(baseline_model, unlearned_model, layer_name):
    """Compare features before and after unlearning"""
    # Implementation for feature comparison
    pass
'''
        
        with open('experiments/random_forgetting/lucent_visualizer.py', 'w') as f:
            f.write(lucent_script)
        
        print("✓ Lucent visualization setup complete")
    
    def run_full_pipeline(self):
        """Run the complete research pipeline"""
        print("🚀 Starting Random Data Forgetting Analysis Pipeline")
        print("=" * 60)
        
        self.run_baseline_training()
        self.generate_random_forget_indices()
        self.generate_saliency_masks()
        self.run_unlearning_experiments()
        self.analyze_weight_changes()
        self.setup_lucent_visualization()
        
        print("=" * 60)
        print("✅ Pipeline complete! Check experiments/random_forgetting/ for results")

def main():
    # Configuration for experiments
    config = {
        'arch': 'resnet18',
        'dataset': 'cifar10',
        'train_epochs': 100,
        'train_lr': 0.1,
        'unlearn_epochs': 10,
        'unlearn_lr': 0.013,
        'batch_size': 128
    }
    
    analyzer = RandomDataForgettingAnalyzer(config)
    analyzer.run_full_pipeline()

if __name__ == "__main__":
    main()