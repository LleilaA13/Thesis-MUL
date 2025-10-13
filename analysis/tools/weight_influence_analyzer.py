#!/usr/bin/env python3
"""
Weight Influence Analysis for Random Data Forgetting
Analyzes which specific weights and layers are most influenced by random data forgetting
"""

import torch
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib/seaborn not available. Plotting will be disabled.")

from collections import defaultdict
import json
import os
from datetime import datetime

class WeightInfluenceAnalyzer:
    """Analyzes weight changes caused by random data forgetting"""
    
    def __init__(self, baseline_model_path, experiment_models_dir):
        self.baseline_model_path = baseline_model_path
        self.experiment_models_dir = experiment_models_dir
        self.baseline_weights = None
        self.load_baseline()
    
    def load_baseline(self):
        """Load baseline model weights"""
        checkpoint = torch.load(self.baseline_model_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            self.baseline_weights = checkpoint['state_dict']
        else:
            self.baseline_weights = checkpoint
        print(f"✓ Loaded baseline model with {len(self.baseline_weights)} layers")
    
    def analyze_layer_sensitivity(self, experiment_name):
        """Analyze which layers are most sensitive to forgetting"""
        # Updated to work with your directory structure
        model_path = os.path.join(self.experiment_models_dir, experiment_name, 'RLcheckpoint.pth.tar')
        
        # Try alternative paths if the first doesn't exist
        if not os.path.exists(model_path):
            # Try without RLcheckpoint naming
            alt_paths = [
                os.path.join(self.experiment_models_dir, experiment_name, 'checkpoint.pth.tar'),
                os.path.join(self.experiment_models_dir, experiment_name, 'model_best.pth.tar'),
                os.path.join(self.experiment_models_dir, experiment_name, 'unlearned_model.pth.tar')
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                print(f"❌ Model not found for {experiment_name}")
                print(f"   Tried: {model_path}")
                for alt_path in alt_paths:
                    print(f"   Tried: {alt_path}")
                return None
        
        print(f"📊 Analyzing: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            unlearn_weights = checkpoint['state_dict']
        else:
            unlearn_weights = checkpoint
        
        layer_analysis = {}
        
        for layer_name in self.baseline_weights.keys():
            if layer_name in unlearn_weights:
                try:
                    baseline_layer = self.baseline_weights[layer_name]
                    unlearn_layer = unlearn_weights[layer_name]
                    
                    # Ensure tensors are float for calculations
                    if baseline_layer.dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                        baseline_layer = baseline_layer.float()
                    if unlearn_layer.dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                        unlearn_layer = unlearn_layer.float()
                    
                    # Calculate various metrics
                    diff = torch.abs(baseline_layer - unlearn_layer)
                    relative_diff = diff / (torch.abs(baseline_layer) + 1e-8)
                    
                    # Handle NaN values and convert to safe float values
                    def safe_float(tensor_value):
                        """Convert tensor value to safe float, handling NaN/inf"""
                        val = tensor_value.item()
                        if np.isnan(val) or np.isinf(val):
                            return 0.0
                        return val
                    
                    def safe_percentage(tensor):
                        """Calculate safe percentage, handling NaN"""
                        result = (tensor > 1e-6).float().mean()
                        val = result.item()
                        if np.isnan(val) or np.isinf(val):
                            return 0.0
                        return val * 100
                    
                    layer_analysis[layer_name] = {
                        'mean_absolute_change': safe_float(diff.mean()),
                        'max_absolute_change': safe_float(diff.max()),
                        'std_absolute_change': safe_float(diff.std()),
                        'mean_relative_change': safe_float(relative_diff.mean()),
                        'max_relative_change': safe_float(relative_diff.max()),
                        'percentage_changed': safe_percentage(diff),
                        'layer_shape': list(baseline_layer.shape),
                        'total_parameters': baseline_layer.numel()
                    }
                except Exception as e:
                    print(f"⚠️  Skipping layer {layer_name} due to error: {e}")
                    print(f"   Baseline dtype: {self.baseline_weights[layer_name].dtype}")
                    print(f"   Unlearn dtype: {unlearn_weights[layer_name].dtype}")
                    continue
        
        return layer_analysis
    
    def analyze_weight_magnitude_distribution(self, experiment_name):
        """Analyze how weight magnitude distributions change"""
        # Updated to work with your directory structure
        model_path = os.path.join(self.experiment_models_dir, experiment_name, 'RLcheckpoint.pth.tar')
        
        # Try alternative paths if the first doesn't exist
        if not os.path.exists(model_path):
            alt_paths = [
                os.path.join(self.experiment_models_dir, experiment_name, 'checkpoint.pth.tar'),
                os.path.join(self.experiment_models_dir, experiment_name, 'model_best.pth.tar'),
                os.path.join(self.experiment_models_dir, experiment_name, 'unlearned_model.pth.tar')
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                return None
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            unlearn_weights = checkpoint['state_dict']
        else:
            unlearn_weights = checkpoint
        
        distribution_analysis = {}
        
        for layer_name in self.baseline_weights.keys():
            if layer_name in unlearn_weights and 'weight' in layer_name:
                try:
                    baseline_layer = self.baseline_weights[layer_name].flatten()
                    unlearn_layer = unlearn_weights[layer_name].flatten()
                    
                    # Ensure tensors are float for calculations
                    if baseline_layer.dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                        baseline_layer = baseline_layer.float()
                    if unlearn_layer.dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                        unlearn_layer = unlearn_layer.float()
                    
                    # Handle NaN values and convert to safe float values
                    def safe_float(tensor_value):
                        """Convert tensor value to safe float, handling NaN/inf"""
                        val = tensor_value.item()
                        if np.isnan(val) or np.isinf(val):
                            return 0.0
                        return val
                    
                    distribution_analysis[layer_name] = {
                        'baseline_mean': safe_float(baseline_layer.mean()),
                        'baseline_std': safe_float(baseline_layer.std()),
                        'unlearn_mean': safe_float(unlearn_layer.mean()),
                        'unlearn_std': safe_float(unlearn_layer.std()),
                        'kl_divergence': self.calculate_kl_divergence(baseline_layer, unlearn_layer),
                        'wasserstein_distance': self.calculate_wasserstein_distance(baseline_layer, unlearn_layer)
                    }
                except Exception as e:
                    print(f"⚠️  Skipping distribution analysis for {layer_name}: {e}")
                    continue
        
        return distribution_analysis
    
    def calculate_kl_divergence(self, p, q, bins=100):
        """Calculate KL divergence between two weight distributions"""
        # Convert to numpy for histogram calculation
        p_np = p.detach().numpy()
        q_np = q.detach().numpy()
        
        # Create histograms
        min_val = min(p_np.min(), q_np.min())
        max_val = max(p_np.max(), q_np.max())
        
        p_hist, _ = np.histogram(p_np, bins=bins, range=(min_val, max_val), density=True)
        q_hist, _ = np.histogram(q_np, bins=bins, range=(min_val, max_val), density=True)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p_hist = p_hist + epsilon
        q_hist = q_hist + epsilon
        
        # Normalize
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()
        
        # Calculate KL divergence
        kl_div = np.sum(p_hist * np.log(p_hist / q_hist))
        return kl_div
    
    def calculate_wasserstein_distance(self, p, q):
        """Calculate Wasserstein distance between two distributions"""
        try:
            from scipy.stats import wasserstein_distance
            p_np = p.detach().numpy()
            q_np = q.detach().numpy()
            return wasserstein_distance(p_np, q_np)
        except ImportError:
            print("⚠️  scipy not available - using L2 distance instead")
            # Fallback to L2 distance
            return torch.norm(p - q, p=2).item()
    
    def identify_most_affected_weights(self, experiment_name, top_k=10):
        """Identify individual weights most affected by forgetting"""
        # Updated to work with your directory structure
        model_path = os.path.join(self.experiment_models_dir, experiment_name, 'RLcheckpoint.pth.tar')
        
        # Try alternative paths if the first doesn't exist
        if not os.path.exists(model_path):
            alt_paths = [
                os.path.join(self.experiment_models_dir, experiment_name, 'checkpoint.pth.tar'),
                os.path.join(self.experiment_models_dir, experiment_name, 'model_best.pth.tar'),
                os.path.join(self.experiment_models_dir, experiment_name, 'unlearned_model.pth.tar')
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                return []
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            unlearn_weights = checkpoint['state_dict']
        else:
            unlearn_weights = checkpoint
        
        all_changes = []
        weight_locations = []
        
        for layer_name in self.baseline_weights.keys():
            if layer_name in unlearn_weights and 'weight' in layer_name:
                baseline_layer = self.baseline_weights[layer_name]
                unlearn_layer = unlearn_weights[layer_name]
                
                # Ensure tensors are float for calculations
                if baseline_layer.dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                    baseline_layer = baseline_layer.float()
                if unlearn_layer.dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                    unlearn_layer = unlearn_layer.float()
                
                diff = torch.abs(baseline_layer - unlearn_layer)
                relative_diff = diff / (torch.abs(baseline_layer) + 1e-8)
                
                # Flatten and get positions
                flat_diff = relative_diff.flatten()
                for i, change in enumerate(flat_diff):
                    all_changes.append(change.item())
                    weight_locations.append((layer_name, i))
        
        # Get top-k most changed weights
        sorted_indices = np.argsort(all_changes)[::-1]
        top_changes = []
        
        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            layer_name, weight_idx = weight_locations[idx]
            change_magnitude = all_changes[idx]
            
            top_changes.append({
                'layer': layer_name,
                'weight_index': weight_idx,
                'relative_change': change_magnitude,
                'rank': i + 1
            })
        
        return top_changes
    
    def visualize_layer_changes(self, analysis_results, save_dir='experiments/random_forgetting/visualizations'):
        """Create visualizations of layer changes"""
        if not PLOTTING_AVAILABLE:
            print("⚠️  Skipping visualization - matplotlib not available")
            return
            
        os.makedirs(save_dir, exist_ok=True)
        
        for exp_name, layer_data in analysis_results.items():
            if not layer_data:  # Skip if no data
                continue
                
            # Extract data for plotting
            layer_names = list(layer_data.keys())
            mean_changes = [layer_data[name]['mean_relative_change'] for name in layer_names]
            max_changes = [layer_data[name]['max_relative_change'] for name in layer_names]
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Mean changes
            ax1.bar(range(len(layer_names)), mean_changes)
            ax1.set_title(f'Mean Relative Weight Changes - {exp_name}')
            ax1.set_xlabel('Layer')
            ax1.set_ylabel('Mean Relative Change')
            ax1.set_xticks(range(len(layer_names)))
            ax1.set_xticklabels(layer_names, rotation=45, ha='right')
            
            # Max changes
            ax2.bar(range(len(layer_names)), max_changes)
            ax2.set_title(f'Maximum Relative Weight Changes - {exp_name}')
            ax2.set_xlabel('Layer')
            ax2.set_ylabel('Max Relative Change')
            ax2.set_xticks(range(len(layer_names)))
            ax2.set_xticklabels(layer_names, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'layer_changes_{exp_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_comprehensive_report(self, output_dir='experiments/random_forgetting/weight_analysis'):
        """Generate comprehensive analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all experiment directories - Updated for your structure
        exp_dirs = [d for d in os.listdir(self.experiment_models_dir) 
                   if os.path.isdir(os.path.join(self.experiment_models_dir, d)) 
                   and 'random_forgetting' in d]
        
        print(f"Found {len(exp_dirs)} random forgetting experiments:")
        for exp in exp_dirs:
            print(f"  - {exp}")
        
        full_analysis = {}
        
        for exp_name in exp_dirs:
            print(f"📊 Analyzing experiment: {exp_name}")
            
            # Layer sensitivity analysis
            layer_analysis = self.analyze_layer_sensitivity(exp_name)
            if layer_analysis is None:
                continue
            
            # Weight distribution analysis
            distribution_analysis = self.analyze_weight_magnitude_distribution(exp_name)
            
            # Most affected weights
            top_affected = self.identify_most_affected_weights(exp_name)
            
            full_analysis[exp_name] = {
                'layer_sensitivity': layer_analysis,
                'distribution_changes': distribution_analysis,
                'top_affected_weights': top_affected
            }
        
        # Save comprehensive analysis
        with open(os.path.join(output_dir, 'comprehensive_weight_analysis.json'), 'w') as f:
            json.dump(full_analysis, f, indent=2)
        
        # Generate visualizations
        layer_analyses = {name: data['layer_sensitivity'] for name, data in full_analysis.items() 
                         if data.get('layer_sensitivity')}
        if layer_analyses:
            vis_dir = output_dir.replace('weight_analysis', 'visualizations')
            self.visualize_layer_changes(layer_analyses, vis_dir)
        
        # Generate summary report
        self.generate_summary_report(full_analysis, output_dir)
        
        print("✅ Comprehensive weight analysis complete!")
        return full_analysis
    
    def generate_summary_report(self, analysis_data, output_dir):
        """Generate a human-readable summary report"""
        report_lines = []
        report_lines.append("# Random Data Forgetting - Weight Analysis Summary")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("\n")
        
        for exp_name, data in analysis_data.items():
            report_lines.append(f"## Experiment: {exp_name}")
            
            # Parse experiment parameters from name
            parts = exp_name.split('_')
            if len(parts) >= 6:
                ratio = parts[1]
                mask = parts[3]
                method = parts[5]
                report_lines.append(f"- **Forget Ratio**: {ratio}")
                report_lines.append(f"- **Mask Threshold**: {mask}")
                report_lines.append(f"- **Unlearn Method**: {method}")
            
            # Layer sensitivity summary
            layer_data = data['layer_sensitivity']
            most_affected_layer = max(layer_data.keys(), key=lambda x: layer_data[x]['mean_relative_change'])
            least_affected_layer = min(layer_data.keys(), key=lambda x: layer_data[x]['mean_relative_change'])
            
            report_lines.append(f"- **Most Affected Layer**: {most_affected_layer} ({layer_data[most_affected_layer]['mean_relative_change']:.6f})")
            report_lines.append(f"- **Least Affected Layer**: {least_affected_layer} ({layer_data[least_affected_layer]['mean_relative_change']:.6f})")
            
            # Top affected weights
            top_weights = data['top_affected_weights'][:3]
            report_lines.append("- **Top 3 Most Changed Weights**:")
            for weight in top_weights:
                report_lines.append(f"  - {weight['layer']} (change: {weight['relative_change']:.6f})")
            
            report_lines.append("\n")
        
        # Save report
        with open(os.path.join(output_dir, 'summary_report.md'), 'w') as f:
            f.write('\n'.join(report_lines))

def main():
    # Configuration - Updated for your directory structure
    baseline_model = "models/resnet50_pretrained.pth"  # Your baseline pretrained model
    results_dir = "results"  # Your results directory
    
    if not os.path.exists(baseline_model):
        print("❌ Baseline model not found. Please check the path.")
        print(f"   Looking for: {baseline_model}")
        return
    
    if not os.path.exists(results_dir):
        print("❌ Results directory not found. Please check the path.")
        print(f"   Looking for: {results_dir}")
        return
    
    analyzer = WeightInfluenceAnalyzer(baseline_model, results_dir)
    analysis_results = analyzer.generate_comprehensive_report()
    
    print("🎉 Weight influence analysis complete!")
    print("📂 Check experiments/random_forgetting/weight_analysis/ for results")

if __name__ == "__main__":
    main()