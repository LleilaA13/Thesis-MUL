#!/usr/bin/env python3
"""
Channel-Level Weight Analysis for Lucent Integration
Analyzes which specific channels within layers are most affected by forgetting
"""

import torch
import json
import numpy as np
import os
from collections import defaultdict

class ChannelLevelAnalyzer:
    """Analyzes weight changes at the channel level for Lucent targeting"""
    
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
    
    def analyze_channel_sensitivity(self, experiment_name, top_k_channels=10):
        """Analyze which channels within each layer are most affected"""
        model_path = os.path.join(self.experiment_models_dir, experiment_name, 'RLcheckpoint.pth.tar')
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None
        
        print(f"📊 Analyzing channels in: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            unlearn_weights = checkpoint['state_dict']
        else:
            unlearn_weights = checkpoint
        
        channel_analysis = {}
        
        for layer_name in self.baseline_weights.keys():
            if layer_name in unlearn_weights and 'weight' in layer_name:
                try:
                    baseline_layer = self.baseline_weights[layer_name]
                    unlearn_layer = unlearn_weights[layer_name]
                    
                    # Ensure tensors are float
                    if baseline_layer.dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                        baseline_layer = baseline_layer.float()
                    if unlearn_layer.dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                        unlearn_layer = unlearn_layer.float()
                    
                    # Analyze different layer types
                    if len(baseline_layer.shape) == 4:  # Conv layers: [out_channels, in_channels, H, W]
                        channel_changes = self.analyze_conv_channels(baseline_layer, unlearn_layer, layer_name)
                    elif len(baseline_layer.shape) == 2:  # FC layers: [out_features, in_features]
                        channel_changes = self.analyze_fc_channels(baseline_layer, unlearn_layer, layer_name)
                    elif len(baseline_layer.shape) == 1:  # BN/Bias layers: [features]
                        channel_changes = self.analyze_1d_channels(baseline_layer, unlearn_layer, layer_name)
                    else:
                        continue  # Skip unsupported layer types
                    
                    if channel_changes:
                        channel_analysis[layer_name] = channel_changes
                        
                except Exception as e:
                    print(f"⚠️  Skipping channel analysis for {layer_name}: {e}")
                    continue
        
        return channel_analysis
    
    def analyze_conv_channels(self, baseline, unlearn, layer_name):
        """Analyze convolutional layer channels [out_channels, in_channels, H, W]"""
        diff = torch.abs(baseline - unlearn)
        
        # Calculate change per output channel (sum over in_channels, H, W)
        out_channel_changes = diff.sum(dim=(1, 2, 3))  # Sum over [in_channels, H, W]
        
        # Calculate relative changes
        baseline_magnitudes = torch.abs(baseline).sum(dim=(1, 2, 3))
        relative_changes = out_channel_changes / (baseline_magnitudes + 1e-8)
        
        # Get top changed channels
        top_channels = torch.topk(relative_changes, min(10, len(relative_changes)))
        
        channel_info = {
            'layer_type': 'conv',
            'layer_shape': list(baseline.shape),
            'total_out_channels': baseline.shape[0],
            'most_affected_channels': [],
            'least_affected_channels': [],
            'channel_statistics': {
                'mean_change_per_channel': self.safe_float(out_channel_changes.mean()),
                'std_change_per_channel': self.safe_float(out_channel_changes.std()),
                'max_change_channel': int(torch.argmax(out_channel_changes).item()),
                'min_change_channel': int(torch.argmin(out_channel_changes).item())
            }
        }
        
        # Most affected channels
        for i, (change_val, channel_idx) in enumerate(zip(top_channels.values, top_channels.indices)):
            channel_info['most_affected_channels'].append({
                'channel_index': int(channel_idx.item()),
                'absolute_change': self.safe_float(out_channel_changes[channel_idx]),
                'relative_change': self.safe_float(change_val),
                'lucent_target': f"{layer_name.replace('.weight', '')}:{channel_idx.item()}",
                'rank': i + 1
            })
        
        # Least affected channels
        bottom_channels = torch.topk(relative_changes, min(5, len(relative_changes)), largest=False)
        for i, (change_val, channel_idx) in enumerate(zip(bottom_channels.values, bottom_channels.indices)):
            channel_info['least_affected_channels'].append({
                'channel_index': int(channel_idx.item()),
                'absolute_change': self.safe_float(out_channel_changes[channel_idx]),
                'relative_change': self.safe_float(change_val),
                'lucent_target': f"{layer_name.replace('.weight', '')}:{channel_idx.item()}",
                'rank': i + 1
            })
        
        return channel_info
    
    def analyze_fc_channels(self, baseline, unlearn, layer_name):
        """Analyze fully connected layer channels [out_features, in_features]"""
        diff = torch.abs(baseline - unlearn)
        
        # Calculate change per output neuron (sum over input features)
        neuron_changes = diff.sum(dim=1)  # Sum over input features
        
        baseline_magnitudes = torch.abs(baseline).sum(dim=1)
        relative_changes = neuron_changes / (baseline_magnitudes + 1e-8)
        
        # Get top changed neurons
        top_neurons = torch.topk(relative_changes, min(10, len(relative_changes)))
        
        channel_info = {
            'layer_type': 'fc',
            'layer_shape': list(baseline.shape),
            'total_neurons': baseline.shape[0],
            'most_affected_neurons': [],
            'least_affected_neurons': [],
            'neuron_statistics': {
                'mean_change_per_neuron': self.safe_float(neuron_changes.mean()),
                'std_change_per_neuron': self.safe_float(neuron_changes.std()),
                'max_change_neuron': int(torch.argmax(neuron_changes).item()),
                'min_change_neuron': int(torch.argmin(neuron_changes).item())
            }
        }
        
        # Most affected neurons
        for i, (change_val, neuron_idx) in enumerate(zip(top_neurons.values, top_neurons.indices)):
            channel_info['most_affected_neurons'].append({
                'neuron_index': int(neuron_idx.item()),
                'absolute_change': self.safe_float(neuron_changes[neuron_idx]),
                'relative_change': self.safe_float(change_val),
                'lucent_target': f"{layer_name.replace('.weight', '')}:{neuron_idx.item()}",
                'rank': i + 1
            })
        
        # Least affected neurons
        bottom_neurons = torch.topk(relative_changes, min(5, len(relative_changes)), largest=False)
        for i, (change_val, neuron_idx) in enumerate(zip(bottom_neurons.values, bottom_neurons.indices)):
            channel_info['least_affected_neurons'].append({
                'neuron_index': int(neuron_idx.item()),
                'absolute_change': self.safe_float(neuron_changes[neuron_idx]),
                'relative_change': self.safe_float(change_val),
                'lucent_target': f"{layer_name.replace('.weight', '')}:{neuron_idx.item()}",
                'rank': i + 1
            })
        
        return channel_info
    
    def analyze_1d_channels(self, baseline, unlearn, layer_name):
        """Analyze 1D layers like BatchNorm [features]"""
        diff = torch.abs(baseline - unlearn)
        relative_changes = diff / (torch.abs(baseline) + 1e-8)
        
        # Get top changed elements
        top_elements = torch.topk(relative_changes, min(10, len(relative_changes)))
        
        channel_info = {
            'layer_type': '1d',
            'layer_shape': list(baseline.shape),
            'total_elements': baseline.shape[0],
            'most_affected_elements': [],
            'least_affected_elements': [],
            'element_statistics': {
                'mean_change': self.safe_float(diff.mean()),
                'std_change': self.safe_float(diff.std()),
                'max_change_element': int(torch.argmax(diff).item()),
                'min_change_element': int(torch.argmin(diff).item())
            }
        }
        
        # Most affected elements
        for i, (change_val, elem_idx) in enumerate(zip(top_elements.values, top_elements.indices)):
            channel_info['most_affected_elements'].append({
                'element_index': int(elem_idx.item()),
                'absolute_change': self.safe_float(diff[elem_idx]),
                'relative_change': self.safe_float(change_val),
                'rank': i + 1
            })
        
        return channel_info
    
    def safe_float(self, tensor_value):
        """Convert tensor value to safe float, handling NaN/inf"""
        if torch.is_tensor(tensor_value):
            val = tensor_value.item()
        else:
            val = float(tensor_value)
        
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    
    def get_lucent_targets_for_experiment(self, experiment_name, top_k=20):
        """Get the most important Lucent targets for an experiment"""
        channel_data = self.analyze_channel_sensitivity(experiment_name)
        
        if not channel_data:
            return []
        
        lucent_targets = []
        
        for layer_name, layer_info in channel_data.items():
            layer_type = layer_info['layer_type']
            
            if layer_type == 'conv' and 'most_affected_channels' in layer_info:
                for channel in layer_info['most_affected_channels']:
                    lucent_targets.append({
                        'target': channel['lucent_target'],
                        'layer': layer_name,
                        'channel': channel['channel_index'],
                        'change_magnitude': channel['relative_change'],
                        'type': 'conv_channel'
                    })
            
            elif layer_type == 'fc' and 'most_affected_neurons' in layer_info:
                for neuron in layer_info['most_affected_neurons']:
                    lucent_targets.append({
                        'target': neuron['lucent_target'],
                        'layer': layer_name,
                        'neuron': neuron['neuron_index'],
                        'change_magnitude': neuron['relative_change'],
                        'type': 'fc_neuron'
                    })
        
        # Sort by change magnitude and return top-k
        lucent_targets.sort(key=lambda x: x['change_magnitude'], reverse=True)
        return lucent_targets[:top_k]
    
    def generate_lucent_visualization_script(self, experiment_name, output_file='lucent_channel_targets.py'):
        """Generate a ready-to-run Lucent visualization script"""
        targets = self.get_lucent_targets_for_experiment(experiment_name)
        
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated Lucent visualization script for {experiment_name}
Targets the most affected channels/neurons identified by weight analysis
Run from repository root directory: python experiments/channel_analysis/{experiment_name}/visualize_{experiment_name}.py
"""

import torch
from torchvision import models
from lucent.optvis import render, objectives
import os

def load_model(model_path, device='auto'):
    """Load the unlearned model"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 200)  # Tiny ImageNet classes
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    return model

def visualize_top_affected_channels():
    """Visualize the most affected channels"""
    model_path = "experiments/results/good_results/{experiment_name}/RLcheckpoint.pth.tar"
    output_dir = "experiments/channel_visualizations/{experiment_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎨 Loading model: {{model_path}}")
    model = load_model(model_path)
    
    print(f"🎯 Visualizing top {{len(targets)}} most affected channels/neurons...")
    
    # Top affected targets from weight analysis
    targets = {targets}
    
    for i, target_info in enumerate(targets):
        target = target_info['target']
        change_mag = target_info['change_magnitude']
        target_type = target_info['type']
        
        print(f"\\n{{i+1:2d}}. {{target}} ({{target_type}}, change: {{change_mag:.4f}})")
        
        try:
            # Render visualization
            img = render.render_vis(model, target, show_inline=False, thresholds=(512,))
            
            # Save with descriptive name
            safe_target = target.replace(':', '_').replace('.', '_')
            filename = f"rank{{i+1:02d}}_{{safe_target}}_change{{change_mag:.4f}}.png"
            save_path = os.path.join(output_dir, filename)
            
            if hasattr(img, 'save'):
                img.save(save_path)
            
            print(f"✅ Saved: {{filename}}")
            
        except Exception as e:
            print(f"❌ Failed to visualize {{target}}: {{e}}")
    
    print(f"\\n🎉 Visualization complete! Check {{output_dir}}")

if __name__ == "__main__":
    visualize_top_affected_channels()
'''
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        print(f"✅ Generated Lucent script: {output_file}")
        print(f"📊 Script targets {len(targets)} most affected channels/neurons")
        
        return output_file
    
    def analyze_all_experiments(self, experiments_dir='experiments/results/good_results', 
                               output_dir='experiments/channel_analysis'):
        """Analyze all experiments and generate comprehensive channel analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all experiment directories
        exp_dirs = [d for d in os.listdir(experiments_dir) 
                   if os.path.isdir(os.path.join(experiments_dir, d))]
        
        print(f"🔬 CHANNEL-LEVEL ANALYSIS")
        print("="*50)
        print(f"Found {len(exp_dirs)} experiments:")
        for exp in exp_dirs:
            print(f"  - {exp}")
        
        all_results = {}
        
        for exp_name in exp_dirs:
            print(f"\\n📊 Analyzing {exp_name}...")
            
            # Set the correct path for analysis
            self.experiment_models_dir = experiments_dir
            
            # Analyze channels
            channel_data = self.analyze_channel_sensitivity(exp_name)
            if channel_data:
                all_results[exp_name] = channel_data
                
                # Generate Lucent targets
                targets = self.get_lucent_targets_for_experiment(exp_name)
                
                # Save experiment-specific results
                exp_output_dir = os.path.join(output_dir, exp_name)
                os.makedirs(exp_output_dir, exist_ok=True)
                
                # Save channel analysis
                with open(os.path.join(exp_output_dir, 'channel_analysis.json'), 'w') as f:
                    json.dump(channel_data, f, indent=2)
                
                # Save Lucent targets
                with open(os.path.join(exp_output_dir, 'lucent_targets.json'), 'w') as f:
                    json.dump(targets, f, indent=2)
                
                # Generate visualization script
                script_path = os.path.join(exp_output_dir, f'visualize_{exp_name}.py')
                self.generate_lucent_visualization_script(exp_name, script_path)
                
                print(f"✅ {exp_name}: {len(targets)} Lucent targets identified")
        
        # Save comprehensive results
        with open(os.path.join(output_dir, 'comprehensive_channel_analysis.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate summary report
        self.generate_channel_summary_report(all_results, output_dir)
        
        print(f"\\n🎉 Channel analysis complete!")
        print(f"📂 Results saved to: {output_dir}")
        
        return all_results
    
    def generate_channel_summary_report(self, results, output_dir):
        """Generate summary report of channel analysis"""
        report = []
        report.append("# Channel-Level Weight Analysis Summary")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## Key Findings")
        report.append("")
        
        for exp_name, exp_data in results.items():
            report.append(f"### {exp_name}")
            
            # Count different layer types
            conv_layers = sum(1 for layer_info in exp_data.values() if layer_info.get('layer_type') == 'conv')
            fc_layers = sum(1 for layer_info in exp_data.values() if layer_info.get('layer_type') == 'fc')
            
            report.append(f"- **Analyzed layers**: {len(exp_data)} total ({conv_layers} conv, {fc_layers} fc)")
            
            # Find most affected layer
            max_change_layer = None
            max_change = 0
            for layer_name, layer_info in exp_data.items():
                if layer_info['layer_type'] == 'conv' and 'most_affected_channels' in layer_info:
                    if layer_info['most_affected_channels']:
                        change = layer_info['most_affected_channels'][0]['relative_change']
                        if change > max_change:
                            max_change = change
                            max_change_layer = layer_name
            
            if max_change_layer:
                report.append(f"- **Most affected layer**: {max_change_layer} (change: {max_change:.4f})")
            
            report.append("")
        
        report.append("## Lucent Visualization Targets")
        report.append("")
        report.append("Each experiment has generated:")
        report.append("- `channel_analysis.json`: Detailed per-channel analysis")
        report.append("- `lucent_targets.json`: Top 20 targets for Lucent")
        report.append("- `visualize_[experiment].py`: Ready-to-run visualization script")
        report.append("")
        
        report.append("## Usage")
        report.append("")
        report.append("```bash")
        report.append("# Run visualization for specific experiment (from repository root)")
        report.append("python experiments/channel_analysis/[experiment_name]/visualize_[experiment_name].py")
        report.append("")
        report.append("# Or run channel analysis from analysis tools directory")
        report.append("cd analysis/tools")
        report.append("python channel_level_analyzer.py")
        report.append("```")
        
        # Save report
        report_path = os.path.join(output_dir, 'channel_analysis_summary.md')
        with open(report_path, 'w') as f:
            f.write('\\n'.join(report))
        
        print(f"✅ Channel summary report saved: {report_path}")

def main():
    """Main function to run channel-level analysis"""
    baseline_model = "experiments/models/resnet50_pretrained.pth"
    
    if not os.path.exists(baseline_model):
        print(f"❌ Baseline model not found: {baseline_model}")
        return
    
    analyzer = ChannelLevelAnalyzer(baseline_model, "experiments/results/good_results")
    analyzer.analyze_all_experiments()

if __name__ == "__main__":
    from datetime import datetime
    main()