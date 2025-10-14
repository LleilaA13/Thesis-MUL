#!/usr/bin/env python3
"""
Real Influence Analyzer for Machine Unlearning

This script analyzes actual model weights to find the most influenced:
1. Layers (by comparing baseline vs unlearned models)
2. Channels within those layers
3. Individual weights within those channels

Author: Research Pipeline
Usage: python real_influence_analyzer.py --baseline_model path/to/baseline.pth --unlearned_model path/to/unlearned.pth
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Any, Optional
import torchvision.models as models

class RealInfluenceAnalyzer:
    """
    Analyzes real model weights to find most influenced components
    """
    
    def __init__(self, output_dir: str = "analysis/results/real_influence_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.baseline_model = None
        self.unlearned_model = None
        
        # Analysis results storage
        self.layer_influences = {}
        self.channel_influences = {}
        self.weight_influences = {}
        
        print(f"🔍 Real Influence Analyzer initialized")
        print(f"📊 Output directory: {self.output_dir}")
    
    def load_models(self, baseline_path: str, unlearned_path: str, num_classes: int = 200):
        """Load baseline and unlearned models for comparison"""
        
        print(f"📥 Loading baseline model from: {baseline_path}")
        print(f"📥 Loading unlearned model from: {unlearned_path}")
        
        # Create model architecture (ResNet50 for Tiny ImageNet)
        def create_model():
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        
        # Load baseline model
        self.baseline_model = create_model()
        if baseline_path.endswith('.tar'):
            baseline_checkpoint = torch.load(baseline_path, map_location='cpu')
            if 'state_dict' in baseline_checkpoint:
                self.baseline_model.load_state_dict(baseline_checkpoint['state_dict'], strict=False)
            else:
                self.baseline_model.load_state_dict(baseline_checkpoint, strict=False)
        else:
            self.baseline_model.load_state_dict(torch.load(baseline_path, map_location='cpu'), strict=False)
        
        # Load unlearned model
        self.unlearned_model = create_model()
        if unlearned_path.endswith('.tar'):
            unlearned_checkpoint = torch.load(unlearned_path, map_location='cpu')
            if 'state_dict' in unlearned_checkpoint:
                self.unlearned_model.load_state_dict(unlearned_checkpoint['state_dict'], strict=False)
            else:
                self.unlearned_model.load_state_dict(unlearned_checkpoint, strict=False)
        else:
            self.unlearned_model.load_state_dict(torch.load(unlearned_path, map_location='cpu'), strict=False)
        
        self.baseline_model.eval()
        self.unlearned_model.eval()
        
        print(f"✅ Models loaded successfully")
        print(f"📊 Model architecture: {self.baseline_model.__class__.__name__}")
        
        # Verify models have same architecture
        baseline_params = dict(self.baseline_model.named_parameters())
        unlearned_params = dict(self.unlearned_model.named_parameters())
        
        if set(baseline_params.keys()) != set(unlearned_params.keys()):
            raise ValueError("Models have different architectures!")
        
        print(f"🔗 Total parameters: {sum(p.numel() for p in self.baseline_model.parameters()):,}")
        return True
    
    def analyze_layer_influences(self, top_k: int = 30) -> pd.DataFrame:
        """Analyze which layers are most influenced by unlearning"""
        
        print(f"🎯 Analyzing layer influences (top {top_k})...")
        
        layer_data = []
        
        baseline_params = dict(self.baseline_model.named_parameters())
        unlearned_params = dict(self.unlearned_model.named_parameters())
        
        for layer_name in baseline_params.keys():
            baseline_weight = baseline_params[layer_name].data
            unlearned_weight = unlearned_params[layer_name].data
            
            # Calculate influence metrics
            diff = torch.abs(baseline_weight - unlearned_weight)
            
            # Avoid division by zero
            baseline_abs = torch.abs(baseline_weight)
            rel_diff = torch.where(baseline_abs > 1e-8, diff / baseline_abs, diff)
            
            layer_influence = {
                'layer_name': layer_name,
                'mean_absolute_change': diff.mean().item(),
                'max_absolute_change': diff.max().item(),
                'mean_relative_change': rel_diff.mean().item(),
                'max_relative_change': rel_diff.max().item(),
                'std_absolute_change': diff.std().item(),
                'percentage_changed': (diff > 1e-6).float().mean().item() * 100,
                'total_parameters': baseline_weight.numel(),
                'layer_type': self._get_layer_type(layer_name),
                'layer_depth': self._get_layer_depth(layer_name),
                'shape': list(baseline_weight.shape)
            }
            
            layer_data.append(layer_influence)
        
        # Create DataFrame and sort by relative change
        df = pd.DataFrame(layer_data)
        df_sorted = df.sort_values('mean_relative_change', ascending=False)
        
        # Store top layers
        top_layers = df_sorted.head(top_k)
        self.layer_influences = top_layers.to_dict('records')
        
        print(f"✅ Analyzed {len(df)} layers, found top {len(top_layers)} most influenced")
        
        return top_layers
    
    def analyze_channel_influences(self, top_layers: pd.DataFrame, top_k_channels: int = 50) -> pd.DataFrame:
        """Analyze which channels within top layers are most influenced"""
        
        print(f"🎯 Analyzing channel influences for top layers...")
        
        channel_data = []
        
        baseline_params = dict(self.baseline_model.named_parameters())
        unlearned_params = dict(self.unlearned_model.named_parameters())
        
        # Focus on conv layers from top influenced layers
        conv_layers = top_layers[top_layers['layer_name'].str.contains('conv') & 
                                top_layers['layer_name'].str.contains('weight')].head(10)
        
        for _, layer_info in conv_layers.iterrows():
            layer_name = layer_info['layer_name']
            
            baseline_weight = baseline_params[layer_name].data  # Shape: [out_channels, in_channels, H, W]
            unlearned_weight = unlearned_params[layer_name].data
            
            if len(baseline_weight.shape) != 4:  # Skip non-conv weights
                continue
            
            out_channels = baseline_weight.shape[0]
            
            # Analyze each output channel
            for channel_idx in range(out_channels):
                channel_baseline = baseline_weight[channel_idx]  # [in_channels, H, W]
                channel_unlearned = unlearned_weight[channel_idx]
                
                # Calculate channel-specific influence
                diff = torch.abs(channel_baseline - channel_unlearned)
                baseline_abs = torch.abs(channel_baseline)
                rel_diff = torch.where(baseline_abs > 1e-8, diff / baseline_abs, diff)
                
                channel_influence = {
                    'layer_name': layer_name,
                    'channel_index': channel_idx,
                    'channel_id': f"{layer_name}_ch{channel_idx}",
                    'mean_absolute_change': diff.mean().item(),
                    'max_absolute_change': diff.max().item(),
                    'mean_relative_change': rel_diff.mean().item(),
                    'max_relative_change': rel_diff.max().item(),
                    'std_absolute_change': diff.std().item(),
                    'percentage_changed': (diff > 1e-6).float().mean().item() * 100,
                    'channel_parameters': channel_baseline.numel(),
                    'layer_total_relative_change': layer_info['mean_relative_change']
                }
                
                channel_data.append(channel_influence)
        
        if not channel_data:
            print("⚠️  No conv layers found for channel analysis")
            return pd.DataFrame()
        
        # Create DataFrame and sort by relative change
        df = pd.DataFrame(channel_data)
        df_sorted = df.sort_values('mean_relative_change', ascending=False)
        
        # Store top channels
        top_channels = df_sorted.head(top_k_channels)
        self.channel_influences = top_channels.to_dict('records')
        
        print(f"✅ Analyzed {len(df)} channels, found top {len(top_channels)} most influenced")
        
        return top_channels
    
    def analyze_weight_influences(self, top_channels: pd.DataFrame, top_k_weights: int = 100) -> pd.DataFrame:
        """Analyze individual weights within top channels"""
        
        print(f"🎯 Analyzing individual weight influences...")
        
        weight_data = []
        
        baseline_params = dict(self.baseline_model.named_parameters())
        unlearned_params = dict(self.unlearned_model.named_parameters())
        
        # Focus on top channels
        top_channel_subset = top_channels.head(20)  # Top 20 channels for detailed weight analysis
        
        for _, channel_info in top_channel_subset.iterrows():
            layer_name = channel_info['layer_name']
            channel_idx = channel_info['channel_index']
            
            baseline_weight = baseline_params[layer_name].data
            unlearned_weight = unlearned_params[layer_name].data
            
            if len(baseline_weight.shape) != 4:  # Skip non-conv weights
                continue
            
            # Get specific channel weights
            channel_baseline = baseline_weight[channel_idx]  # [in_channels, H, W]
            channel_unlearned = unlearned_weight[channel_idx]
            
            # Flatten to analyze individual weights
            flat_baseline = channel_baseline.flatten()
            flat_unlearned = channel_unlearned.flatten()
            
            # Calculate per-weight influences
            diff = torch.abs(flat_baseline - flat_unlearned)
            baseline_abs = torch.abs(flat_baseline)
            rel_diff = torch.where(baseline_abs > 1e-8, diff / baseline_abs, diff)
            
            # Get top weights within this channel
            top_indices = torch.topk(rel_diff, min(10, len(rel_diff)), largest=True).indices
            
            for rank, weight_idx in enumerate(top_indices):
                weight_influence = {
                    'layer_name': layer_name,
                    'channel_index': channel_idx,
                    'weight_index': weight_idx.item(),
                    'weight_id': f"{layer_name}_ch{channel_idx}_w{weight_idx.item()}",
                    'baseline_value': flat_baseline[weight_idx].item(),
                    'unlearned_value': flat_unlearned[weight_idx].item(),
                    'absolute_change': diff[weight_idx].item(),
                    'relative_change': rel_diff[weight_idx].item(),
                    'rank_in_channel': rank + 1,
                    'channel_total_change': channel_info['mean_relative_change']
                }
                
                weight_data.append(weight_influence)
        
        if not weight_data:
            print("⚠️  No weights found for analysis")
            return pd.DataFrame()
        
        # Create DataFrame and sort by relative change
        df = pd.DataFrame(weight_data)
        df_sorted = df.sort_values('relative_change', ascending=False)
        
        # Store top weights
        top_weights = df_sorted.head(top_k_weights)
        self.weight_influences = top_weights.to_dict('records')
        
        print(f"✅ Analyzed {len(df)} individual weights, found top {len(top_weights)} most influenced")
        
        return top_weights
    
    def generate_lucent_targets(self, top_layers: pd.DataFrame, top_channels: pd.DataFrame) -> List[Dict]:
        """Generate Lucent-compatible targeting strings"""
        
        print(f"🎨 Generating Lucent targets...")
        
        lucent_targets = []
        
        # Layer-level targets (from conv layers)
        conv_layers = top_layers[top_layers['layer_name'].str.contains('conv') & 
                                top_layers['layer_name'].str.contains('weight')].head(10)
        
        for _, layer in conv_layers.iterrows():
            layer_name = layer['layer_name']
            lucent_layer = layer_name.replace('.weight', '')
            
            lucent_targets.append({
                'target': lucent_layer,
                'type': 'layer',
                'influence_score': layer['mean_relative_change'],
                'description': f"Layer {lucent_layer} (Change: {layer['mean_relative_change']:.4f})",
                'layer_name': layer_name
            })
        
        # Channel-level targets
        if not top_channels.empty:
            for _, channel in top_channels.head(15).iterrows():
                layer_name = channel['layer_name']
                channel_idx = channel['channel_index']
                lucent_layer = layer_name.replace('.weight', '')
                
                target_str = f"{lucent_layer}:{channel_idx}"
                
                lucent_targets.append({
                    'target': target_str,
                    'type': 'channel',
                    'influence_score': channel['mean_relative_change'],
                    'description': f"Channel {channel_idx} in {lucent_layer} (Change: {channel['mean_relative_change']:.4f})",
                    'layer_name': layer_name,
                    'channel_index': channel_idx
                })
        
        # Sort by influence score
        lucent_targets.sort(key=lambda x: x['influence_score'], reverse=True)
        
        print(f"✅ Generated {len(lucent_targets)} Lucent targets")
        return lucent_targets
    
    def create_visualizations(self, top_layers: pd.DataFrame, top_channels: pd.DataFrame, top_weights: pd.DataFrame):
        """Create comprehensive visualizations"""
        
        print(f"📊 Creating visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Top Layers by Influence
        plt.subplot(2, 4, 1)
        top_10_layers = top_layers.head(10)
        y_pos = range(len(top_10_layers))
        plt.barh(y_pos, top_10_layers['mean_relative_change'])
        plt.yticks(y_pos, [name.split('.')[-2] + '.' + name.split('.')[-1] for name in top_10_layers['layer_name']])
        plt.xlabel('Mean Relative Change')
        plt.title('Top 10 Most Influenced Layers')
        plt.gca().invert_yaxis()
        
        # 2. Layer Types Distribution
        plt.subplot(2, 4, 2)
        layer_type_counts = top_layers['layer_type'].value_counts()
        plt.pie(layer_type_counts.values, labels=layer_type_counts.index, autopct='%1.1f%%')
        plt.title('Layer Types Distribution')
        
        # 3. Influence by Layer Depth
        plt.subplot(2, 4, 3)
        depth_influence = top_layers.groupby('layer_depth')['mean_relative_change'].mean().sort_index()
        plt.bar(depth_influence.index, depth_influence.values)
        plt.xlabel('Layer Depth')
        plt.ylabel('Average Relative Change')
        plt.title('Influence by Layer Depth')
        
        # 4. Parameter Count vs Influence
        plt.subplot(2, 4, 4)
        plt.scatter(top_layers['total_parameters'], top_layers['mean_relative_change'], alpha=0.6)
        plt.xlabel('Total Parameters')
        plt.ylabel('Mean Relative Change')
        plt.title('Parameters vs Influence')
        plt.xscale('log')
        
        # 5. Top Channels (if available)
        plt.subplot(2, 4, 5)
        if not top_channels.empty:
            top_10_channels = top_channels.head(10)
            y_pos = range(len(top_10_channels))
            plt.barh(y_pos, top_10_channels['mean_relative_change'])
            plt.yticks(y_pos, [f"Ch{ch['channel_index']}" for _, ch in top_10_channels.iterrows()])
            plt.xlabel('Mean Relative Change')
            plt.title('Top 10 Most Influenced Channels')
            plt.gca().invert_yaxis()
        else:
            plt.text(0.5, 0.5, 'No Channel Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Channel Analysis')
        
        # 6. Channel Distribution per Layer
        plt.subplot(2, 4, 6)
        if not top_channels.empty:
            channel_layer_counts = top_channels['layer_name'].value_counts().head(8)
            plt.bar(range(len(channel_layer_counts)), channel_layer_counts.values)
            plt.xticks(range(len(channel_layer_counts)), 
                      [name.split('.')[-2] for name in channel_layer_counts.index], rotation=45)
            plt.ylabel('Number of Influenced Channels')
            plt.title('Influenced Channels per Layer')
        else:
            plt.text(0.5, 0.5, 'No Channel Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Channels per Layer')
        
        # 7. Weight Change Distribution
        plt.subplot(2, 4, 7)
        if not top_weights.empty:
            plt.hist(top_weights['relative_change'], bins=20, alpha=0.7)
            plt.xlabel('Relative Change')
            plt.ylabel('Frequency')
            plt.title('Weight Change Distribution')
        else:
            plt.text(0.5, 0.5, 'No Weight Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Weight Changes')
        
        # 8. Summary Statistics
        plt.subplot(2, 4, 8)
        plt.axis('off')
        
        summary_text = """REAL INFLUENCE ANALYSIS SUMMARY

📊 Components Analyzed:
• Layers: {}
• Channels: {}
• Weights: {}

🎯 Most Influenced:
• Layer: {}
• Change: {}

🏗️ Most Affected Type:
• {}

💡 Key Insight:
• {} depth most affected""".format(
            len(top_layers),
            len(top_channels), 
            len(top_weights),
            (top_layers.iloc[0]['layer_name'].split('.')[-1] if not top_layers.empty else 'N/A'),
            (f"{top_layers.iloc[0]['mean_relative_change']:.4f}" if not top_layers.empty else 'N/A'),
            (top_layers['layer_type'].value_counts().index[0] if not top_layers.empty else 'N/A'),
            (depth_influence.idxmax() if not depth_influence.empty else 'N/A')
        )
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / 'real_influence_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved visualization to: {viz_path}")
        
        plt.show()
    
    def save_results(self, top_layers: pd.DataFrame, top_channels: pd.DataFrame, 
                    top_weights: pd.DataFrame, lucent_targets: List[Dict]):
        """Save all analysis results"""
        
        print(f"💾 Saving results...")
        
        # Save comprehensive summary
        summary = {
            'analysis_type': 'real_weight_comparison',
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_info': {
                'architecture': 'ResNet50',
                'total_parameters': sum(p.numel() for p in self.baseline_model.parameters()),
                'analyzed_layers': len(top_layers)
            },
            'top_influences': {
                'most_influenced_layer': top_layers.iloc[0]['layer_name'] if not top_layers.empty else None,
                'max_layer_change': top_layers.iloc[0]['mean_relative_change'] if not top_layers.empty else 0,
                'most_influenced_channel': (top_channels.iloc[0]['channel_id'] if not top_channels.empty else None),
                'max_channel_change': (top_channels.iloc[0]['mean_relative_change'] if not top_channels.empty else 0),
                'most_influenced_weight': (top_weights.iloc[0]['weight_id'] if not top_weights.empty else None),
                'max_weight_change': (top_weights.iloc[0]['relative_change'] if not top_weights.empty else 0)
            },
            'statistics': {
                'layers_analyzed': len(self.layer_influences),
                'channels_analyzed': len(self.channel_influences),
                'weights_analyzed': len(self.weight_influences),
                'avg_layer_change': top_layers['mean_relative_change'].mean() if not top_layers.empty else 0,
                'avg_channel_change': top_channels['mean_relative_change'].mean() if not top_channels.empty else 0,
                'avg_weight_change': top_weights['relative_change'].mean() if not top_weights.empty else 0
            }
        }
        
        # Save files
        with open(self.output_dir / 'real_influence_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        with open(self.output_dir / 'lucent_targets_real.json', 'w') as f:
            json.dump(lucent_targets, f, indent=2, default=str)
        
        top_layers.to_csv(self.output_dir / 'top_influenced_layers_real.csv', index=False)
        if not top_channels.empty:
            top_channels.to_csv(self.output_dir / 'top_influenced_channels_real.csv', index=False)
        if not top_weights.empty:
            top_weights.to_csv(self.output_dir / 'top_influenced_weights_real.csv', index=False)
        
        # Create Lucent commands file
        self._create_lucent_commands(lucent_targets)
        
        print(f"✅ Results saved to: {self.output_dir}")
    
    def _create_lucent_commands(self, lucent_targets: List[Dict]):
        """Create ready-to-use Lucent commands"""
        
        with open(self.output_dir / 'lucent_commands_real.py', 'w') as f:
            f.write("# Real Influence Analysis - Lucent Visualization Commands\n")
            f.write("# Copy these into Google Colab\n\n")
            f.write("from lucent.optvis import render\n")
            f.write("import matplotlib.pyplot as plt\n\n")
            
            f.write("# Top individual targets\n")
            for i, target in enumerate(lucent_targets[:10], 1):
                f.write(f'# {i}. {target["description"]}\n')
                f.write(f'img_{i} = render.render_vis(model, "{target["target"]}", show_inline=True, thresholds=(512,))\n\n')
            
            f.write("\n# Batch visualization of top 6\n")
            f.write("targets = [\n")
            for target in lucent_targets[:6]:
                f.write(f'    "{target["target"]}",  # {target["description"]}\n')
            f.write("]\n\n")
            
            f.write("""fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, target in enumerate(targets):
    img = render.render_vis(model, target, show_inline=False, thresholds=(256,))
    if hasattr(img, 'cpu'):
        img_np = img.cpu().numpy().transpose(1, 2, 0)
    else:
        img_np = np.array(img)
    axes[i].imshow(img_np)
    axes[i].set_title(target, fontsize=10)
    axes[i].axis('off')

plt.suptitle('Real Analysis: Top 6 Most Influenced Components', fontsize=16)
plt.tight_layout()
plt.show()
""")
    
    def run_complete_analysis(self, baseline_path: str, unlearned_path: str, 
                            top_k_layers: int = 30, top_k_channels: int = 50, top_k_weights: int = 100):
        """Run the complete real influence analysis"""
        
        print("🚀 Starting Real Influence Analysis")
        print("=" * 60)
        
        # Load models
        self.load_models(baseline_path, unlearned_path)
        
        # Run analyses
        top_layers = self.analyze_layer_influences(top_k_layers)
        top_channels = self.analyze_channel_influences(top_layers, top_k_channels)
        top_weights = self.analyze_weight_influences(top_channels, top_k_weights)
        
        # Generate Lucent targets
        lucent_targets = self.generate_lucent_targets(top_layers, top_channels)
        
        # Create visualizations
        self.create_visualizations(top_layers, top_channels, top_weights)
        
        # Save results
        self.save_results(top_layers, top_channels, top_weights, lucent_targets)
        
        print("\n🎉 Real Influence Analysis Complete!")
        print(f"📊 Analyzed {len(top_layers)} layers, {len(top_channels)} channels, {len(top_weights)} weights")
        print(f"🎨 Generated {len(lucent_targets)} Lucent targets")
        print(f"📁 Results saved to: {self.output_dir}")
        
        return {
            'top_layers': top_layers,
            'top_channels': top_channels,
            'top_weights': top_weights,
            'lucent_targets': lucent_targets
        }
    
    # Helper methods
    def _get_layer_type(self, layer_name: str) -> str:
        """Get layer type from name"""
        if 'conv' in layer_name:
            return 'Convolution'
        elif 'bn' in layer_name:
            return 'BatchNorm'
        elif 'fc' in layer_name or 'classifier' in layer_name:
            return 'Fully Connected'
        elif 'downsample' in layer_name:
            return 'Downsample'
        else:
            return 'Other'
    
    def _get_layer_depth(self, layer_name: str) -> int:
        """Get layer depth from name"""
        if 'layer1' in layer_name:
            return 1
        elif 'layer2' in layer_name:
            return 2
        elif 'layer3' in layer_name:
            return 3
        elif 'layer4' in layer_name:
            return 4
        elif 'conv1' in layer_name or 'bn1' in layer_name:
            return 0
        elif 'fc' in layer_name:
            return 5
        else:
            return 6


def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description='Real Influence Analysis for Machine Unlearning')
    parser.add_argument('--baseline_model', required=True, 
                       help='Path to baseline model (.pth or .tar)')
    parser.add_argument('--unlearned_model', required=True,
                       help='Path to unlearned model (.pth or .tar)')
    parser.add_argument('--output_dir', default='analysis/results/real_influence_analysis',
                       help='Output directory for results')
    parser.add_argument('--num_classes', type=int, default=200,
                       help='Number of classes in the model')
    parser.add_argument('--top_k_layers', type=int, default=30,
                       help='Number of top layers to analyze')
    parser.add_argument('--top_k_channels', type=int, default=50,
                       help='Number of top channels to analyze')
    parser.add_argument('--top_k_weights', type=int, default=100,
                       help='Number of top weights to analyze')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RealInfluenceAnalyzer(output_dir=args.output_dir)
    
    # Run analysis
    results = analyzer.run_complete_analysis(
        baseline_path=args.baseline_model,
        unlearned_path=args.unlearned_model,
        top_k_layers=args.top_k_layers,
        top_k_channels=args.top_k_channels,
        top_k_weights=args.top_k_weights
    )
    
    print(f"\n✨ Real analysis complete! Check results in: {args.output_dir}")


if __name__ == "__main__":
    main()