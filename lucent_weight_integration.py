#!/usr/bin/env python3
"""
Lucent-Weight Analysis Integration
Combines weight influence analysis with Lucent feature visualization for comprehensive insights
"""

import torch
import json
import os
import numpy as np
from collections import defaultdict
from torchvision import models

try:
    from lucent.optvis import render, objectives, param, transform
    from lucent.modelzoo.util import get_model_layers
    LUCENT_AVAILABLE = True
except ImportError:
    LUCENT_AVAILABLE = False
    print("⚠️  Lucent not available. Install with: pip install lucent")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib/seaborn not available. Plotting will be disabled.")

class LucentWeightAnalyzer:
    """Integrates weight analysis with Lucent feature visualization"""
    
    def __init__(self, weight_analysis_path, model_paths):
        self.weight_analysis_path = weight_analysis_path
        self.model_paths = model_paths
        self.weight_data = self.load_weight_analysis()
        self.models = {}
        
    def load_weight_analysis(self):
        """Load the comprehensive weight analysis"""
        with open(self.weight_analysis_path, 'r') as f:
            return json.load(f)
    
    def load_models(self, device='auto'):
        """Load all models for visualization"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🚀 Loading models on device: {device}")
        
        for exp_name, model_path in self.model_paths.items():
            print(f"Loading {exp_name}...")
            
            # Create ResNet50 for Tiny ImageNet (200 classes)
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 200)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            model = model.to(device)
            model.eval()
            self.models[exp_name] = model
            print(f"✅ {exp_name} loaded successfully")
    
    def get_most_affected_layers(self, exp_name, top_k=5):
        """Get the layers most affected by forgetting for an experiment"""
        if exp_name not in self.weight_data:
            return []
        
        layer_data = self.weight_data[exp_name]['layer_sensitivity']
        
        # Sort by mean relative change
        sorted_layers = sorted(layer_data.items(), 
                             key=lambda x: x[1]['mean_relative_change'], 
                             reverse=True)
        
        return [(name, data) for name, data in sorted_layers[:top_k]]
    
    def get_least_affected_layers(self, exp_name, top_k=5):
        """Get the layers least affected by forgetting"""
        if exp_name not in self.weight_data:
            return []
        
        layer_data = self.weight_data[exp_name]['layer_sensitivity']
        
        # Sort by mean relative change (ascending)
        sorted_layers = sorted(layer_data.items(), 
                             key=lambda x: x[1]['mean_relative_change'])
        
        return [(name, data) for name, data in sorted_layers[:top_k]]
    
    def visualize_targeted_features(self, exp_name, class_indices=[0, 66, 102, 131], 
                                  output_dir='experiments/lucent_weight_analysis'):
        """Visualize features focusing on most/least affected layers"""
        if not LUCENT_AVAILABLE:
            print("❌ Lucent not available for visualization")
            return
        
        if exp_name not in self.models:
            print(f"❌ Model {exp_name} not loaded")
            return
        
        model = self.models[exp_name]
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🎨 TARGETED FEATURE VISUALIZATION: {exp_name}")
        print("="*60)
        
        # Get most and least affected layers
        most_affected = self.get_most_affected_layers(exp_name, 3)
        least_affected = self.get_least_affected_layers(exp_name, 3)
        
        print(f"🔥 Most affected layers:")
        for name, data in most_affected:
            print(f"   {name}: {data['mean_relative_change']:.6f}")
        
        print(f"🛡️  Least affected layers:")
        for name, data in least_affected:
            print(f"   {name}: {data['mean_relative_change']:.6f}")
        
        # Class names for Tiny ImageNet
        class_names = {0: "Egyptian_Cat", 66: "Tabby_Cat", 102: "Cougar", 131: "Persian_Cat"}
        
        # Visualize each class
        for class_idx in class_indices:
            class_name = class_names.get(class_idx, f"Class_{class_idx}")
            print(f"\n🎯 Visualizing {class_name} (index {class_idx})")
            
            try:
                # Standard class visualization
                img = render.render_vis(model, f"labels:{class_idx}", 
                                      show_inline=False, thresholds=(512,))
                
                # Save the visualization
                save_path = os.path.join(output_dir, f"{exp_name}_{class_name}_class_{class_idx}.png")
                if hasattr(img, 'save'):
                    img.save(save_path)
                print(f"✅ Saved: {save_path}")
                
            except Exception as e:
                print(f"❌ Failed to visualize {class_name}: {e}")
    
    def compare_forgetting_effects(self, class_indices=[0, 66, 102, 131],
                                 output_dir='experiments/lucent_weight_comparison'):
        """Compare visualizations across different forgetting ratios"""
        if not LUCENT_AVAILABLE:
            print("❌ Lucent not available for visualization")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🔍 FORGETTING COMPARISON ANALYSIS")
        print("="*60)
        
        class_names = {0: "Egyptian_Cat", 66: "Tabby_Cat", 102: "Cougar", 131: "Persian_Cat"}
        
        # Create comparison report
        comparison_report = []
        comparison_report.append("# Lucent-Weight Analysis Comparison Report")
        comparison_report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        comparison_report.append("")
        
        for class_idx in class_indices:
            class_name = class_names.get(class_idx, f"Class_{class_idx}")
            print(f"\n🎯 Comparing {class_name} across all experiments...")
            
            comparison_report.append(f"## {class_name} (Index {class_idx})")
            comparison_report.append("")
            
            for exp_name, model in self.models.items():
                print(f"   Visualizing {exp_name}...")
                
                try:
                    # Get weight analysis for this experiment
                    if exp_name in self.weight_data:
                        layer_data = self.weight_data[exp_name]['layer_sensitivity']
                        most_affected = max(layer_data.keys(), 
                                          key=lambda x: layer_data[x]['mean_relative_change'])
                        change_magnitude = layer_data[most_affected]['mean_relative_change']
                        
                        comparison_report.append(f"### {exp_name}")
                        comparison_report.append(f"- **Most affected layer**: {most_affected}")
                        comparison_report.append(f"- **Change magnitude**: {change_magnitude:.6f}")
                        
                    # Generate visualization
                    img = render.render_vis(model, f"labels:{class_idx}", 
                                          show_inline=False, thresholds=(512,))
                    
                    # Save visualization
                    save_path = os.path.join(output_dir, f"{exp_name}_{class_name}.png")
                    if hasattr(img, 'save'):
                        img.save(save_path)
                    
                    comparison_report.append(f"- **Visualization**: {save_path}")
                    comparison_report.append("")
                    
                except Exception as e:
                    print(f"❌ Failed {exp_name} for {class_name}: {e}")
                    comparison_report.append(f"- **Error**: {e}")
                    comparison_report.append("")
        
        # Save comparison report
        report_path = os.path.join(output_dir, 'lucent_weight_comparison_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(comparison_report))
        
        print(f"✅ Comparison report saved: {report_path}")
    
    def analyze_layer_feature_correlation(self, exp_name, layer_targets=None):
        """Analyze how weight changes correlate with feature visualization quality"""
        if not LUCENT_AVAILABLE:
            print("❌ Lucent not available for analysis")
            return
        
        if exp_name not in self.models:
            print(f"❌ Model {exp_name} not loaded")
            return
        
        model = self.models[exp_name]
        
        if layer_targets is None:
            # Use most/least affected layers as targets
            most_affected = self.get_most_affected_layers(exp_name, 2)
            least_affected = self.get_least_affected_layers(exp_name, 2)
            layer_targets = [name for name, _ in most_affected + least_affected]
        
        print(f"\n🔬 LAYER-FEATURE CORRELATION ANALYSIS: {exp_name}")
        print("="*60)
        
        analysis_results = {}
        
        for layer_name in layer_targets:
            print(f"\n🎯 Analyzing layer: {layer_name}")
            
            # Get weight change info
            if exp_name in self.weight_data:
                layer_data = self.weight_data[exp_name]['layer_sensitivity']
                if layer_name in layer_data:
                    weight_change = layer_data[layer_name]['mean_relative_change']
                    print(f"   Weight change: {weight_change:.6f}")
                    
                    # Try to visualize this layer
                    try:
                        # Get layer structure for visualization targets
                        layer_info = self.get_layer_visualization_targets(model, layer_name)
                        analysis_results[layer_name] = {
                            'weight_change': weight_change,
                            'visualization_targets': layer_info
                        }
                        
                    except Exception as e:
                        print(f"   ❌ Visualization failed: {e}")
                        analysis_results[layer_name] = {
                            'weight_change': weight_change,
                            'visualization_error': str(e)
                        }
        
        return analysis_results
    
    def get_layer_visualization_targets(self, model, layer_name):
        """Get appropriate visualization targets for a specific layer"""
        # Get all layers in the model
        all_layers = get_model_layers(model)
        
        # Find the layer and get its info
        for name, layer in all_layers:
            if layer_name in name:
                return {
                    'layer_type': type(layer).__name__,
                    'layer_path': name,
                    'has_neurons': hasattr(layer, 'out_features') or hasattr(layer, 'num_features')
                }
        
        return {'error': 'Layer not found in model structure'}
    
    def generate_comprehensive_report(self, output_dir='experiments/comprehensive_lucent_weight_analysis'):
        """Generate comprehensive analysis combining weight changes and feature visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n🎯 COMPREHENSIVE LUCENT-WEIGHT ANALYSIS")
        print("="*70)
        
        # Run all analyses
        self.compare_forgetting_effects(output_dir=os.path.join(output_dir, 'comparisons'))
        
        for exp_name in self.models.keys():
            exp_dir = os.path.join(output_dir, exp_name)
            self.visualize_targeted_features(exp_name, output_dir=exp_dir)
            self.analyze_layer_feature_correlation(exp_name)
        
        print(f"\n✅ Comprehensive analysis complete!")
        print(f"📂 Results saved to: {output_dir}")

def main():
    """Main function to run Lucent-Weight analysis on good results"""
    print("🎨 LUCENT-WEIGHT ANALYSIS INTEGRATION")
    print("="*50)
    
    # Configuration
    weight_analysis_path = "experiments/good_results_weight_analysis/comprehensive_weight_analysis.json"
    
    model_paths = {
        'baseline': 'models/resnet50_pretrained.pth',
        'random_forgetting_10percent_RL_conservative': 'results/good_results/random_forgetting_10percent_RL_conservative/RLcheckpoint.pth.tar',
        'random_forgetting_20percent_RL_tweak_conservative': 'results/good_results/random_forgetting_20percent_RL_tweak_conservative/RLcheckpoint.pth.tar',
        'random_forgetting_30percent_RL_tweak_conservative': 'results/good_results/random_forgetting_30percent_RL_tweak_conservative/RLcheckpoint.pth.tar'
    }
    
    # Verify files exist
    if not os.path.exists(weight_analysis_path):
        print(f"❌ Weight analysis not found: {weight_analysis_path}")
        print("   Run 'python analyze_good_results.py' first")
        return
    
    missing_models = [name for name, path in model_paths.items() if not os.path.exists(path)]
    if missing_models:
        print(f"❌ Missing models: {missing_models}")
        return
    
    # Initialize analyzer
    analyzer = LucentWeightAnalyzer(weight_analysis_path, model_paths)
    
    # Load models
    analyzer.load_models()
    
    # Run comprehensive analysis
    analyzer.generate_comprehensive_report()

if __name__ == "__main__":
    from datetime import datetime
    main()