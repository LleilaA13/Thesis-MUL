#!/usr/bin/env python3
"""
Lucent Feature Visualization for Random Data Forgetting Analysis
Visualizes how random data forgetting affects learned features using Lucent
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
from PIL import Image
import torchvision.transforms as transforms

# Add Classification directory to path
sys.path.append('./Classification')
sys.path.append('./Classification/models')

try:
    from lucent.optvis import render, param, transform, objectives
    from lucent.misc.io import show
    LUCENT_AVAILABLE = True
except ImportError:
    print("⚠️  Lucent not available. Install with: pip install lucent")
    LUCENT_AVAILABLE = False

class FeatureVisualizationAnalyzer:
    """Analyzes feature changes caused by random data forgetting using Lucent"""
    
    def __init__(self, baseline_model_path, experiment_models_dir):
        self.baseline_model_path = baseline_model_path
        self.experiment_models_dir = experiment_models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self, model_path, arch='resnet18'):
        """Load model from checkpoint"""
        # Import model architecture
        if arch == 'resnet18':
            from models.ResNets import resnet18
            model = resnet18(num_classes=10)  # CIFAR-10
        else:
            raise ValueError(f"Architecture {arch} not implemented")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        return model
    
    def get_layer_names(self, model):
        """Get names of convolutional layers for visualization"""
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)
        return conv_layers
    
    def visualize_feature_maps(self, model, layer_name, channels=None, num_iterations=512, image_size=224):
        """Visualize feature maps for specific layer and channels"""
        if not LUCENT_AVAILABLE:
            print("❌ Lucent not available for feature visualization")
            return None
        
        if channels is None:
            # Get number of output channels for this layer
            layer = dict(model.named_modules())[layer_name]
            if hasattr(layer, 'out_channels'):
                num_channels = min(16, layer.out_channels)
            else:
                num_channels = 16
            channels = range(num_channels)
        
        visualizations = {}
        
        for channel in channels:
            try:
                # Render feature visualization
                result = render.render_vis(
                    model,
                    objectives.channel(layer_name, channel),
                    param_f=lambda: param.image(image_size, fft=True, decorrelate=True),
                    optimizer=torch.optim.Adam,
                    transforms=transform.standard_transforms,
                    steps=num_iterations,
                    show_inline=False
                )
                
                visualizations[channel] = result[0]
                
            except Exception as e:
                print(f"⚠️  Failed to visualize {layer_name} channel {channel}: {e}")
                continue
        
        return visualizations
    
    def compare_layer_features_before_after(self, baseline_model, unlearned_model, layer_name, 
                                          channels=None, save_dir=None):
        """Compare features before and after unlearning"""
        if not LUCENT_AVAILABLE:
            print("❌ Lucent not available for feature comparison")
            return
        
        print(f"🎨 Comparing features for layer: {layer_name}")
        
        # Visualize baseline features
        baseline_features = self.visualize_feature_maps(baseline_model, layer_name, channels)
        
        # Visualize unlearned features
        unlearned_features = self.visualize_feature_maps(unlearned_model, layer_name, channels)
        
        if baseline_features is None or unlearned_features is None:
            return
        
        # Create comparison visualization
        if save_dir:
            self.create_comparison_plot(baseline_features, unlearned_features, layer_name, save_dir)
        
        return baseline_features, unlearned_features
    
    def create_comparison_plot(self, baseline_features, unlearned_features, layer_name, save_dir):
        """Create side-by-side comparison plot"""
        common_channels = set(baseline_features.keys()) & set(unlearned_features.keys())
        
        if not common_channels:
            print("⚠️  No common channels to compare")
            return
        
        # Limit to first 8 channels for visualization
        channels_to_plot = sorted(list(common_channels))[:8]
        
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, len(channels_to_plot), figure=fig)
        
        for i, channel in enumerate(channels_to_plot):
            # Baseline features
            ax1 = fig.add_subplot(gs[0, i])
            ax1.imshow(np.array(baseline_features[channel]))
            ax1.set_title(f'Baseline\nChannel {channel}')
            ax1.axis('off')
            
            # Unlearned features
            ax2 = fig.add_subplot(gs[1, i])
            ax2.imshow(np.array(unlearned_features[channel]))
            ax2.set_title(f'After Unlearning\nChannel {channel}')
            ax2.axis('off')
        
        plt.suptitle(f'Feature Comparison: {layer_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the comparison
        os.makedirs(save_dir, exist_ok=True)
        safe_layer_name = layer_name.replace('.', '_')
        plt.savefig(os.path.join(save_dir, f'feature_comparison_{safe_layer_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved comparison for {layer_name}")
    
    def analyze_input_attribution(self, model, input_images, target_layer, save_dir=None):
        """Analyze how specific input images affect different layers"""
        if not LUCENT_AVAILABLE:
            print("❌ Lucent not available for attribution analysis")
            return
        
        attributions = {}
        
        for i, img in enumerate(input_images):
            try:
                # Convert PIL image to tensor if needed
                if isinstance(img, Image.Image):
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
                    img_tensor = transform(img).unsqueeze(0).to(self.device)
                else:
                    img_tensor = img
                
                # Perform attribution analysis using integrated gradients or similar
                # This is a simplified version - you might want to use more sophisticated methods
                img_tensor.requires_grad_(True)
                
                # Forward pass
                output = model(img_tensor)
                
                # Get activations from target layer
                activation = {}
                def get_activation(name):
                    def hook(model, input, output):
                        activation[name] = output.detach()
                    return hook
                
                # Register hook
                layer = dict(model.named_modules())[target_layer]
                handle = layer.register_forward_hook(get_activation(target_layer))
                
                # Forward pass to get activations
                _ = model(img_tensor)
                
                # Remove hook
                handle.remove()
                
                attributions[f'image_{i}'] = {
                    'activation_map': activation[target_layer].cpu().numpy(),
                    'original_image': img_tensor.cpu().numpy()
                }
                
            except Exception as e:
                print(f"⚠️  Failed attribution analysis for image {i}: {e}")
                continue
        
        if save_dir:
            self.save_attribution_analysis(attributions, target_layer, save_dir)
        
        return attributions
    
    def save_attribution_analysis(self, attributions, layer_name, save_dir):
        """Save attribution analysis results"""
        os.makedirs(save_dir, exist_ok=True)
        
        for img_name, data in attributions.items():
            # Save activation maps as heatmaps
            activation = data['activation_map'][0]  # Remove batch dimension
            
            # Average across channels for visualization
            avg_activation = np.mean(activation, axis=0)
            
            plt.figure(figsize=(10, 5))
            
            # Original image
            plt.subplot(1, 2, 1)
            orig_img = data['original_image'][0].transpose(1, 2, 0)
            # Denormalize
            orig_img = orig_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            orig_img = np.clip(orig_img, 0, 1)
            plt.imshow(orig_img)
            plt.title(f'Original Image - {img_name}')
            plt.axis('off')
            
            # Activation heatmap
            plt.subplot(1, 2, 2)
            plt.imshow(avg_activation, cmap='hot', interpolation='nearest')
            plt.title(f'Activation Map - {layer_name}')
            plt.colorbar()
            plt.axis('off')
            
            safe_layer_name = layer_name.replace('.', '_')
            plt.savefig(os.path.join(save_dir, f'attribution_{safe_layer_name}_{img_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_comprehensive_visualization_analysis(self, arch='resnet18'):
        """Run comprehensive feature visualization analysis"""
        print("🎨 Starting comprehensive feature visualization analysis...")
        
        # Load baseline model
        baseline_model = self.load_model(self.baseline_model_path, arch)
        print("✅ Loaded baseline model")
        
        # Get layer names
        conv_layers = self.get_layer_names(baseline_model)
        print(f"📋 Found {len(conv_layers)} convolutional layers: {conv_layers}")
        
        # Focus on key layers (first, middle, last)
        if len(conv_layers) >= 3:
            key_layers = [conv_layers[0], conv_layers[len(conv_layers)//2], conv_layers[-1]]
        else:
            key_layers = conv_layers
        
        # Get experiment directories
        exp_dirs = [d for d in os.listdir(self.experiment_models_dir) if d != 'baseline']
        
        for exp_name in exp_dirs:
            print(f"🔍 Analyzing experiment: {exp_name}")
            
            model_path = os.path.join(self.experiment_models_dir, exp_name, 'unlearn', 'model_best.pth.tar')
            if not os.path.exists(model_path):
                print(f"⚠️  Model not found: {model_path}")
                continue
            
            # Load unlearned model
            unlearned_model = self.load_model(model_path, arch)
            
            # Create save directory
            save_dir = f'experiments/random_forgetting/visualizations/features/{exp_name}'
            
            # Compare features for key layers
            for layer_name in key_layers:
                try:
                    self.compare_layer_features_before_after(
                        baseline_model, unlearned_model, layer_name, 
                        channels=range(8), save_dir=save_dir
                    )
                except Exception as e:
                    print(f"⚠️  Failed to compare {layer_name}: {e}")
                    continue
        
        print("✅ Comprehensive visualization analysis complete!")

def main():
    if not LUCENT_AVAILABLE:
        print("❌ Lucent is required for feature visualization.")
        print("Install with: pip install lucent")
        return
    
    # Configuration
    baseline_model = "experiments/random_forgetting/models/baseline/model_best.pth.tar"
    experiments_dir = "experiments/random_forgetting/models"
    
    if not os.path.exists(baseline_model):
        print("❌ Baseline model not found. Please run the research pipeline first.")
        return
    
    analyzer = FeatureVisualizationAnalyzer(baseline_model, experiments_dir)
    analyzer.run_comprehensive_visualization_analysis()
    
    print("🎉 Feature visualization analysis complete!")
    print("📂 Check experiments/random_forgetting/visualizations/features/ for results")

if __name__ == "__main__":
    main()