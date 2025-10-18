#!/usr/bin/env python3
"""
Auto-generated Lucent visualization script for random_forgetting_10percent_RL_conservative
Targets the most affected channels/neurons identified by weight analysis
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
    model_path = "results/good_results/random_forgetting_10percent_RL_conservative/RLcheckpoint.pth.tar"
    output_dir = "experiments/channel_visualizations/random_forgetting_10percent_RL_conservative"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎨 Loading model: {model_path}")
    model = load_model(model_path)
    
    print(f"🎯 Visualizing top {len(targets)} most affected channels/neurons...")
    
    # Top affected targets from weight analysis
    targets = [{'target': 'layer3.0.conv3:301', 'layer': 'layer3.0.conv3.weight', 'channel': 301, 'change_magnitude': 1.0077000856399536, 'type': 'conv_channel'}, {'target': 'layer3.0.downsample.0:17', 'layer': 'layer3.0.downsample.0.weight', 'channel': 17, 'change_magnitude': 0.9000861644744873, 'type': 'conv_channel'}, {'target': 'layer2.1.conv3:63', 'layer': 'layer2.1.conv3.weight', 'channel': 63, 'change_magnitude': 0.8620054125785828, 'type': 'conv_channel'}, {'target': 'layer3.0.downsample.0:52', 'layer': 'layer3.0.downsample.0.weight', 'channel': 52, 'change_magnitude': 0.7820558547973633, 'type': 'conv_channel'}, {'target': 'fc:0', 'layer': 'fc.weight', 'neuron': 0, 'change_magnitude': 0.7796670794487, 'type': 'fc_neuron'}, {'target': 'layer3.3.conv3:411', 'layer': 'layer3.3.conv3.weight', 'channel': 411, 'change_magnitude': 0.7717757225036621, 'type': 'conv_channel'}, {'target': 'layer3.0.conv3:17', 'layer': 'layer3.0.conv3.weight', 'channel': 17, 'change_magnitude': 0.741048276424408, 'type': 'conv_channel'}, {'target': 'fc:2', 'layer': 'fc.weight', 'neuron': 2, 'change_magnitude': 0.7366511225700378, 'type': 'fc_neuron'}, {'target': 'fc:8', 'layer': 'fc.weight', 'neuron': 8, 'change_magnitude': 0.7234655618667603, 'type': 'fc_neuron'}, {'target': 'layer3.0.conv3:1010', 'layer': 'layer3.0.conv3.weight', 'channel': 1010, 'change_magnitude': 0.7218040227890015, 'type': 'conv_channel'}, {'target': 'fc:1', 'layer': 'fc.weight', 'neuron': 1, 'change_magnitude': 0.7122533917427063, 'type': 'fc_neuron'}, {'target': 'layer3.0.downsample.0:393', 'layer': 'layer3.0.downsample.0.weight', 'channel': 393, 'change_magnitude': 0.7046143412590027, 'type': 'conv_channel'}, {'target': 'layer1.2.conv3:239', 'layer': 'layer1.2.conv3.weight', 'channel': 239, 'change_magnitude': 0.7025042176246643, 'type': 'conv_channel'}, {'target': 'layer1.2.conv3:30', 'layer': 'layer1.2.conv3.weight', 'channel': 30, 'change_magnitude': 0.7001792788505554, 'type': 'conv_channel'}, {'target': 'layer3.1.conv3:324', 'layer': 'layer3.1.conv3.weight', 'channel': 324, 'change_magnitude': 0.6948227882385254, 'type': 'conv_channel'}, {'target': 'layer3.3.conv3:186', 'layer': 'layer3.3.conv3.weight', 'channel': 186, 'change_magnitude': 0.6907543540000916, 'type': 'conv_channel'}, {'target': 'layer3.4.conv3:339', 'layer': 'layer3.4.conv3.weight', 'channel': 339, 'change_magnitude': 0.6907244920730591, 'type': 'conv_channel'}, {'target': 'fc:5', 'layer': 'fc.weight', 'neuron': 5, 'change_magnitude': 0.6906498670578003, 'type': 'fc_neuron'}, {'target': 'layer3.4.conv3:248', 'layer': 'layer3.4.conv3.weight', 'channel': 248, 'change_magnitude': 0.6892299056053162, 'type': 'conv_channel'}, {'target': 'fc:6', 'layer': 'fc.weight', 'neuron': 6, 'change_magnitude': 0.6878623366355896, 'type': 'fc_neuron'}]
    
    for i, target_info in enumerate(targets):
        target = target_info['target']
        change_mag = target_info['change_magnitude']
        target_type = target_info['type']
        
        print(f"\n{i+1:2d}. {target} ({target_type}, change: {change_mag:.4f})")
        
        try:
            # Render visualization
            img = render.render_vis(model, target, show_inline=False, thresholds=(512,))
            
            # Save with descriptive name
            safe_target = target.replace(':', '_').replace('.', '_')
            filename = f"rank{i+1:02d}_{safe_target}_change{change_mag:.4f}.png"
            save_path = os.path.join(output_dir, filename)
            
            if hasattr(img, 'save'):
                img.save(save_path)
            
            print(f"✅ Saved: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to visualize {target}: {e}")
    
    print(f"\n🎉 Visualization complete! Check {output_dir}")

if __name__ == "__main__":
    visualize_top_affected_channels()
