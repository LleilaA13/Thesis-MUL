#!/usr/bin/env python3
"""
Auto-generated Lucent visualization script for random_forgetting_10percent_RL_tweak_conservative
Targets the most affected channels/neurons identified by weight analysis
Run from repository root directory: python experiments/channel_analysis/random_forgetting_10percent_RL_tweak_conservative/visualize_random_forgetting_10percent_RL_tweak_conservative.py
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
    model_path = "experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/RLcheckpoint.pth.tar"
    output_dir = "experiments/channel_visualizations/random_forgetting_10percent_RL_tweak_conservative"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎨 Loading model: {model_path}")
    model = load_model(model_path)
    
    print(f"🎯 Visualizing top {len(targets)} most affected channels/neurons...")
    
    # Top affected targets from weight analysis
    targets = [{'target': 'layer3.0.conv3:301', 'layer': 'layer3.0.conv3.weight', 'channel': 301, 'change_magnitude': 1.0583423376083374, 'type': 'conv_channel'}, {'target': 'layer3.0.downsample.0:17', 'layer': 'layer3.0.downsample.0.weight', 'channel': 17, 'change_magnitude': 0.9091891050338745, 'type': 'conv_channel'}, {'target': 'layer3.5.conv3:359', 'layer': 'layer3.5.conv3.weight', 'channel': 359, 'change_magnitude': 0.9052949547767639, 'type': 'conv_channel'}, {'target': 'layer1.2.conv3:239', 'layer': 'layer1.2.conv3.weight', 'channel': 239, 'change_magnitude': 0.8739693760871887, 'type': 'conv_channel'}, {'target': 'layer2.1.conv3:63', 'layer': 'layer2.1.conv3.weight', 'channel': 63, 'change_magnitude': 0.8629945516586304, 'type': 'conv_channel'}, {'target': 'layer3.0.conv3:17', 'layer': 'layer3.0.conv3.weight', 'channel': 17, 'change_magnitude': 0.8246787786483765, 'type': 'conv_channel'}, {'target': 'layer3.0.downsample.0:393', 'layer': 'layer3.0.downsample.0.weight', 'channel': 393, 'change_magnitude': 0.8167803883552551, 'type': 'conv_channel'}, {'target': 'layer3.0.downsample.0:52', 'layer': 'layer3.0.downsample.0.weight', 'channel': 52, 'change_magnitude': 0.8140313029289246, 'type': 'conv_channel'}, {'target': 'layer3.3.conv3:411', 'layer': 'layer3.3.conv3.weight', 'channel': 411, 'change_magnitude': 0.8079807162284851, 'type': 'conv_channel'}, {'target': 'layer4.0.conv3:1076', 'layer': 'layer4.0.conv3.weight', 'channel': 1076, 'change_magnitude': 0.8039549589157104, 'type': 'conv_channel'}, {'target': 'layer3.3.conv3:591', 'layer': 'layer3.3.conv3.weight', 'channel': 591, 'change_magnitude': 0.8018151521682739, 'type': 'conv_channel'}, {'target': 'fc:0', 'layer': 'fc.weight', 'neuron': 0, 'change_magnitude': 0.8005838394165039, 'type': 'fc_neuron'}, {'target': 'layer4.0.downsample.0:1076', 'layer': 'layer4.0.downsample.0.weight', 'channel': 1076, 'change_magnitude': 0.7816384434700012, 'type': 'conv_channel'}, {'target': 'layer4.0.conv3:818', 'layer': 'layer4.0.conv3.weight', 'channel': 818, 'change_magnitude': 0.7788787484169006, 'type': 'conv_channel'}, {'target': 'layer1.2.conv3:184', 'layer': 'layer1.2.conv3.weight', 'channel': 184, 'change_magnitude': 0.7716384530067444, 'type': 'conv_channel'}, {'target': 'layer3.3.conv3:442', 'layer': 'layer3.3.conv3.weight', 'channel': 442, 'change_magnitude': 0.7676535844802856, 'type': 'conv_channel'}, {'target': 'layer3.4.conv3:881', 'layer': 'layer3.4.conv3.weight', 'channel': 881, 'change_magnitude': 0.7668554782867432, 'type': 'conv_channel'}, {'target': 'layer3.0.conv3:1010', 'layer': 'layer3.0.conv3.weight', 'channel': 1010, 'change_magnitude': 0.7643892765045166, 'type': 'conv_channel'}, {'target': 'fc:8', 'layer': 'fc.weight', 'neuron': 8, 'change_magnitude': 0.7600323557853699, 'type': 'fc_neuron'}, {'target': 'layer3.4.conv3:248', 'layer': 'layer3.4.conv3.weight', 'channel': 248, 'change_magnitude': 0.7595192193984985, 'type': 'conv_channel'}]
    
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
