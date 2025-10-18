#!/usr/bin/env python3
"""
Auto-generated Lucent visualization script for random_forgetting_20percent_RL_tweak_conservative
Targets the most affected channels/neurons identified by weight analysis
Run from repository root directory: python experiments/channel_analysis/random_forgetting_20percent_RL_tweak_conservative/visualize_random_forgetting_20percent_RL_tweak_conservative.py
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
    model_path = "experiments/results/good_results/random_forgetting_20percent_RL_tweak_conservative/RLcheckpoint.pth.tar"
    output_dir = "experiments/channel_visualizations/random_forgetting_20percent_RL_tweak_conservative"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎨 Loading model: {model_path}")
    model = load_model(model_path)
    
    print(f"🎯 Visualizing top {len(targets)} most affected channels/neurons...")
    
    # Top affected targets from weight analysis
    targets = [{'target': 'layer3.1.conv3:324', 'layer': 'layer3.1.conv3.weight', 'channel': 324, 'change_magnitude': 1.1719740629196167, 'type': 'conv_channel'}, {'target': 'layer1.2.conv3:184', 'layer': 'layer1.2.conv3.weight', 'channel': 184, 'change_magnitude': 1.113129734992981, 'type': 'conv_channel'}, {'target': 'layer3.4.conv3:138', 'layer': 'layer3.4.conv3.weight', 'channel': 138, 'change_magnitude': 1.0424468517303467, 'type': 'conv_channel'}, {'target': 'layer3.3.conv3:716', 'layer': 'layer3.3.conv3.weight', 'channel': 716, 'change_magnitude': 1.0162662267684937, 'type': 'conv_channel'}, {'target': 'layer3.0.conv3:17', 'layer': 'layer3.0.conv3.weight', 'channel': 17, 'change_magnitude': 0.9972174763679504, 'type': 'conv_channel'}, {'target': 'layer3.5.conv3:359', 'layer': 'layer3.5.conv3.weight', 'channel': 359, 'change_magnitude': 0.9936762452125549, 'type': 'conv_channel'}, {'target': 'fc:0', 'layer': 'fc.weight', 'neuron': 0, 'change_magnitude': 0.9581729769706726, 'type': 'fc_neuron'}, {'target': 'layer1.2.conv3:38', 'layer': 'layer1.2.conv3.weight', 'channel': 38, 'change_magnitude': 0.9433958530426025, 'type': 'conv_channel'}, {'target': 'layer3.4.conv3:467', 'layer': 'layer3.4.conv3.weight', 'channel': 467, 'change_magnitude': 0.9379315972328186, 'type': 'conv_channel'}, {'target': 'layer3.0.conv3:947', 'layer': 'layer3.0.conv3.weight', 'channel': 947, 'change_magnitude': 0.9255580902099609, 'type': 'conv_channel'}, {'target': 'layer4.0.conv3:941', 'layer': 'layer4.0.conv3.weight', 'channel': 941, 'change_magnitude': 0.9187935590744019, 'type': 'conv_channel'}, {'target': 'fc:2', 'layer': 'fc.weight', 'neuron': 2, 'change_magnitude': 0.9068165421485901, 'type': 'fc_neuron'}, {'target': 'layer3.4.conv3:339', 'layer': 'layer3.4.conv3.weight', 'channel': 339, 'change_magnitude': 0.9035474061965942, 'type': 'conv_channel'}, {'target': 'layer3.4.conv3:595', 'layer': 'layer3.4.conv3.weight', 'channel': 595, 'change_magnitude': 0.8982837796211243, 'type': 'conv_channel'}, {'target': 'fc:8', 'layer': 'fc.weight', 'neuron': 8, 'change_magnitude': 0.8937801122665405, 'type': 'fc_neuron'}, {'target': 'layer3.3.conv3:761', 'layer': 'layer3.3.conv3.weight', 'channel': 761, 'change_magnitude': 0.8872107863426208, 'type': 'conv_channel'}, {'target': 'layer3.4.conv3:1017', 'layer': 'layer3.4.conv3.weight', 'channel': 1017, 'change_magnitude': 0.8749878406524658, 'type': 'conv_channel'}, {'target': 'layer3.3.conv3:626', 'layer': 'layer3.3.conv3.weight', 'channel': 626, 'change_magnitude': 0.869099497795105, 'type': 'conv_channel'}, {'target': 'layer4.0.conv3:1076', 'layer': 'layer4.0.conv3.weight', 'channel': 1076, 'change_magnitude': 0.8661670684814453, 'type': 'conv_channel'}, {'target': 'fc:5', 'layer': 'fc.weight', 'neuron': 5, 'change_magnitude': 0.8657568097114563, 'type': 'fc_neuron'}]
    
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
