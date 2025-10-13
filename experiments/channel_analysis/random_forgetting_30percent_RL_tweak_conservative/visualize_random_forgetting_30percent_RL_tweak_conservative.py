#!/usr/bin/env python3
"""
Auto-generated Lucent visualization script for random_forgetting_30percent_RL_tweak_conservative
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
    model_path = "results/good_results/random_forgetting_30percent_RL_tweak_conservative/RLcheckpoint.pth.tar"
    output_dir = "experiments/channel_visualizations/random_forgetting_30percent_RL_tweak_conservative"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎨 Loading model: {model_path}")
    model = load_model(model_path)
    
    print(f"🎯 Visualizing top {len(targets)} most affected channels/neurons...")
    
    # Top affected targets from weight analysis
    targets = [{'target': 'layer3.0.conv3:301', 'layer': 'layer3.0.conv3.weight', 'channel': 301, 'change_magnitude': 1.3145579099655151, 'type': 'conv_channel'}, {'target': 'layer3.5.conv3:359', 'layer': 'layer3.5.conv3.weight', 'channel': 359, 'change_magnitude': 1.2473665475845337, 'type': 'conv_channel'}, {'target': 'layer3.0.downsample.0:17', 'layer': 'layer3.0.downsample.0.weight', 'channel': 17, 'change_magnitude': 1.1311867237091064, 'type': 'conv_channel'}, {'target': 'layer3.4.conv3:784', 'layer': 'layer3.4.conv3.weight', 'channel': 784, 'change_magnitude': 1.0910719633102417, 'type': 'conv_channel'}, {'target': 'layer3.4.conv3:889', 'layer': 'layer3.4.conv3.weight', 'channel': 889, 'change_magnitude': 1.0373238325119019, 'type': 'conv_channel'}, {'target': 'layer3.0.conv3:1010', 'layer': 'layer3.0.conv3.weight', 'channel': 1010, 'change_magnitude': 1.0278098583221436, 'type': 'conv_channel'}, {'target': 'layer3.4.conv3:361', 'layer': 'layer3.4.conv3.weight', 'channel': 361, 'change_magnitude': 1.0243712663650513, 'type': 'conv_channel'}, {'target': 'layer3.1.conv3:324', 'layer': 'layer3.1.conv3.weight', 'channel': 324, 'change_magnitude': 1.0195856094360352, 'type': 'conv_channel'}, {'target': 'fc:0', 'layer': 'fc.weight', 'neuron': 0, 'change_magnitude': 0.9954951405525208, 'type': 'fc_neuron'}, {'target': 'layer3.5.conv3:547', 'layer': 'layer3.5.conv3.weight', 'channel': 547, 'change_magnitude': 0.9887124300003052, 'type': 'conv_channel'}, {'target': 'layer3.0.conv3:17', 'layer': 'layer3.0.conv3.weight', 'channel': 17, 'change_magnitude': 0.9847847819328308, 'type': 'conv_channel'}, {'target': 'fc:8', 'layer': 'fc.weight', 'neuron': 8, 'change_magnitude': 0.9771416187286377, 'type': 'fc_neuron'}, {'target': 'layer1.2.conv3:184', 'layer': 'layer1.2.conv3.weight', 'channel': 184, 'change_magnitude': 0.9757413864135742, 'type': 'conv_channel'}, {'target': 'fc:6', 'layer': 'fc.weight', 'neuron': 6, 'change_magnitude': 0.9585286974906921, 'type': 'fc_neuron'}, {'target': 'fc:2', 'layer': 'fc.weight', 'neuron': 2, 'change_magnitude': 0.9559330344200134, 'type': 'fc_neuron'}, {'target': 'layer4.0.conv3:941', 'layer': 'layer4.0.conv3.weight', 'channel': 941, 'change_magnitude': 0.9443097114562988, 'type': 'conv_channel'}, {'target': 'fc:1', 'layer': 'fc.weight', 'neuron': 1, 'change_magnitude': 0.936276376247406, 'type': 'fc_neuron'}, {'target': 'fc:5', 'layer': 'fc.weight', 'neuron': 5, 'change_magnitude': 0.9312119483947754, 'type': 'fc_neuron'}, {'target': 'fc:9', 'layer': 'fc.weight', 'neuron': 9, 'change_magnitude': 0.9250946044921875, 'type': 'fc_neuron'}, {'target': 'fc:4', 'layer': 'fc.weight', 'neuron': 4, 'change_magnitude': 0.9247284531593323, 'type': 'fc_neuron'}]
    
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
