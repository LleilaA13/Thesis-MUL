#!/usr/bin/env python3
"""
Demo: Channel-Specific Lucent Visualization
Tests the most affected channels identified by weight analysis
"""

import json
import os
import sys

# Add path for lucent if needed
sys.path.append('/path/to/lucent')  # Update if needed

def demo_channel_visualization():
    """Demo the top affected channels from 10% forgetting experiment"""
    print("🎨 DEMO: Channel-Specific Lucent Visualization")
    print("="*60)
    
    # Load the top targets from 10% forgetting (most stable)
    targets_file = "experiments/channel_analysis/random_forgetting_10percent_RL_conservative/lucent_targets.json"
    
    if not os.path.exists(targets_file):
        print(f"❌ Targets file not found: {targets_file}")
        print("   Run 'python channel_level_analyzer.py' first")
        return
    
    with open(targets_file, 'r') as f:
        targets = json.load(f)
    
    print(f"📊 Found {len(targets)} channel targets")
    print("\n🔝 TOP 10 MOST AFFECTED CHANNELS:")
    print("-" * 60)
    
    for i, target in enumerate(targets[:10]):
        channel_target = target['target']
        layer = target['layer'].replace('.weight', '')
        change = target['change_magnitude']
        target_type = target['type']
        
        # Get channel or neuron index depending on type
        if 'channel' in target:
            index = target['channel']
            index_type = "Channel"
        elif 'neuron' in target:
            index = target['neuron']
            index_type = "Neuron"
        else:
            index = "N/A"
            index_type = "Index"
        
        print(f"{i+1:2d}. {channel_target:<25} | Change: {change:.4f}")
        print(f"    Layer: {layer}, {index_type}: {index} ({target_type})")
        
        if i < 3:  # Show Lucent command for top 3
            print(f"    💡 Lucent: render.render_vis(model, '{channel_target}')")
        print()
    
    print("\n🎯 RECOMMENDED LUCENT ANALYSIS:")
    print("="*60)
    
    # Group by layer depth
    early_layers = [t for t in targets if any(x in t['layer'] for x in ['layer1', 'layer2'])]
    mid_layers = [t for t in targets if 'layer3' in t['layer']]
    late_layers = [t for t in targets if any(x in t['layer'] for x in ['layer4', 'fc'])]
    
    print(f"🔹 Early layers (layer1-2): {len(early_layers)} affected channels")
    if early_layers:
        top_early = early_layers[0]
        print(f"   Most affected: {top_early['target']} (change: {top_early['change_magnitude']:.4f})")
    
    print(f"🔹 Mid layers (layer3): {len(mid_layers)} affected channels") 
    if mid_layers:
        top_mid = mid_layers[0]
        print(f"   Most affected: {top_mid['target']} (change: {top_mid['change_magnitude']:.4f})")
    
    print(f"🔹 Late layers (layer4+): {len(late_layers)} affected channels")
    if late_layers:
        top_late = late_layers[0]
        print(f"   Most affected: {top_late['target']} (change: {top_late['change_magnitude']:.4f})")
    
    print("\n💡 SUGGESTED LUCENT EXPERIMENTS:")
    print("="*60)
    print("1. **Channel Comparison**: Visualize same channel across all 3 forgetting ratios")
    print("2. **Layer Depth Analysis**: Compare early vs mid vs late layer channels")
    print("3. **High vs Low Impact**: Visualize most vs least affected channels")
    print("4. **Class-Specific Analysis**: Check if certain classes are more affected")
    
    # Generate sample code
    print("\n🔬 SAMPLE LUCENT CODE:")
    print("="*60)
    print("```python")
    print("# Load your models")
    print("baseline_model = load_model('models/resnet50_pretrained.pth')")
    print("unlearned_model = load_model('results/good_results/.../RLcheckpoint.pth.tar')")
    print()
    print("# Compare most affected channel across models")
    top_target = targets[0]['target']
    print(f"target = '{top_target}'  # Most affected channel")
    print("baseline_viz = render.render_vis(baseline_model, target)")
    print("unlearned_viz = render.render_vis(unlearned_model, target)")
    print()
    print("# Visualize specific class through affected channel")
    print("class_viz = render.render_vis(unlearned_model, 'labels:131')  # Persian Cat")
    print(f"channel_viz = render.render_vis(unlearned_model, '{top_target}')  # Affected channel")
    print("```")
    
    return targets

def compare_experiments():
    """Compare channel targets across all experiments"""
    print("\n🔍 CROSS-EXPERIMENT COMPARISON:")
    print("="*60)
    
    experiments = [
        "random_forgetting_10percent_RL_conservative",
        "random_forgetting_20percent_RL_tweak_conservative", 
        "random_forgetting_30percent_RL_tweak_conservative"
    ]
    
    all_targets = {}
    
    for exp in experiments:
        targets_file = f"experiments/channel_analysis/{exp}/lucent_targets.json"
        if os.path.exists(targets_file):
            with open(targets_file, 'r') as f:
                all_targets[exp] = json.load(f)
    
    if not all_targets:
        print("❌ No targets found")
        return
    
    print("📊 Top affected channel per experiment:")
    for exp, targets in all_targets.items():
        if targets:
            top = targets[0]
            ratio = "10%" if "10percent" in exp else "20%" if "20percent" in exp else "30%"
            print(f"  {ratio:<4} | {top['target']:<25} | Change: {top['change_magnitude']:.4f}")
    
    # Find common affected layers
    all_layers = set()
    for targets in all_targets.values():
        for target in targets[:5]:  # Top 5 from each
            layer = target['layer'].split('.')[0] + '.' + target['layer'].split('.')[1]  # e.g., "layer3.0"
            all_layers.add(layer)
    
    print(f"\n🎯 Most commonly affected layer groups: {len(all_layers)}")
    for layer in sorted(all_layers):
        print(f"  - {layer}")

if __name__ == "__main__":
    targets = demo_channel_visualization()
    if targets:
        compare_experiments()
    
    print("\n" + "="*60)
    print("🎉 READY FOR LUCENT ANALYSIS!")
    print("="*60)
    print("Next steps:")
    print("1. Install Lucent: pip install lucent")
    print("2. Use the channel targets above in your Lucent experiments")
    print("3. Compare baseline vs unlearned models on same channels")
    print("4. Analyze how forgetting affects specific feature detectors")
    print("="*60)