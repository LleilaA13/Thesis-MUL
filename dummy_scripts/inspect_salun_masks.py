#!/usr/bin/env python3
"""
Quick mask inspection to verify mask quality
"""

import torch
import os

def inspect_masks():
    """Inspect the generated SalUn masks"""
    
    mask_dir = "masks/resnet50_dogs_forgetting"
    
    print("🔍 MASK INSPECTION REPORT")
    print("="*50)
    
    if not os.path.exists(mask_dir):
        print(f"❌ Mask directory not found: {mask_dir}")
        return
    
    # Check available masks
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.pt')]
    print(f"📁 Available masks: {len(mask_files)}")
    
    for mask_file in sorted(mask_files):
        threshold = mask_file.replace('with_', '').replace('.pt', '')
        print(f"   {mask_file} → threshold={threshold}")
    
    # Inspect a specific mask
    if "with_0.5.pt" in mask_files:
        print(f"\n🔬 DETAILED ANALYSIS: with_0.5.pt")
        
        mask_path = os.path.join(mask_dir, "with_0.5.pt")
        mask = torch.load(mask_path, map_location='cpu')
        
        print(f"📊 Mask contains {len(mask)} layers:")
        
        total_params = 0
        allowed_params = 0
        layer_info = []
        
        for layer_name, layer_mask in mask.items():
            layer_total = layer_mask.numel()
            layer_allowed = (layer_mask == 1).sum().item()
            layer_blocked = layer_total - layer_allowed
            
            total_params += layer_total
            allowed_params += layer_allowed
            
            percentage_allowed = (layer_allowed / layer_total) * 100
            
            layer_info.append({
                'name': layer_name,
                'total': layer_total,
                'allowed': layer_allowed,
                'blocked': layer_blocked,
                'pct_allowed': percentage_allowed
            })
        
        # Sort by parameter count (largest first)
        layer_info.sort(key=lambda x: x['total'], reverse=True)
        
        print(f"\n📈 TOP 10 LARGEST LAYERS:")
        for i, info in enumerate(layer_info[:10]):
            print(f"  {i+1:2d}. {info['name']:<20} | {info['total']:>8,} params | {info['pct_allowed']:>5.1f}% allowed")
        
        overall_percentage = (allowed_params / total_params) * 100
        print(f"\n🎯 OVERALL STATISTICS:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Allowed (mask=1): {allowed_params:,} ({overall_percentage:.1f}%)")
        print(f"   Blocked (mask=0): {total_params - allowed_params:,} ({100-overall_percentage:.1f}%)")
        
        # Check if this seems reasonable
        if overall_percentage < 30:
            print(f"⚠️  WARNING: Very restrictive mask ({overall_percentage:.1f}% allowed)")
            print(f"   This may explain retain accuracy issues!")
        elif overall_percentage > 70:
            print(f"⚠️  WARNING: Very permissive mask ({overall_percentage:.1f}% allowed)")
            print(f"   May not provide sufficient selectivity")
        else:
            print(f"✅ REASONABLE: {overall_percentage:.1f}% of weights allowed")
        
        # Check if final layers are targeted
        final_layers = [info for info in layer_info if 'fc' in info['name'] or 'classifier' in info['name']]
        if final_layers:
            print(f"\n🎯 FINAL CLASSIFICATION LAYERS:")
            for info in final_layers:
                print(f"   {info['name']}: {info['pct_allowed']:.1f}% allowed")
        
        return mask, overall_percentage
    
    else:
        print("❌ with_0.5.pt not found for detailed analysis")
        return None, None

if __name__ == "__main__":
    mask, pct = inspect_masks()
    
    if pct is not None:
        print(f"\n💡 RECOMMENDATIONS:")
        if pct < 40:
            print(f"   Try with_0.6.pt or with_0.7.pt for less restrictive masking")
        elif pct > 60:
            print(f"   Current mask should be fine for retain accuracy")
        else:
            print(f"   Mask seems balanced - check other parameters")