#!/usr/bin/env python3
"""
Ultra-aggressive SalUn configuration for 5% forget accuracy with high retain accuracy
Based on paper evidence showing 15.6% prob of forgotten class (excellent forgetting)
"""

import os
import sys

def main():
    print("🎯 CLASSIFICATION-SPECIFIC SALUN FOR 5% FORGET ACCURACY")
    print("="*60)
    
    # STRATEGY 1: Multiple complementary methods (based on paper's classification results)
    strategies = {
        'Strategy 1 - Paper-based GA + Restrictive Mask': {
            'method': 'GA',
            'lr': 0.05,     # Higher than paper's 0.013 but not extreme
            'epochs': 25,   # More epochs for stronger forgetting
            'threshold': 0.2,  # Very restrictive (block 80% weights)
            'alpha': 8,     # Strong forgetting signal
            'rationale': 'GA with restrictive mask - paper shows GA works well for classification'
        },
        
        'Strategy 2 - Boundary Expanding (Classification)': {
            'method': 'boundary_expanding',
            'lr': 0.03,     # Moderate LR for boundary pushing
            'epochs': 20,
            'threshold': 0.3,
            'alpha': 6,
            'rationale': 'Push forget data to invalid class (200 for TinyImageNet), designed for classification'
        },
        
        'Strategy 3 - Multi-stage Training': {
            'method': 'sequential',
            'steps': [
                {'method': 'GA', 'lr': 0.04, 'epochs': 15, 'threshold': 0.2, 'alpha': 6},
                {'method': 'RL', 'lr': 0.02, 'epochs': 10, 'threshold': 0.4, 'alpha': 4}
            ],
            'rationale': 'First GA for strong forgetting, then RL for stability - classification optimized'
        },
        
        'Strategy 4 - Paper-inspired Conservative': {
            'method': 'GA',
            'lr': 0.02,     # Closer to paper's 0.013 but still higher
            'epochs': 35,   # Many epochs for gradual forgetting
            'threshold': 0.3,
            'alpha': 5,
            'rationale': 'Based on paper examples, longer training with moderate parameters'
        }
    }
    
    print("📈 STRATEGIES FOR ULTRA-LOW FORGET ACCURACY:")
    for name, config in strategies.items():
        print(f"\n{name}:")
        if 'steps' in config:
            for i, step in enumerate(config['steps']):
                print(f"  Step {i+1}: {step}")
        else:
            print(f"  Method: {config['method']}")
            print(f"  LR: {config['lr']}")
            print(f"  Epochs: {config['epochs']}")
            print(f"  Threshold: {config['threshold']}")
        print(f"  Rationale: {config['rationale']}")
    
    # STRATEGY 2: Mask optimization for retain accuracy preservation
    print(f"\n🎯 MASK STRATEGIES FOR HIGH RETAIN ACCURACY:")
    mask_strategies = [
        "Progressive masking: Start with 0.2, gradually increase to 0.5",
        "Layer-specific masks: More restrictive on FC layers, less on conv layers",
        "Dynamic thresholding: Adapt based on retain accuracy during training"
    ]
    
    for i, strategy in enumerate(mask_strategies, 1):
        print(f"  {i}. {strategy}")
    
    # STRATEGY 3: Training dynamics
    print(f"\n⚡ TRAINING DYNAMICS:")
    dynamics = [
        "Higher alpha values (5-10) for stronger forgetting signal",
        "Learning rate scheduling: High initially, decay to preserve retain",
        "Early stopping based on retain accuracy threshold",
        "Batch size tuning: Smaller batches for more gradient updates"
    ]
    
    for i, dynamic in enumerate(dynamics, 1):
        print(f"  {i}. {dynamic}")
    
    # Generate the classification-optimized command
    print(f"\n🚀 RECOMMENDED CLASSIFICATION-OPTIMIZED COMMAND:")
    cmd = """
conda run -n salUN python resnet50_unlearn_dogs.py \\
  --unlearn GA \\
  --unlearn_lr 0.05 \\
  --num_epochs 25 \\
  --mask with_0.2.pt \\
  --batch_size 128 \\
  --alpha 8 \\
  --save_model
"""
    
    print(cmd)
    
    print(f"\n📊 EXPECTED RESULTS:")
    print(f"   • Forget Accuracy: 5-15% (excellent unlearning)")
    print(f"   • Retain Accuracy: 55-65% (acceptable preservation)")
    print(f"   • Risk: High - may damage model significantly")
    
    print(f"\n⚠️  BACKUP STRATEGY IF RETAIN ACCURACY TOO LOW:")
    backup_cmd = """
# If retain drops below 50%, use paper-inspired approach:
conda run -n salUN python resnet50_unlearn_dogs.py \\
  --unlearn GA \\
  --unlearn_lr 0.02 \\
  --num_epochs 35 \\
  --mask with_0.3.pt \\
  --batch_size 128 \\
  --alpha 5 \\
  --save_model
"""
    print(backup_cmd)
    
    response = input(f"\n❓ Run the ultra-aggressive experiment? [y/N]: ").strip().lower()
    
    if response == 'y':
        print(f"\n🏃 LAUNCHING ULTRA-AGGRESSIVE EXPERIMENT...")
        
        # Change to correct directory and run
        os.chdir('/media/hdd/usr/leyla/Unlearn-Saliency')
        
        # First check if we have the restrictive mask
        if not os.path.exists('masks/resnet50_dogs_forgetting/with_0.2.pt'):
            print("⚠️  with_0.2.pt not found, using with_0.3.pt instead...")
            mask_file = 'with_0.3.pt'
        else:
            mask_file = 'with_0.2.pt'
        
        final_cmd = f"""conda run -n salUN python resnet50_unlearn_dogs.py \
--unlearn GA \
--unlearn_lr 0.05 \
--num_epochs 25 \
--mask {mask_file} \
--batch_size 128 \
--alpha 8 \
--save_model"""
        
        print(f"Executing: {final_cmd}")
        os.system(final_cmd.replace('\\\n', ' '))
    else:
        print(f"\n✋ Experiment cancelled. Use the commands above when ready.")

if __name__ == "__main__":
    main()