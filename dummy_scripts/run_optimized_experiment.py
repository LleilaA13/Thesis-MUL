#!/usr/bin/env python3
"""
Run optimized SalUn experiment with higher learning rate and GA method
"""

import os
import sys

def main():
    print("🚀 STARTING OPTIMIZED SALUN EXPERIMENT")
    print("="*50)
    
    # Experiment parameters - AGGRESSIVE BUT CONTROLLED
    config = {
        'method': 'GA',  # Gradient Ascent instead of Random Labels
        'lr': 0.05,      # DOUBLE the learning rate (was 0.025)
        'epochs': 20,    # More epochs
        'threshold': 0.5, # Keep the balanced mask
        'batch_size': 256
    }
    
    print("📊 EXPERIMENT CONFIGURATION:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print(f"\n🎯 RATIONALE:")
    print(f"   • GA method: More precise than Random Labels")
    print(f"   • LR=0.05: Double previous rate for stronger forgetting")
    print(f"   • 20 epochs: Sufficient time for convergence")
    print(f"   • threshold=0.5: Balanced weight modification")
    
    # Build the command
    cmd_parts = [
        "conda run -n salUN python resnet50_unlearn_dogs.py",
        f"--unlearn {config['method']}",
        f"--unlearn_lr {config['lr']}",
        f"--num_epochs {config['epochs']}",
        f"--mask with_{config['threshold']}.pt",
        f"--batch_size {config['batch_size']}",
        "--save_model",
        "--alpha 5"  # Keep high alpha for strong forgetting
    ]
    
    full_cmd = " ".join(cmd_parts)
    
    print(f"\n🔥 COMMAND TO RUN:")
    print(f"   {full_cmd}")
    
    print(f"\n📈 EXPECTED OUTCOMES:")
    print(f"   • Forget Accuracy: <50% (target achieved)")
    print(f"   • Retain Accuracy: >60% (improved from 38%)")
    print(f"   • Time: ~20-25 minutes")
    
    response = input(f"\n❓ Start optimized experiment? [y/N]: ").strip().lower()
    
    if response == 'y':
        print(f"\n🏃 LAUNCHING EXPERIMENT...")
        print(f"   You can monitor in tmux or check results_tracker.py")
        
        # Change to correct directory and run
        os.chdir('/media/hdd/usr/leyla/Unlearn-Saliency')
        os.system(full_cmd)
    else:
        print(f"\n✋ Experiment cancelled. Command saved above for manual execution.")

if __name__ == "__main__":
    main()