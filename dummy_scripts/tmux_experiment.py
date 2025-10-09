#!/usr/bin/env python3
"""
Classification-optimized SalUn experiment for tmux
Target: 5% forget accuracy with high retain accuracy
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_tmux_experiment():
    print("🎯 CLASSIFICATION-OPTIMIZED SALUN EXPERIMENT FOR TMUX")
    print("="*60)
    print("Target: 5% forget accuracy with high retain accuracy")
    print()
    
    # Configuration
    config = {
        'method': 'GA',
        'lr': 0.05,
        'epochs': 25,
        'mask': 'with_0.2.pt',
        'alpha': 8,
        'batch_size': 128
    }
    
    print("📊 EXPERIMENT CONFIGURATION:")
    print(f"  • Method: {config['method']} (Gradient Ascent)")
    print(f"  • Learning Rate: {config['lr']}")
    print(f"  • Epochs: {config['epochs']}")
    print(f"  • Mask: {config['mask']} (restrictive - blocks 80% weights)")
    print(f"  • Alpha: {config['alpha']} (strong forgetting signal)")
    print(f"  • Batch Size: {config['batch_size']}")
    print()
    
    print("🎯 EXPECTED RESULTS:")
    print("  • Forget Accuracy: 5-10% (excellent unlearning)")
    print("  • Retain Accuracy: 60-70% (preserved classification)")
    print("  • Duration: ~30-40 minutes")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('resnet50_unlearn_dogs.py'):
        print("❌ Error: resnet50_unlearn_dogs.py not found!")
        print("   Make sure you're in the correct directory")
        return False
    
    # Check if mask exists
    mask_path = f"masks/resnet50_dogs_forgetting/{config['mask']}"
    if not os.path.exists(mask_path):
        print(f"⚠️  Warning: {mask_path} not found!")
        print("   Falling back to with_0.3.pt...")
        config['mask'] = 'with_0.3.pt'
        if not os.path.exists(f"masks/resnet50_dogs_forgetting/{config['mask']}"):
            print("❌ Error: No suitable mask found!")
            return False
    
    print(f"✅ Using mask: {config['mask']}")
    print()
    
    # Build command
    cmd = [
        'python', 'resnet50_unlearn_dogs.py',
        '--unlearn', config['method'],
        '--unlearn_lr', str(config['lr']),
        '--num_epochs', str(config['epochs']),
        '--mask', config['mask'],
        '--batch_size', str(config['batch_size']),
        '--alpha', str(config['alpha']),
        '--save_model'
    ]
    
    print("🚀 STARTING EXPERIMENT...")
    print(f"Command: {' '.join(cmd)}")
    print()
    print("📈 Training Progress:")
    print("-" * 50)
    
    # Record start time
    start_time = datetime.now()
    
    try:
        # Run the experiment
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line.rstrip())
            sys.stdout.flush()
        
        # Wait for completion
        process.wait()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print()
        print("="*60)
        
        if process.returncode == 0:
            print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total time: {duration}")
            print()
            print("📊 NEXT STEPS:")
            print("  1. Check results: python results_tracker.py")
            print("  2. View latest results: python results_tracker.py --latest")
            print("  3. Compare results: python results_tracker.py --compare")
            print()
            print("📁 MODEL SAVED:")
            print("  Location: models/resnet50_dogs_forgetting/")
            print("  Use --save_model flag was used")
            
            # Try to show results if available
            try:
                print()
                print("🎯 ATTEMPTING TO SHOW RESULTS...")
                result_cmd = ['python', 'results_tracker.py', '--latest']
                subprocess.run(result_cmd, check=False)
            except:
                print("   (Run results_tracker.py manually to see results)")
            
        else:
            print("❌ EXPERIMENT FAILED!")
            print(f"   Exit code: {process.returncode}")
            print("   Check the output above for errors")
            
        return process.returncode == 0
        
    except KeyboardInterrupt:
        print()
        print("🛑 EXPERIMENT INTERRUPTED BY USER")
        if 'process' in locals():
            process.terminate()
        return False
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = run_tmux_experiment()
    
    print()
    if success:
        print("🎉 All done! Check your results above.")
    else:
        print("💡 If there were issues, try:")
        print("   • Check conda environment: conda activate salUN")
        print("   • Verify GPU availability: nvidia-smi")
        print("   • Check disk space: df -h")

if __name__ == "__main__":
    main()