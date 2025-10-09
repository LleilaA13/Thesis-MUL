#!/usr/bin/env python3
"""
Monitor the classification-optimized SalUn experiment running in tmux
"""

import time
import json
import os
from datetime import datetime

def monitor_experiment():
    print("🔍 CLASSIFICATION EXPERIMENT MONITOR")
    print("="*50)
    print(f"Monitoring started at: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Monitor the results file
    results_file = "unlearn_results_log.json"
    
    print("📊 Expected Progress:")
    print("  • Training should take ~30-40 minutes")
    print("  • 25 epochs with GA method")
    print("  • Target: Forget acc <10%, Retain acc >60%")
    print()
    
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"📈 LATEST RESULTS ({len(results)} experiments):")
            if results:
                latest = results[-1]
                print(f"  Last run: {latest.get('timestamp', 'Unknown')}")
                print(f"  Forget acc: {latest.get('results', {}).get('forget_acc', 'N/A')}%")
                print(f"  Retain acc: {latest.get('results', {}).get('retain_acc', 'N/A')}%")
                print(f"  Method: {latest.get('parameters', {}).get('lr', 'N/A')} LR")
        except:
            print("📋 Results file exists but couldn't parse")
    else:
        print("📋 No results file yet - experiment still starting")
    
    print()
    print("🔧 TMUX COMMANDS:")
    print("  To attach to tmux session:")
    print("    tmux attach")
    print("  To check if training is running:")
    print("    ps aux | grep python")
    print("  To monitor GPU usage:")
    print("    nvidia-smi")
    print()
    print("📁 FILES TO WATCH:")
    print(f"  • {results_file} - Results will appear here")
    print(f"  • resnet50_unlearn_dogs.py - Training script")
    print(f"  • models/resnet50_dogs_forgetting/ - Saved models")

if __name__ == "__main__":
    monitor_experiment()