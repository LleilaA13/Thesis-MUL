#!/usr/bin/env python3
"""
Results tracking system for SalUn unlearning experiments
"""

import json
import os
from datetime import datetime

RESULTS_FILE = "unlearn_results_log.json"

def log_experiment(forget_type, parameters, results, notes=""):
    """
    Log experiment parameters and results
    
    Args:
        forget_type: 'dogs', 'cats', 'vehicles'
        parameters: dict with 'epochs', 'lr', 'mask_threshold'
        results: dict with 'forget_acc', 'retain_acc', 'train_acc', 'loss'
        notes: any additional notes
    """
    
    # Load existing results
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    # Create new entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "forget_type": forget_type,
        "parameters": parameters,
        "results": results,
        "notes": notes
    }
    
    # Add to results
    all_results.append(entry)
    
    # Save back
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Logged experiment: {forget_type} with {parameters}")

def view_results(forget_type=None):
    """View all results or filter by forget_type"""
    
    if not os.path.exists(RESULTS_FILE):
        print("No results logged yet.")
        return
    
    with open(RESULTS_FILE, 'r') as f:
        all_results = json.load(f)
    
    # Filter if needed
    if forget_type:
        all_results = [r for r in all_results if r['forget_type'] == forget_type]
    
    print(f"=== UNLEARNING EXPERIMENT RESULTS ===")
    if forget_type:
        print(f"Filtered for: {forget_type.upper()}")
    print()
    
    for i, result in enumerate(all_results, 1):
        params = result['parameters']
        res = result['results']
        
        print(f"#{i} - {result['forget_type'].upper()} ({result['timestamp'][:19]})")
        print(f"  Parameters: {params['epochs']} epochs, LR={params['lr']}, mask={params['mask_threshold']}")
        
        # Handle both numeric and string values
        forget_val = res.get('forget_acc', 'N/A')
        retain_val = res.get('retain_acc', 'N/A')
        train_val = res.get('train_acc', 'N/A')
        
        if isinstance(forget_val, (int, float)):
            forget_str = f"{forget_val:.1f}%"
        else:
            forget_str = str(forget_val)
            
        if isinstance(retain_val, (int, float)):
            retain_str = f"{retain_val:.1f}%"
        else:
            retain_str = str(retain_val)
            
        if isinstance(train_val, (int, float)):
            train_str = f"{train_val:.1f}%"
        else:
            train_str = str(train_val)
        
        print(f"  Results: Forget={forget_str}, Retain={retain_str}, Train={train_str}")
        if result['notes']:
            print(f"  Notes: {result['notes']}")
        print()

def compare_best_results():
    """Show best forget accuracy for each category"""
    
    if not os.path.exists(RESULTS_FILE):
        print("No results to compare yet.")
        return
    
    with open(RESULTS_FILE, 'r') as f:
        all_results = json.load(f)
    
    categories = ['dogs', 'cats', 'vehicles']
    
    print("=== BEST RESULTS BY CATEGORY ===")
    for category in categories:
        cat_results = [r for r in all_results if r['forget_type'] == category]
        if not cat_results:
            print(f"{category.upper()}: No experiments yet")
            continue
            
        # Find best (lowest forget accuracy)
        best = min(cat_results, key=lambda x: x['results'].get('forget_acc', 100))
        params = best['parameters']
        res = best['results']
        
        print(f"{category.upper()}: {res.get('forget_acc', 'N/A'):.1f}% forget acc")
        print(f"  Best params: {params['epochs']} epochs, LR={params['lr']}, mask={params['mask_threshold']}")
        print(f"  Date: {best['timestamp'][:19]}")
        print()

def quick_log_from_terminal():
    """Quick function to log results from terminal output"""
    print("=== Quick Log from Terminal Output ===")
    
    # Your recent results from tmux
    forget_acc = 62.0
    retain_acc = 66.52
    
    print(f"Logging results: forget={forget_acc}%, retain={retain_acc}%")
    
    # Default parameters - adjust these if needed
    parameters = {
        "epochs": 5,
        "lr": 0.01,
        "mask_threshold": 0.5
    }
    
    results = {
        "forget_acc": forget_acc,
        "retain_acc": retain_acc,
        "train_acc": "not_available",
        "loss": "not_available"
    }
    
    notes = "Moderate forget accuracy (62%) - some unlearning achieved but not optimal"
    
    log_experiment("dogs", parameters, results, notes)
    print("✅ Results saved successfully!")
    return True

def log_custom(forget_acc, retain_acc, epochs=5, lr=0.01, mask=0.5, forget_type="dogs", notes=""):
    """Log custom results with specified parameters"""
    parameters = {
        "epochs": int(epochs),
        "lr": float(lr), 
        "mask_threshold": float(mask)
    }
    
    results = {
        "forget_acc": float(forget_acc),
        "retain_acc": float(retain_acc),
        "train_acc": "not_available",
        "loss": "not_available"
    }
    
    log_experiment(forget_type, parameters, results, notes)
    print(f"✅ Logged: {forget_type} forget={forget_acc}% retain={retain_acc}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "log":
            quick_log_from_terminal()
        elif command == "custom" and len(sys.argv) >= 4:
            # Usage: python results_tracker.py custom 62.0 66.52 [epochs] [lr] [mask] [type] [notes]
            forget_acc = float(sys.argv[2])
            retain_acc = float(sys.argv[3])
            epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
            lr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01
            mask = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
            forget_type = sys.argv[7] if len(sys.argv) > 7 else "dogs"
            notes = " ".join(sys.argv[8:]) if len(sys.argv) > 8 else ""
            log_custom(forget_acc, retain_acc, epochs, lr, mask, forget_type, notes)
        elif command == "view":
            forget_type = sys.argv[2] if len(sys.argv) > 2 else None
            view_results(forget_type)
        elif command == "compare":
            compare_best_results()
        else:
            print(f"Unknown command: {command}")
            print("Usage examples:")
            print("  python results_tracker.py log")
            print("  python results_tracker.py custom 62.0 66.52 5 0.01 0.5 dogs 'Recent tmux run'")
    else:
        # Example usage
        print("SalUn Results Tracker")
        print("Usage:")
        print("  python results_tracker.py log")
        print("  python results_tracker.py custom <forget_acc> <retain_acc> [epochs] [lr] [mask] [type] [notes]")
        print("  python results_tracker.py view [dogs|cats|vehicles]")
        print("  python results_tracker.py compare")