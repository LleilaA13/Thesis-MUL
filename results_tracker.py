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
        print(f"  Results: Forget={res.get('forget_acc', 'N/A'):.1f}%, Retain={res.get('retain_acc', 'N/A'):.1f}%, Train={res.get('train_acc', 'N/A'):.1f}%")
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

if __name__ == "__main__":
    # Example usage
    print("SalUn Results Tracker")
    print("Usage:")
    print("  python results_tracker.py log")
    print("  python results_tracker.py view [dogs|cats|vehicles]")
    print("  python results_tracker.py compare")