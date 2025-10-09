#!/usr/bin/env python3
"""
Debug and analyze unlearning issues - why forget accuracy is still high
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_results():
    """Analyze the trend in unlearning results"""
    
    print("=== UNLEARNING PERFORMANCE ANALYSIS ===\n")
    
    # Load results
    with open("unlearn_results_log.json", 'r') as f:
        results = json.load(f)
    
    dogs_results = [r for r in results if r['forget_type'] == 'dogs']
    
    print("📊 TREND ANALYSIS:")
    print("Experiment #  | Forget Acc | Retain Acc | LR    | Epochs | Mask")
    print("-" * 65)
    
    for i, result in enumerate(dogs_results, 1):
        params = result['parameters']
        res = result['results']
        forget_acc = res.get('forget_acc', 'N/A')
        retain_acc = res.get('retain_acc', 'N/A')
        
        print(f"#{i:2d}           | {forget_acc:>8}%  | {retain_acc:>8}%  | {params['lr']:<5} | {params['epochs']:<6} | {params['mask_threshold']}")
    
    print("\n🔍 KEY OBSERVATIONS:")
    
    # Find best and worst
    valid_results = [r for r in dogs_results if isinstance(r['results'].get('forget_acc'), (int, float))]
    if valid_results:
        best = min(valid_results, key=lambda x: x['results']['forget_acc'])
        worst = max(valid_results, key=lambda x: x['results']['forget_acc'])
        
        print(f"✅ Best unlearning: {best['results']['forget_acc']}% forget acc")
        print(f"   Parameters: LR={best['parameters']['lr']}, epochs={best['parameters']['epochs']}, mask={best['parameters']['mask_threshold']}")
        
        print(f"❌ Worst unlearning: {worst['results']['forget_acc']}% forget acc") 
        print(f"   Parameters: LR={worst['parameters']['lr']}, epochs={worst['parameters']['epochs']}, mask={worst['parameters']['mask_threshold']}")
    
    print("\n🚨 PROBLEM DIAGNOSIS:")
    
    latest = dogs_results[-1]
    latest_forget = latest['results']['forget_acc']
    
    if latest_forget > 60:
        print("❌ MAJOR ISSUE: Forget accuracy too high (>60%)")
        print("   Possible causes:")
        print("   1. Learning rate too low - model not changing enough")
        print("   2. Not enough epochs - insufficient unlearning time") 
        print("   3. Mask threshold too high - not blocking enough neurons")
        print("   4. SalUn mask quality issues - wrong neurons being targeted")
        print("   5. Evaluation bug - labels being restored incorrectly")
        
    elif latest_forget > 55:
        print("⚠️  MODERATE ISSUE: Some unlearning but not sufficient")
        print("   Need more aggressive parameters")
        
    else:
        print("✅ GOOD: Forget accuracy approaching random performance")

def suggest_next_experiments():
    """Suggest next experimental parameters"""
    
    print("\n🎯 RECOMMENDED NEXT STEPS:")
    
    print("\n1. 🔥 AGGRESSIVE UNLEARNING (try first):")
    print("   python results_tracker.py custom X X 15 0.1 0.2 dogs 'Aggressive: 15 epochs, LR=0.1, mask=0.2'")
    print("   → Very high LR, many epochs, very restrictive mask (80% neurons blocked)")
    
    print("\n2. 🎯 TARGETED DEBUGGING:")
    print("   a) Check if SalUn masks are correct:")
    print("      python inspect_mask.py  # Check mask quality")
    print("   b) Verify evaluation logic:")
    print("      python debug_forget_accuracy.py  # Check label restoration")
    
    print("\n3. 📊 ALTERNATIVE APPROACHES:")
    print("   a) Try different mask thresholds:")
    print("      - mask=0.1 (block 90% of neurons)")
    print("      - mask=0.05 (block 95% of neurons)")
    print("   b) Much longer training:")
    print("      - 20-30 epochs with moderate LR (0.05)")
    print("   c) Learning rate scheduling:")
    print("      - Start high (0.1) then decay")
    
    print("\n4. 🔧 POTENTIAL FIXES TO CHECK:")
    print("   a) Model loading issues - are we loading the right checkpoint?")
    print("   b) Mask application - are masks being applied correctly during training?")
    print("   c) Loss function - is the unlearning loss working properly?")
    print("   d) Dataset issues - are forget/retain splits correct?")

def plot_results():
    """Create a simple plot of results"""
    
    with open("unlearn_results_log.json", 'r') as f:
        results = json.load(f)
    
    dogs_results = [r for r in results if r['forget_type'] == 'dogs']
    
    # Extract data
    forget_accs = []
    retain_accs = []
    labels = []
    
    for i, result in enumerate(dogs_results):
        res = result['results']
        if isinstance(res.get('forget_acc'), (int, float)):
            forget_accs.append(res['forget_acc'])
            retain_accs.append(res.get('retain_acc', 0))
            labels.append(f"#{i+1}")
    
    if forget_accs:
        print(f"\n📈 RESULTS VISUALIZATION:")
        print("Experiment | Forget% | Retain% | Status")
        print("-" * 40)
        for i, (f, r, l) in enumerate(zip(forget_accs, retain_accs, labels)):
            status = "✅ Good" if f < 50 else "⚠️ OK" if f < 55 else "❌ Poor"
            print(f"{l:>10} | {f:>6.1f}% | {r:>6.1f}% | {status}")
        
        print(f"\nTarget: <50% forget accuracy (random performance)")
        print(f"Current best: {min(forget_accs):.1f}% forget accuracy")

if __name__ == "__main__":
    analyze_results()
    suggest_next_experiments() 
    plot_results()