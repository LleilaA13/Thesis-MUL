#!/usr/bin/env python3
"""
Quick script to run the improved SalUn unlearning
"""

import subprocess
import sys

def main():
    print("🚀 RUNNING IMPROVED SALUN UNLEARNING")
    print("Based on official SalUn repository analysis")
    print()
    
    choice = input("Choose option:\n1. Run single experiment with optimal params\n2. Run multiple experiments\n3. Just test current model\nChoice (1-3): ").strip()
    
    if choice == "1":
        print("\n[*] Running single experiment with optimal SalUn parameters...")
        subprocess.run(["python", "resnet50_unlearn_dogs.py"])
        
    elif choice == "2":
        print("\n[*] Running multiple experiments...")
        subprocess.run(["python", "advanced_salun_experiments.py"])
        
    elif choice == "3":
        print("\n[*] Testing current model...")
        subprocess.run(["python", "test_unlearned_model.py"])
        
    else:
        print("Invalid choice")
        return
    
    print(f"\n[*] Next steps:")
    print(f"1. Check results with: python test_unlearned_model.py")
    print(f"2. View all results: python results_tracker.py view dogs")
    print(f"3. Compare performance: python results_tracker.py compare")

if __name__ == "__main__":
    main()