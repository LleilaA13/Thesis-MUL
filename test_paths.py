#!/usr/bin/env python3
"""
Path Verification Script
Tests if all the updated paths work correctly after repository reorganization
"""

import os
import sys
from pathlib import Path

def test_file_exists(path, description):
    """Test if a file or directory exists"""
    if os.path.exists(path):
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} (NOT FOUND)")
        return False

def test_import(import_path, module_name, description):
    """Test if an import works"""
    try:
        original_path = sys.path.copy()
        sys.path.append(import_path)
        __import__(module_name)
        sys.path = original_path
        print(f"✅ {description}: Import successful")
        return True
    except ImportError as e:
        sys.path = original_path
        print(f"❌ {description}: Import failed - {e}")
        return False

def main():
    print("🧪 REPOSITORY PATH VERIFICATION")
    print("="*50)
    
    # Get current directory and repository root
    current_dir = Path.cwd()
    repo_root = current_dir
    
    # Try to find repository root
    while repo_root.parent != repo_root:
        if (repo_root / "core").exists() and (repo_root / "analysis").exists():
            break
        repo_root = repo_root.parent
    
    print(f"📁 Current directory: {current_dir}")
    print(f"📁 Repository root: {repo_root}")
    print()
    
    os.chdir(repo_root)
    
    # Test 1: Core repository structure
    print("1️⃣  TESTING CORE STRUCTURE")
    print("-" * 30)
    structure_ok = True
    structure_ok &= test_file_exists("core/Classification", "Core Classification directory")
    structure_ok &= test_file_exists("analysis/tools", "Analysis tools directory")
    structure_ok &= test_file_exists("analysis/visualizations", "Analysis visualizations directory")
    structure_ok &= test_file_exists("analysis/weight_analysis", "Weight analysis directory")
    structure_ok &= test_file_exists("experiments", "Experiments directory")
    structure_ok &= test_file_exists("scripts", "Scripts directory")
    structure_ok &= test_file_exists("docs", "Documentation directory")
    print()
    
    # Test 2: Key model and data files
    print("2️⃣  TESTING KEY FILES")
    print("-" * 30)
    files_ok = True
    files_ok &= test_file_exists("experiments/models/resnet50_pretrained.pth", "Baseline model")
    files_ok &= test_file_exists("experiments/results/good_results", "Good results directory")
    files_ok &= test_file_exists("experiments/good_results_weight_analysis", "Weight analysis results")
    print()
    
    # Test 3: Analysis script imports (from analysis/weight_analysis/)
    print("3️⃣  TESTING WEIGHT ANALYSIS IMPORTS")
    print("-" * 30)
    os.chdir("analysis/weight_analysis")
    import_ok = True
    import_ok &= test_import("../tools", "weight_influence_analyzer", "Weight Influence Analyzer")
    import_ok &= test_import("../visualizations", "enhanced_visualization", "Enhanced Visualization")
    os.chdir(repo_root)
    print()
    
    # Test 4: Channel analysis imports (from analysis/tools/)
    print("4️⃣  TESTING CHANNEL ANALYSIS IMPORTS")
    print("-" * 30)
    os.chdir("analysis/tools")
    channel_ok = True
    channel_ok &= test_import(".", "channel_level_analyzer", "Channel Level Analyzer")
    channel_ok &= test_import(".", "weight_influence_analyzer", "Weight Influence Analyzer")
    os.chdir(repo_root)
    print()
    
    # Test 5: Core Classification access (from scripts/)
    print("5️⃣  TESTING CORE CLASSIFICATION ACCESS")
    print("-" * 30)
    os.chdir("scripts")
    core_ok = True
    core_ok &= test_file_exists("../core/Classification/main_train.py", "Training script")
    core_ok &= test_file_exists("../core/Classification/main_forget.py", "Forgetting script")
    core_ok &= test_file_exists("../core/Classification/models", "Models directory")
    os.chdir(repo_root)
    print()
    
    # Test 6: Relative path resolution
    print("6️⃣  TESTING RELATIVE PATHS")
    print("-" * 30)
    
    # From analysis/weight_analysis/ perspective
    os.chdir("analysis/weight_analysis")
    paths_ok = True
    paths_ok &= test_file_exists("../../experiments/models/resnet50_pretrained.pth", "Baseline (from weight_analysis)")
    paths_ok &= test_file_exists("../../experiments/results/good_results", "Good results (from weight_analysis)")
    os.chdir(repo_root)
    
    # From analysis/tools/ perspective  
    os.chdir("analysis/tools")
    paths_ok &= test_file_exists("../../experiments/models/resnet50_pretrained.pth", "Baseline (from tools)")
    os.chdir(repo_root)
    print()
    
    # Summary
    print("📊 SUMMARY")
    print("="*50)
    
    all_tests = [
        ("Core Structure", structure_ok),
        ("Key Files", files_ok), 
        ("Weight Analysis Imports", import_ok),
        ("Channel Analysis Imports", channel_ok),
        ("Core Classification Access", core_ok),
        ("Relative Paths", paths_ok)
    ]
    
    passed = sum(1 for _, ok in all_tests if ok)
    total = len(all_tests)
    
    for test_name, ok in all_tests:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All paths updated successfully! Repository organization is working correctly.")
    else:
        print("⚠️  Some paths need fixing. Check the failed tests above.")
    
    # Restore original directory
    os.chdir(current_dir)

if __name__ == "__main__":
    main()