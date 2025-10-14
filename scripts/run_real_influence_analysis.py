#!/usr/bin/env python3
"""
Quick runner for real influence analysis using your existing models
"""

import os
import sys
import importlib.util
from pathlib import Path

def main():
    print("🚀 Real Influence Analysis Runner")
    print("=" * 50)
    
    # Define paths to your models
    project_root = Path(__file__).parent.parent
    
    # Check available models
    experiments_dir = project_root / "experiments"
    
    print("🔍 Looking for ResNet50 models...")
    
    # Look for baseline and unlearned models
    baseline_path = experiments_dir / "models" / "resnet50_pretrained.pth"
    
    possible_unlearned_paths = [
        experiments_dir / "results" / "good_results" / "random_forgetting_10percent_RL_tweak_conservative" / "RLcheckpoint.pth.tar",
        experiments_dir / "results" / "good_results" / "random_forgetting_20percent_RL_tweak_conservative" / "RLcheckpoint.pth.tar", 
        experiments_dir / "results" / "good_results" / "random_forgetting_30percent_RL_tweak_conservative" / "RLcheckpoint.pth.tar",
    ]
    
    # Find existing models
    unlearned_path = None
    
    # Check baseline model
    if baseline_path.exists():
        print(f"✅ Found baseline model: {baseline_path}")
    else:
        print(f"❌ Baseline model not found: {baseline_path}")
        return False
    
    # Check for unlearned models
    for path in possible_unlearned_paths:
        if path.exists():
            unlearned_path = path
            print(f"✅ Found unlearned model: {path}")
            break
    
    if not unlearned_path:
        print("❌ No unlearned model found!")
        print("Expected locations:")
        for path in possible_unlearned_paths:
            print(f"   - {path}")
        return False
    
    # Import the analyzer
    analyzer_path = project_root / "analysis/tools/real_influence_analyzer.py"
    spec = importlib.util.spec_from_file_location("real_influence_analyzer", analyzer_path)
    analyzer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer_module)
    RealInfluenceAnalyzer = analyzer_module.RealInfluenceAnalyzer
    
    # Set output directory
    output_dir = project_root / "analysis/results/real_influence_analysis"
    
    print(f"📊 Output directory: {output_dir}")
    print()
    
    # Initialize analyzer
    analyzer = RealInfluenceAnalyzer(output_dir=str(output_dir))
    
    try:
        # Run analysis
        print("🚀 Starting real model comparison analysis...")
        results = analyzer.run_complete_analysis(
            baseline_path=str(baseline_path),
            unlearned_path=str(unlearned_path),
            top_k_layers=25,       # Top 25 layers
            top_k_channels=40,     # Top 40 channels
            top_k_weights=100      # Top 100 weights
        )
        
        print("\n🎉 Real Analysis Complete!")
        print("=" * 40)
        
        # Print summary
        top_layers = results['top_layers']
        top_channels = results['top_channels']
        top_weights = results['top_weights']
        lucent_targets = results['lucent_targets']
        
        print("📋 ANALYSIS SUMMARY:")
        print(f"   • Layers analyzed: {len(top_layers)}")
        print(f"   • Channels analyzed: {len(top_channels)}")
        print(f"   • Weights analyzed: {len(top_weights)}")
        print(f"   • Lucent targets: {len(lucent_targets)}")
        print()
        
        if not top_layers.empty:
            print("🎯 MOST INFLUENCED COMPONENTS:")
            print(f"   • Layer: {top_layers.iloc[0]['layer_name']}")
            print(f"   • Change: {top_layers.iloc[0]['mean_relative_change']:.6f}")
            
            if not top_channels.empty:
                print(f"   • Top Channel: {top_channels.iloc[0]['channel_id']}")
                print(f"   • Channel Change: {top_channels.iloc[0]['mean_relative_change']:.6f}")
            
            if not top_weights.empty:
                print(f"   • Top Weight: {top_weights.iloc[0]['weight_id']}")
                print(f"   • Weight Change: {top_weights.iloc[0]['relative_change']:.6f}")
        
        print()
        print("🎨 TOP 5 LUCENT TARGETS:")
        for i, target in enumerate(lucent_targets[:5], 1):
            print(f"   {i}. {target['target']} ({target['type']}) - Score: {target['influence_score']:.6f}")
        
        print()
        print("📄 FILES CREATED:")
        print(f"   • real_influence_summary.json")
        print(f"   • lucent_targets_real.json") 
        print(f"   • lucent_commands_real.py")
        print(f"   • top_influenced_layers_real.csv")
        print(f"   • top_influenced_channels_real.csv")
        print(f"   • top_influenced_weights_real.csv")
        print(f"   • real_influence_analysis.png")
        
        print()
        print("💡 NEXT STEPS:")
        print("   1. Upload lucent_targets_real.json to Google Drive")
        print("   2. Copy lucent_commands_real.py contents to Colab")
        print("   3. Use real_influence_summary.json for detailed analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)