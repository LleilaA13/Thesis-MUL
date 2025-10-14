#!/usr/bin/env python3
"""
Easy script to run weight influence analysis on your unlearning experiments
"""

import os
import sys
sys.path.append('../tools')
from weight_influence_analyzer import WeightInfluenceAnalyzer

def main():
    print("🔬 WEIGHT INFLUENCE ANALYSIS")
    print("="*50)
    
    # Check your current setup
    baseline_model = "../../experiments/models/resnet50_pretrained.pth"
    results_dir = "../../experiments/results/good_results"
    
    print(f"📂 Baseline model: {baseline_model}")
    print(f"📂 Results directory: {results_dir}")
    
    # Verify paths exist
    if not os.path.exists(baseline_model):
        print("❌ Baseline model not found!")
        print("   Please ensure you have models/resnet50_pretrained.pth")
        return
    
    if not os.path.exists(results_dir):
        print("❌ Results directory not found!")
        return
    
    # List available experiments
    experiments = [d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d)) 
                  and 'random_forgetting' in d]
    
    print(f"\n🧪 Found {len(experiments)} random forgetting experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"   {i}. {exp}")
    
    if not experiments:
        print("❌ No random forgetting experiments found!")
        print("   Make sure your experiment directories contain 'random_forgetting' in the name")
        return
    
    # Run analysis
    print(f"\n🚀 Starting analysis on {len(experiments)} experiments...")
    
    analyzer = WeightInfluenceAnalyzer(baseline_model, results_dir)
    
    try:
        analysis_results = analyzer.generate_comprehensive_report()
        
        print("\n✅ Analysis complete!")
        print("📊 Results saved to:")
        print("   - experiments/random_forgetting/weight_analysis/comprehensive_weight_analysis.json")
        print("   - experiments/random_forgetting/weight_analysis/summary_report.md")
        print("   - experiments/random_forgetting/visualizations/")
        
        # Quick summary
        print(f"\n📈 Quick Summary:")
        print(f"   - Analyzed {len(analysis_results)} experiments")
        for exp_name in analysis_results.keys():
            print(f"   - {exp_name}: ✓")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("   Check that your model files are compatible")

if __name__ == "__main__":
    main()