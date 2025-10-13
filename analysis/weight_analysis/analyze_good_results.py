#!/usr/bin/env python3
"""
Weight Analysis for Good Results
Analyzes the weight changes in your best performing unlearning experiments
"""

import os
from weight_influence_analyzer import WeightInfluenceAnalyzer

def main():
    print("🏆 WEIGHT ANALYSIS FOR GOOD RESULTS")
    print("="*50)
    
    # Updated paths for good results
    baseline_model = "models/resnet50_pretrained.pth"
    good_results_dir = "results/good_results"
    
    print(f"📂 Baseline model: {baseline_model}")
    print(f"📂 Good results directory: {good_results_dir}")
    
    # Verify paths exist
    if not os.path.exists(baseline_model):
        print("❌ Baseline model not found!")
        print("   Please ensure you have models/resnet50_pretrained.pth")
        return
    
    if not os.path.exists(good_results_dir):
        print("❌ Good results directory not found!")
        return
    
    # List available good experiments
    experiments = [d for d in os.listdir(good_results_dir) 
                  if os.path.isdir(os.path.join(good_results_dir, d))]
    
    print(f"\n🌟 Found {len(experiments)} good experiments:")
    for i, exp in enumerate(experiments, 1):
        exp_path = os.path.join(good_results_dir, exp)
        files = os.listdir(exp_path)
        print(f"   {i}. {exp}")
        print(f"      Files: {', '.join(files)}")
    
    if not experiments:
        print("❌ No experiments found in good_results!")
        return
    
    # Run analysis on good results
    print(f"\n🚀 Starting weight analysis on {len(experiments)} good experiments...")
    
    analyzer = WeightInfluenceAnalyzer(baseline_model, good_results_dir)
    
    try:
        analysis_results = analyzer.generate_comprehensive_report(
            output_dir='experiments/good_results_weight_analysis'
        )
        
        print("\n✅ Good Results Analysis Complete!")
        print("📊 Results saved to:")
        print("   - experiments/good_results_weight_analysis/comprehensive_weight_analysis.json")
        print("   - experiments/good_results_weight_analysis/summary_report.md")
        print("   - experiments/random_forgetting/visualizations/")
        
        # Detailed summary for good results
        print(f"\n🏆 GOOD RESULTS SUMMARY:")
        print(f"   📊 Analyzed {len(analysis_results)} best experiments")
        
        for exp_name, data in analysis_results.items():
            print(f"\n   🔬 {exp_name}:")
            if 'layer_sensitivity' in data:
                layer_data = data['layer_sensitivity']
                if layer_data:
                    # Find most and least affected layers
                    most_affected = max(layer_data.keys(), key=lambda x: layer_data[x]['mean_relative_change'])
                    least_affected = min(layer_data.keys(), key=lambda x: layer_data[x]['mean_relative_change'])
                    
                    most_change = layer_data[most_affected]['mean_relative_change']
                    least_change = layer_data[least_affected]['mean_relative_change']
                    
                    print(f"      🔥 Most affected layer: {most_affected} ({most_change:.6f})")
                    print(f"      🛡️  Least affected layer: {least_affected} ({least_change:.6f})")
                    
                    # Count total layers analyzed
                    total_layers = len(layer_data)
                    print(f"      📊 Total layers analyzed: {total_layers}")
            
            if 'top_affected_weights' in data:
                top_weights = data['top_affected_weights'][:3]
                if top_weights:
                    print(f"      🎯 Top 3 changed weights:")
                    for weight in top_weights:
                        print(f"         - {weight['layer']}: {weight['relative_change']:.6f}")
        
        print(f"\n🎉 Analysis complete! Check the output directory for detailed results.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()