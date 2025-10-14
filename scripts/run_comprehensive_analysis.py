#!/usr/bin/env python3
"""
Quick runner for comprehensive influence analysis

This script provides an easy way to run the comprehensive analysis
with your existing results.
"""

import os
import sys
import importlib.util
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the analyzer
analyzer_path = project_root / "analysis/tools/comprehensive_influence_analyzer.py"
spec = importlib.util.spec_from_file_location("comprehensive_influence_analyzer", analyzer_path)
analyzer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer_module)
ComprehensiveInfluenceAnalyzer = analyzer_module.ComprehensiveInfluenceAnalyzer

def main():
    print("🚀 Quick Comprehensive Influence Analysis")
    print("=" * 50)
    
    # Define paths
    project_root = Path(__file__).parent.parent
    weight_analysis_path = project_root / "experiments" / "good_results_weight_analysis" / "comprehensive_weight_analysis.json"
    channel_analysis_path = project_root / "experiments" / "channel_analysis" / "comprehensive_channel_analysis.json"
    output_dir = project_root / "analysis" / "results" / "comprehensive_analysis"
    
    # Check if files exist
    if not weight_analysis_path.exists():
        print(f"❌ Weight analysis file not found: {weight_analysis_path}")
        print("Please run the weight analysis first or update the path.")
        return
    
    print(f"✅ Found weight analysis: {weight_analysis_path}")
    
    if channel_analysis_path.exists():
        print(f"✅ Found channel analysis: {channel_analysis_path}")
    else:
        print(f"⚠️  Channel analysis not found: {channel_analysis_path}")
        print("Will extract channel data from weight analysis.")
        channel_analysis_path = None
    
    # Initialize analyzer
    analyzer = ComprehensiveInfluenceAnalyzer(output_dir=str(output_dir))
    
    # Set analysis parameters
    analyzer.top_k_layers = 25      # Top 25 layers
    analyzer.top_k_channels = 40    # Top 40 channels  
    analyzer.top_k_weights = 60     # Top 60 weight groups
    
    print(f"📊 Analysis parameters:")
    print(f"   • Top layers: {analyzer.top_k_layers}")
    print(f"   • Top channels: {analyzer.top_k_channels}")
    print(f"   • Top weights: {analyzer.top_k_weights}")
    print()
    
    # Run comprehensive analysis
    try:
        results = analyzer.run_comprehensive_analysis(
            weight_analysis_path=str(weight_analysis_path),
            channel_analysis_path=str(channel_analysis_path) if channel_analysis_path else None
        )
        
        print("\n🎉 Analysis Complete!")
        print("=" * 30)
        print(f"📁 Results saved to: {output_dir}")
        print()
        
        # Print quick summary
        summary = results['summary']
        lucent_targets = results['lucent_targets']
        
        print("📋 QUICK SUMMARY:")
        print(f"   • Experiments analyzed: {len(summary['analysis_overview']['experiments_analyzed'])}")
        print(f"   • Top influenced layer: {summary['most_influenced_components']['layers']['top_layer']}")
        print(f"   • Max change: {summary['most_influenced_components']['layers']['max_change']:.4f}")
        print(f"   • Lucent targets generated: {len(lucent_targets)}")
        print()
        
        print("🎯 TOP 5 LUCENT TARGETS:")
        for i, target in enumerate(lucent_targets[:5], 1):
            print(f"   {i}. {target['target']} ({target['type']}) - Score: {target['influence_score']:.4f}")
        print()
        
        print("📄 FILES CREATED:")
        print(f"   • comprehensive_influence_summary.json")
        print(f"   • lucent_targets.json")
        print(f"   • lucent_commands.py")
        print(f"   • top_influenced_layers.csv")
        print(f"   • top_influenced_weights.csv")
        print(f"   • comprehensive_influence_analysis.png")
        print()
        
        print("💡 NEXT STEPS:")
        print("   1. Upload lucent_targets.json to Google Drive")
        print("   2. Copy lucent_commands.py contents to Colab")
        print("   3. Use the comprehensive_influence_summary.json for analysis")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)