#!/usr/bin/env python3
"""
Multi-Experiment Real Influence Analyzer

Analyzes multiple forgetting experiments (10%, 20%, 30%) and generates
separate JSON files for each experiment to compare influence patterns.
"""

import os
import sys
import importlib.util
from pathlib import Path

def main():
    print("🚀 Multi-Experiment Real Influence Analysis")
    print("=" * 60)
    
    # Define paths to your models
    project_root = Path(__file__).parent.parent
    experiments_dir = project_root / "experiments"
    
    # Define all experiments to analyze
    experiments = {
        "10percent": {
            "name": "random_forgetting_10percent_RL_tweak_conservative",
            "path": experiments_dir / "results" / "good_results" / "random_forgetting_10percent_RL_tweak_conservative" / "RLcheckpoint.pth.tar",
            "description": "10% Random Data Forgetting"
        },
        "20percent": {
            "name": "random_forgetting_20percent_RL_tweak_conservative", 
            "path": experiments_dir / "results" / "good_results" / "random_forgetting_20percent_RL_tweak_conservative" / "RLcheckpoint.pth.tar",
            "description": "20% Random Data Forgetting"
        },
        "30percent": {
            "name": "random_forgetting_30percent_RL_tweak_conservative",
            "path": experiments_dir / "results" / "good_results" / "random_forgetting_30percent_RL_tweak_conservative" / "RLcheckpoint.pth.tar", 
            "description": "30% Random Data Forgetting"
        }
    }
    
    # Baseline model
    baseline_path = experiments_dir / "models" / "resnet50_pretrained.pth"
    
    print("🔍 Checking available models...")
    
    # Check baseline
    if not baseline_path.exists():
        print(f"❌ Baseline model not found: {baseline_path}")
        return False
    print(f"✅ Found baseline model: {baseline_path}")
    
    # Check all experiments
    available_experiments = {}
    for exp_key, exp_info in experiments.items():
        if exp_info["path"].exists():
            available_experiments[exp_key] = exp_info
            print(f"✅ Found {exp_info['description']}: {exp_info['path']}")
        else:
            print(f"❌ Missing {exp_info['description']}: {exp_info['path']}")
    
    if not available_experiments:
        print("❌ No experiment models found!")
        return False
    
    print(f"\n📊 Will analyze {len(available_experiments)} experiments")
    
    # Import the analyzer
    analyzer_path = project_root / "analysis/tools/real_influence_analyzer.py"
    spec = importlib.util.spec_from_file_location("real_influence_analyzer", analyzer_path)
    analyzer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer_module)
    RealInfluenceAnalyzer = analyzer_module.RealInfluenceAnalyzer
    
    # Results storage
    all_results = {}
    
    # Analyze each experiment
    for exp_key, exp_info in available_experiments.items():
        print(f"\n🎯 Analyzing {exp_info['description']}")
        print("=" * 50)
        
        # Set output directory for this experiment
        output_dir = project_root / "analysis" / "results" / "multi_experiment_analysis" / exp_key
        
        # Initialize analyzer for this experiment
        analyzer = RealInfluenceAnalyzer(output_dir=str(output_dir))
        
        try:
            # Run analysis
            results = analyzer.run_complete_analysis(
                baseline_path=str(baseline_path),
                unlearned_path=str(exp_info["path"]),
                top_k_layers=30,       # More layers for comparison
                top_k_channels=50,     # More channels for comparison
                top_k_weights=150      # More weights for comparison
            )
            
            # Store results with experiment info
            all_results[exp_key] = {
                "experiment_info": exp_info,
                "analysis_results": results,
                "output_directory": str(output_dir)
            }
            
            print(f"✅ {exp_info['description']} analysis complete!")
            
        except Exception as e:
            print(f"❌ {exp_info['description']} analysis failed: {e}")
            continue
    
    if not all_results:
        print("❌ No experiments completed successfully!")
        return False
    
    # Create comparison analysis
    print(f"\n📊 Creating Cross-Experiment Comparison")
    print("=" * 50)
    
    # Generate comparison summary
    comparison_summary = create_comparison_summary(all_results)
    
    # Save comparison results
    comparison_dir = project_root / "analysis" / "results" / "multi_experiment_analysis"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(comparison_dir / "experiment_comparison.json", 'w') as f:
        json.dump(comparison_summary, f, indent=2, default=str)
    
    # Create unified Lucent targets
    create_unified_lucent_targets(all_results, comparison_dir)
    
    # Print final summary
    print_final_summary(all_results, comparison_summary)
    
    return True

def create_comparison_summary(all_results):
    """Create a summary comparing all experiments"""
    
    comparison = {
        "analysis_info": {
            "total_experiments": len(all_results),
            "experiments_analyzed": list(all_results.keys()),
            "timestamp": __import__('pandas').Timestamp.now().isoformat()
        },
        "experiment_comparison": {},
        "cross_experiment_insights": {},
        "severity_ranking": []
    }
    
    severity_scores = []
    
    for exp_key, exp_data in all_results.items():
        results = exp_data["analysis_results"]
        top_layers = results["top_layers"]
        top_channels = results["top_channels"] 
        top_weights = results["top_weights"]
        
        # Calculate summary metrics
        exp_summary = {
            "experiment_name": exp_data["experiment_info"]["description"],
            "layers_analyzed": len(top_layers),
            "channels_analyzed": len(top_channels),
            "weights_analyzed": len(top_weights),
            "statistics": {
                "avg_layer_change": top_layers['mean_relative_change'].mean() if not top_layers.empty else 0,
                "max_layer_change": top_layers['mean_relative_change'].max() if not top_layers.empty else 0,
                "avg_channel_change": top_channels['mean_relative_change'].mean() if not top_channels.empty else 0,
                "max_channel_change": top_channels['mean_relative_change'].max() if not top_channels.empty else 0,
                "avg_weight_change": top_weights['relative_change'].mean() if not top_weights.empty else 0,
                "max_weight_change": top_weights['relative_change'].max() if not top_weights.empty else 0
            },
            "most_affected_components": {
                "top_layer": top_layers.iloc[0]['layer_name'] if not top_layers.empty else None,
                "top_channel": top_channels.iloc[0]['channel_id'] if not top_channels.empty else None,
                "top_weight": top_weights.iloc[0]['weight_id'] if not top_weights.empty else None
            }
        }
        
        comparison["experiment_comparison"][exp_key] = exp_summary
        
        # Calculate severity score for ranking
        severity_score = (
            exp_summary["statistics"]["avg_layer_change"] * 0.3 +
            exp_summary["statistics"]["avg_channel_change"] * 0.4 +
            exp_summary["statistics"]["avg_weight_change"] * 0.3
        )
        severity_scores.append((exp_key, severity_score, exp_summary["experiment_name"]))
    
    # Sort by severity
    severity_scores.sort(key=lambda x: x[1], reverse=True)
    comparison["severity_ranking"] = [
        {
            "rank": i + 1,
            "experiment": score[0],
            "name": score[2],
            "severity_score": score[1]
        }
        for i, score in enumerate(severity_scores)
    ]
    
    # Generate insights
    comparison["cross_experiment_insights"] = {
        "most_severe_experiment": severity_scores[0][0] if severity_scores else None,
        "least_severe_experiment": severity_scores[-1][0] if severity_scores else None,
        "severity_trend": "Increasing with forgetting ratio" if len(severity_scores) >= 2 and 
                         severity_scores[0][0] == "30percent" else "Variable",
        "common_affected_layers": find_common_affected_layers(all_results),
        "forgetting_ratio_impact": analyze_forgetting_ratio_impact(all_results)
    }
    
    return comparison

def find_common_affected_layers(all_results):
    """Find layers that are consistently affected across experiments"""
    
    layer_counts = {}
    
    for exp_key, exp_data in all_results.items():
        top_layers = exp_data["analysis_results"]["top_layers"]
        
        for _, layer in top_layers.head(10).iterrows():  # Top 10 from each
            layer_name = layer['layer_name']
            if layer_name not in layer_counts:
                layer_counts[layer_name] = 0
            layer_counts[layer_name] += 1
    
    # Find layers appearing in multiple experiments
    common_layers = [
        {"layer": layer, "appearances": count}
        for layer, count in layer_counts.items()
        if count > 1
    ]
    
    # Sort by appearances
    common_layers.sort(key=lambda x: x["appearances"], reverse=True)
    
    return common_layers[:10]  # Top 10 most common

def analyze_forgetting_ratio_impact(all_results):
    """Analyze how forgetting ratio affects influence patterns"""
    
    impact_analysis = {}
    
    # Extract ratios and their effects
    ratio_effects = []
    for exp_key, exp_data in all_results.items():
        ratio = int(exp_key.replace("percent", ""))
        results = exp_data["analysis_results"]
        
        avg_layer_effect = results["top_layers"]['mean_relative_change'].mean() if not results["top_layers"].empty else 0
        avg_channel_effect = results["top_channels"]['mean_relative_change'].mean() if not results["top_channels"].empty else 0
        
        ratio_effects.append({
            "ratio": ratio,
            "avg_layer_effect": avg_layer_effect,
            "avg_channel_effect": avg_channel_effect
        })
    
    # Sort by ratio
    ratio_effects.sort(key=lambda x: x["ratio"])
    
    impact_analysis = {
        "ratio_progression": ratio_effects,
        "layer_effect_trend": "increasing" if len(ratio_effects) >= 2 and 
                             ratio_effects[-1]["avg_layer_effect"] > ratio_effects[0]["avg_layer_effect"] else "variable",
        "channel_effect_trend": "increasing" if len(ratio_effects) >= 2 and 
                               ratio_effects[-1]["avg_channel_effect"] > ratio_effects[0]["avg_channel_effect"] else "variable"
    }
    
    return impact_analysis

def create_unified_lucent_targets(all_results, output_dir):
    """Create unified Lucent targets across all experiments"""
    
    unified_targets = {
        "unified_analysis": {
            "description": "Combined Lucent targets from all forgetting experiments",
            "experiments_included": list(all_results.keys()),
            "timestamp": __import__('pandas').Timestamp.now().isoformat()
        },
        "by_experiment": {},
        "top_cross_experiment_targets": [],
        "experiment_specific_targets": {}
    }
    
    all_targets = []
    
    # Collect targets from all experiments
    for exp_key, exp_data in all_results.items():
        targets = exp_data["analysis_results"]["lucent_targets"]
        
        # Store experiment-specific targets
        unified_targets["by_experiment"][exp_key] = {
            "experiment_name": exp_data["experiment_info"]["description"],
            "total_targets": len(targets),
            "top_10_targets": targets[:10]
        }
        
        # Add to global list with experiment info
        for target in targets:
            target_copy = target.copy()
            target_copy["source_experiment"] = exp_key
            target_copy["experiment_name"] = exp_data["experiment_info"]["description"]
            all_targets.append(target_copy)
    
    # Sort all targets by influence score
    all_targets.sort(key=lambda x: x["influence_score"], reverse=True)
    
    # Top targets across all experiments
    unified_targets["top_cross_experiment_targets"] = all_targets[:25]
    
    # Create experiment-specific high-impact targets
    for exp_key in all_results.keys():
        exp_targets = [t for t in all_targets if t["source_experiment"] == exp_key]
        unified_targets["experiment_specific_targets"][exp_key] = exp_targets[:15]
    
    # Save unified targets
    import json
    with open(output_dir / "unified_lucent_targets.json", 'w') as f:
        json.dump(unified_targets, f, indent=2, default=str)
    
    # Create unified Lucent commands
    create_unified_lucent_commands(unified_targets, output_dir)

def create_unified_lucent_commands(unified_targets, output_dir):
    """Create Lucent commands for all experiments"""
    
    with open(output_dir / "unified_lucent_commands.py", 'w') as f:
        f.write("# Unified Multi-Experiment Lucent Visualization Commands\n")
        f.write("# Generated from 10%, 20%, 30% forgetting analysis\n\n")
        f.write("from lucent.optvis import render\n")
        f.write("import matplotlib.pyplot as plt\n\n")
        
        # Top targets across all experiments
        f.write("# TOP TARGETS ACROSS ALL EXPERIMENTS\n")
        f.write("# These are the most influenced components from any experiment\n\n")
        
        top_targets = unified_targets["top_cross_experiment_targets"][:10]
        for i, target in enumerate(top_targets, 1):
            f.write(f'# {i}. {target["description"]} (from {target["experiment_name"]})\n')
            f.write(f'img_all_{i} = render.render_vis(model, "{target["target"]}", show_inline=True, thresholds=(512,))\n\n')
        
        # Experiment-specific sections
        for exp_key, exp_data in unified_targets["by_experiment"].items():
            f.write(f'\n# {exp_data["experiment_name"].upper()} SPECIFIC TARGETS\n')
            f.write(f'# Top targets specifically from {exp_data["experiment_name"]}\n\n')
            
            for i, target in enumerate(exp_data["top_10_targets"][:5], 1):
                f.write(f'# {exp_key}_{i}. {target["description"]}\n')
                f.write(f'img_{exp_key}_{i} = render.render_vis(model, "{target["target"]}", show_inline=True, thresholds=(512,))\n\n')
        
        # Batch comparison visualization
        f.write('''\n# BATCH COMPARISON VISUALIZATION
# Compare top targets from each experiment side by side

def compare_experiments():
    experiments = {
''')
        
        for exp_key, exp_data in unified_targets["by_experiment"].items():
            targets = [t["target"] for t in exp_data["top_10_targets"][:3]]
            f.write(f'        "{exp_data["experiment_name"]}": {targets},\n')
        
        f.write('''    }
    
    fig, axes = plt.subplots(len(experiments), 3, figsize=(15, 5*len(experiments)))
    if len(experiments) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (exp_name, targets) in enumerate(experiments.items()):
        for j, target in enumerate(targets):
            img = render.render_vis(model, target, show_inline=False, thresholds=(256,))
            if hasattr(img, 'cpu'):
                img_np = img.cpu().numpy().transpose(1, 2, 0)
            else:
                img_np = np.array(img)
            axes[i, j].imshow(img_np)
            axes[i, j].set_title(f"{exp_name}\\n{target}", fontsize=10)
            axes[i, j].axis('off')
    
    plt.suptitle('Multi-Experiment Comparison: Most Influenced Components', fontsize=16)
    plt.tight_layout()
    plt.show()

# Run the comparison
compare_experiments()
''')

def print_final_summary(all_results, comparison_summary):
    """Print comprehensive final summary"""
    
    print("\n🎉 Multi-Experiment Analysis Complete!")
    print("=" * 60)
    
    print("📊 EXPERIMENTS ANALYZED:")
    for exp_key, exp_data in all_results.items():
        print(f"   ✅ {exp_data['experiment_info']['description']}")
        results = exp_data["analysis_results"]
        print(f"      📈 Layers: {len(results['top_layers'])}, Channels: {len(results['top_channels'])}, Weights: {len(results['top_weights'])}")
    
    print(f"\n🏆 SEVERITY RANKING:")
    for rank_info in comparison_summary["severity_ranking"]:
        print(f"   {rank_info['rank']}. {rank_info['name']} (Score: {rank_info['severity_score']:.2f})")
    
    print(f"\n🎯 CROSS-EXPERIMENT INSIGHTS:")
    insights = comparison_summary["cross_experiment_insights"]
    print(f"   • Most severe: {insights['most_severe_experiment']}")
    print(f"   • Least severe: {insights['least_severe_experiment']}") 
    print(f"   • Severity trend: {insights['severity_trend']}")
    print(f"   • Common affected layers: {len(insights['common_affected_layers'])}")
    
    print(f"\n📄 FILES GENERATED:")
    base_dir = Path(__file__).parent.parent / "analysis" / "results" / "multi_experiment_analysis"
    print(f"   📁 Main directory: {base_dir}")
    print(f"   📊 experiment_comparison.json - Cross-experiment comparison")
    print(f"   🎨 unified_lucent_targets.json - All Lucent targets")
    print(f"   🐍 unified_lucent_commands.py - Ready-to-use Colab commands")
    
    for exp_key in all_results.keys():
        print(f"   📂 {exp_key}/ - Individual experiment results")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Upload unified_lucent_targets.json to Google Drive")
    print(f"   2. Use unified_lucent_commands.py in Google Colab")
    print(f"   3. Compare experiment_comparison.json for insights")
    print(f"   4. Analyze individual experiment directories for details")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)