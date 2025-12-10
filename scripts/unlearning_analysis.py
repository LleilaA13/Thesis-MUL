#!/usr/bin/env python3
#
# unlearning_analyzer.py
#
# This script performs a deterministic analysis of the structural impact of machine
# unlearning on a neural network. It compares a baseline model to one or more
# unlearned models to quantify changes at the layer, channel, and weight levels.
#
# It is designed to be fully reproducible by using stable sorting algorithms
# and deterministic tie-breaking for all comparisons.
#
# The script generates detailed quantitative reports (CSV, JSON) and Python
# scripts for high-resolution feature visualization with Lucent.
#

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Any
import torchvision.models as models

# --- Script Configuration ---

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "analysis/results/unlearning_analysis"

EXPERIMENTS = {
    "10percent": {
        "name": "10% Random Forgetting",
        "path": EXPERIMENTS_DIR / "results/good_results/random_forgetting_10percent_RL_tweak_conservative/RLcheckpoint.pth.tar",
    },
    "20percent": {
        "name": "20% Random Forgetting",
        "path": EXPERIMENTS_DIR / "results/good_results/random_forgetting_20percent_RL_tweak_conservative/RLcheckpoint.pth.tar",
    },
    "30percent": {
        "name": "30% Random Forgetting",
        "path": EXPERIMENTS_DIR / "results/good_results/random_forgetting_30percent_RL_tweak_conservative/RLcheckpoint.pth.tar",
    },
}

BASELINE_MODEL_PATH = EXPERIMENTS_DIR / "models/resnet50_pretrained.pth"


class DeterministicInfluenceAnalyzer:
    """
    Analyzes model weights to find the most influenced components with guaranteed
    reproducibility.
    """
    
    def __init__(self):
        self.baseline_model = None
        print("Unlearning Influence Analyzer Initialized")

    def load_model(self, path: Path, num_classes: int = 200) -> nn.Module:
        """Loads a single ResNet-50 model from a checkpoint."""
        print(f"  -> Loading model from: {path}")
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        state_dict.pop('normalize.mean', None)
        state_dict.pop('normalize.std', None)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        return model

    def run_analysis(self, baseline_model: nn.Module, unlearned_model: nn.Module) -> Dict[str, pd.DataFrame]:
        """Performs the full hierarchical analysis for one model pair."""
        print("    -> Analyzing layers, channels, and weights...")
        top_layers = self._analyze_layers(baseline_model, unlearned_model)
        top_channels = self._analyze_channels(baseline_model, unlearned_model, top_layers)
        top_weights = self._analyze_weights(baseline_model, unlearned_model, top_channels)
        print("    -> Analysis complete for this pair.")
        return {"layers": top_layers, "channels": top_channels, "weights": top_weights}

    def _analyze_layers(self, m1: nn.Module, m2: nn.Module) -> pd.DataFrame:
        """Analyzes layer influences with a deterministic stable sort."""
        layer_data = []
        p1 = dict(m1.named_parameters())
        p2 = dict(m2.named_parameters())

        for name, w1 in p1.items():
            w2 = p2[name]
            diff = torch.abs(w1.data - w2.data)
            rel_diff = torch.where(torch.abs(w1.data) > 1e-8, diff / torch.abs(w1.data), diff)
            layer_data.append({
                'layer_name': name,
                'mean_absolute_change': diff.mean().item(),
                'mean_relative_change': rel_diff.mean().item(),
            })
        
        df = pd.DataFrame(layer_data)
        return df.sort_values('mean_relative_change', ascending=False, kind='stable')

    def _analyze_channels(self, m1: nn.Module, m2: nn.Module, top_layers: pd.DataFrame) -> pd.DataFrame:
        """Analyzes channel influences with a deterministic stable sort."""
        channel_data = []
        p1 = dict(m1.named_parameters())
        p2 = dict(m2.named_parameters())

        conv_layers = top_layers[top_layers['layer_name'].str.contains('conv') & 
                                 top_layers['layer_name'].str.contains('weight')].head(15)

        for _, layer_info in conv_layers.iterrows():
            name = layer_info['layer_name']
            w1, w2 = p1[name].data, p2[name].data
            if w1.dim() != 4: continue

            for ch_idx in range(w1.shape[0]):
                ch1, ch2 = w1[ch_idx], w2[ch_idx]
                diff = torch.abs(ch1 - ch2)
                rel_diff = torch.where(torch.abs(ch1) > 1e-8, diff / torch.abs(ch1), diff)
                channel_data.append({
                    'layer_name': name,
                    'channel_index': ch_idx,
                    'channel_id': f"{name}_ch{ch_idx}",
                    'mean_relative_change': rel_diff.mean().item(),
                })
        
        if not channel_data: return pd.DataFrame()
        df = pd.DataFrame(channel_data)
        return df.sort_values('mean_relative_change', ascending=False, kind='stable')

    def _analyze_weights(self, m1: nn.Module, m2: nn.Module, top_channels: pd.DataFrame) -> pd.DataFrame:
        """Analyzes weight influences with a deterministic stable sort."""
        weight_data = []
        p1 = dict(m1.named_parameters())
        p2 = dict(m2.named_parameters())
        
        for _, ch_info in top_channels.head(20).iterrows():
            name, ch_idx = ch_info['layer_name'], ch_info['channel_index']
            w1, w2 = p1[name].data, p2[name].data
            if w1.dim() != 4: continue

            ch1, ch2 = w1[ch_idx].flatten(), w2[ch_idx].flatten()
            diff = torch.abs(ch1 - ch2)
            rel_diff = torch.where(torch.abs(ch1) > 1e-8, diff / torch.abs(ch1), diff)
            
            top_indices = torch.topk(rel_diff, min(10, len(rel_diff))).indices
            for weight_idx in top_indices:
                weight_data.append({
                    'channel_id': ch_info['channel_id'],
                    'weight_index': weight_idx.item(),
                    'weight_id': f"{ch_info['channel_id']}_w{weight_idx.item()}",
                    'relative_change': rel_diff[weight_idx].item(),
                })

        if not weight_data: return pd.DataFrame()
        df = pd.DataFrame(weight_data)
        return df.sort_values('relative_change', ascending=False, kind='stable')


def create_individual_summary(results: Dict[str, pd.DataFrame], baseline_model: nn.Module) -> Dict:
    """Creates a detailed summary for a single experiment, matching the old format."""
    layers = results['layers']
    channels = results['channels']
    weights = results['weights']

    model_info = {
        "architecture": "ResNet50",
        "total_parameters": sum(p.numel() for p in baseline_model.parameters()),
        "analyzed_layers": len(layers)
    }

    top_influences = {}
    if not layers.empty:
        top_layer = layers.iloc[0]
        top_influences["most_influenced_layer"] = top_layer['layer_name']
        top_influences["max_layer_change"] = top_layer['mean_relative_change']
    
    if not channels.empty:
        top_channel = channels.iloc[0]
        top_influences["most_influenced_channel"] = top_channel['channel_id']
        top_influences["max_channel_change"] = top_channel['mean_relative_change']
        
    if not weights.empty:
        top_weight = weights.iloc[0]
        top_influences["most_influenced_weight"] = top_weight.get('weight_id', 'N/A')
        top_influences["max_weight_change"] = top_weight['relative_change']
        
    statistics = {
        "layers_analyzed": len(layers),
        "channels_analyzed": len(channels),
        "weights_analyzed": len(weights),
        "avg_layer_change": layers['mean_relative_change'].mean() if not layers.empty else 0,
        "avg_channel_change": channels['mean_relative_change'].mean() if not channels.empty else 0,
        "avg_weight_change": weights['relative_change'].mean() if not weights.empty else 0
    }
    
    summary = {
        "analysis_type": "real_weight_comparison",
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_info": model_info,
        "top_influences": top_influences,
        "statistics": statistics
    }
    
    return summary


def create_comparison_summary(all_results: Dict) -> Dict:
    """Creates a summary comparing all experiments."""
    summary = {"statistics": {}, "severity_ranking": [], "cross_experiment_insights": {}}
    layer_counts = defaultdict(int)
    
    for exp_key, results in all_results.items():
        layers = results['layers']
        channels = results['channels']
        
        stats = {
            'avg_layer_change': layers['mean_relative_change'].mean(),
            'max_layer_change': layers['mean_relative_change'].max(),
            'avg_channel_change': channels['mean_relative_change'].mean() if not channels.empty else 0,
        }
        summary["statistics"][exp_key] = stats
        
        for layer_name in layers.head(30)['layer_name']:
            layer_counts[layer_name] += 1

    severity = {k: v['avg_layer_change'] + v['avg_channel_change'] for k, v in summary["statistics"].items()}
    sorted_severity = sorted(severity.items(), key=lambda item: item[1], reverse=True)
    
    for i, (exp_key, score) in enumerate(sorted_severity):
        summary["severity_ranking"].append({
            "rank": i + 1,
            "name": EXPERIMENTS[exp_key]["name"],
            "severity_score": score
        })

    sorted_layers = sorted(layer_counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    
    summary["cross_experiment_insights"] = {
        "most_severe_experiment": sorted_severity[0][0],
        "least_severe_experiment": sorted_severity[-1][0],
        "common_affected_layers": [
            {"layer": name, "count": count} for name, count in sorted_layers if count > 1
        ][:10]
    }
    return summary

def generate_lucent_files(all_results: Dict, output_dir: Path):
    """Generates Lucent files with high-resolution rendering settings."""
    print("\nGenerating Lucent visualization files with high-resolution settings...")
    
    hq_settings_py_code = """
# Define high-resolution, high-quality rendering parameters
from lucent.optvis import param, transform, objectives
IMG_SIZE = 300
transforms = [
    transform.pad(16, mode='constant', constant_value=.5),
    transform.jitter(8),
    transform.random_scale([1 + (i - 5) / 50. for i in range(11)]),
    transform.random_rotate(list(range(-10, 11)) + 5 * [0]),
    transform.jitter(4),
    transform.crop_or_pad_to(IMG_SIZE, IMG_SIZE)
]
param_f = lambda: param.image(IMG_SIZE, batch=1, decorrelate=True)
"""

    for exp_key, results in all_results.items():
        exp_output_dir = output_dir / exp_key
        channels = results.get('channels')
        if channels is None or channels.empty: continue

        individual_targets = []
        for _, channel in channels.head(15).iterrows():
            layer, ch_idx = channel['layer_name'], channel['channel_index']
            lucent_layer = layer.replace('.', '_').replace('[', '_').replace(']', '').replace('_weight', '')
            target_str = f"{lucent_layer}:{ch_idx}"
            # --- FIX STARTS HERE ---
            # Added the 'type': 'channel' key to match the notebook's expectation.
            individual_targets.append({
                "target": target_str,
                "type": "channel" 
            })
            # --- FIX ENDS HERE ---
        
        (exp_output_dir / f"lucent_targets_{exp_key}.json").write_text(json.dumps(individual_targets, indent=2))

        py_commands = f"# Lucent Commands for the {exp_key} Experiment\n\n"
        py_commands += "import matplotlib.pyplot as plt\nfrom lucent.optvis import render\n"
        py_commands += hq_settings_py_code
        py_commands += f"targets = {[t['target'] for t in individual_targets]}\n\n"
        py_commands += f"""
# Visualize top 15 targets for this experiment
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
fig.suptitle(f'Top 15 Most Affected Features for the {{exp_key}} Experiment', fontsize=18)
for ax, target in zip(axes.flatten(), targets):
    img = render.render_vis(unlearned_model, target, param_f=param_f, transforms=transforms, thresholds=(2048,))
    ax.imshow(img[0][0])
    ax.set_title(target, fontsize=10)
    ax.axis('off')
for i in range(len(targets), len(axes.flatten())):
    axes.flatten()[i].axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
"""
        (exp_output_dir / f"lucent_commands_{exp_key}.py").write_text(py_commands)
        print(f"  -> Individual HR Lucent file created for {exp_key}")

    most_severe_exp_key = sorted(severity.items(), key=lambda item: item[1], reverse=True)[0][0]
    top_channels_from_severe_exp = all_results[most_severe_exp_key]['channels']
    if not top_channels_from_severe_exp.empty:
        top_channel = top_channels_from_severe_exp.iloc[0]
        layer, ch_idx = top_channel['layer_name'], top_channel['channel_index']
        lucent_layer = layer.replace('.', '_').replace('[', '_').replace(']', '').replace('_weight', '')
        target_to_compare = f"{lucent_layer}:{ch_idx}"
    else:
        target_to_compare = "layer4_2_conv3:100"

    unified_py_commands = "# Unified Lucent Commands for Cross-Experiment Comparison\n\n"
    unified_py_commands += "import matplotlib.pyplot as plt\nfrom lucent.optvis import render\n"
    unified_py_commands += hq_settings_py_code
    unified_py_commands += "# models = {'Baseline': model_baseline, '10% Forget': model_10, ...}\n\n"
    unified_py_commands += f"target_to_compare = '{target_to_compare}'\n\n"
    unified_py_commands += """
# Plot one target across all models to see the trend
fig, axes = plt.subplots(1, len(models), figsize=(20, 5))
fig.suptitle(f'Visualization for: {target_to_compare}', fontsize=16)
for ax, (name, model) in zip(axes, models.items()):
    img = render.render_vis(model, target_to_compare, param_f=param_f, transforms=transforms, thresholds=(2048,))
    ax.imshow(img[0][0])
    ax.set_title(name)
    ax.axis('off')
plt.show()
"""
    (output_dir / "unified_lucent_commands.py").write_text(unified_py_commands)
    print("  -> Unified HR Lucent comparison file created.")


def main():
    """Main function to run the complete deterministic analysis."""
    global severity
    
    print("Starting Deterministic Multi-Experiment Impact Analysis")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analyzer = DeterministicInfluenceAnalyzer()

    baseline_model = analyzer.load_model(BASELINE_MODEL_PATH)

    all_results = {}
    for exp_key, exp_info in EXPERIMENTS.items():
        print(f"\nAnalyzing Experiment: {exp_info['name']}")
        if not exp_info['path'].exists():
            print(f"  -> ERROR: Model file not found, skipping: {exp_info['path']}")
            continue
        
        unlearned_model = analyzer.load_model(exp_info['path'])
        results = analyzer.run_analysis(baseline_model, unlearned_model)
        all_results[exp_key] = results
        
        exp_dir = OUTPUT_DIR / exp_key
        exp_dir.mkdir(exist_ok=True)
        
        for name, df in results.items():
            df.to_csv(exp_dir / f"{name}.csv", index=False)
        print(f"  -> Individual CSV results saved to: {exp_dir}")

        individual_summary = create_individual_summary(results, baseline_model)
        summary_path = exp_dir / "influence_summary.json"
        summary_path.write_text(json.dumps(individual_summary, indent=2))
        print(f"  -> Individual influence summary saved to: {summary_path}")


    if not all_results:
        print("\nNo experiments were analyzed. Please check model paths.")
        return

    print("\nCreating cross-experiment comparison summary...")
    comparison_summary = create_comparison_summary(all_results)
    (OUTPUT_DIR / "experiment_comparison.json").write_text(json.dumps(comparison_summary, indent=2))
    print("  -> Comparison summary saved.")
    
    severity = {k: v['avg_layer_change'] + v['avg_channel_change'] for k, v in comparison_summary["statistics"].items()}

    generate_lucent_files(all_results, OUTPUT_DIR)
    
    print("\n\nAnalysis Complete!")
    print("-" * 60)
    print("SEVERITY RANKING (Most to Least Impactful):")
    for rank_info in comparison_summary["severity_ranking"]:
        print(f"  {rank_info['rank']}. {rank_info['name']} (Score: {rank_info['severity_score']:.4f})")
    
    print("\nTOP 5 COMMON AFFECTED LAYERS:")
    for layer_info in comparison_summary["cross_experiment_insights"]["common_affected_layers"][:5]:
        print(f"  - {layer_info['layer']} (Found in {layer_info['count']} experiments)")
        
    print(f"\nAll results have been saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()

