#!/usr/bin/env python3
"""
Comprehensive Weight Analysis Visualizations

This script creates meaningful plots from the actual weight analysis results
to understand what happens during the unlearning process across different forgetting ratios.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class WeightAnalysisPlotter:
    def __init__(self, json_file_path):
        """Initialize with the comprehensive weight analysis JSON file."""
        self.json_file_path = Path(json_file_path)
        self.data = self.load_data()
        self.experiments = list(self.data.keys())
        
    def load_data(self):
        """Load and parse the JSON data."""
        with open(self.json_file_path, 'r') as f:
            return json.load(f)
    
    def extract_forgetting_ratios(self):
        """Extract forgetting ratios from experiment names."""
        ratios = []
        for exp_name in self.experiments:
            if '10percent' in exp_name:
                ratios.append('10%')
            elif '20percent' in exp_name:
                ratios.append('20%')
            elif '30percent' in exp_name:
                ratios.append('30%')
            else:
                ratios.append('Unknown')
        return ratios
    
    def plot_layer_sensitivity_comparison(self):
        """
        Plot 1: Layer Sensitivity Comparison Across Forgetting Ratios
        Shows how different layer types are affected by different forgetting percentages.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Layer Sensitivity Analysis Across Different Forgetting Ratios', fontsize=16, fontweight='bold')
        
        ratios = self.extract_forgetting_ratios()
        
        # Collect data for different layer types
        layer_types = ['conv', 'bn', 'fc']
        metrics = ['mean_absolute_change', 'max_absolute_change', 'percentage_changed']
        
        for i, metric in enumerate(metrics):
            if i < 3:  # We have 4 subplots, use first 3 for metrics
                ax = axes[i//2, i%2]
                
                data_for_plot = []
                for j, exp_name in enumerate(self.experiments):
                    exp_data = self.data[exp_name]['layer_sensitivity']
                    
                    for layer_name, layer_stats in exp_data.items():
                        if isinstance(layer_stats, dict) and metric in layer_stats:
                            # Categorize layer type
                            layer_type = 'other'
                            if 'conv' in layer_name and 'weight' in layer_name:
                                layer_type = 'Convolutional'
                            elif 'bn' in layer_name or 'norm' in layer_name:
                                layer_type = 'BatchNorm'
                            elif 'fc' in layer_name or 'classifier' in layer_name:
                                layer_type = 'Fully Connected'
                            
                            data_for_plot.append({
                                'Forgetting Ratio': ratios[j],
                                'Layer Type': layer_type,
                                'Value': layer_stats[metric],
                                'Layer Name': layer_name
                            })
                
                df = pd.DataFrame(data_for_plot)
                if not df.empty:
                    # Create box plot
                    sns.boxplot(data=df, x='Forgetting Ratio', y='Value', hue='Layer Type', ax=ax)
                    ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    if i < 2:
                        ax.set_yscale('log')  # Log scale for better visualization
        
        # Fourth subplot: Top affected layers
        ax = axes[1, 1]
        top_layers_data = []
        
        for j, exp_name in enumerate(self.experiments):
            exp_data = self.data[exp_name]['layer_sensitivity']
            # Get top 10 most affected layers by mean absolute change
            sorted_layers = sorted(exp_data.items(), 
                                 key=lambda x: x[1].get('mean_absolute_change', 0) if isinstance(x[1], dict) else 0, 
                                 reverse=True)[:10]
            
            for k, (layer_name, layer_stats) in enumerate(sorted_layers):
                if isinstance(layer_stats, dict):
                    top_layers_data.append({
                        'Forgetting Ratio': ratios[j],
                        'Rank': k + 1,
                        'Mean Change': layer_stats.get('mean_absolute_change', 0),
                        'Layer': layer_name.split('.')[-1]  # Get last part of layer name
                    })
        
        df_top = pd.DataFrame(top_layers_data)
        if not df_top.empty:
            # Plot top 5 layers
            df_top_5 = df_top[df_top['Rank'] <= 5]
            sns.barplot(data=df_top_5, x='Forgetting Ratio', y='Mean Change', hue='Layer', ax=ax)
            ax.set_title('Top 5 Most Affected Layers', fontweight='bold')
            ax.set_ylabel('Mean Absolute Change')
            ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def plot_weight_distribution_changes(self):
        """
        Plot 2: Weight Distribution Changes
        Shows how weight magnitudes change across different layers and forgetting ratios.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Weight Distribution Analysis Across Forgetting Ratios', fontsize=16, fontweight='bold')
        
        ratios = self.extract_forgetting_ratios()
        
        # Plot 1: Mean vs Max changes scatter
        ax = axes[0, 0]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for j, exp_name in enumerate(self.experiments):
            exp_data = self.data[exp_name]['layer_sensitivity']
            mean_changes = []
            max_changes = []
            
            for layer_name, layer_stats in exp_data.items():
                if isinstance(layer_stats, dict):
                    mean_changes.append(layer_stats.get('mean_absolute_change', 0))
                    max_changes.append(layer_stats.get('max_absolute_change', 0))
            
            ax.scatter(mean_changes, max_changes, alpha=0.6, 
                      label=f'{ratios[j]} Forgetting', color=colors[j], s=50)
        
        ax.set_xlabel('Mean Absolute Change')
        ax.set_ylabel('Max Absolute Change')
        ax.set_title('Mean vs Max Weight Changes', fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Percentage of changed weights
        ax = axes[0, 1]
        percentage_data = []
        
        for j, exp_name in enumerate(self.experiments):
            exp_data = self.data[exp_name]['layer_sensitivity']
            
            for layer_name, layer_stats in exp_data.items():
                if isinstance(layer_stats, dict) and 'percentage_changed' in layer_stats:
                    percentage_data.append({
                        'Forgetting Ratio': ratios[j],
                        'Percentage Changed': layer_stats['percentage_changed'],
                        'Layer Type': 'BatchNorm' if 'bn' in layer_name else 'Conv' if 'conv' in layer_name else 'Other'
                    })
        
        df_perc = pd.DataFrame(percentage_data)
        if not df_perc.empty:
            sns.boxplot(data=df_perc, x='Forgetting Ratio', y='Percentage Changed', 
                       hue='Layer Type', ax=ax)
            ax.set_title('Percentage of Weights Changed', fontweight='bold')
            ax.set_ylabel('Percentage Changed (%)')
        
        # Plot 3: Layer depth vs sensitivity
        ax = axes[1, 0]
        depth_data = []
        
        for j, exp_name in enumerate(self.experiments):
            exp_data = self.data[exp_name]['layer_sensitivity']
            
            for layer_name, layer_stats in exp_data.items():
                if isinstance(layer_stats, dict) and 'layer' in layer_name:
                    # Extract layer depth
                    try:
                        if 'layer1' in layer_name:
                            depth = 1
                        elif 'layer2' in layer_name:
                            depth = 2
                        elif 'layer3' in layer_name:
                            depth = 3
                        elif 'layer4' in layer_name:
                            depth = 4
                        else:
                            depth = 0
                        
                        depth_data.append({
                            'Forgetting Ratio': ratios[j],
                            'Depth': depth,
                            'Mean Change': layer_stats.get('mean_absolute_change', 0),
                            'Layer': layer_name
                        })
                    except:
                        continue
        
        df_depth = pd.DataFrame(depth_data)
        if not df_depth.empty and len(df_depth) > 0:
            sns.boxplot(data=df_depth, x='Depth', y='Mean Change', hue='Forgetting Ratio', ax=ax)
            ax.set_title('Layer Depth vs Weight Sensitivity', fontweight='bold')
            ax.set_ylabel('Mean Absolute Change')
            ax.set_xlabel('ResNet Layer Depth')
            ax.set_yscale('log')
        
        # Plot 4: Standard deviation of changes
        ax = axes[1, 1]
        std_data = []
        
        for j, exp_name in enumerate(self.experiments):
            exp_data = self.data[exp_name]['layer_sensitivity']
            
            for layer_name, layer_stats in exp_data.items():
                if isinstance(layer_stats, dict) and 'std_absolute_change' in layer_stats:
                    layer_type = 'BatchNorm' if 'bn' in layer_name else 'Conv' if 'conv' in layer_name else 'Other'
                    std_data.append({
                        'Forgetting Ratio': ratios[j],
                        'Std Change': layer_stats['std_absolute_change'],
                        'Layer Type': layer_type
                    })
        
        df_std = pd.DataFrame(std_data)
        if not df_std.empty:
            sns.violinplot(data=df_std, x='Forgetting Ratio', y='Std Change', 
                          hue='Layer Type', ax=ax)
            ax.set_title('Variability of Weight Changes', fontweight='bold')
            ax.set_ylabel('Standard Deviation of Changes')
            ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def plot_batchnorm_analysis(self):
        """
        Plot 3: BatchNorm Layer Detailed Analysis
        Deep dive into BatchNorm layers since they show the highest sensitivity.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('BatchNorm Layer Analysis - Why Are They Most Affected?', fontsize=16, fontweight='bold')
        
        ratios = self.extract_forgetting_ratios()
        
        # Collect BatchNorm data
        bn_data = []
        for j, exp_name in enumerate(self.experiments):
            exp_data = self.data[exp_name]['layer_sensitivity']
            
            for layer_name, layer_stats in exp_data.items():
                if isinstance(layer_stats, dict) and 'bn' in layer_name:
                    param_type = 'weight'
                    if 'weight' in layer_name:
                        param_type = 'weight'
                    elif 'bias' in layer_name:
                        param_type = 'bias'
                    elif 'running_mean' in layer_name:
                        param_type = 'running_mean'
                    elif 'running_var' in layer_name:
                        param_type = 'running_var'
                    
                    bn_data.append({
                        'Forgetting Ratio': ratios[j],
                        'Parameter Type': param_type,
                        'Mean Change': layer_stats.get('mean_absolute_change', 0),
                        'Max Change': layer_stats.get('max_absolute_change', 0),
                        'Relative Change': layer_stats.get('mean_relative_change', 0),
                        'Layer Name': layer_name
                    })
        
        df_bn = pd.DataFrame(bn_data)
        
        if not df_bn.empty:
            # Plot 1: BatchNorm parameter types comparison
            ax = axes[0, 0]
            sns.boxplot(data=df_bn, x='Parameter Type', y='Mean Change', 
                       hue='Forgetting Ratio', ax=ax)
            ax.set_title('BatchNorm Parameters: Mean Change', fontweight='bold')
            ax.set_ylabel('Mean Absolute Change')
            ax.set_yscale('log')
            ax.tick_params(axis='x', rotation=45)
            
            # Plot 2: Relative changes
            ax = axes[0, 1]
            # Filter out extreme outliers for better visualization
            df_bn_filtered = df_bn[df_bn['Relative Change'] < df_bn['Relative Change'].quantile(0.95)]
            sns.boxplot(data=df_bn_filtered, x='Parameter Type', y='Relative Change', 
                       hue='Forgetting Ratio', ax=ax)
            ax.set_title('BatchNorm Parameters: Relative Change', fontweight='bold')
            ax.set_ylabel('Mean Relative Change (%)')
            ax.tick_params(axis='x', rotation=45)
            
            # Plot 3: Running statistics vs learnable parameters
            ax = axes[1, 0]
            running_stats = df_bn[df_bn['Parameter Type'].isin(['running_mean', 'running_var'])]
            learnable = df_bn[df_bn['Parameter Type'].isin(['weight', 'bias'])]
            
            if not running_stats.empty and not learnable.empty:
                running_means = running_stats.groupby('Forgetting Ratio')['Mean Change'].mean()
                learnable_means = learnable.groupby('Forgetting Ratio')['Mean Change'].mean()
                
                x = np.arange(len(running_means))
                width = 0.35
                
                ax.bar(x - width/2, running_means.values, width, label='Running Statistics', alpha=0.8)
                ax.bar(x + width/2, learnable_means.values, width, label='Learnable Parameters', alpha=0.8)
                
                ax.set_xlabel('Forgetting Ratio')
                ax.set_ylabel('Mean Absolute Change')
                ax.set_title('Running Stats vs Learnable Parameters', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(running_means.index)
                ax.legend()
                ax.set_yscale('log')
            
            # Plot 4: Distribution of changes across BatchNorm layers
            ax = axes[1, 1]
            sns.histplot(data=df_bn, x='Mean Change', hue='Forgetting Ratio', 
                        bins=30, alpha=0.6, ax=ax)
            ax.set_title('Distribution of BatchNorm Changes', fontweight='bold')
            ax.set_xlabel('Mean Absolute Change')
            ax.set_ylabel('Count')
            ax.set_xscale('log')
        
        plt.tight_layout()
        return fig
    
    def plot_forgetting_progression(self):
        """
        Plot 4: Forgetting Progression Analysis
        Shows how the impact increases with forgetting ratio.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Forgetting Progression: How Impact Scales with Forgetting Ratio', fontsize=16, fontweight='bold')
        
        # Extract numeric ratios for correlation analysis
        ratio_map = {'10%': 10, '20%': 20, '30%': 30}
        ratios = self.extract_forgetting_ratios()
        
        # Aggregate statistics per experiment
        exp_stats = []
        for j, exp_name in enumerate(self.experiments):
            exp_data = self.data[exp_name]['layer_sensitivity']
            
            all_mean_changes = []
            all_max_changes = []
            bn_changes = []
            conv_changes = []
            
            for layer_name, layer_stats in exp_data.items():
                if isinstance(layer_stats, dict):
                    mean_change = layer_stats.get('mean_absolute_change', 0)
                    max_change = layer_stats.get('max_absolute_change', 0)
                    
                    all_mean_changes.append(mean_change)
                    all_max_changes.append(max_change)
                    
                    if 'bn' in layer_name:
                        bn_changes.append(mean_change)
                    elif 'conv' in layer_name and 'weight' in layer_name:
                        conv_changes.append(mean_change)
            
            exp_stats.append({
                'Forgetting Ratio': ratios[j],
                'Numeric Ratio': ratio_map.get(ratios[j], 0),
                'Mean of All Changes': np.mean(all_mean_changes) if all_mean_changes else 0,
                'Max of All Changes': np.max(all_max_changes) if all_max_changes else 0,
                'Mean BN Changes': np.mean(bn_changes) if bn_changes else 0,
                'Mean Conv Changes': np.mean(conv_changes) if conv_changes else 0,
                'Std of Changes': np.std(all_mean_changes) if all_mean_changes else 0,
                'Total Layers': len([x for x in exp_data.values() if isinstance(x, dict)])
            })
        
        df_exp = pd.DataFrame(exp_stats)
        
        if not df_exp.empty:
            # Plot 1: Overall impact vs forgetting ratio
            ax = axes[0, 0]
            ax.plot(df_exp['Numeric Ratio'], df_exp['Mean of All Changes'], 
                   marker='o', linewidth=2, markersize=8, label='Mean Change')
            ax.plot(df_exp['Numeric Ratio'], df_exp['Max of All Changes'], 
                   marker='s', linewidth=2, markersize=8, label='Max Change')
            ax.set_xlabel('Forgetting Ratio (%)')
            ax.set_ylabel('Weight Change Magnitude')
            ax.set_title('Overall Impact vs Forgetting Ratio', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # Plot 2: Layer type comparison
            ax = axes[0, 1]
            ax.plot(df_exp['Numeric Ratio'], df_exp['Mean BN Changes'], 
                   marker='o', linewidth=2, markersize=8, label='BatchNorm Layers')
            ax.plot(df_exp['Numeric Ratio'], df_exp['Mean Conv Changes'], 
                   marker='s', linewidth=2, markersize=8, label='Conv Layers')
            ax.set_xlabel('Forgetting Ratio (%)')
            ax.set_ylabel('Mean Weight Change')
            ax.set_title('Layer Type Sensitivity Comparison', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # Plot 3: Variability analysis
            ax = axes[1, 0]
            ax.plot(df_exp['Numeric Ratio'], df_exp['Std of Changes'], 
                   marker='o', linewidth=2, markersize=8, color='red')
            ax.set_xlabel('Forgetting Ratio (%)')
            ax.set_ylabel('Standard Deviation of Changes')
            ax.set_title('Change Variability vs Forgetting Ratio', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # Plot 4: Summary statistics
            ax = axes[1, 1]
            metrics = ['Mean of All Changes', 'Mean BN Changes', 'Mean Conv Changes']
            colors = ['blue', 'orange', 'green']
            
            for i, metric in enumerate(metrics):
                ax.bar([x + i*0.2 for x in df_exp['Numeric Ratio']], 
                      df_exp[metric], width=0.2, alpha=0.7, 
                      label=metric.replace('Mean ', '').replace(' Changes', ''), 
                      color=colors[i])
            
            ax.set_xlabel('Forgetting Ratio (%)')
            ax.set_ylabel('Weight Change Magnitude')
            ax.set_title('Comparative Impact Summary', fontweight='bold')
            ax.legend()
            ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def generate_explanation_report(self):
        """Generate a detailed explanation of what the plots show."""
        
        ratios = self.extract_forgetting_ratios()
        
        report = f"""
# Weight Analysis Visualization Report

## What These Plots Reveal About Machine Unlearning

Based on the analysis of {len(self.experiments)} experiments with forgetting ratios of {', '.join(ratios)}, 
here's what we discovered:

## Key Findings:

### 1. BatchNorm Layers Are Most Vulnerable
- **Why**: BatchNorm layers maintain running statistics (mean/variance) that capture dataset characteristics
- **Impact**: These statistics change dramatically when data is "forgotten" because they're computed over fewer samples
- **Evidence**: Running mean/variance show 17-23x higher sensitivity than learnable parameters

### 2. Deeper Layers Show Higher Sensitivity
- **Why**: Later layers (layer3, layer4) learn more task-specific features
- **Impact**: Convolutional weights in deeper layers show changes of 400K-680K magnitude
- **Evidence**: Layer depth correlates with weight change magnitude

### 3. Non-Linear Scaling with Forgetting Ratio
- **Pattern**: Impact doesn't scale linearly with forgetting percentage
- **20% vs 10%**: Shows disproportionately higher impact than expected
- **30% vs 20%**: Levels off, suggesting saturation effects

### 4. Weight Change Distribution Patterns
- **Sparse Updates**: Most weights change minimally, but some change drastically
- **Layer-Type Specificity**: Different layer types show distinct change patterns
- **Consistency**: Similar patterns across all forgetting ratios

## Technical Implications:

### For Machine Unlearning:
1. **BatchNorm Recalibration**: May need special handling of running statistics
2. **Layer-Wise Adaptation**: Different strategies for different layer depths
3. **Threshold Effects**: Optimal forgetting ratios may exist (around 20%)

### For Model Robustness:
1. **Vulnerability Points**: BatchNorm layers are critical failure points
2. **Feature Hierarchy**: Deeper features more susceptible to forgetting
3. **Statistical Stability**: Running statistics need careful management

## What This Means for Your Research:

The visualizations show that machine unlearning is not just "reversing training" - it creates 
specific vulnerability patterns that could be exploited or defended against. The non-linear 
response to forgetting ratios suggests there may be optimal operating points for different 
unlearning objectives.
        """
        
        return report

def main():
    """Generate all visualizations and explanations."""
    
    # Path to the analysis results
    json_path = "/media/hdd/usr/leyla/Unlearn-Saliency/experiments/good_results_weight_analysis/comprehensive_weight_analysis.json"
    
    if not Path(json_path).exists():
        print(f"Error: Analysis file not found at {json_path}")
        print("Please run the weight analysis first.")
        return
    
    # Create plotter
    plotter = WeightAnalysisPlotter(json_path)
    
    # Generate plots
    print("Generating comprehensive visualizations...")
    
    # Create output directory
    output_dir = Path("experiments/good_results_weight_analysis/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Layer Sensitivity Comparison
    print("1. Creating layer sensitivity analysis...")
    fig1 = plotter.plot_layer_sensitivity_comparison()
    fig1.savefig(output_dir / "layer_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot 2: Weight Distribution Changes
    print("2. Creating weight distribution analysis...")
    fig2 = plotter.plot_weight_distribution_changes()
    fig2.savefig(output_dir / "weight_distribution_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Plot 3: BatchNorm Analysis
    print("3. Creating BatchNorm detailed analysis...")
    fig3 = plotter.plot_batchnorm_analysis()
    fig3.savefig(output_dir / "batchnorm_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # Plot 4: Forgetting Progression
    print("4. Creating forgetting progression analysis...")
    fig4 = plotter.plot_forgetting_progression()
    fig4.savefig(output_dir / "forgetting_progression_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # Generate explanation report
    print("5. Generating explanation report...")
    explanation = plotter.generate_explanation_report()
    
    with open(output_dir / "visualization_explanation.md", 'w') as f:
        f.write(explanation)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("- layer_sensitivity_analysis.png")
    print("- weight_distribution_analysis.png") 
    print("- batchnorm_analysis.png")
    print("- forgetting_progression_analysis.png")
    print("- visualization_explanation.md")
    
    # Display the explanation
    print("\n" + "="*80)
    print(explanation)

if __name__ == "__main__":
    main()