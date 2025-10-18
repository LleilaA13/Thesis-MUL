#!/usr/bin/env python3
"""
Improved Layer Visualization Script

Creates readable and meaningful visualizations for layer changes during unlearning.
Addresses the issue of unreadable layer names by using better grouping and display strategies.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for clean, readable plots
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

class ImprovedLayerVisualizer:
    def __init__(self, json_file_path):
        """Initialize with the comprehensive weight analysis JSON file."""
        self.json_file_path = Path(json_file_path)
        self.data = self.load_data()
        self.experiments = list(self.data.keys())
        
    def load_data(self):
        """Load and parse the JSON data."""
        with open(self.json_file_path, 'r') as f:
            return json.load(f)
    
    def categorize_layer(self, layer_name):
        """Categorize layers into meaningful groups."""
        layer_name = layer_name.lower()
        
        if 'conv1' in layer_name and not 'layer' in layer_name:
            return 'Initial Conv', 'conv1', 0
        elif 'bn1' in layer_name and not 'layer' in layer_name:
            return 'Initial BN', 'bn1', 1
        elif 'layer1' in layer_name:
            return 'Block 1', self._extract_sublayer(layer_name), 2
        elif 'layer2' in layer_name:
            return 'Block 2', self._extract_sublayer(layer_name), 3
        elif 'layer3' in layer_name:
            return 'Block 3', self._extract_sublayer(layer_name), 4
        elif 'layer4' in layer_name:
            return 'Block 4', self._extract_sublayer(layer_name), 5
        elif 'fc' in layer_name or 'classifier' in layer_name:
            return 'Classifier', 'fc', 6
        elif 'normalize' in layer_name:
            return 'Normalization', 'normalize', 7
        else:
            return 'Other', 'other', 8
    
    def _extract_sublayer(self, layer_name):
        """Extract sublayer information for better categorization."""
        if 'conv1' in layer_name:
            return 'conv1'
        elif 'conv2' in layer_name:
            return 'conv2'
        elif 'conv3' in layer_name:
            return 'conv3'
        elif 'bn1' in layer_name:
            return 'bn1'
        elif 'bn2' in layer_name:
            return 'bn2'
        elif 'bn3' in layer_name:
            return 'bn3'
        elif 'downsample' in layer_name:
            return 'downsample'
        else:
            return 'other'
    
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
    
    def create_readable_heatmap(self):
        """Create a heatmap showing layer sensitivity across experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('Layer Sensitivity Heatmaps - Readable Format', fontsize=18, fontweight='bold')
        
        ratios = self.extract_forgetting_ratios()
        
        # Prepare data for different metrics
        metrics = ['mean_absolute_change', 'max_absolute_change', 'percentage_changed', 'mean_relative_change']
        metric_names = ['Mean Absolute Change', 'Max Absolute Change', 'Percentage Changed', 'Mean Relative Change']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx//2, idx%2]
            
            # Collect data for heatmap
            heatmap_data = []
            layer_categories = []
            
            # Get top 15 most affected layers across all experiments for this metric
            all_layers = {}
            for exp_name in self.experiments:
                exp_data = self.data[exp_name]['layer_sensitivity']
                for layer_name, layer_stats in exp_data.items():
                    if isinstance(layer_stats, dict) and metric in layer_stats:
                        if layer_name not in all_layers:
                            all_layers[layer_name] = []
                        all_layers[layer_name].append(layer_stats[metric])
            
            # Calculate average impact and select top layers
            avg_impact = {layer: np.mean(values) for layer, values in all_layers.items()}
            top_layers = sorted(avg_impact.items(), key=lambda x: x[1], reverse=True)[:15]
            
            # Create heatmap data
            heatmap_matrix = []
            readable_labels = []
            
            for layer_name, _ in top_layers:
                category, sublayer, _ = self.categorize_layer(layer_name)
                readable_label = f"{category}_{sublayer}"
                if len(readable_label) > 20:
                    readable_label = readable_label[:17] + "..."
                readable_labels.append(readable_label)
                
                row = []
                for exp_name in self.experiments:
                    exp_data = self.data[exp_name]['layer_sensitivity']
                    if layer_name in exp_data and isinstance(exp_data[layer_name], dict):
                        value = exp_data[layer_name].get(metric, 0)
                        # Log transform for better visualization
                        if value > 0:
                            row.append(np.log10(value + 1e-10))
                        else:
                            row.append(0)
                    else:
                        row.append(0)
                heatmap_matrix.append(row)
            
            if heatmap_matrix:
                # Create heatmap
                im = ax.imshow(heatmap_matrix, cmap='YlOrRd', aspect='auto')
                
                # Set labels
                ax.set_xticks(range(len(ratios)))
                ax.set_xticklabels(ratios)
                ax.set_yticks(range(len(readable_labels)))
                ax.set_yticklabels(readable_labels, fontsize=9)
                
                ax.set_xlabel('Forgetting Ratio')
                ax.set_ylabel('Layer (Category_Type)')
                ax.set_title(f'{metric_name}\n(Log Scale)', fontweight='bold')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)
                
                # Add value annotations for better readability
                for i in range(len(readable_labels)):
                    for j in range(len(ratios)):
                        if i < len(heatmap_matrix) and j < len(heatmap_matrix[i]):
                            value = heatmap_matrix[i][j]
                            if value > 0:
                                # Show original value, not log
                                orig_value = 10**(value) - 1e-10
                                if orig_value >= 1:
                                    text = f'{orig_value:.1f}'
                                else:
                                    text = f'{orig_value:.2e}'
                                ax.text(j, i, text, ha='center', va='center', 
                                       fontsize=8, color='black' if value < np.max(heatmap_matrix)*0.7 else 'white')
        
        plt.tight_layout()
        return fig
    
    def create_layer_block_summary(self):
        """Create a summary plot by ResNet blocks."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ResNet Block-wise Analysis', fontsize=16, fontweight='bold')
        
        ratios = self.extract_forgetting_ratios()
        
        # Group data by blocks
        block_data = []
        for j, exp_name in enumerate(self.experiments):
            exp_data = self.data[exp_name]['layer_sensitivity']
            
            block_stats = {
                'Initial': {'mean_changes': [], 'max_changes': []},
                'Block 1': {'mean_changes': [], 'max_changes': []},
                'Block 2': {'mean_changes': [], 'max_changes': []},
                'Block 3': {'mean_changes': [], 'max_changes': []},
                'Block 4': {'mean_changes': [], 'max_changes': []},
                'Classifier': {'mean_changes': [], 'max_changes': []}
            }
            
            for layer_name, layer_stats in exp_data.items():
                if isinstance(layer_stats, dict):
                    category, sublayer, _ = self.categorize_layer(layer_name)
                    
                    # Map categories to blocks
                    if category in ['Initial Conv', 'Initial BN']:
                        block_key = 'Initial'
                    elif category in ['Block 1', 'Block 2', 'Block 3', 'Block 4']:
                        block_key = category
                    elif category == 'Classifier':
                        block_key = 'Classifier'
                    else:
                        continue
                    
                    if block_key in block_stats:
                        block_stats[block_key]['mean_changes'].append(layer_stats.get('mean_absolute_change', 0))
                        block_stats[block_key]['max_changes'].append(layer_stats.get('max_absolute_change', 0))
            
            # Calculate block averages
            for block, stats in block_stats.items():
                if stats['mean_changes']:
                    block_data.append({
                        'Forgetting Ratio': ratios[j],
                        'Block': block,
                        'Avg Mean Change': np.mean(stats['mean_changes']),
                        'Avg Max Change': np.mean(stats['max_changes']),
                        'Std Mean Change': np.std(stats['mean_changes']),
                        'Layer Count': len(stats['mean_changes'])
                    })
        
        df_blocks = pd.DataFrame(block_data)
        
        if not df_blocks.empty:
            # Plot 1: Average mean change by block
            ax = axes[0, 0]
            sns.barplot(data=df_blocks, x='Block', y='Avg Mean Change', hue='Forgetting Ratio', ax=ax)
            ax.set_title('Average Mean Change by Block', fontweight='bold')
            ax.set_ylabel('Average Mean Change')
            ax.set_yscale('log')
            ax.tick_params(axis='x', rotation=45)
            
            # Plot 2: Average max change by block
            ax = axes[0, 1]
            sns.barplot(data=df_blocks, x='Block', y='Avg Max Change', hue='Forgetting Ratio', ax=ax)
            ax.set_title('Average Max Change by Block', fontweight='bold')
            ax.set_ylabel('Average Max Change')
            ax.set_yscale('log')
            ax.tick_params(axis='x', rotation=45)
            
            # Plot 3: Standard deviation (variability within blocks)
            ax = axes[1, 0]
            sns.barplot(data=df_blocks, x='Block', y='Std Mean Change', hue='Forgetting Ratio', ax=ax)
            ax.set_title('Variability Within Blocks', fontweight='bold')
            ax.set_ylabel('Standard Deviation of Changes')
            ax.set_yscale('log')
            ax.tick_params(axis='x', rotation=45)
            
            # Plot 4: Layer count per block
            ax = axes[1, 1]
            layer_counts = df_blocks.groupby('Block')['Layer Count'].first().reset_index()
            bars = ax.bar(layer_counts['Block'], layer_counts['Layer Count'], alpha=0.7)
            ax.set_title('Number of Layers per Block', fontweight='bold')
            ax.set_ylabel('Layer Count')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_top_layers_detailed(self):
        """Create a detailed view of the most affected layers."""
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle('Top Most Affected Layers - Detailed Analysis', fontsize=16, fontweight='bold')
        
        ratios = self.extract_forgetting_ratios()
        
        # Get top 10 layers by average impact across all experiments
        all_layer_impacts = {}
        for exp_name in self.experiments:
            exp_data = self.data[exp_name]['layer_sensitivity']
            for layer_name, layer_stats in exp_data.items():
                if isinstance(layer_stats, dict):
                    if layer_name not in all_layer_impacts:
                        all_layer_impacts[layer_name] = []
                    all_layer_impacts[layer_name].append(layer_stats.get('mean_absolute_change', 0))
        
        # Calculate average and select top layers
        avg_impacts = {layer: np.mean(impacts) for layer, impacts in all_layer_impacts.items()}
        top_10_layers = sorted(avg_impacts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create readable labels
        top_layer_data = []
        readable_names = []
        
        for layer_name, avg_impact in top_10_layers:
            category, sublayer, _ = self.categorize_layer(layer_name)
            readable_name = f"{category}\n{sublayer}"
            readable_names.append(readable_name)
            
            for j, exp_name in enumerate(self.experiments):
                exp_data = self.data[exp_name]['layer_sensitivity']
                if layer_name in exp_data:
                    layer_stats = exp_data[layer_name]
                    if isinstance(layer_stats, dict):
                        top_layer_data.append({
                            'Layer': readable_name,
                            'Full Name': layer_name,
                            'Forgetting Ratio': ratios[j],
                            'Mean Change': layer_stats.get('mean_absolute_change', 0),
                            'Max Change': layer_stats.get('max_absolute_change', 0),
                            'Percentage Changed': layer_stats.get('percentage_changed', 0)
                        })
        
        df_top = pd.DataFrame(top_layer_data)
        
        if not df_top.empty:
            # Plot 1: Mean changes for top layers
            ax = axes[0]
            sns.barplot(data=df_top, x='Layer', y='Mean Change', hue='Forgetting Ratio', ax=ax)
            ax.set_title('Top 10 Most Affected Layers - Mean Change', fontweight='bold')
            ax.set_ylabel('Mean Absolute Change')
            ax.set_yscale('log')
            ax.tick_params(axis='x', rotation=45)
            
            # Plot 2: Percentage changed for top layers
            ax = axes[1]
            sns.barplot(data=df_top, x='Layer', y='Percentage Changed', hue='Forgetting Ratio', ax=ax)
            ax.set_title('Top 10 Most Affected Layers - Percentage Changed', fontweight='bold')
            ax.set_ylabel('Percentage of Weights Changed (%)')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_layer_type_comparison(self):
        """Create a comparison of different layer parameter types."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Layer Parameter Type Analysis', fontsize=16, fontweight='bold')
        
        ratios = self.extract_forgetting_ratios()
        
        # Categorize by parameter type
        param_data = []
        for j, exp_name in enumerate(self.experiments):
            exp_data = self.data[exp_name]['layer_sensitivity']
            
            for layer_name, layer_stats in exp_data.items():
                if isinstance(layer_stats, dict):
                    # Determine parameter type
                    param_type = 'Other'
                    if 'weight' in layer_name:
                        if 'conv' in layer_name:
                            param_type = 'Conv Weights'
                        elif 'bn' in layer_name:
                            param_type = 'BN Weights'
                        elif 'fc' in layer_name:
                            param_type = 'FC Weights'
                        else:
                            param_type = 'Other Weights'
                    elif 'bias' in layer_name:
                        if 'conv' in layer_name:
                            param_type = 'Conv Bias'
                        elif 'bn' in layer_name:
                            param_type = 'BN Bias'
                        elif 'fc' in layer_name:
                            param_type = 'FC Bias'
                        else:
                            param_type = 'Other Bias'
                    elif 'running_mean' in layer_name:
                        param_type = 'BN Running Mean'
                    elif 'running_var' in layer_name:
                        param_type = 'BN Running Var'
                    elif 'num_batches' in layer_name:
                        param_type = 'BN Batch Count'
                    
                    param_data.append({
                        'Forgetting Ratio': ratios[j],
                        'Parameter Type': param_type,
                        'Mean Change': layer_stats.get('mean_absolute_change', 0),
                        'Max Change': layer_stats.get('max_absolute_change', 0),
                        'Percentage Changed': layer_stats.get('percentage_changed', 0),
                        'Relative Change': layer_stats.get('mean_relative_change', 0)
                    })
        
        df_params = pd.DataFrame(param_data)
        
        if not df_params.empty:
            # Plot 1: Mean change by parameter type
            ax = axes[0, 0]
            sns.boxplot(data=df_params, x='Parameter Type', y='Mean Change', 
                       hue='Forgetting Ratio', ax=ax)
            ax.set_title('Mean Change by Parameter Type', fontweight='bold')
            ax.set_ylabel('Mean Absolute Change')
            ax.set_yscale('log')
            ax.tick_params(axis='x', rotation=45)
            
            # Plot 2: Percentage changed by parameter type
            ax = axes[0, 1]
            sns.boxplot(data=df_params, x='Parameter Type', y='Percentage Changed', 
                       hue='Forgetting Ratio', ax=ax)
            ax.set_title('Percentage Changed by Parameter Type', fontweight='bold')
            ax.set_ylabel('Percentage Changed (%)')
            ax.tick_params(axis='x', rotation=45)
            
            # Plot 3: Focus on learnable vs non-learnable parameters
            ax = axes[1, 0]
            learnable_types = ['Conv Weights', 'BN Weights', 'FC Weights', 'Conv Bias', 'BN Bias', 'FC Bias']
            running_types = ['BN Running Mean', 'BN Running Var']
            
            learnable_data = df_params[df_params['Parameter Type'].isin(learnable_types)]
            running_data = df_params[df_params['Parameter Type'].isin(running_types)]
            
            if not learnable_data.empty and not running_data.empty:
                learnable_avg = learnable_data.groupby('Forgetting Ratio')['Mean Change'].mean()
                running_avg = running_data.groupby('Forgetting Ratio')['Mean Change'].mean()
                
                x = np.arange(len(learnable_avg))
                width = 0.35
                
                ax.bar(x - width/2, learnable_avg.values, width, label='Learnable Params', alpha=0.8)
                ax.bar(x + width/2, running_avg.values, width, label='Running Stats', alpha=0.8)
                
                ax.set_xlabel('Forgetting Ratio')
                ax.set_ylabel('Mean Absolute Change')
                ax.set_title('Learnable vs Running Statistics', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(learnable_avg.index)
                ax.legend()
                ax.set_yscale('log')
            
            # Plot 4: Distribution of changes
            ax = axes[1, 1]
            # Focus on main parameter types for clarity
            main_types = ['Conv Weights', 'BN Weights', 'BN Running Mean', 'BN Running Var']
            main_data = df_params[df_params['Parameter Type'].isin(main_types)]
            
            if not main_data.empty:
                sns.violinplot(data=main_data, x='Parameter Type', y='Mean Change', 
                              hue='Forgetting Ratio', ax=ax)
                ax.set_title('Distribution of Changes - Main Types', fontweight='bold')
                ax.set_ylabel('Mean Absolute Change')
                ax.set_yscale('log')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig

def main():
    """Generate improved, readable visualizations."""
    
    # Path to the analysis results
    json_path = "/media/hdd/usr/leyla/Unlearn-Saliency/experiments/good_results_weight_analysis/comprehensive_weight_analysis.json"
    
    if not Path(json_path).exists():
        print(f"Error: Analysis file not found at {json_path}")
        return
    
    # Create visualizer
    visualizer = ImprovedLayerVisualizer(json_path)
    
    # Create output directory
    output_dir = Path("experiments/good_results_weight_analysis/improved_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating improved, readable visualizations...")
    
    # Generate improved plots
    try:
        # Plot 1: Readable heatmap
        print("1. Creating readable layer sensitivity heatmap...")
        fig1 = visualizer.create_readable_heatmap()
        fig1.savefig(output_dir / "readable_layer_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Plot 2: Block-wise summary
        print("2. Creating ResNet block-wise analysis...")
        fig2 = visualizer.create_layer_block_summary()
        fig2.savefig(output_dir / "resnet_block_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # Plot 3: Top layers detailed
        print("3. Creating detailed top layers analysis...")
        fig3 = visualizer.create_top_layers_detailed()
        fig3.savefig(output_dir / "top_layers_detailed.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        # Plot 4: Parameter type comparison
        print("4. Creating parameter type comparison...")
        fig4 = visualizer.create_layer_type_comparison()
        fig4.savefig(output_dir / "parameter_type_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        print(f"\n✅ All improved visualizations saved to: {output_dir}")
        print("\nGenerated files:")
        print("- readable_layer_heatmap.png")
        print("- resnet_block_analysis.png") 
        print("- top_layers_detailed.png")
        print("- parameter_type_analysis.png")
        
        print("\n" + "="*60)
        print("VISUALIZATION IMPROVEMENTS MADE:")
        print("="*60)
        print("1. ✅ Readable layer names using category grouping")
        print("2. ✅ Heatmaps with value annotations")
        print("3. ✅ Block-wise analysis instead of individual layers")
        print("4. ✅ Parameter type categorization")
        print("5. ✅ Log scale for better visualization")
        print("6. ✅ Top 10 most affected layers focus")
        print("7. ✅ Clear legends and titles")
        print("="*60)
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()