#!/usr/bin/env python3
"""
Enhanced Good Results Visualization
Creates publication-quality visualizations of weight analysis results
"""

import json
import numpy as np
import os

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("❌ matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")

class EnhancedVisualizationGenerator:
    """Creates enhanced visualizations for weight analysis results"""
    
    def __init__(self, analysis_path):
        self.analysis_path = analysis_path
        self.data = self.load_analysis()
        self.setup_style()
    
    def load_analysis(self):
        """Load the weight analysis data"""
        with open(self.analysis_path, 'r') as f:
            return json.load(f)
    
    def setup_style(self):
        """Setup publication-quality plotting style"""
        if PLOTTING_AVAILABLE:
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
            plt.rcParams.update({
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 11,
                'figure.titlesize': 16
            })
    
    def create_forgetting_ratio_comparison(self, save_dir='experiments/enhanced_visualizations'):
        """Create comprehensive forgetting ratio comparison"""
        if not PLOTTING_AVAILABLE:
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract forgetting ratios and key metrics
        experiments = {}
        for exp_name, exp_data in self.data.items():
            # Extract forgetting percentage from name
            if '10percent' in exp_name:
                ratio = 10
            elif '20percent' in exp_name:
                ratio = 20
            elif '30percent' in exp_name:
                ratio = 30
            else:
                continue
            
            layer_data = exp_data['layer_sensitivity']
            
            # Calculate summary statistics
            changes = [layer['mean_relative_change'] for layer in layer_data.values()]
            max_changes = [layer['max_relative_change'] for layer in layer_data.values()]
            
            experiments[ratio] = {
                'name': exp_name,
                'mean_change': np.mean(changes),
                'max_change': np.max(max_changes),
                'std_change': np.std(changes),
                'median_change': np.median(changes),
                'q75_change': np.percentile(changes, 75),
                'q25_change': np.percentile(changes, 25)
            }
        
        # Create comprehensive comparison plot
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
        
        ratios = sorted(experiments.keys())
        colors = ['#3498db', '#e74c3c', '#f39c12']
        
        # 1. Mean Changes Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        means = [experiments[r]['mean_change'] for r in ratios]
        stds = [experiments[r]['std_change'] for r in ratios]
        bars1 = ax1.bar(ratios, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax1.set_title('Mean Weight Changes by Forgetting Ratio')
        ax1.set_xlabel('Forgetting Ratio (%)')
        ax1.set_ylabel('Mean Relative Change')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars1, means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(stds)*0.1,
                    f'{mean:.4f}', ha='center', va='bottom')
        
        # 2. Maximum Changes Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        max_changes = [experiments[r]['max_change'] for r in ratios]
        bars2 = ax2.bar(ratios, max_changes, color=colors, alpha=0.7)
        ax2.set_title('Maximum Weight Changes by Forgetting Ratio')
        ax2.set_xlabel('Forgetting Ratio (%)')
        ax2.set_ylabel('Maximum Relative Change')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, max_val in zip(bars2, max_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(max_changes)*0.02,
                    f'{max_val:.2f}', ha='center', va='bottom')
        
        # 3. Box plot comparison
        ax3 = fig.add_subplot(gs[1, :])
        box_data = []
        for ratio in ratios:
            exp_name = experiments[ratio]['name']
            layer_data = self.data[exp_name]['layer_sensitivity']
            changes = [layer['mean_relative_change'] for layer in layer_data.values()]
            box_data.append(changes)
        
        bp = ax3.boxplot(box_data, labels=[f'{r}%' for r in ratios], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_title('Distribution of Weight Changes by Forgetting Ratio')
        ax3.set_xlabel('Forgetting Ratio (%)')
        ax3.set_ylabel('Relative Change Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Layer type analysis
        ax4 = fig.add_subplot(gs[2, :])
        
        # Categorize layers by type
        layer_types = {}
        for ratio in ratios:
            exp_name = experiments[ratio]['name']
            layer_data = self.data[exp_name]['layer_sensitivity']
            
            types = {'conv': [], 'bn': [], 'fc': [], 'other': []}
            for layer_name, layer_info in layer_data.items():
                change = layer_info['mean_relative_change']
                if 'conv' in layer_name:
                    types['conv'].append(change)
                elif 'bn' in layer_name or 'batch' in layer_name:
                    types['bn'].append(change)
                elif 'fc' in layer_name or 'linear' in layer_name:
                    types['fc'].append(change)
                else:
                    types['other'].append(change)
            
            layer_types[ratio] = {k: np.mean(v) if v else 0 for k, v in types.items()}
        
        # Plot layer type comparison
        x = np.arange(len(ratios))
        width = 0.2
        type_names = ['conv', 'bn', 'fc', 'other']
        type_colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, layer_type in enumerate(type_names):
            values = [layer_types[ratio][layer_type] for ratio in ratios]
            ax4.bar(x + i*width, values, width, label=layer_type, color=type_colors[i], alpha=0.7)
        
        ax4.set_title('Average Weight Changes by Layer Type and Forgetting Ratio')
        ax4.set_xlabel('Forgetting Ratio (%)')
        ax4.set_ylabel('Average Relative Change')
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels([f'{r}%' for r in ratios])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'comprehensive_forgetting_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive comparison saved: {save_path}")
        
        return experiments
    
    def create_layer_hierarchy_analysis(self, save_dir='experiments/enhanced_visualizations'):
        """Analyze and visualize how different parts of the network are affected"""
        if not PLOTTING_AVAILABLE:
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Layer Hierarchy Analysis: How Forgetting Affects Different Network Parts', fontsize=16)
        
        for idx, (exp_name, exp_data) in enumerate(self.data.items()):
            if idx >= 4:  # Only plot first 4 experiments
                break
            
            ax = axes[idx // 2, idx % 2]
            layer_data = exp_data['layer_sensitivity']
            
            # Categorize layers by network depth/type
            early_layers = []  # conv1, layer1
            mid_layers = []    # layer2, layer3
            late_layers = []   # layer4, fc
            norm_layers = []   # bn, running_mean, etc.
            
            for layer_name, layer_info in layer_data.items():
                change = layer_info['mean_relative_change']
                
                if any(x in layer_name for x in ['conv1', 'layer1']):
                    early_layers.append(change)
                elif any(x in layer_name for x in ['layer2', 'layer3']):
                    mid_layers.append(change)
                elif any(x in layer_name for x in ['layer4', 'fc']):
                    late_layers.append(change)
                elif any(x in layer_name for x in ['bn', 'running', 'batch', 'norm']):
                    norm_layers.append(change)
            
            # Create violin plot
            data_to_plot = []
            labels = []
            
            if early_layers:
                data_to_plot.append(early_layers)
                labels.append('Early\n(conv1, layer1)')
            if mid_layers:
                data_to_plot.append(mid_layers)
                labels.append('Middle\n(layer2, layer3)')
            if late_layers:
                data_to_plot.append(late_layers)
                labels.append('Late\n(layer4, fc)')
            if norm_layers:
                data_to_plot.append(norm_layers)
                labels.append('Normalization\n(bn, running)')
            
            if data_to_plot:
                parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)))
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('Relative Change')
                
                # Extract forgetting ratio for title
                if '10percent' in exp_name:
                    ratio = '10%'
                elif '20percent' in exp_name:
                    ratio = '20%'
                elif '30percent' in exp_name:
                    ratio = '30%'
                else:
                    ratio = 'Unknown'
                
                ax.set_title(f'Forgetting Ratio: {ratio}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'layer_hierarchy_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Layer hierarchy analysis saved: {save_path}")
    
    def create_weight_magnitude_heatmap(self, save_dir='experiments/enhanced_visualizations'):
        """Create heatmap showing weight changes across experiments and layers"""
        if not PLOTTING_AVAILABLE:
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Collect data for heatmap
        all_layers = set()
        for exp_data in self.data.values():
            all_layers.update(exp_data['layer_sensitivity'].keys())
        
        # Filter to interesting layers (avoid too many normalization layers)
        interesting_layers = [layer for layer in all_layers 
                            if any(x in layer for x in ['conv', 'layer', 'fc']) 
                            and 'running' not in layer][:20]  # Limit to 20 most interesting
        
        # Create matrix
        exp_names = list(self.data.keys())
        matrix = np.zeros((len(exp_names), len(interesting_layers)))
        
        for i, exp_name in enumerate(exp_names):
            layer_data = self.data[exp_name]['layer_sensitivity']
            for j, layer_name in enumerate(interesting_layers):
                if layer_name in layer_data:
                    matrix[i, j] = layer_data[layer_name]['mean_relative_change']
        
        # Create heatmap
        plt.figure(figsize=(16, 8))
        
        # Use log scale for better visualization if values are very different
        matrix_log = np.log10(matrix + 1e-10)  # Add small value to avoid log(0)
        
        sns.heatmap(matrix_log, 
                   xticklabels=[layer.replace('layer', 'L').replace('.weight', '') for layer in interesting_layers],
                   yticklabels=[name.replace('random_forgetting_', '').replace('_RL_', '_') for name in exp_names],
                   annot=False, 
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Log10(Relative Change)'})
        
        plt.title('Weight Changes Heatmap: Experiments vs Layers')
        plt.xlabel('Layers')
        plt.ylabel('Experiments')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'weight_changes_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Weight changes heatmap saved: {save_path}")
    
    def generate_all_visualizations(self, save_dir='experiments/enhanced_visualizations'):
        """Generate all enhanced visualizations"""
        print("\n🎨 GENERATING ENHANCED VISUALIZATIONS")
        print("="*50)
        
        if not PLOTTING_AVAILABLE:
            print("❌ matplotlib/seaborn not available")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate all visualizations
        experiments = self.create_forgetting_ratio_comparison(save_dir)
        self.create_layer_hierarchy_analysis(save_dir)
        self.create_weight_magnitude_heatmap(save_dir)
        
        # Create summary report
        self.create_visualization_summary(experiments, save_dir)
        
        print(f"\n✅ All enhanced visualizations generated!")
        print(f"📂 Results saved to: {save_dir}")
    
    def create_visualization_summary(self, experiments, save_dir):
        """Create summary report of visualizations"""
        report = []
        report.append("# Enhanced Weight Analysis Visualizations Summary")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## Key Findings")
        report.append("")
        
        # Find optimal forgetting ratio
        if experiments:
            optimal_ratio = min(experiments.keys(), key=lambda x: experiments[x]['mean_change'])
            report.append(f"- **Optimal Forgetting Ratio**: {optimal_ratio}% (lowest mean change: {experiments[optimal_ratio]['mean_change']:.6f})")
            
            max_impact_ratio = max(experiments.keys(), key=lambda x: experiments[x]['max_change'])
            report.append(f"- **Highest Impact Ratio**: {max_impact_ratio}% (max change: {experiments[max_impact_ratio]['max_change']:.2f})")
        
        report.append("")
        report.append("## Generated Visualizations")
        report.append("")
        report.append("1. **comprehensive_forgetting_comparison.png**: Complete comparison across forgetting ratios")
        report.append("2. **layer_hierarchy_analysis.png**: How different network parts are affected")
        report.append("3. **weight_changes_heatmap.png**: Detailed layer-by-layer analysis")
        report.append("")
        
        report.append("## Recommendations for Lucent Analysis")
        report.append("")
        if experiments:
            optimal = experiments[optimal_ratio]
            report.append(f"- Focus Lucent analysis on **{optimal_ratio}% forgetting ratio** (most stable)")
            report.append("- Pay special attention to **normalization layers** (highest changes)")
            report.append("- Compare **early vs late layer** feature visualizations")
            report.append("- Investigate why **BatchNorm statistics** change so dramatically")
        
        # Save report
        report_path = os.path.join(save_dir, 'visualization_summary.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Visualization summary saved: {report_path}")

def main():
    """Main function to generate enhanced visualizations"""
    analysis_path = "experiments/good_results_weight_analysis/comprehensive_weight_analysis.json"
    
    if not os.path.exists(analysis_path):
        print(f"❌ Weight analysis not found: {analysis_path}")
        print("   Run 'python analyze_good_results.py' first")
        return
    
    generator = EnhancedVisualizationGenerator(analysis_path)
    generator.generate_all_visualizations()

if __name__ == "__main__":
    from datetime import datetime
    main()