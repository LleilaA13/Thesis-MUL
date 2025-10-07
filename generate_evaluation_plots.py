#!/usr/bin/env python3
"""
Evaluation Metrics Plotting Script for Machine Unlearning Thesis
Extracts and visualizes evaluation metrics from MIA and other assessments
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# Add src to path for imports
sys.path.append('/media/hdd/usr/leyla/Unlearn-Saliency/src')

class EvaluationPlotter:
    def __init__(self, base_dir="/media/hdd/usr/leyla/Unlearn-Saliency"):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "thesis_figures" / "evaluation_metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Evaluation Plotter initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def generate_mia_data(self):
        """Generate synthetic MIA evaluation data"""
        np.random.seed(42)
        
        # Shadow model performance
        shadow_train_size = 1000
        shadow_test_size = 1000
        num_classes = 200
        
        # Generate synthetic predictions
        shadow_train_logits = np.random.dirichlet(np.ones(num_classes), shadow_train_size)
        shadow_train_labels = np.random.randint(0, num_classes, shadow_train_size)
        
        shadow_test_logits = np.random.dirichlet(np.ones(num_classes), shadow_test_size)
        shadow_test_labels = np.random.randint(0, num_classes, shadow_test_size)
        
        # Target model performance
        target_train_size = 800
        target_test_size = 800
        
        # Original model - higher confidence on training data
        target_train_logits_orig = np.random.dirichlet(np.ones(num_classes) * 0.5, target_train_size)
        target_train_labels = np.random.randint(0, num_classes, target_train_size)
        
        target_test_logits_orig = np.random.dirichlet(np.ones(num_classes), target_test_size)
        target_test_labels = np.random.randint(0, num_classes, target_test_size)
        
        # Unlearned model - reduced confidence on forgotten data
        target_train_logits_unlearn = target_train_logits_orig.copy()
        # Reduce confidence for forgotten samples (first 100 samples simulate forgotten data)
        forgotten_indices = np.arange(100)
        for idx in forgotten_indices:
            # Make predictions more uniform (less confident)
            target_train_logits_unlearn[idx] = np.random.dirichlet(np.ones(num_classes) * 2)
        
        target_test_logits_unlearn = target_test_logits_orig * np.random.uniform(0.8, 1.0, 
                                                                               (target_test_size, num_classes))
        
        return {
            'shadow_train': (shadow_train_logits, shadow_train_labels),
            'shadow_test': (shadow_test_logits, shadow_test_labels),
            'target_train_orig': (target_train_logits_orig, target_train_labels),
            'target_test_orig': (target_test_logits_orig, target_test_labels),
            'target_train_unlearn': (target_train_logits_unlearn, target_train_labels),
            'target_test_unlearn': (target_test_logits_unlearn, target_test_labels),
            'forgotten_indices': forgotten_indices
        }
    
    def calculate_mia_metrics(self, data):
        """Calculate MIA attack success rates"""
        
        def confidence_based_attack(train_logits, test_logits, train_labels, test_labels):
            # Use max confidence as the attack feature
            train_conf = np.max(train_logits, axis=1)
            test_conf = np.max(test_logits, axis=1)
            
            # Threshold-based attack
            threshold = np.median(np.concatenate([train_conf, test_conf]))
            
            # Predictions: high confidence -> member, low confidence -> non-member
            train_pred = train_conf >= threshold  # True positives
            test_pred = test_conf >= threshold   # False positives
            
            # Calculate accuracy
            tp = np.sum(train_pred)
            tn = np.sum(~test_pred)
            total = len(train_pred) + len(test_pred)
            accuracy = (tp + tn) / total
            
            return accuracy, train_conf, test_conf, threshold
        
        # Original model MIA
        orig_acc, orig_train_conf, orig_test_conf, orig_thresh = confidence_based_attack(
            data['target_train_orig'][0], data['target_test_orig'][0],
            data['target_train_orig'][1], data['target_test_orig'][1]
        )
        
        # Unlearned model MIA
        unlearn_acc, unlearn_train_conf, unlearn_test_conf, unlearn_thresh = confidence_based_attack(
            data['target_train_unlearn'][0], data['target_test_unlearn'][0],
            data['target_train_unlearn'][1], data['target_test_unlearn'][1]
        )
        
        return {
            'original_mia_acc': orig_acc,
            'unlearned_mia_acc': unlearn_acc,
            'original_train_conf': orig_train_conf,
            'original_test_conf': orig_test_conf,
            'unlearned_train_conf': unlearn_train_conf,
            'unlearned_test_conf': unlearn_test_conf,
            'original_threshold': orig_thresh,
            'unlearned_threshold': unlearn_thresh
        }
    
    def plot_mia_evaluation(self):
        """Plot comprehensive MIA evaluation results"""
        
        # Generate synthetic data
        data = self.generate_mia_data()
        metrics = self.calculate_mia_metrics(data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. MIA Attack Accuracy Comparison
        models = ['Original Model', 'Unlearned Model']
        mia_accuracies = [metrics['original_mia_acc'] * 100, metrics['unlearned_mia_acc'] * 100]
        
        bars = ax1.bar(models, mia_accuracies, color=['red', 'green'], alpha=0.7)
        ax1.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Random Baseline')
        ax1.set_title('Membership Inference Attack Success Rate', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Attack Accuracy (%)')
        ax1.set_ylim(45, 75)
        ax1.legend()
        
        for bar, acc in zip(bars, mia_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Confidence Distribution Comparison
        ax2.hist(metrics['original_train_conf'], bins=30, alpha=0.5, 
                label='Original - Train', color='blue', density=True)
        ax2.hist(metrics['original_test_conf'], bins=30, alpha=0.5, 
                label='Original - Test', color='red', density=True)
        ax2.hist(metrics['unlearned_train_conf'], bins=30, alpha=0.5, 
                label='Unlearned - Train', color='lightblue', density=True)
        ax2.hist(metrics['unlearned_test_conf'], bins=30, alpha=0.5, 
                label='Unlearned - Test', color='lightcoral', density=True)
        
        ax2.set_title('Confidence Score Distributions', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        # 3. ROC Curves
        # Generate ROC data for original model
        train_labels_orig = np.ones(len(metrics['original_train_conf']))
        test_labels_orig = np.zeros(len(metrics['original_test_conf']))
        
        y_true_orig = np.concatenate([train_labels_orig, test_labels_orig])
        y_scores_orig = np.concatenate([metrics['original_train_conf'], 
                                       metrics['original_test_conf']])
        
        fpr_orig, tpr_orig, _ = roc_curve(y_true_orig, y_scores_orig)
        roc_auc_orig = auc(fpr_orig, tpr_orig)
        
        # Generate ROC data for unlearned model
        y_scores_unlearn = np.concatenate([metrics['unlearned_train_conf'], 
                                          metrics['unlearned_test_conf']])
        
        fpr_unlearn, tpr_unlearn, _ = roc_curve(y_true_orig, y_scores_unlearn)
        roc_auc_unlearn = auc(fpr_unlearn, tpr_unlearn)
        
        ax3.plot(fpr_orig, tpr_orig, label=f'Original (AUC = {roc_auc_orig:.3f})', 
                linewidth=2, color='red')
        ax3.plot(fpr_unlearn, tpr_unlearn, label=f'Unlearned (AUC = {roc_auc_unlearn:.3f})', 
                linewidth=2, color='green')
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        ax3.set_title('ROC Curves for MIA', fontsize=14, fontweight='bold')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Privacy-Utility Trade-off
        sparsity_levels = np.linspace(0, 0.8, 9)
        utility_scores = 85 - 15 * sparsity_levels + np.random.normal(0, 1, 9)
        privacy_scores = 50 - 15 * (1 - sparsity_levels) + np.random.normal(0, 1, 9)
        
        scatter = ax4.scatter(utility_scores, privacy_scores, s=100, 
                             c=sparsity_levels, cmap='viridis', alpha=0.8)
        ax4.set_title('Privacy-Utility Trade-off', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Model Utility (Accuracy %)')
        ax4.set_ylabel('Privacy Protection (Lower MIA Accuracy %)')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Sparsity Level')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mia_evaluation.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'mia_evaluation.pdf', bbox_inches='tight')
        print("✅ MIA evaluation plots saved")
    
    def plot_unlearning_metrics(self):
        """Plot comprehensive unlearning effectiveness metrics"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Forget vs Retain Performance
        categories = ['Forget Set', 'Retain Set', 'Test Set']
        before_acc = [85.2, 84.8, 82.1]
        after_acc = [12.3, 83.1, 81.5]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, before_acc, width, label='Before Unlearning', 
                       color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, after_acc, width, label='After Unlearning', 
                       color='lightcoral', alpha=0.8)
        
        ax1.set_title('Unlearning Effectiveness', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 2. Confusion Matrix for Forgotten Classes
        np.random.seed(42)
        # Simulate confusion matrix for vehicle classes
        vehicle_classes = ['Car', 'Truck', 'Bus']
        n_classes = len(vehicle_classes)
        
        # Before unlearning - good classification
        cm_before = np.array([[45, 3, 2], [2, 42, 6], [1, 4, 45]])
        # After unlearning - poor classification (more confusion)
        cm_after = np.array([[12, 18, 20], [15, 10, 25], [20, 22, 8]])
        
        # Plot confusion matrices
        im1 = ax2.imshow(cm_before, interpolation='nearest', cmap='Blues')
        ax2.set_title('Confusion Matrix: Before Unlearning\n(Vehicle Classes)', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                ax2.text(j, i, str(cm_before[i, j]), ha="center", va="center", 
                        color="white" if cm_before[i, j] > cm_before.max()/2 else "black")
        
        ax2.set_xticks(np.arange(n_classes))
        ax2.set_yticks(np.arange(n_classes))
        ax2.set_xticklabels(vehicle_classes)
        ax2.set_yticklabels(vehicle_classes)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # 3. Forgetting Progress Over Epochs
        epochs = np.arange(1, 21)
        forget_accuracy = 85 - 70 * (1 - np.exp(-epochs/5)) + np.random.normal(0, 2, 20)
        retain_accuracy = 84 - 3 * (1 - np.exp(-epochs/10)) + np.random.normal(0, 1, 20)
        
        ax3.plot(epochs, forget_accuracy, 'o-', label='Forget Set Accuracy', 
                linewidth=2, color='red', markersize=6)
        ax3.plot(epochs, retain_accuracy, 's-', label='Retain Set Accuracy', 
                linewidth=2, color='green', markersize=6)
        
        ax3.set_title('Unlearning Progress', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Unlearning Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 90)
        
        # 4. Class-wise Forgetting Effectiveness
        classes = ['Car', 'Truck', 'Bus', 'Dog', 'Cat', 'Bird', 'Tree', 'House']
        forgetting_rates = [85.5, 82.3, 87.1, 2.1, 1.8, 3.2, 2.5, 1.9]  # High for vehicles, low for others
        colors = ['red' if rate > 50 else 'green' for rate in forgetting_rates]
        
        bars = ax4.bar(classes, forgetting_rates, color=colors, alpha=0.7)
        ax4.set_title('Class-wise Forgetting Effectiveness', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Forgetting Rate (%)')
        ax4.set_xlabel('Class')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add threshold line
        ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5, 
                   label='Effective Forgetting Threshold')
        ax4.legend()
        
        # Add value labels
        for bar, rate in zip(bars, forgetting_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'unlearning_metrics.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'unlearning_metrics.pdf', bbox_inches='tight')
        print("✅ Unlearning metrics plots saved")
    
    def plot_computational_efficiency(self):
        """Plot computational efficiency metrics"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Training Time Comparison
        methods = ['From Scratch', 'Transfer Learning', 'Unlearning']
        training_times = [120, 20, 15]  # minutes
        colors = ['red', 'blue', 'green']
        
        bars = ax1.bar(methods, training_times, color=colors, alpha=0.7)
        ax1.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Training Time (minutes)')
        
        for bar, time in zip(bars, training_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{time} min', ha='center', va='bottom', fontweight='bold')
        
        # 2. Memory Usage
        sparsity_levels = np.linspace(0, 0.9, 10)
        memory_usage = 100 * (1 - sparsity_levels)  # Percentage of original
        storage_size = memory_usage  # GB
        
        ax2.plot(sparsity_levels, memory_usage, 'o-', linewidth=2, markersize=8, color='purple')
        ax2.set_title('Memory Usage vs Sparsity', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Sparsity Level')
        ax2.set_ylabel('Memory Usage (% of Original)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 110)
        
        # 3. Model Size vs Performance
        model_sizes = [100, 70, 50, 30, 20, 15, 10]  # MB
        performances = [85.2, 84.1, 82.5, 79.8, 75.2, 68.5, 55.3]  # Accuracy
        
        ax3.scatter(model_sizes, performances, s=100, alpha=0.7, c=range(len(model_sizes)), 
                   cmap='viridis')
        ax3.plot(model_sizes, performances, '--', alpha=0.5)
        ax3.set_title('Model Size vs Performance Trade-off', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Model Size (MB)')
        ax3.set_ylabel('Performance (Accuracy %)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Computational Complexity
        operations = ['Forward Pass', 'Gradient Computation', 'Mask Generation', 'Weight Update']
        original_flops = [100, 150, 0, 80]  # Relative FLOPs
        unlearning_flops = [30, 45, 25, 20]  # Reduced due to sparsity
        
        x = np.arange(len(operations))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, original_flops, width, label='Original Model', alpha=0.8)
        bars2 = ax4.bar(x + width/2, unlearning_flops, width, label='Unlearned Model', alpha=0.8)
        
        ax4.set_title('Computational Complexity Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Relative FLOPs')
        ax4.set_xticks(x)
        ax4.set_xticklabels(operations, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'computational_efficiency.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'computational_efficiency.pdf', bbox_inches='tight')
        print("✅ Computational efficiency plots saved")
    
    def create_evaluation_summary(self):
        """Create comprehensive evaluation summary dashboard"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Machine Unlearning: Comprehensive Evaluation Summary', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Generate some synthetic data for demonstration
        np.random.seed(42)
        
        # 1. Overall Performance Metrics
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        before = [85.2, 84.8, 85.1, 84.9]
        after_retain = [83.1, 82.5, 83.3, 82.9]
        after_forget = [12.3, 15.2, 10.8, 12.8]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        ax1.bar(x - width, before, width, label='Before', alpha=0.8)
        ax1.bar(x, after_retain, width, label='After (Retain)', alpha=0.8)
        ax1.bar(x + width, after_forget, width, label='After (Forget)', alpha=0.8)
        
        ax1.set_title('Performance Metrics', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45)
        ax1.legend(fontsize=8)
        ax1.set_ylabel('Score (%)')
        
        # 2. Privacy Protection
        ax2 = fig.add_subplot(gs[0, 1])
        privacy_methods = ['MIA Defense', 'Confidence\nReduction', 'Feature\nObfuscation']
        effectiveness = [68.5, 75.2, 62.8]
        
        bars = ax2.bar(privacy_methods, effectiveness, color='green', alpha=0.7)
        ax2.set_title('Privacy Protection Effectiveness', fontweight='bold')
        ax2.set_ylabel('Protection Level (%)')
        ax2.set_ylim(0, 100)
        
        for bar, eff in zip(bars, effectiveness):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{eff:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. Model Compression
        ax3 = fig.add_subplot(gs[0, 2])
        compression_ratios = [1.0, 0.7, 0.5, 0.3, 0.2]
        accuracy_retention = [100, 98.5, 95.2, 88.7, 78.3]
        
        ax3.plot(compression_ratios, accuracy_retention, 'o-', linewidth=2, markersize=8)
        ax3.set_title('Compression vs Accuracy', fontweight='bold')
        ax3.set_xlabel('Model Size (Fraction)')
        ax3.set_ylabel('Accuracy Retention (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Training Efficiency
        ax4 = fig.add_subplot(gs[0, 3])
        approaches = ['Scratch', 'Transfer', 'Unlearn']
        times = [120, 20, 15]
        colors = ['red', 'blue', 'green']
        
        bars = ax4.bar(approaches, times, color=colors, alpha=0.7)
        ax4.set_title('Training Time', fontweight='bold')
        ax4.set_ylabel('Time (minutes)')
        
        # 5. Forgetting Effectiveness Heatmap
        ax5 = fig.add_subplot(gs[1, :2])
        classes = ['Car', 'Truck', 'Bus', 'Dog', 'Cat', 'Bird', 'Tree', 'Flower']
        metrics_eval = ['Accuracy Drop', 'Confidence Drop', 'Feature Change', 'Gradient Norm']
        
        # Create synthetic heatmap data
        heatmap_data = np.random.uniform(0, 100, (len(metrics_eval), len(classes)))
        # Make vehicle classes show higher forgetting
        heatmap_data[:, :3] *= 1.5  # Vehicles
        heatmap_data = np.clip(heatmap_data, 0, 100)
        
        im = ax5.imshow(heatmap_data, cmap='Reds', aspect='auto')
        ax5.set_title('Forgetting Effectiveness by Class and Metric', fontweight='bold')
        ax5.set_xticks(range(len(classes)))
        ax5.set_yticks(range(len(metrics_eval)))
        ax5.set_xticklabels(classes)
        ax5.set_yticklabels(metrics_eval)
        
        # Add text annotations
        for i in range(len(metrics_eval)):
            for j in range(len(classes)):
                ax5.text(j, i, f'{heatmap_data[i, j]:.0f}', ha="center", va="center", 
                        color="white" if heatmap_data[i, j] > 50 else "black", fontsize=8)
        
        plt.colorbar(im, ax=ax5, label='Effectiveness (%)')
        
        # 6. ROC Curves
        ax6 = fig.add_subplot(gs[1, 2:])
        
        # Generate synthetic ROC data
        fpr_orig = np.linspace(0, 1, 100)
        tpr_orig = 0.3 + 0.7 * fpr_orig + 0.2 * np.sin(fpr_orig * np.pi)
        tpr_orig = np.clip(tpr_orig, 0, 1)
        
        fpr_unlearn = fpr_orig
        tpr_unlearn = 0.1 + 0.4 * fpr_orig + 0.1 * np.sin(fpr_orig * np.pi)
        tpr_unlearn = np.clip(tpr_unlearn, 0, 1)
        
        ax6.plot(fpr_orig, tpr_orig, label='Original Model (AUC=0.73)', linewidth=2)
        ax6.plot(fpr_unlearn, tpr_unlearn, label='Unlearned Model (AUC=0.52)', linewidth=2)
        ax6.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        ax6.set_title('ROC Curves: Membership Inference Attack', fontweight='bold')
        ax6.set_xlabel('False Positive Rate')
        ax6.set_ylabel('True Positive Rate')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Summary Statistics Table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        summary_data = [
            ['Metric', 'Original Model', 'Unlearned Model', 'Improvement'],
            ['Forget Set Accuracy', '85.2%', '12.3%', '-85.6%'],
            ['Retain Set Accuracy', '84.8%', '83.1%', '-2.0%'],
            ['MIA Attack Success', '68.5%', '52.1%', '-23.9%'],
            ['Model Size', '45.2 MB', '13.6 MB', '-70.0%'],
            ['Training Time', '120 min', '15 min', '-87.5%'],
            ['Memory Usage', '2.1 GB', '0.7 GB', '-66.7%'],
            ['Inference Speed', '100 ms', '65 ms', '+35.0%']
        ]
        
        table = ax7.table(cellText=summary_data[1:], colLabels=summary_data[0],
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Color code improvements
        for i in range(1, len(summary_data)):
            improvement = summary_data[i][3]
            if improvement.startswith('-') and 'Accuracy' in summary_data[i][0]:
                if 'Forget' in summary_data[i][0]:
                    table[(i, 3)].set_facecolor('#90EE90')  # Good for forgetting
                else:
                    table[(i, 3)].set_facecolor('#FFB6C1')  # Bad for retention
            elif improvement.startswith('-') and 'Accuracy' not in summary_data[i][0]:
                table[(i, 3)].set_facecolor('#90EE90')  # Good reduction
            elif improvement.startswith('+'):
                table[(i, 3)].set_facecolor('#90EE90')  # Good improvement
        
        ax7.set_title('Comprehensive Performance Summary', fontweight='bold', pad=30, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'evaluation_summary.pdf', bbox_inches='tight')
        print("✅ Evaluation summary saved")
    
    def generate_all_evaluations(self):
        """Generate all evaluation plots"""
        print("📊 Generating evaluation metric plots...")
        
        self.plot_mia_evaluation()
        self.plot_unlearning_metrics()
        self.plot_computational_efficiency()
        self.create_evaluation_summary()
        
        print(f"\n✨ All evaluation plots saved to: {self.output_dir}")
        print("\n📊 Generated files:")
        for file in sorted(self.output_dir.glob("*.png")):
            print(f"  📈 {file.name}")
        
        return self.output_dir

if __name__ == "__main__":
    plotter = EvaluationPlotter()
    output_dir = plotter.generate_all_evaluations()
    
    print(f"\n🎯 Next steps:")
    print(f"1. Review generated evaluation plots in: {output_dir}")
    print(f"2. Run actual MIA evaluation using your src/classification/evaluation/MIA.py")
    print(f"3. Replace synthetic data with real experimental results")
    print(f"4. Customize metrics based on your specific evaluation criteria")