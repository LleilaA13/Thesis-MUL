#!/bin/bash
# Random Data Forgetting Research Pipeline Executor
# This script runs the complete research pipeline

echo "🚀 Random Data Forgetting + Feature Visualization Research Pipeline"
echo "================================================================="

# Check if we're in the right directory
if [ ! -d "../core/Classification" ]; then
    echo "❌ Error: Please run this script from the Unlearn-Saliency/scripts directory"
    echo "   Or run from root: bash scripts/run_research_pipeline.sh"
    exit 1
fi






# Make scripts executable
chmod +x research_pipeline.py
chmod +x ../analysis/tools/weight_influence_analyzer.py
chmod +x ../analysis/visualizations/lucent_visualizer.py

echo "✅ Environment setup complete!"
echo ""

# Ask user what to run
echo "🎯 What would you like to run?"
echo "1) Complete pipeline (training + analysis + visualization)"
echo "2) Weight influence analysis only (requires existing models)"
echo "3) Feature visualization only (requires existing models)"
echo "4) Quick test run (small subset)"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🏃 Running complete pipeline..."
        python research_pipeline.py
        echo "📊 Running weight analysis..."
        python weight_influence_analyzer.py
        echo "🎨 Running feature visualization..."
        python lucent_visualizer.py
        ;;
    2)
        echo "📊 Running weight influence analysis..."
        python weight_influence_analyzer.py
        ;;
    3)
        echo "🎨 Running feature visualization..."
        python lucent_visualizer.py
        ;;
    4)
        echo "🧪 Running quick test..."
        echo "This will train a small model for testing..."
        # Quick test with fewer epochs
        python -c "
import sys
sys.path.append('.')
from research_pipeline import RandomDataForgettingAnalyzer

config = {
    'arch': 'resnet18',
    'dataset': 'cifar10', 
    'train_epochs': 5,  # Reduced for testing
    'train_lr': 0.1,
    'unlearn_epochs': 3,  # Reduced for testing
    'unlearn_lr': 0.013,
    'batch_size': 128
}

analyzer = RandomDataForgettingAnalyzer(config)
print('🧪 Running quick test with reduced epochs...')
analyzer.run_baseline_training()
analyzer.generate_random_forget_indices([0.1])  # Only test 10% forgetting
"
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "🎉 Pipeline execution complete!"
echo "📂 Results are saved in: experiments/random_forgetting/"
echo ""
echo "📋 Next steps:"
echo "   1. Check experiments/random_forgetting/weight_analysis/ for weight analysis"
echo "   2. Check experiments/random_forgetting/visualizations/ for feature visualizations" 
echo "   3. Review the summary report in weight_analysis/summary_report.md"
echo ""
echo "🔬 For your thesis, focus on:"
echo "   - Which layers are most affected by random forgetting"
echo "   - How feature representations change"
echo "   - Comparison between RL and GA methods"
echo "   - Effect of different mask thresholds"