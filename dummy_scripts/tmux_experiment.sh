#!/bin/bash

echo "🎯 CLASSIFICATION-OPTIMIZED SALUN EXPERIMENT FOR TMUX"
echo "===================================================="
echo "Target: 5% forget accuracy with high retain accuracy"
echo ""
echo "Configuration:"
echo "  • Method: GA (Gradient Ascent)"
echo "  • Learning Rate: 0.05"
echo "  • Epochs: 25" 
echo "  • Mask: with_0.2.pt (restrictive - blocks 80% weights)"
echo "  • Alpha: 8 (strong forgetting signal)"
echo "  • Batch Size: 128"
echo ""
echo "🚀 Starting in 3 seconds..."
sleep 3

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salUN

# Change to correct directory
cd /media/hdd/usr/leyla/Unlearn-Saliency

# Run the experiment
echo "▶️  EXPERIMENT STARTING NOW..."
echo ""

python resnet50_unlearn_dogs.py \
  --unlearn GA \
  --unlearn_lr 0.05 \
  --num_epochs 25 \
  --mask with_0.2.pt \
  --batch_size 128 \
  --alpha 8 \
  --save_model

echo ""
echo "✅ EXPERIMENT COMPLETED!"
echo "📊 Check results with: python results_tracker.py"
echo "📁 Model saved in models/resnet50_dogs_forgetting/"