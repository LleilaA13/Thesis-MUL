#!/bin/bash
# Classification-optimized SalUn experiment runner for tmux

echo "🎯 CLASSIFICATION-OPTIMIZED SALUN EXPERIMENT"
echo "============================================="
echo "Configuration:"
echo "  Method: GA (Gradient Ascent)"
echo "  Learning Rate: 0.05"
echo "  Epochs: 25"
echo "  Mask: with_0.2.pt (blocks 80% weights)"
echo "  Alpha: 8 (strong forgetting signal)"
echo "  Batch Size: 128"
echo ""
echo "Expected Results:"
echo "  Target Forget Accuracy: 5-10%"
echo "  Target Retain Accuracy: 60-70%"
echo ""
echo "🚀 Starting experiment..."
echo ""

# Activate environment and run
conda activate salUN

# Run the experiment
python resnet50_unlearn_dogs.py \
  --unlearn GA \
  --unlearn_lr 0.05 \
  --num_epochs 25 \
  --mask with_0.2.pt \
  --batch_size 128 \
  --alpha 8 \
  --save_model

echo ""
echo "✅ Experiment completed!"
echo "📊 Check results_tracker.py for results"
echo "📁 Model saved with --save_model flag"