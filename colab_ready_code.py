# =============================================================================
# LUCENT FEATURE VISUALIZATION FOR TINY IMAGENET - COLAB READY
# Complete code for comparing original vs unlearned models
# =============================================================================

# 1. INSTALLATION AND SETUP
# -------------------------
!pip install --quiet git+https://github.com/greentfrapp/lucent.git
!pip install --quiet torch torchvision matplotlib

# Import libraries
import torch
from torchvision import models
import matplotlib.pyplot as plt
from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo.util import get_model_layers
import numpy as np

print("✅ All libraries installed and imported successfully!")

# 2. DEVICE SETUP (FIXES THE DEVICE MISMATCH ERROR)
# -------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")

# 3. MODEL LOADING FUNCTION (WITH DEVICE FIX)
# -------------------------------------------
def load_resnet50_tinyimagenet(model_path, model_name="model", device='auto'):
    """Load ResNet50 for Tiny ImageNet with proper device handling"""
    # Automatic device detection
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading {model_name} on device: {device}")
    
    model = models.resnet50(weights=None)  # Updated syntax for newer PyTorch
    # Key: 200 classes for Tiny ImageNet, NOT 1000!
    model.fc = torch.nn.Linear(model.fc.in_features, 200)
    
    # Load your trained weights
    # Fix for PyTorch 2.6+ weights_only security feature
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    try:
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"Warning: Loading {model_name} with strict=False due to: {e}")
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    # Move model to device AFTER loading weights (CRITICAL FOR DEVICE FIX)
    model = model.to(device)
    model.eval()
    print(f"✅ Successfully loaded {model_name} on {device}")
    return model

# 4. TINY IMAGENET CLASS MAPPINGS
# -------------------------------
TINYIMAGENET_CLASSES = {
    'cats': {
        'indices': [0, 66, 102, 131],
        'names': ['Egyptian Cat', 'Tabby Cat', 'Cougar/Mountain Lion', 'Persian Cat']
    },
    'dogs': {
        'indices': [182, 135, 78, 39, 11, 194],
        'names': ['Chihuahua', 'Yorkshire Terrier', 'Golden Retriever', 'Labrador Retriever', 'German Shepherd', 'Standard Poodle']
    },
    'vehicles': {
        'indices': [8, 61, 84, 90, 96, 133, 166],
        'names': ['Convertible', 'Jeep', 'Pickup Truck', 'Sports Car', 'Truck', 'Race Car', 'Taxi']
    }
}

def print_class_info():
    """Print available class mappings"""
    print("\n📋 TINY IMAGENET CLASS MAPPINGS:")
    print("="*50)
    for category, info in TINYIMAGENET_CLASSES.items():
        print(f"\n🔸 {category.upper()}:")
        for idx, name in zip(info['indices'], info['names']):
            print(f"   {idx}: {name}")

print_class_info()

# 5. LOAD YOUR MODELS (UPDATE THESE PATHS!)
# -----------------------------------------
# 🚨 IMPORTANT: Update these paths to your actual model files
original_model_path = "/content/drive/MyDrive/path/to/your/original_model.pth"
unlearned_model_path = "/content/drive/MyDrive/path/to/your/unlearned_model.pth"

# Uncomment and run when you have the correct paths:
# print("Loading models...")
# original_model = load_resnet50_tinyimagenet(original_model_path, "Original Model", device=device)
# unlearned_model = load_resnet50_tinyimagenet(unlearned_model_path, "Unlearned Model", device=device)
# print("Both models loaded successfully! 🎉")

# 6. MODEL PREDICTION TESTING (DEBUG FUNCTION)
# --------------------------------------------
def debug_model_predictions(model, model_name="Model", test_classes=[0, 66, 102, 131]):
    """Test what your model actually predicts for specific classes"""
    print(f"\n🧪 TESTING {model_name.upper()} PREDICTIONS:")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Create a dummy input (random noise)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        outputs = model(dummy_input)
        probs = torch.softmax(outputs, dim=1)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Input device: {dummy_input.device}")
        
        # Check top predictions
        top5_probs, top5_indices = torch.topk(probs, 5)
        print(f"\n🔝 Top 5 predictions for random input:")
        for i in range(5):
            class_idx = top5_indices[0][i].item()
            prob = top5_probs[0][i].item()
            print(f"   Class {class_idx}: {prob:.4f}")
        
        # Check specific class activations
        print(f"\n🎯 Target class activations:")
        for class_idx in test_classes:
            prob = probs[0][class_idx].item()
            activation = outputs[0][class_idx].item()
            class_name = "Unknown"
            
            # Find class name
            for category, info in TINYIMAGENET_CLASSES.items():
                if class_idx in info['indices']:
                    idx_pos = info['indices'].index(class_idx)
                    class_name = info['names'][idx_pos]
                    break
            
            print(f"   Class {class_idx} ({class_name}): prob={prob:.6f}, raw={activation:.3f}")

# 7. VISUALIZATION TESTING FUNCTION
# ---------------------------------
def test_visualization_approaches(model, class_idx=0, class_name="Egyptian Cat", model_name="Model"):
    """Test different ways to visualize the same class"""
    print(f"\n🧪 TESTING {model_name.upper()} VISUALIZATIONS FOR {class_name} (Index {class_idx}):")
    print("="*80)
    
    approaches = [
        ("Direct Label", f"labels:{class_idx}"),
        ("FC Layer Neuron", f"fc:{class_idx}"),
        ("Objective Function", objectives.neuron("fc", class_idx)),
    ]
    
    for approach_name, target in approaches:
        print(f"\n🔹 {approach_name}")
        try:
            img = render.render_vis(model, target, show_inline=True, thresholds=(512,))
            print(f"✅ {approach_name} worked")
        except Exception as e:
            print(f"❌ {approach_name} failed: {e}")

# 7B. DEMONSTRATION OF SECTION 7 (RUNS IMMEDIATELY)
# -------------------------------------------------
print("\n📋 SECTION 7 DEMONSTRATION:")
print("Creating dummy model to test visualization approaches...")

section7_dummy = models.resnet50(weights=None)
section7_dummy.fc = torch.nn.Linear(section7_dummy.fc.in_features, 200)
section7_dummy = section7_dummy.to(device)
section7_dummy.eval()

print("✅ Section 7 function defined and ready!")
print("📝 Demo: Testing visualization approaches on dummy model...")

# Quick test (just one approach to save time)
print("\n🎨 Testing one visualization approach:")
try:
    img = render.render_vis(section7_dummy, "labels:0", show_inline=True, thresholds=(128,))
    print("✅ Section 7 visualization test successful!")
except Exception as e:
    print(f"❌ Section 7 visualization test failed: {e}")

print("✅ Section 7 demonstration complete!")

# 8. COMPLETE COMPARISON FUNCTION
# ------------------------------
def compare_models_visualization(original_model, unlearned_model, forgotten_classes):
    """Complete comparison between original and unlearned models"""
    print("\n🎯 MACHINE UNLEARNING VISUAL ANALYSIS")
    print("="*80)
    
    for class_idx, class_name in forgotten_classes.items():
        print(f"\n📊 COMPARING CLASS {class_idx}: {class_name}")
        print("-" * 60)
        
        # Test predictions first
        debug_model_predictions(original_model, "Original", [class_idx])
        debug_model_predictions(unlearned_model, "Unlearned", [class_idx])
        
        # Test visualizations
        print(f"\n🔵 ORIGINAL MODEL - {class_name}:")
        test_visualization_approaches(original_model, class_idx, class_name, "Original")
        
        print(f"\n🔴 UNLEARNED MODEL - {class_name}:")
        test_visualization_approaches(unlearned_model, class_idx, class_name, "Unlearned")
        
        print("\n" + "="*80)

# 8B. DEMONSTRATION OF SECTION 8 (RUNS IMMEDIATELY)
# -------------------------------------------------
print("\n📋 SECTION 8 DEMONSTRATION:")
print("Creating two dummy models to show how compare_models_visualization() works...")

# Create two dummy models (simulating original vs unlearned)
print("Creating dummy 'original' model...")
dummy_original = models.resnet50(weights=None)
dummy_original.fc = torch.nn.Linear(dummy_original.fc.in_features, 200)
dummy_original = dummy_original.to(device)
dummy_original.eval()

print("Creating dummy 'unlearned' model...")
dummy_unlearned = models.resnet50(weights=None)
dummy_unlearned.fc = torch.nn.Linear(dummy_unlearned.fc.in_features, 200)
dummy_unlearned = dummy_unlearned.to(device)
dummy_unlearned.eval()

# Test with a small subset of classes
test_classes = {0: "Egyptian Cat", 131: "Persian Cat"}

print("✅ Section 8 function defined and ready!")
print("📝 Demo: Running comparison on dummy models with classes:", test_classes)

# Run a quick demo (just predictions, skip visualizations to save time)
print("\n🔍 QUICK DEMO - PREDICTIONS ONLY:")
for class_idx, class_name in test_classes.items():
    print(f"\n📊 Class {class_idx}: {class_name}")
    debug_model_predictions(dummy_original, f"Dummy Original", [class_idx])
    debug_model_predictions(dummy_unlearned, f"Dummy Unlearned", [class_idx])

print("✅ Section 8 demonstration complete!")

# 9. EXAMPLE USAGE WHEN MODELS ARE LOADED
# ---------------------------------------
# Uncomment this when your models are loaded:

# forgotten_classes = {
#     0: "Egyptian Cat",
#     66: "Tabby Cat", 
#     102: "Cougar",
#     131: "Persian Cat"
# }

# # Debug predictions first
# print("🔍 DEBUGGING MODEL PREDICTIONS:")
# debug_model_predictions(original_model, "Original Model", list(forgotten_classes.keys()))
# debug_model_predictions(unlearned_model, "Unlearned Model", list(forgotten_classes.keys()))

# # Run complete comparison
# compare_models_visualization(original_model, unlearned_model, forgotten_classes)

# 10. QUICK TEST WITH DUMMY MODEL (WORKS IMMEDIATELY)
# ---------------------------------------------------
print("\n🧪 QUICK TEST WITH DUMMY MODEL:")
print("="*50)

# Create a dummy model to test the pipeline
dummy_model = models.resnet50(weights=None)
dummy_model.fc = torch.nn.Linear(dummy_model.fc.in_features, 200)
dummy_model = dummy_model.to(device)
dummy_model.eval()

print("✅ Dummy model created successfully")
debug_model_predictions(dummy_model, "Dummy Model", [0, 131])

# Test visualization on dummy model
print("\n🎨 Testing visualization on dummy model:")
try:
    img = render.render_vis(dummy_model, "labels:0", show_inline=True, thresholds=(256,))
    print("✅ Visualization test successful!")
except Exception as e:
    print(f"❌ Visualization test failed: {e}")

print("\n" + "="*80)
print("🎉 SETUP COMPLETE!")
print("📝 TO USE WITH YOUR MODELS:")
print("1. Update the model paths in section 5")
print("2. Uncomment the model loading code")
print("3. Uncomment the comparison code in section 9")
print("4. Run the complete analysis!")
print("="*80)