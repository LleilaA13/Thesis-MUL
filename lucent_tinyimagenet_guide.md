# Lucent Feature Visualization for Tiny ImageNet Models

This guide shows how to use Lucent for feature visualization with models trained on Tiny ImageNet (200 classes) instead of standard ImageNet (1000 classes).

## Key Differences: Tiny ImageNet vs ImageNet

| Dataset | Classes | Class Indices | Example |
|---------|---------|---------------|---------|
| ImageNet | 1000 | 0-999 | `labels:949` (strawberry) |
| Tiny ImageNet | 200 | 0-199 | `labels:0` (Egyptian Cat) |

## 1. Setup and Installation

```python
# Install Lucent
!pip install --quiet git+https://github.com/greentfrapp/lucent.git

# Import required libraries
import torch
from torchvision import models
import matplotlib.pyplot as plt
from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo.util import get_model_layers
```

## 2. Load Both Original and Unlearned Models

**CRITICAL:** Your models must have 200 output classes, not 1000!

```python
def load_resnet50_tinyimagenet(model_path, model_name="model", device='auto'):
    """Load a ResNet50 model trained on Tiny ImageNet with proper device handling"""
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
    
    # Move model to device AFTER loading weights
    model = model.to(device)
    model.eval()
    print(f"✅ Successfully loaded {model_name} on {device}")
    return model

# Load both models for comparison
original_model_path = "/path/to/your/original_model.pth"  # Update this path
unlearned_model_path = "/path/to/your/unlearned_model.pth"  # Update this path

# Choose device (automatic detection by default)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print("Loading models...")
original_model = load_resnet50_tinyimagenet(original_model_path, "Original Model", device=device)
unlearned_model = load_resnet50_tinyimagenet(unlearned_model_path, "Unlearned Model", device=device)
print("Both models loaded successfully! 🎉")
```

## ⚠️ Important: Device Compatibility Fix

**THE ERROR YOU ENCOUNTERED:** `RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same`

**THE SOLUTION:** Make sure your models are loaded on the same device. The updated `load_resnet50_tinyimagenet()` function above handles this automatically by:

1. Detecting if CUDA is available (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`)
2. Loading checkpoint to the correct device (`map_location=device`)
3. Moving model to device after loading weights (`model = model.to(device)`)

**Quick Fix for Existing Code:**
```python
# If you're getting device mismatch errors, ensure both model and inputs are on same device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Move existing models to correct device
original_model = original_model.to(device)
unlearned_model = unlearned_model.to(device)

# Lucent handles device automatically, but if you create custom inputs:
# your_input_tensor = your_input_tensor.to(device)
```

### Model Path Examples

Based on your directory structure, your model paths might look like:

```python
# Examples of typical model paths in your setup
original_model_path = "/content/drive/MyDrive/models/resnet50_cats_forgetting/original/checkpoint.pth.tar"
unlearned_model_path = "/content/drive/MyDrive/models/resnet50_cats_forgetting/mask0_5_salun/RLcheckpoint.pth.tar"

# Or for dogs:
# original_model_path = "/content/drive/MyDrive/models/resnet50_dogs_forgetting/original/checkpoint.pth.tar"
# unlearned_model_path = "/content/drive/MyDrive/models/resnet50_dogs_forgetting/mask0_5_GA_method/checkpoint.pth.tar"

# Or for vehicles:
# original_model_path = "/content/drive/MyDrive/models/resnet50_vehicles_forgetting/original/checkpoint.pth.tar"
# unlearned_model_path = "/content/drive/MyDrive/models/resnet50_vehicles_forgetting/mask0_5_salun/RLcheckpoint.pth.tar"
```

## 3. Tiny ImageNet Class Mapping

Based on your unlearn configuration, here are the class indices:

```python
# Tiny ImageNet class indices from your unlearn_config.py
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
        'indices': [147, 31, 52, 64, 90, 15, 117, 152],
        'names': ['Beach Wagon', 'Car', 'Golf Cart', 'Go-kart', 'Police Van', 'School Bus', 'Sports Car', 'Taxi']
    }
}

# Print available classes
for category, info in TINYIMAGENET_CLASSES.items():
    print(f"\n{category.upper()}:")
    for idx, name in zip(info['indices'], info['names']):
        print(f"  Index {idx}: {name}")
```

### 🐛 Debug: Why Am I Not Getting Cat-Like Results?

```python
def debug_model_predictions(model, model_name="Model"):
    """Debug why label visualizations aren't working"""
    print(f"\n🔍 DEBUGGING {model_name.upper()}:")
    print("="*50)
    
    # 1. Check model architecture
    print(f"📊 Model final layer: {model.fc}")
    print(f"📊 Output classes: {model.fc.out_features}")
    
    # 2. Test with dummy input
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"📊 Output shape: {output.shape}")
        print(f"📊 Output range: {output.min().item():.3f} to {output.max().item():.3f}")
        
        # Get top predictions for random input
        probs = torch.softmax(output, dim=1)
        top5_values, top5_indices = torch.topk(probs, 5)
        print(f"📊 Top 5 predictions for random input:")
        for i, (idx, prob) in enumerate(zip(top5_indices[0], top5_values[0])):
            print(f"     {i+1}. Class {idx.item()}: {prob.item():.4f}")
    
    # 3. Test specific cat classes
    cat_indices = [0, 66, 102, 131]
    cat_names = ['Egyptian Cat', 'Tabby Cat', 'Cougar', 'Persian Cat']
    
    print(f"\n🐱 Testing Cat Class Activations:")
    for idx, name in zip(cat_indices, cat_names):
        if idx < model.fc.out_features:
            print(f"✅ Class {idx} ({name}): Valid index")
        else:
            print(f"❌ Class {idx} ({name}): INVALID - exceeds {model.fc.out_features} classes!")

# Debug both models
debug_model_predictions(original_model, "Original Model")
debug_model_predictions(unlearned_model, "Unlearned Model")
```

### 🎯 Test Different Visualization Approaches

```python
def test_visualization_approaches(model, class_idx=0, class_name="Egyptian Cat"):
    """Test different ways to visualize the same class"""
    print(f"\n🧪 TESTING DIFFERENT APPROACHES FOR {class_name} (Index {class_idx}):")
    print("="*60)
    
    # Approach 1: Direct label
    print("🔹 Approach 1: Direct Label")
    try:
        _ = render.render_vis(model, f"labels:{class_idx}", show_inline=True)
        print("✅ Direct label worked")
    except Exception as e:
        print(f"❌ Direct label failed: {e}")
    
    # Approach 2: FC layer neuron
    print(f"\n🔹 Approach 2: FC Layer Neuron")
    try:
        _ = render.render_vis(model, f"fc:{class_idx}", show_inline=True)
        print("✅ FC neuron worked")
    except Exception as e:
        print(f"❌ FC neuron failed: {e}")
    
    # Approach 3: Objective function
    print(f"\n🔹 Approach 3: Objective Function")
    try:
        obj = objectives.neuron("fc", class_idx)
        _ = render.render_vis(model, obj, show_inline=True)
        print("✅ Objective function worked")
    except Exception as e:
        print(f"❌ Objective function failed: {e}")
    
    # Approach 4: Different thresholds
    print(f"\n🔹 Approach 4: Higher Thresholds")
    try:
        _ = render.render_vis(model, f"labels:{class_idx}", 
                            thresholds=(512, 1024), show_inline=True)
        print("✅ Higher thresholds worked")
    except Exception as e:
        print(f"❌ Higher thresholds failed: {e}")

# Test Egyptian Cat (index 0) on both models
print("🔵 ORIGINAL MODEL TESTS:")
test_visualization_approaches(original_model, 0, "Egyptian Cat")

print("\n🔴 UNLEARNED MODEL TESTS:")
test_visualization_approaches(unlearned_model, 0, "Egyptian Cat")
```

### 🏗️ ResNet50 Architecture Deep Dive

```python
def explore_resnet50_architecture(model):
    """Complete exploration of ResNet50 structure for Lucent"""
    print("🏗️ RESNET50 COMPLETE ARCHITECTURE:")
    print("="*60)
    
    # Print model summary
    print(f"📊 Model type: {type(model).__name__}")
    print(f"📊 Final layer: {model.fc}")
    print(f"📊 Output classes: {model.fc.out_features}")
    
    print("\n🔍 DETAILED LAYER STRUCTURE:")
    print("-" * 40)
    
    # Group layers by blocks
    layer_groups = {
        "Initial Layers": ["conv1", "bn1", "relu", "maxpool"],
        "Layer1 (64 channels)": [],
        "Layer2 (128 channels)": [],
        "Layer3 (256 channels)": [],
        "Layer4 (512 channels)": [],
        "Final Layers": ["avgpool", "fc"]
    }
    
    # Collect all layer names
    all_layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            all_layers.append(name)
            
            # Categorize layers
            if name.startswith("layer1"):
                layer_groups["Layer1 (64 channels)"].append(name)
            elif name.startswith("layer2"):
                layer_groups["Layer2 (128 channels)"].append(name)
            elif name.startswith("layer3"):
                layer_groups["Layer3 (256 channels)"].append(name)
            elif name.startswith("layer4"):
                layer_groups["Layer4 (512 channels)"].append(name)
    
    # Print categorized layers
    for group_name, layers in layer_groups.items():
        print(f"\n📁 {group_name}:")
        for layer in layers:
            print(f"   {layer}")
    
    print(f"\n📊 Total layers found: {len(all_layers)}")
    
    return all_layers

def get_resnet50_visualization_targets():
    """Get the best layers for visualization in ResNet50"""
    print("\n🎯 BEST LAYERS FOR VISUALIZATION:")
    print("="*50)
    
    viz_targets = {
        "Early Features (Low-level)": {
            "conv1": "Initial edge/texture detection",
            "layer1.0.conv1": "Basic shape detection",
            "layer1.0.conv2": "Simple pattern combinations",
            "layer1.1.conv1": "Enhanced edge detection",
        },
        "Mid-level Features": {
            "layer2.0.conv1": "Complex shapes and textures",
            "layer2.0.conv2": "Object parts detection",
            "layer2.1.conv1": "Refined part detection",
            "layer2.2.conv1": "Part combinations",
        },
        "High-level Features": {
            "layer3.0.conv1": "Object-like patterns",
            "layer3.0.conv2": "Complex object parts",
            "layer3.2.conv1": "Abstract object features",
            "layer3.4.conv1": "Class-specific patterns",
        },
        "Abstract Features": {
            "layer4.0.conv1": "High-level semantic features",
            "layer4.0.conv2": "Class-discriminative patterns",
            "layer4.1.conv1": "Very abstract representations",
            "layer4.2.conv1": "Near-classification features",
        },
        "Classification": {
            "fc": "Final class predictions (use neuron index)"
        }
    }
    
    for category, layers in viz_targets.items():
        print(f"\n🔹 {category}:")
        for layer, description in layers.items():
            print(f"   {layer:20} → {description}")
    
    return viz_targets

# Explore your original model
all_layers = explore_resnet50_architecture(original_model)
viz_targets = get_resnet50_visualization_targets()
```

### 🧪 Test ResNet50 Layer Visualizations

```python
def test_resnet50_layers(model, model_name="Model"):
    """Test visualization on key ResNet50 layers"""
    print(f"\n🧪 TESTING {model_name.upper()} LAYER VISUALIZATIONS:")
    print("="*60)
    
    # Define test layers with expected channel counts
    test_layers = [
        ("conv1", 32, "Should show edge/texture detectors"),
        ("layer1.0.conv1", 32, "Basic shape detectors"),
        ("layer1.0.conv2", 32, "Simple pattern combinations"),
        ("layer2.0.conv1", 64, "Complex shapes and textures"),
        ("layer2.0.conv2", 64, "Object parts"),
        ("layer3.0.conv1", 128, "Object-like patterns"),
        ("layer3.0.conv2", 128, "Complex object parts"),
        ("layer4.0.conv1", 256, "High-level semantic features"),
        ("layer4.0.conv2", 256, "Class-discriminative patterns"),
    ]
    
    successful_layers = []
    
    for layer_name, test_channel, description in test_layers:
        print(f"\n🔍 Testing {layer_name} (channel {test_channel})")
        print(f"   Expected: {description}")
        
        try:
            # Test if layer exists
            layer_module = model
            for part in layer_name.split('.'):
                layer_module = getattr(layer_module, part)
            
            # Get layer output channels
            if hasattr(layer_module, 'out_channels'):
                max_channels = layer_module.out_channels
                # Use a safe channel index
                safe_channel = min(test_channel, max_channels - 1)
                print(f"   ✅ Layer found - {max_channels} channels, testing channel {safe_channel}")
                
                # Try visualization
                _ = render.render_vis(model, f"{layer_name}:{safe_channel}", 
                                    thresholds=(256,), show_inline=True)
                successful_layers.append((layer_name, safe_channel, description))
                print(f"   ✅ Visualization successful!")
                
            else:
                print(f"   ⚠️  Layer found but no out_channels attribute")
                
        except AttributeError:
            print(f"   ❌ Layer not found in model")
        except Exception as e:
            print(f"   ❌ Visualization failed: {str(e)[:100]}...")
    
    print(f"\n📊 SUMMARY: {len(successful_layers)} layers visualized successfully")
    return successful_layers

# Test your original model
successful_original = test_resnet50_layers(original_model, "Original Model")
```

### 🐱 Focus on Cat Classification Path

```python
def trace_cat_classification_path(model):
    """Trace how cat features develop through ResNet50"""
    print("\n🐱 TRACING CAT FEATURE DEVELOPMENT:")
    print("="*50)
    
    cat_indices = [0, 66, 102, 131]  # Your cat classes
    cat_names = ['Egyptian Cat', 'Tabby Cat', 'Cougar', 'Persian Cat']
    
    # Test different layers for cat-specific features
    cat_relevant_layers = [
        ("layer1.0.conv2", 16, "Basic cat shapes (ears, whiskers)"),
        ("layer2.0.conv2", 32, "Cat facial features"),
        ("layer2.2.conv2", 48, "Cat body parts"),
        ("layer3.0.conv2", 64, "Cat-like patterns"),
        ("layer3.2.conv2", 96, "Specific cat features"),
        ("layer3.4.conv2", 128, "Cat breed differences"),
        ("layer4.0.conv2", 160, "Abstract cat concepts"),
        ("layer4.2.conv2", 256, "Pre-classification cat features"),
    ]
    
    print("🔍 Testing layers for cat-relevant features:")
    for layer_name, channel, description in cat_relevant_layers:
        print(f"\n📍 {layer_name}:{channel} - {description}")
        try:
            _ = render.render_vis(model, f"{layer_name}:{channel}", 
                                thresholds=(256,), show_inline=True)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n🎯 Testing final classification neurons for cats:")
    for idx, name in zip(cat_indices, cat_names):
        print(f"\n🐱 {name} (neuron {idx}):")
        try:
            _ = render.render_vis(model, f"fc:{idx}", 
                                thresholds=(256, 512), show_inline=True)
        except Exception as e:
            print(f"   ❌ Failed: {e}")

### 🔍 Compare Your ResNet50 vs Pretrained ResNet50

```python
def compare_resnet50_architectures():
    """Compare your custom ResNet50 with standard pretrained ResNet50"""
    print("🔍 COMPARING RESNET50 ARCHITECTURES:")
    print("="*70)
    
    # Load standard pretrained ResNet50 for comparison
    print("📥 Loading standard pretrained ResNet50...")
    pretrained_resnet50 = models.resnet50(weights='IMAGENET1K_V1')
    pretrained_resnet50.eval()
    
    print(f"\n📊 ARCHITECTURE COMPARISON:")
    print("-" * 50)
    
    models_to_compare = [
        ("Your Original Model", original_model),
        ("Standard Pretrained", pretrained_resnet50)
    ]
    
    for model_name, model in models_to_compare:
        print(f"\n🏗️ {model_name}:")
        print(f"   Final layer: {model.fc}")
        print(f"   Output classes: {model.fc.out_features}")
        print(f"   Input features to FC: {model.fc.in_features}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Check if model is in eval mode
        print(f"   Training mode: {model.training}")
    
    return pretrained_resnet50

def test_pretrained_vs_custom_visualization():
    """Test visualization on both pretrained and your custom model"""
    print(f"\n🧪 VISUALIZATION COMPARISON:")
    print("="*60)
    
    # Load pretrained for comparison
    pretrained_model = models.resnet50(weights='IMAGENET1K_V1')
    pretrained_model.eval()
    
    # Test same layer on both models
    test_layer = "layer3.0.conv1"
    test_channel = 64
    
    print(f"🔍 Testing {test_layer}:{test_channel} on both models:")
    
    print(f"\n🔵 Your Original Model ({test_layer}:{test_channel}):")
    try:
        _ = render.render_vis(original_model, f"{test_layer}:{test_channel}", 
                            thresholds=(256,), show_inline=True)
        print("✅ Your model visualization successful")
    except Exception as e:
        print(f"❌ Your model failed: {e}")
    
    print(f"\n🟢 Pretrained Model ({test_layer}:{test_channel}):")
    try:
        _ = render.render_vis(pretrained_model, f"{test_layer}:{test_channel}", 
                            thresholds=(256,), show_inline=True)
        print("✅ Pretrained model visualization successful")
    except Exception as e:
        print(f"❌ Pretrained model failed: {e}")
    
    return pretrained_model

def analyze_layer_differences(custom_model, pretrained_model):
    """Analyze differences between your model and pretrained model layers"""
    print(f"\n🔬 DETAILED LAYER ANALYSIS:")
    print("="*60)
    
    # Get layer information for both models
    def get_layer_info(model, model_name):
        layer_info = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and len(list(module.children())) == 0:
                layer_info[name] = {
                    'shape': module.weight.shape if hasattr(module, 'weight') else None,
                    'type': type(module).__name__,
                    'requires_grad': module.weight.requires_grad if hasattr(module, 'weight') else None
                }
        return layer_info
    
    custom_layers = get_layer_info(custom_model, "Custom")
    pretrained_layers = get_layer_info(pretrained_model, "Pretrained")
    
    # Compare key layers
    key_layers = [
        "conv1", "layer1.0.conv1", "layer1.0.conv2", 
        "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1", "fc"
    ]
    
    print(f"📋 Key Layer Comparison:")
    print(f"{'Layer':<20} {'Custom Shape':<20} {'Pretrained Shape':<20} {'Match'}")
    print("-" * 70)
    
    for layer in key_layers:
        custom_shape = custom_layers.get(layer, {}).get('shape', 'Missing')
        pretrained_shape = pretrained_layers.get(layer, {}).get('shape', 'Missing')
        match = "✅" if custom_shape == pretrained_shape else "❌"
        
        print(f"{layer:<20} {str(custom_shape):<20} {str(pretrained_shape):<20} {match}")

def test_imagenet_vs_tinyimagenet_classes():
    """Test visualization of similar classes between ImageNet and TinyImageNet"""
    print(f"\n🎯 IMAGENET VS TINY-IMAGENET CLASS COMPARISON:")
    print("="*60)
    
    # Load pretrained for ImageNet comparison
    pretrained_model = models.resnet50(weights='IMAGENET1K_V1')
    pretrained_model.eval()
    
    # ImageNet has different cat classes - let's find similar ones
    # Egyptian cat might be similar to tabby cat in ImageNet
    imagenet_cat_classes = {
        281: "tabby cat",
        282: "tiger cat", 
        283: "Persian cat",
        284: "Siamese cat",
        285: "Egyptian cat"
    }
    
    tinyimagenet_cat_classes = {
        0: "Egyptian Cat",
        66: "Tabby Cat",
        102: "Cougar/Mountain Lion", 
        131: "Persian Cat"
    }
    
    print(f"🐱 ImageNet Cat Classes (1000 total classes):")
    for idx, name in imagenet_cat_classes.items():
        print(f"   Class {idx}: {name}")
        try:
            print(f"   Visualizing ImageNet {name}:")
            _ = render.render_vis(pretrained_model, f"labels:{idx}", 
                                thresholds=(256,), show_inline=True)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n🐱 Your TinyImageNet Cat Classes (200 total classes):")
    for idx, name in tinyimagenet_cat_classes.items():
        print(f"   Class {idx}: {name}")
        try:
            print(f"   Visualizing TinyImageNet {name}:")
            _ = render.render_vis(original_model, f"labels:{idx}", 
                                thresholds=(256,), show_inline=True)
        except Exception as e:
            print(f"   ❌ Failed: {e}")

# Run all comparisons
pretrained_model = compare_resnet50_architectures()
test_pretrained_vs_custom_visualization()
analyze_layer_differences(original_model, pretrained_model)
### 🚨 Debug: Original Model Not Showing Cat Features

```python
def debug_original_model_cat_features():
    """Debug why original model doesn't show cat-like features"""
    print("🚨 DEBUGGING ORIGINAL MODEL - CAT VISUALIZATION ISSUES:")
    print("="*70)
    
    # 1. Verify model predictions on actual data
    print("📊 Step 1: Check model's actual predictions")
    
    # Create a dummy input to test model behavior
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        output = original_model(dummy_input)
        probs = torch.softmax(output, dim=1)
        
        # Check if cat classes have reasonable probabilities
        cat_indices = [0, 66, 102, 131]
        cat_names = ['Egyptian Cat', 'Tabby Cat', 'Cougar', 'Persian Cat']
        
        print(f"🐱 Cat class predictions for random input:")
        for idx, name in zip(cat_indices, cat_names):
            prob = probs[0, idx].item()
            print(f"   Class {idx} ({name}): {prob:.6f} ({prob*100:.4f}%)")
        
        # Get top 10 predictions
        top10_values, top10_indices = torch.topk(probs, 10)
        print(f"\n🔝 Top 10 predictions for random input:")
        for i, (idx, prob) in enumerate(zip(top10_indices[0], top10_values[0])):
            print(f"   {i+1}. Class {idx.item()}: {prob.item():.6f}")
    
    # 2. Check if the model was trained properly on cats
    print(f"\n📈 Step 2: Model training analysis")
    
    # Test different visualization approaches for Persian Cat
    print(f"\n🐱 Step 3: Testing different approaches for Persian Cat (131):")
    
    approaches = [
        ("labels:131", "Direct label approach"),
        ("fc:131", "Final layer neuron approach"),
        (f"fc:{131}", "FC layer with specific neuron"),
    ]
    
    for approach, description in approaches:
        print(f"\n🔍 Testing: {approach} ({description})")
        try:
            _ = render.render_vis(original_model, approach, 
                                thresholds=(512, 1024), show_inline=True)
            print(f"✅ {approach} worked")
        except Exception as e:
            print(f"❌ {approach} failed: {e}")

def check_class_mapping_accuracy():
    """Verify if class indices are correctly mapped"""
    print(f"\n🗺️ VERIFYING CLASS MAPPING:")
    print("="*50)
    
    print(f"📋 Your class mapping from unlearn_config.py:")
    cat_mapping = {
        0: "Egyptian Cat",
        66: "Tabby Cat", 
        102: "Cougar/Mountain Lion",
        131: "Persian Cat"
    }
    
    for idx, name in cat_mapping.items():
        print(f"   Index {idx}: {name}")
    
    print(f"\n❓ CRITICAL QUESTIONS:")
    print(f"1. Was your model actually trained to recognize these specific cat breeds?")
    print(f"2. Are these the correct Tiny ImageNet class indices?")
    print(f"3. Was the training successful for cat classes?")
    
    # Test if model has learned meaningful features for any class
    print(f"\n🧪 Testing some non-cat classes for comparison:")
    
    # Test some vehicle classes (should be easier to visualize)
    vehicle_classes = {31: "Car", 15: "School Bus"}
    
    for idx, name in vehicle_classes.items():
        print(f"\n🚗 Testing {name} (Class {idx}):")
        try:
            _ = render.render_vis(original_model, f"labels:{idx}", 
                                thresholds=(1024,), show_inline=True)
            print(f"✅ {name} visualization successful")
        except Exception as e:
            print(f"❌ {name} failed: {e}")

def compare_with_known_working_class():
    """Compare with a class that should definitely work"""
    print(f"\n🔍 COMPARING WITH DIFFERENT THRESHOLDS AND CLASSES:")
    print("="*60)
    
    # Test Persian Cat with different settings
    test_configs = [
        (256, "Low threshold - should show basic patterns"),
        (512, "Medium threshold - should show clearer features"),
        (1024, "High threshold - should show detailed features"),
        (2048, "Very high threshold - maximum detail"),
    ]
    
    print(f"🐱 Persian Cat (131) with different thresholds:")
    for threshold, description in test_configs:
        print(f"\n📊 Threshold {threshold}: {description}")
        try:
            _ = render.render_vis(original_model, "labels:131", 
                                thresholds=(threshold,), show_inline=True)
        except Exception as e:
            print(f"❌ Threshold {threshold} failed: {e}")
    
    # Test a completely different approach - layer visualization
    print(f"\n🔬 Testing layer-based visualization instead of class-based:")
    try:
        print(f"Testing layer3.0.conv2:50 (should show mid-level features):")
        _ = render.render_vis(original_model, "layer3.0.conv2:50", 
                            thresholds=(512,), show_inline=True)
    except Exception as e:
        print(f"❌ Layer visualization failed: {e}")

# Run all debugging steps
debug_original_model_cat_features()
check_class_mapping_accuracy()
compare_with_known_working_class()
```

### 🎯 Key Questions to Answer:

```python
def analyze_potential_issues():
    """Analyze why original model doesn't show cats"""
    print(f"\n🤔 POTENTIAL ISSUES ANALYSIS:")
    print("="*50)
    
    issues = [
        "🔸 Model wasn't trained properly on cat classes",
        "🔸 Class indices are wrong (not actually cats in Tiny ImageNet)",
        "🔸 Model was overtrained/underfitted on cats", 
        "🔸 Lucent parameters need adjustment",
        "🔸 Model architecture has issues",
        "🔸 The 'original' model is actually already modified"
    ]
    
    for issue in issues:
        print(issue)
    
    print(f"\n💡 NEXT STEPS:")
    print(f"1. Check if ANY class produces recognizable features")
    print(f"2. Verify the model was actually trained on cats")
    print(f"3. Test with a known-good pretrained model")
    print(f"4. Check your training logs/accuracy on cat classes")

### 🏷️ Verify Actual Tiny ImageNet Class Labels

```python
def verify_tinyimagenet_class_labels():
    """Check the actual class labels in your Tiny ImageNet dataset"""
    print("🏷️ VERIFYING TINY IMAGENET CLASS LABELS:")
    print("="*60)
    
    # Read the wnids.txt file (WordNet IDs in order)
    try:
        wnids_path = "/media/hdd/usr/leyla/Unlearn-Saliency/datasets/tiny-imagenet-200/wnids.txt"
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        
        print(f"✅ Found {len(wnids)} class IDs in wnids.txt")
        
        # Check specific indices we're interested in
        cat_indices = [0, 66, 102, 131]
        cat_names_claimed = ['Egyptian Cat', 'Tabby Cat', 'Cougar/Mountain Lion', 'Persian Cat']
        
        print(f"\n🐱 CHECKING YOUR CAT CLASS MAPPINGS:")
        print("-" * 50)
        
        for idx, claimed_name in zip(cat_indices, cat_names_claimed):
            if idx < len(wnids):
                actual_wnid = wnids[idx]
                print(f"Index {idx:3d}: {claimed_name:20} → WordNet ID: {actual_wnid}")
            else:
                print(f"Index {idx:3d}: {claimed_name:20} → ❌ OUT OF RANGE!")
        
        # Show all wnids for manual verification
        print(f"\n📋 ALL WORDNET IDs (first 20):")
        for i, wnid in enumerate(wnids[:20]):
            print(f"   {i:3d}: {wnid}")
        
        print(f"\n📋 WORDNET IDs AROUND YOUR CAT INDICES:")
        for idx in cat_indices:
            start = max(0, idx-2)
            end = min(len(wnids), idx+3)
            print(f"\n   Around index {idx}:")
            for i in range(start, end):
                marker = " 👈" if i == idx else "   "
                print(f"     {i:3d}: {wnids[i]}{marker}")
        
        return wnids
        
    except FileNotFoundError:
        print(f"❌ Could not find wnids.txt at expected path")
        print(f"   Expected: {wnids_path}")
        print(f"   Please check if the path is correct")
        return None

def lookup_wordnet_meanings():
    """Try to decode WordNet IDs to human-readable names"""
    print(f"\n🔍 LOOKING UP WORDNET MEANINGS:")
    print("="*50)
    
    # Read words.txt if available for WordNet ID meanings
    try:
        words_path = "/media/hdd/usr/leyla/Unlearn-Saliency/datasets/tiny-imagenet-200/words.txt"
        with open(words_path, 'r') as f:
            word_mappings = {}
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    wnid, description = parts
                    word_mappings[wnid] = description
        
        print(f"✅ Found {len(word_mappings)} WordNet definitions")
        
        # Look up your cat class WordNet IDs
        wnids = verify_tinyimagenet_class_labels()
        if wnids:
            cat_indices = [0, 66, 102, 131]
            cat_names_claimed = ['Egyptian Cat', 'Tabby Cat', 'Cougar/Mountain Lion', 'Persian Cat']
            
            print(f"\n🐱 ACTUAL MEANINGS OF YOUR CAT CLASSES:")
            print("-" * 60)
            
            for idx, claimed_name in zip(cat_indices, cat_names_claimed):
                if idx < len(wnids):
                    wnid = wnids[idx]
                    actual_meaning = word_mappings.get(wnid, "❌ Not found in words.txt")
                    match = "✅" if any(cat_word in actual_meaning.lower() 
                                     for cat_word in ['cat', 'feline', 'tiger', 'lion', 'cougar', 'persian']) else "❌"
                    
                    print(f"Index {idx:3d}: {claimed_name}")
                    print(f"         WordNet ID: {wnid}")
                    print(f"         Actual meaning: {actual_meaning}")
                    print(f"         Is cat-related? {match}")
                    print()
        
        return word_mappings
        
    except FileNotFoundError:
        print(f"❌ Could not find words.txt")
        return None

def check_unlearn_config_accuracy():
    """Cross-check your unlearn_config.py with actual dataset"""
    print(f"\n🔍 CHECKING UNLEARN_CONFIG.PY ACCURACY:")
    print("="*50)
    
    # Try to read your unlearn_config.py
    try:
        import sys
        sys.path.append('/media/hdd/usr/leyla/Unlearn-Saliency')
        import unlearn_config
        
        print(f"✅ Successfully imported unlearn_config.py")
        
        # Get cat configuration
        cat_config = unlearn_config.get_forget_class_config('cats')
        
        print(f"\n📋 YOUR UNLEARN_CONFIG CAT MAPPING:")
        for idx, name in zip(cat_config['indices'], cat_config['names']):
            print(f"   Index {idx}: {name}")
        
        print(f"\n🧪 VERIFYING AGAINST ACTUAL DATASET:")
        wnids = verify_tinyimagenet_class_labels()
        word_mappings = lookup_wordnet_meanings()
        
        if wnids and word_mappings:
            print(f"\n📊 ACCURACY CHECK:")
            for idx, claimed_name in zip(cat_config['indices'], cat_config['names']):
                if idx < len(wnids):
                    wnid = wnids[idx]
                    actual_meaning = word_mappings.get(wnid, "Unknown")
                    
                    is_accurate = any(cat_word in actual_meaning.lower() 
                                    for cat_word in ['cat', 'feline', 'tiger', 'lion', 'cougar', 'persian'])
                    
                    status = "✅ CORRECT" if is_accurate else "❌ WRONG"
                    print(f"   Index {idx} ({claimed_name}): {status}")
                    print(f"     → Actually: {actual_meaning}")
        
    except ImportError as e:
        print(f"❌ Could not import unlearn_config.py: {e}")
    except Exception as e:
        print(f"❌ Error checking unlearn_config: {e}")

# Run all verification steps
wnids = verify_tinyimagenet_class_labels()
word_mappings = lookup_wordnet_meanings()
check_unlearn_config_accuracy()
```

## 4. Visual Comparison: Original vs Unlearned Models

This is the key section for analyzing the effect of machine unlearning!

### Side-by-Side Class Comparisons

```python
def compare_models_visualization(original_model, unlearned_model, class_idx, class_name, thresholds=(256,)):
    """Compare feature visualization between original and unlearned models"""
    print(f"\n{'='*60}")
    print(f"COMPARING: {class_name} (Index {class_idx})")
    print(f"{'='*60}")
    
    print(f"🔵 ORIGINAL MODEL - {class_name}:")
    _ = render.render_vis(original_model, f"labels:{class_idx}", 
                         thresholds=thresholds, show_inline=True)
    
    print(f"🔴 UNLEARNED MODEL - {class_name}:")
    _ = render.render_vis(unlearned_model, f"labels:{class_idx}", 
                         thresholds=thresholds, show_inline=True)
    
    print(f"Analysis: Compare the patterns above to see unlearning effects")
    print("-" * 60)

# Compare forgotten classes (cats example)
forgotten_cats = {
    0: "Egyptian Cat",
    66: "Tabby Cat", 
    102: "Cougar/Mountain Lion",
    131: "Persian Cat"
}

print("🐱 ANALYZING CAT FORGETTING RESULTS:")
for idx, name in forgotten_cats.items():
    compare_models_visualization(original_model, unlearned_model, idx, name)
```

### Retained Classes Comparison

```python
# Compare retained classes to ensure they weren't affected
retained_classes = {
    182: "Chihuahua",
    135: "Yorkshire Terrier", 
    31: "Car",
    15: "School Bus"
}

print("\n🔒 ANALYZING RETAINED CLASSES (Should look similar):")
for idx, name in retained_classes.items():
    compare_models_visualization(original_model, unlearned_model, idx, name)
```

### Batch Comparison for All Forgotten Classes

```python
def batch_compare_forgotten_classes(original_model, unlearned_model, class_type="cats"):
    """Compare all classes of a specific type"""
    
    classes_map = {
        "cats": {
            0: "Egyptian Cat", 66: "Tabby Cat", 
            102: "Cougar/Mountain Lion", 131: "Persian Cat"
        },
        "dogs": {
            182: "Chihuahua", 135: "Yorkshire Terrier", 78: "Golden Retriever",
            39: "Labrador Retriever", 11: "German Shepherd", 194: "Standard Poodle"
        },
        "vehicles": {
            147: "Beach Wagon", 31: "Car", 52: "Golf Cart", 64: "Go-kart",
            90: "Police Van", 15: "School Bus", 117: "Sports Car", 152: "Taxi"
        }
    }
    
    if class_type not in classes_map:
        print(f"Error: class_type must be one of {list(classes_map.keys())}")
        return
    
    classes = classes_map[class_type]
    print(f"\n🎯 COMPLETE {class_type.upper()} FORGETTING ANALYSIS:")
    print(f"Analyzing {len(classes)} {class_type} classes...")
    
    for idx, name in classes.items():
        compare_models_visualization(original_model, unlearned_model, idx, name)

# Run comparison for your forgotten class type
batch_compare_forgotten_classes(original_model, unlearned_model, "cats")  # Change to "dogs" or "vehicles" as needed
```

### Layer-Level Comparisons

```python
def compare_layer_activations(original_model, unlearned_model, layer_name, channel_idx, class_idx):
    """Compare specific layer activations between models"""
    print(f"\n🔬 LAYER ANALYSIS: {layer_name}:{channel_idx} for class {class_idx}")
    
    print(f"Original Model - {layer_name}:")
    _ = render.render_vis(original_model, f"{layer_name}:{channel_idx}", show_inline=True)
    
    print(f"Unlearned Model - {layer_name}:")
    _ = render.render_vis(unlearned_model, f"{layer_name}:{channel_idx}", show_inline=True)

# Compare early layer features
compare_layer_activations(original_model, unlearned_model, "layer1.0.conv1", 10, 0)

# Compare deeper layer features  
compare_layer_activations(original_model, unlearned_model, "layer3.0.conv1", 50, 0)
```

### Visualize Convolutional Layers

```python
# Explore available layers
print("Available layers in the model:")
layers = get_model_layers(model)
print(layers)

# Visualize specific convolutional layers
print("Layer1 Conv1 - Channel 0:")
_ = render.render_vis(model, "layer1.0.conv1:0", show_inline=True)

print("Layer2 Conv1 - Channel 10:")
_ = render.render_vis(model, "layer2.0.conv1:10", show_inline=True)

print("Layer3 Conv1 - Channel 50:")
_ = render.render_vis(model, "layer3.0.conv1:50", show_inline=True)
```

## 5. Advanced Visualization Options

### Custom Objectives

```python
# Combine multiple classes
cat_obj = objectives.channel("fc", 0)  # Egyptian Cat
dog_obj = objectives.channel("fc", 182)  # Chihuahua
combined_obj = cat_obj + dog_obj

print("Combined Cat + Dog visualization:")
_ = render.render_vis(model, combined_obj, show_inline=True)
```

### Different Parameterizations

```python
# Fourier parameterization (default)
param_f = lambda: param.image(128, fft=True, decorrelate=True)
_ = render.render_vis(model, "labels:0", param_f, show_inline=True)

# Pixel space parameterization
param_f = lambda: param.image(128, fft=False, decorrelate=False)
_ = render.render_vis(model, "labels:0", param_f, transforms=[], show_inline=True)

# CPPN parameterization
cppn_param_f = lambda: param.cppn(128)
cppn_opt = lambda params: torch.optim.Adam(params, 5e-3)
_ = render.render_vis(model, "labels:0", cppn_param_f, cppn_opt, transforms=[], show_inline=True)
```

### Custom Transforms

```python
# Standard transforms
standard_transforms = [
    transform.pad(12),
    transform.jitter(8),
    transform.random_scale([0.9, 0.95, 1.05, 1.1] + [1]*4),
    transform.random_rotate(list(range(-10, 11)) + [0]*5),
    transform.jitter(4),
]

_ = render.render_vis(
    model, 
    "labels:0", 
    transforms=standard_transforms,
    show_inline=True
)
```

## 6. Batch Visualization for Multiple Classes

```python
# Visualize all cat classes
cat_indices = [0, 66, 102, 131]
cat_names = ['Egyptian Cat', 'Tabby Cat', 'Cougar', 'Persian Cat']

for idx, name in zip(cat_indices, cat_names):
    print(f"\n{name} (index {idx}):")
    _ = render.render_vis(model, f"labels:{idx}", show_inline=True)
```

## 7. Troubleshooting Common Issues

### 🚨 Device Mismatch Errors (MOST COMMON)

**Error:** `RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same`

**Cause:** Your model is on GPU (`torch.cuda.FloatTensor`) but inputs are on CPU (`torch.FloatTensor`), or vice versa.

**Solution:**
```python
# 1. Use the updated load function (automatically handles device)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_resnet50_tinyimagenet(model_path, device=device)

# 2. Or manually fix existing models
model = model.to(device)  # Move model to chosen device

# 3. Check device status
print(f"Model device: {next(model.parameters()).device}")
print(f"Available device: {device}")
```

**Prevention:** Always use the same device for models, inputs, and computations. The updated guide functions handle this automatically.

### Wrong Number of Classes
```python
# ❌ WRONG - This is for ImageNet (1000 classes)
model = models.resnet50(weights='IMAGENET1K_V1')

# ✅ CORRECT - This is for Tiny ImageNet (200 classes)
model = models.resnet50(weights=None)  # No pretrained weights
model.fc = torch.nn.Linear(model.fc.in_features, 200)
```

### Wrong Class Indices
```python
# ❌ WRONG - ImageNet strawberry index
_ = render.render_vis(model, "labels:949", show_inline=True)

# ✅ CORRECT - Tiny ImageNet valid indices (0-199)
_ = render.render_vis(model, "labels:0", show_inline=True)  # Egyptian Cat
```

### Model Loading Issues
```python
# Handle different checkpoint formats and PyTorch version compatibility
try:
    # For PyTorch 2.6+, explicitly set weights_only=False
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
except KeyError:
    model.load_state_dict(checkpoint)
except RuntimeError as e:
    print(f"State dict loading error: {e}")
    # Try with strict=False if there are minor mismatches
    model.load_state_dict(checkpoint, strict=False)
```

## 8. Complete Comparison Example Script

```python
# Complete example for comparing original vs unlearned models
import torch
from torchvision import models
from lucent.optvis import render, objectives

# 1. Load both models with proper device handling
def load_resnet50_tinyimagenet(model_path, model_name="model", device='auto'):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = models.resnet50(weights=None)  # Updated syntax for newer PyTorch
    model.fc = torch.nn.Linear(model.fc.in_features, 200)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=False)
    model = model.to(device)  # Ensure model is on correct device
    model.eval()
    print(f"✅ Loaded {model_name} on {device}")
    return model

# 2. Load your models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

original_model = load_resnet50_tinyimagenet("/path/to/original_model.pth", "Original", device)
unlearned_model = load_resnet50_tinyimagenet("/path/to/unlearned_model.pth", "Unlearned", device)

# 3. Define your forgotten classes
forgotten_classes = {
    0: "Egyptian Cat",
    66: "Tabby Cat", 
    102: "Cougar",
    131: "Persian Cat"
}

# 4. Run complete comparison
print("🎯 MACHINE UNLEARNING VISUAL ANALYSIS")
print("="*50)

for idx, name in forgotten_classes.items():
    print(f"\n📊 ANALYZING: {name} (Index {idx})")
    print("-" * 40)
    
    print(f"🔵 Original Model:")
    _ = render.render_vis(original_model, f"labels:{idx}", show_inline=True)
    
    print(f"🔴 Unlearned Model:")
    _ = render.render_vis(unlearned_model, f"labels:{idx}", show_inline=True)
    
    print(f"💡 Expected: Unlearned model should show degraded/different patterns")

print("\n✅ Visual comparison complete!")
print("Look for differences in pattern quality, coherence, and recognizability")
```

## What to Look For in Comparisons

### Successful Unlearning Indicators:
- 🔴 **Degraded patterns** in unlearned model
- 🔴 **Less coherent features** for forgotten classes  
- 🔴 **Noisy or abstract visualizations** instead of clear class features
- 🔴 **Reduced activation strength** in forgotten classes

### Retention Quality Indicators:
- 🔵 **Similar patterns** between models for retained classes
- 🔵 **Maintained feature quality** for non-forgotten classes
- 🔵 **Clear, recognizable features** still present

### Red Flags:
- ⚠️ **No difference** between models (unlearning failed)
- ⚠️ **All classes degraded** (catastrophic forgetting)
- ⚠️ **Retained classes also affected** (poor selectivity)

## Key Takeaways

1. **Always use `num_classes=200`** for Tiny ImageNet models
2. **Use indices 0-199**, not ImageNet's 0-999
3. **Reference your `unlearn_config.py`** for correct class mappings
4. **Load the correct model weights** with proper error handling
5. **Compare original vs unlearned models** to see the effect of forgetting

This approach will correctly visualize features from your Tiny ImageNet trained models using Lucent!