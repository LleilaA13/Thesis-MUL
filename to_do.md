##TODO LIST FOR THESIS:
	- Scrivere i primi 3 capitoli
	- fixare le visuals su resnet-19
	- aggiungere visuals per inceptionV3
	- fare una run di unlearning su Resnet-50 su dataset da definire

	

##CAPITOLI TESI DA SCRIVERE:
	- Introduction
	- Related works
	- Methods


# Early Layers (conv1, layer1):
# - Edge detection, basic shapes, textures
# - These are UNIVERSAL across all natural images
# ✅ Perfectly transferable ImageNet → TinyImageNet

# Middle Layers (layer2, layer3):  
# - Complex patterns, object parts (wheels, eyes, fur)
# - Still very transferable for natural images
# ✅ Highly transferable ImageNet → TinyImageNet

# Final Layer (layer4 + fc):
# - High-level concepts, class-specific features
# - This is what we replace for new classes
# 🔄 We replace fc: 1000 classes → 200 classes