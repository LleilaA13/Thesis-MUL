import torch

mask = torch.load("masks/inceptionv3_cat_forgetting/with_0.3.pt")

print("\n--- Keys in saliency mask ---")
for key in sorted(mask.keys()):
    print(key)
