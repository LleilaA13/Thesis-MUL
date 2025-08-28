import torch

mask = torch.load("masks/inceptionv3_cat_forgetting/with_0.3.pt")

for key in sorted(mask.keys()):
    print(key)
