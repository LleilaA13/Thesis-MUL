import torch
from torchvision.models import inception_v3, Inception_V3_Weights

# === Load pretrained InceptionV3 ===
weights = Inception_V3_Weights.IMAGENET1K_V1
model = inception_v3(weights=weights, aux_logits=True)

# === Disable auxiliary head if unused ===
model.aux_logits = False
model.AuxLogits = None  # Remove aux branch

# === Prepare checkpoint ===
checkpoint = {"state_dict": model.state_dict()}

# === Save for use in SalUn, evaluations, etc. ===
torch.save(checkpoint, "inceptionv3.pth")
print("[✓] Saved pretrained InceptionV3 checkpoint as 'inceptionv3.pth'")
