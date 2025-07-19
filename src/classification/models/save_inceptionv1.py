import torch
from lucent.modelzoo import inceptionv1

# Load pretrained InceptionV1 from Lucent
model = inceptionv1()

# Save its weights as a standard PyTorch checkpoint
torch.save(model.state_dict(), "inceptionv1_lucent.pth")
