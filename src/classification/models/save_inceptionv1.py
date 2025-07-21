import torch
from lucent.modelzoo import inceptionv1

model = inceptionv1()
checkpoint = {"state_dict": model.state_dict()}
torch.save(checkpoint, "inceptionv1.pth")

