import torch
import torch.nn as nn
from lucent.modelzoo import inceptionv1

class InceptionV1WithHead(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.base = inceptionv1()
        self.base.fc = nn.Identity()
        self.fc = nn.Linear(1008, num_classes)

    def forward(self, x):
        x = self.base(x)
        return self.fc(x)

# Instantiate and save
model = InceptionV1WithHead()
checkpoint = {"state_dict": model.state_dict()}
torch.save(checkpoint, "inceptionv1.pth")
