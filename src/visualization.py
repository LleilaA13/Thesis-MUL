from lucent.modelzoo import inceptionv1
from lucent.optvis import render
import torch
import torchvision.models as models

def load_inception(pretrained=True):
    model = inceptionv1(pretrained=pretrained)
    model.eval()
    return model

def load_resnet18(path=None):
    model = models.resnet18(pretrained=False)
    if path:
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def visualize_neuron(model, layer, channel, save_path=None):
    _, image = render.render_vis(model, layer=layer, channel=channel)
    if save_path:
        image[0].save(save_path)
