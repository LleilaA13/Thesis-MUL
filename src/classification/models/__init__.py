from .ResNet import *
from .ResNets import *
from .VGG import *
from .VGG_LTH import *
from lucent.modelzoo import inceptionv1
from torchvision.models import inception_v3

model_dict = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet20s": resnet20s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "vgg16_bn": vgg16_bn,
    "vgg16_bn_lth": vgg16_bn_lth,
    "inceptionv1": inceptionv1,
    "inceptionv3": inception_v3
}
