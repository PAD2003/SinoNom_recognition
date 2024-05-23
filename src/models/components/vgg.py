import torch
import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self,
                 model_name: str="vgg19_bn",
                 weights: str="DEFAULT",
                 output_shape: list=[2130],
                 unfreeze_depth: int=2):
        super().__init__()

        self.output_shape = output_shape[0]

        if model_name == 'vgg11_bn' or model_name == 'vgg11':
            cnn = models.vgg11_bn(pretrained = True if "DEFAULT" else weights)
        elif model_name == 'vgg13_bn' or model_name == 'vgg13':
            cnn = models.vgg13_bn(pretrained = True)
        elif model_name == 'vgg16_bn' or model_name == 'vgg16':
            cnn = models.vgg13_bn(pretrained = True)

        self.vgg19 = models.vgg19_bn(pretrained=True)

        self.vgg19.classifier[0] = nn.Linear(512 * 2 * 2, 4096)
        self.vgg19.classifier[6] = nn.Linear(4096, output_shape[0])

    def forward(self, x):
        x = self.vgg19.features(x)
        x = torch.flatten(x, 1)
        x = self.vgg19.classifier(x)
        return x
