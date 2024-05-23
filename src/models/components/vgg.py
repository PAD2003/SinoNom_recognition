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
            self.cnn = models.vgg11_bn(pretrained = True if "DEFAULT" else weights)
        elif model_name == 'vgg13_bn' or model_name == 'vgg13':
            self.cnn = models.vgg13_bn(pretrained = True)
        elif model_name == 'vgg16_bn' or model_name == 'vgg16':
            self.cnn = models.vgg16_bn(pretrained = True)
        elif model_name == 'vgg19_bn' or model_name == 'vgg19':
            self.cnn = models.vgg19_bn(pretrained = True)
        else:
            raise Exception("Model not available!")

        self.cnn.classifier[0] = nn.Linear(512 * 2 * 2, 4096)
        self.cnn.classifier[6] = nn.Linear(4096, output_shape[0])

    def forward(self, x):
        x = self.cnn.features(x)
        x = torch.flatten(x, 1)
        x = self.cnn.classifier(x)
        return x
