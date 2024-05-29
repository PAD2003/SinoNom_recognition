import torch
import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self,
                 model_name: str="vgg19_bn",
                 weights: str="DEFAULT",
                 output_shape: list=[2130],
                 unfreeze_depth: int=-1):
        super().__init__()

        model_name = model_name.lower()
        self.output_shape = output_shape[0]

        if model_name == 'vgg11_bn' or model_name == 'vgg11':
            self.cnn = models.vgg11_bn(pretrained = True if "DEFAULT" else weights)
        elif model_name == 'vgg13_bn' or model_name == 'vgg13':
            self.cnn = models.vgg13_bn(pretrained = True)
        elif model_name == 'vgg16_bn' or model_name == 'vgg16':
            self.cnn = models.vgg16_bn(pretrained = True)

        self.cnn.classifier[0] = nn.Linear(512 * 2 * 2, 4096)
        self.cnn.classifier[6] = nn.Linear(4096, output_shape[0])

    def forward(self, x):
        x = self.cnn.features(x)
        x = torch.flatten(x, 1)
        x = self.cnn.classifier(x)
        return x

if __name__ == "__main__":
    from torchinfo import summary
    import torch

    # create a model
    xla_vgg = VGG("vgg19")

    # show model
    batch_size = 16
    summary(model=xla_vgg,
            input_size=(batch_size, 3, 64, 64),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    # test input & output shape
    random_input = torch.randn([16, 3, 64, 64]).to('cpu')
    xla_vgg.to('cpu')
    output = xla_vgg(random_input)
    print(f"\n\nINPUT SHAPE: {random_input.shape}")
    print(f"OUTPUT SHAPE: {output.shape}")