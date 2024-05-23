import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F    

class BasicCNN(nn.Module):
    def __init__(self,
                 model_name: str="bsc_cnn",
                 weights: str="DEFAULT",
                 output_shape: list=[2130],
                 unfreeze_depth: int=2):
        super().__init__()
        self.output_shape = output_shape

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, output_shape[0])

    def forward(self, x): 
        x = F.relu(self.conv1(x)) 
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    import torch

    # create a model
    xla_bsc_cnn = BasicCNN()

    # show model
    batch_size = 16
    summary(model=xla_bsc_cnn,
            input_size=(batch_size, 3, 64, 64),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    # test input & output shape
    random_input = torch.randn([16, 3, 64, 64]).to('cpu')
    xla_bsc_cnn.to('cpu')
    output = xla_bsc_cnn(random_input)
    print(f"\n\nINPUT SHAPE: {random_input.shape}")
    print(f"OUTPUT SHAPE: {output.shape}")