from torch import nn
import torch
import torchvision.models as models

class Mod_Resnet(nn.Module):
    def __init__(self):
        super().__init__()

        # self.input_layers = nn.Conv2d(1, 3, 2, padding=1)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        
        

    def forward(self, x):
        # print(x.shape)
        # x = self.input_layers(x)
        # print(x.shape)
        x = self.model(x)
        return x