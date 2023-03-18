from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: Please design your CNN layers. They are typically made of some Conv2d layers and pool layers
        self.conv_layers = nn.Sequential (
            nn.Conv2d(3, 3, 3, padding=1),
            nn.MaxPool2d(3,2),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.MaxPool2d(3,2),
            nn.ReLU(),
            # nn.Conv2d(3, 1, 3, padding=1)
        ) 
        self.linear_layers = nn.Sequential (
            nn.Flatten(),
            nn.Linear(1587, 512),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        # TODO: Finish it
        x = self.conv_layers(x)
        # print(x.shape)
        return self.linear_layers(x)