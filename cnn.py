import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Conv_Net(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(input, output):
            return nn.Sequential(
                nn.Conv2d(input, output, 3, padding=1),
                nn.BatchNorm2d(output),
                nn.ReLU(),
                nn.Conv2d(output, output, 3, padding=1),
                nn.BatchNorm2d(output),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.2)
            )

        self.block1 = conv_block(1, 64)
        self.block2 = conv_block(64, 128)
        self.block3 = conv_block(128, 256)
        self.block4 = conv_block(256, 512)

        # Global average pooling = remove huge FC layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Much smaller classifier
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.gap(x)    # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
