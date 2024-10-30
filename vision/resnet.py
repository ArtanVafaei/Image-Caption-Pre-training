import torch
import torch.nn as nn
import math

class Resnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        layers = []
        inn = 64
        out = 128
        for i in range(config.num_layers):
          layers.append(Block(inn, out, config.dropout))
          inn = out
          out *= 2

        self.layers = nn.ModuleList(layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(int(math.sqrt(config.adapt_pool)), int(math.sqrt(config.adapt_pool))))

        self.flatten = nn.Flatten(start_dim=-2)
        # self.fc = nn.Linear(inn, num_classes)

    def forward(self, x):
        x = self.first(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        return self.flatten(x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.relu = nn.ReLU()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residuals = self.downsample(x)
        x = self.layer(x)
        x = self.relu(x + residuals)
        return self.dropout(x)