import torch
import torch.nn as nn
import math

# Define a custom ResNet-like neural network model
class Resnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Initial layer: convolution with batch normalization, dropout, and ReLU activation
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Initial convolution layer
            nn.BatchNorm2d(num_features=64),                       # Batch normalization
            nn.Dropout(config.dropout),                            # Dropout for regularization
            nn.ReLU()                                              # ReLU activation function
        )

        # Set up main layers based on config: builds several sequential blocks
        layers = []
        inn = 64
        out = 128
        for _ in range(config.num_layers):
            # Append each Block with specific in and out channels and dropout rate
            layers.append(Block(inn, out, config.dropout))
            inn = out  # update in-channel for next block
            out *= 2   # double out-channel size for the next block

        self.layers = nn.ModuleList(layers)  # Store blocks in a ModuleList
        
        # Adaptive average pooling to a configurable size for input flexibility
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(int(math.sqrt(config.adapt_pool)), int(math.sqrt(config.adapt_pool))))

        # Flattening layer for transitioning to a fully connected layer (if added)
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x):
        # Forward pass through the initial layer
        x = self.first(x)
        
        # Forward pass through each block layer in the network
        for layer in self.layers:
            x = layer(x)
        
        # Average pooling and flattening for classifier input preparation
        x = self.avgpool(x)
        return self.flatten(x)  # Flatten and return output


# Define a single residual block in the network
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        
        # Residual connection path with two convolution layers, batch normalization, and dropout
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),  # First conv layer with stride 2
            nn.BatchNorm2d(num_features=out_channels),                                 # Batch normalization
            nn.ReLU(),                                                                 # ReLU activation
            nn.Dropout(dropout),                                                       # Dropout layer for regularization
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), # Second conv layer
            nn.BatchNorm2d(num_features=out_channels)                                  # Batch normalization
        )
        
        # ReLU activation for post-addition residual connection
        self.relu = nn.ReLU()
        
        # Downsampling layer to match the input and output dimensions of residuals
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),  # 1x1 convolution for downsampling
            nn.BatchNorm2d(num_features=out_channels)                       # Batch normalization
        )
        
        # Additional dropout layer for the end of the residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply downsampling to create residuals
        residuals = self.downsample(x)
        
        # Forward pass through main layer sequence
        x = self.layer(x)
        
        # Add residuals and apply ReLU activation
        x = self.relu(x + residuals)
        
        # Apply final dropout and return output
        return self.dropout(x)
