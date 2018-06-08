"""
author: Caner Mercan
"""

import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9, stride=1, padding=0, args=None):
        super().__init__()
        # convolutional layer
        self.conv2d = nn.Conv2d(
                        in_channels     = in_channels,
                        out_channels    = out_channels,
                        kernel_size     = kernel_size,
                        stride          = stride,
                        padding         = padding)
        # ReLU activation
        self.relu = nn.ReLU(inplace = False)
        
    def forward(self, inp):
        inp = self.relu(self.conv2d(inp))
        return inp
    
