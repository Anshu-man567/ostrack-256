import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import sys

"""Patch embedding layer for vision transformer models

This module converts input images into a sequence of patches, each represented as a vector.
It uses a convolutional layer to extract patches and reshape them into the required format for transformer models

This module consists of the following operations:
1. Convolutional layer for patch extraction
2. Flattening the output to create a sequence of patches
3. Transposing the output to match the expected input shape for transformer models
"""
class InputLayer(nn.Module):
    
    """Initialize the patch embedding layer for vision transformer models.

    Args:
        patch_size: Size of each image patch
        n_channels: Number of input channels
        output_size: Dimension of the output embedding for each patch
        print_stats: Enable printing of debug stats
    """
    def __init__(self, patch_size=16, n_channels=3, output_size=768, print_stats=0):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.output_size = output_size

        self.print_stats = print_stats

        self.proj = nn.Conv2d(in_channels=self.n_channels,
                              out_channels=self.output_size,
                              kernel_size=self.patch_size,
                              stride=self.patch_size)

        if self.print_stats != 0:
            print("size after applying pretrain", self.proj.state_dict()['weight'].size(), self.proj.state_dict()['bias'].size())


    def forward(self, input):
       
        # Here C = no of channels
        # at the point output is B, C, H, W
        output = self.proj(input)
        
        if self.print_stats:
            print("size of output from input layer", output.size())
        
        # flatten B, C, H, W -> B, C, N
        output = torch.flatten(output, 2)
        
        if self.print_stats:
            print("size of output from input layer", output.size())

        # invert C and N => B, C, N -> B, N, C
        output = torch.transpose(output,1, 2)

        if self.print_stats:
            print("size of output from input layer", output.size())

        return output