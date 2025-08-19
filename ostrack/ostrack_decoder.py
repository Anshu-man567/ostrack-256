import torch
import torch.nn as nn
from ostrack.hann import hann2d

from ostrack.frozen_batch_norm import FrozenBatchNorm2d

"""Utility functions for OSTrack Decoder

This module provides utility functions to create single fully connected layers
and the OSTrackDecoder class which implements the decoder for OSTrack

This function creates a layer with the following operations:
1. Convolutional layer on the input
2. Batch normalization layer
3. ReLU activation function
"""
def create_single_fcn_layer(n_in_channels, n_out_channel, kernel_size=3, padding=1, dilation=1, stride=1, freeze_bn=True):
    if freeze_bn is True:
        return nn.Sequential(
            # TODO (Anshu-man567): Since bn right after this, try with bias=False)
            nn.Conv2d(n_in_channels, n_out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(n_out_channel),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(n_in_channels, n_out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(n_out_channel),
            nn.ReLU(inplace=True))


"""OSTrack Decoder implementation

This class implements the OSTrack decoder which consists of multiple heads
to predict classification scores, offsets, and bounding box sizes.

For each of the these predictions, it uses a series of convolutional layers
to process the input features and produce the final outputs.

The sizes of the output channels vary from 768 -> 256 -> 128 -> 64 -> 32 for each prediction head.
But the final output channels are:
1. Classification scores: 1 channel
2. Offsets: 2 channels (center point's [x, y] coordinates)
3. Bounding box sizes: 2 channels (width, height)
"""
class OSTrackDecoder(nn.Module):

    """Initialize the OSTrack Decoder
    
    Args:
        print_stats: Enable printing of debug stats
    """
    def __init__(self, print_stats=False):
        super().__init__()

        self.print_stats = print_stats

        # TODO (Anshu-man567): Enable varying these sizes depending search image sizes!
        self.feat_sz = float(256 / 16)
        self.n_out_channels = [768, 256, 128, 64, 32]
        self.size_d = [1, # For classification scores
                       2] # For offset and bounding box sizes

        idx = 1
        self.conv1_ctr = create_single_fcn_layer(n_out_channel=self.n_out_channels[idx],
                                                      n_in_channels=self.n_out_channels[idx-1])

        self.conv1_offset = create_single_fcn_layer(n_out_channel=self.n_out_channels[idx],
                                                      n_in_channels=self.n_out_channels[idx-1])

        self.conv1_size = create_single_fcn_layer(n_out_channel=self.n_out_channels[idx],
                                                      n_in_channels=self.n_out_channels[idx-1])

        idx += 1

        self.conv2_ctr = create_single_fcn_layer(n_out_channel=self.n_out_channels[idx],
                                                      n_in_channels=self.n_out_channels[idx-1])

        self.conv2_offset = create_single_fcn_layer(n_out_channel=self.n_out_channels[idx],
                                                      n_in_channels=self.n_out_channels[idx-1])

        self.conv2_size = create_single_fcn_layer(n_out_channel=self.n_out_channels[idx],
                                                      n_in_channels=self.n_out_channels[idx-1])

        idx += 1

        self.conv3_ctr = create_single_fcn_layer(n_out_channel=self.n_out_channels[idx],
                                                      n_in_channels=self.n_out_channels[idx-1])

        self.conv3_offset = create_single_fcn_layer(n_out_channel=self.n_out_channels[idx],
                                                      n_in_channels=self.n_out_channels[idx-1])

        self.conv3_size = create_single_fcn_layer(n_out_channel=self.n_out_channels[idx],
                                                      n_in_channels=self.n_out_channels[idx-1])

        idx += 1

        self.conv4_ctr = create_single_fcn_layer(n_out_channel=self.n_out_channels[idx],
                                                      n_in_channels=self.n_out_channels[idx-1])

        self.conv4_offset = create_single_fcn_layer(n_out_channel=self.n_out_channels[idx],
                                                      n_in_channels=self.n_out_channels[idx-1])

        self.conv4_size = create_single_fcn_layer(n_out_channel=self.n_out_channels[idx],
                                                      n_in_channels=self.n_out_channels[idx-1])

        self.conv5_ctr = nn.Conv2d(self.n_out_channels[idx],
                                   out_channels=self.size_d[0],
                                   kernel_size=1)

        self.conv5_offset = nn.Conv2d(self.n_out_channels[idx], 
                                      out_channels=self.size_d[1],
                                      kernel_size=1)

        self.conv5_size = nn.Conv2d(self.n_out_channels[idx], 
                                    out_channels=self.size_d[1], 
                                    kernel_size=1)

    """
    Classifier score prediction methods
    """
    def get_classifier_score(self, input):
        op1 = self.conv1_ctr(input)
        op2 = self.conv2_ctr(op1)
        op3 = self.conv3_ctr(op2)
        op4 = self.conv4_ctr(op3)
        op5 = self.conv5_ctr(op4)

        return op5
    
    """
    Offset prediction method
    """
    def get_offset(self, input):
        op1 = self.conv1_offset(input)
        op2 = self.conv2_offset(op1)
        op3 = self.conv3_offset(op2)
        op4 = self.conv4_offset(op3)
        op5 = self.conv5_offset(op4)

        return op5

    """
    Bounding box size prediction method
    """
    def get_bb_size(self, input):
        op1 = self.conv1_size(input)
        op2 = self.conv2_size(op1)
        op3 = self.conv3_size(op2)
        op4 = self.conv4_size(op3)
        op5 = self.conv5_size(op4)

        return op5

    """
    Sigmoid function to clamp the output preditions for classification scores and bounding box sizes
    """
    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

    def forward(self, input):
        classifier_score = self.sigmoid(self.get_classifier_score(input))
        offset_values = self.get_offset(input)
        pred_bb_size = self.sigmoid(self.get_bb_size(input))

        return classifier_score, pred_bb_size, offset_values

def model_stats(model):
    for key, value in model.state_dict().items():
        print(key, "\tlen:", value.size())
    print("Done")

def test_ostrack_decoder():
    ostrack_decoder = OSTrackDecoder()
    print(model_stats(ostrack_decoder))


if __name__ == "__main__":
    test_ostrack_decoder()
