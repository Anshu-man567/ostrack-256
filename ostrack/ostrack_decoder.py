import torch
import torch.nn as nn
from ostrack.hann import hann2d

from ostrack.frozen_batch_norm import FrozenBatchNorm2d

def create_single_fcn_layer(n_in_channels, n_out_channel, kernel_size=3, padding=1, dilation=1, stride=1, freeze_bn=True):
    if freeze_bn is True:
        return nn.Sequential(
            # TODO (Anshu-man567: Since bn right after this, try with bias=False)
            # nn.Conv2d(n_in_channels, n_out_channel, kernel_size=kernel_size, stride=stride,
            #           padding=padding, dilation=dilation, bias=True),
            nn.Conv2d(n_in_channels, n_out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            FrozenBatchNorm2d(n_out_channel),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(n_in_channels, n_out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(n_out_channel),
            nn.ReLU(inplace=True))

class OSTrackDecoder(nn.Module):
    def __init__(self, is_cand_elimn_en=False, print_stats=False):
        super().__init__()

        self.print_stats = print_stats

        if is_cand_elimn_en is True:
            self.token_padding = None
        else:
            # TODO (Anshu-man567): Implement this one!
            self.token_padding = None

        self.feat_sz = float(256 / 16)

        # self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True)


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

    def get_classifier_score(self, input):
        op1 = self.conv1_ctr(input)
        op2 = self.conv2_ctr(op1)
        op3 = self.conv3_ctr(op2)
        op4 = self.conv4_ctr(op3)
        op5 = self.conv5_ctr(op4)

        return op5

    def get_offset(self, input):
        op1 = self.conv1_offset(input)
        op2 = self.conv2_offset(op1)
        op3 = self.conv3_offset(op2)
        op4 = self.conv4_offset(op3)
        op5 = self.conv5_offset(op4)

        return op5

    def get_bb_size(self, input):
        op1 = self.conv1_size(input)
        op2 = self.conv2_size(op1)
        op3 = self.conv3_size(op2)
        op4 = self.conv4_size(op3)
        op5 = self.conv5_size(op4)

        return op5

    def forward(self, input):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        classifier_score = _sigmoid(self.get_classifier_score(input))

        offset_values = self.get_offset(input)
        pred_bb_size = _sigmoid(self.get_bb_size(input))

        # TODO (Anshu-man567) : Understand why hann windows and add an option to enable or disable
        # classifier_score = self.output_window * classifier_score

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



'''
box_head.conv1_ctr.0.weight 	len: torch.Size([256, 768, 3, 3])
box_head.conv1_ctr.0.bias 	len: torch.Size([256])
box_head.conv1_ctr.1.weight 	len: torch.Size([256])
box_head.conv1_ctr.1.bias 	len: torch.Size([256])
box_head.conv1_ctr.1.running_mean 	len: torch.Size([256])
box_head.conv1_ctr.1.running_var 	len: torch.Size([256])
box_head.conv1_ctr.1.num_batches_tracked 	len: torch.Size([])

box_head.conv2_ctr.0.weight 	len: torch.Size([128, 256, 3, 3])
box_head.conv2_ctr.0.bias 	len: torch.Size([128])
box_head.conv2_ctr.1.weight 	len: torch.Size([128])
box_head.conv2_ctr.1.bias 	len: torch.Size([128])
box_head.conv2_ctr.1.running_mean 	len: torch.Size([128])
box_head.conv2_ctr.1.running_var 	len: torch.Size([128])
box_head.conv2_ctr.1.num_batches_tracked 	len: torch.Size([])

box_head.conv3_ctr.0.weight 	len: torch.Size([64, 128, 3, 3])
box_head.conv3_ctr.0.bias 	len: torch.Size([64])
box_head.conv3_ctr.1.weight 	len: torch.Size([64])
box_head.conv3_ctr.1.bias 	len: torch.Size([64])
box_head.conv3_ctr.1.running_mean 	len: torch.Size([64])
box_head.conv3_ctr.1.running_var 	len: torch.Size([64])
box_head.conv3_ctr.1.num_batches_tracked 	len: torch.Size([])

box_head.conv4_ctr.0.weight 	len: torch.Size([32, 64, 3, 3])
box_head.conv4_ctr.0.bias 	len: torch.Size([32])
box_head.conv4_ctr.1.weight 	len: torch.Size([32])
box_head.conv4_ctr.1.bias 	len: torch.Size([32])
box_head.conv4_ctr.1.running_mean 	len: torch.Size([32])
box_head.conv4_ctr.1.running_var 	len: torch.Size([32])
box_head.conv4_ctr.1.num_batches_tracked 	len: torch.Size([])

box_head.conv5_ctr.weight 	len: torch.Size([1, 32, 1, 1])
box_head.conv5_ctr.bias 	len: torch.Size([1])

box_head.conv1_offset.0.weight 	len: torch.Size([256, 768, 3, 3])
box_head.conv1_offset.0.bias 	len: torch.Size([256])
box_head.conv1_offset.1.weight 	len: torch.Size([256])
box_head.conv1_offset.1.bias 	len: torch.Size([256])
box_head.conv1_offset.1.running_mean 	len: torch.Size([256])
box_head.conv1_offset.1.running_var 	len: torch.Size([256])
box_head.conv1_offset.1.num_batches_tracked 	len: torch.Size([])
box_head.conv2_offset.0.weight 	len: torch.Size([128, 256, 3, 3])
box_head.conv2_offset.0.bias 	len: torch.Size([128])
box_head.conv2_offset.1.weight 	len: torch.Size([128])
box_head.conv2_offset.1.bias 	len: torch.Size([128])
box_head.conv2_offset.1.running_mean 	len: torch.Size([128])
box_head.conv2_offset.1.running_var 	len: torch.Size([128])
box_head.conv2_offset.1.num_batches_tracked 	len: torch.Size([])
box_head.conv3_offset.0.weight 	len: torch.Size([64, 128, 3, 3])
box_head.conv3_offset.0.bias 	len: torch.Size([64])
box_head.conv3_offset.1.weight 	len: torch.Size([64])
box_head.conv3_offset.1.bias 	len: torch.Size([64])
box_head.conv3_offset.1.running_mean 	len: torch.Size([64])
box_head.conv3_offset.1.running_var 	len: torch.Size([64])
box_head.conv3_offset.1.num_batches_tracked 	len: torch.Size([])
box_head.conv4_offset.0.weight 	len: torch.Size([32, 64, 3, 3])
box_head.conv4_offset.0.bias 	len: torch.Size([32])
box_head.conv4_offset.1.weight 	len: torch.Size([32])
box_head.conv4_offset.1.bias 	len: torch.Size([32])
box_head.conv4_offset.1.running_mean 	len: torch.Size([32])
box_head.conv4_offset.1.running_var 	len: torch.Size([32])
box_head.conv4_offset.1.num_batches_tracked 	len: torch.Size([])

box_head.conv5_offset.weight 	len: torch.Size([2, 32, 1, 1])
box_head.conv5_offset.bias 	len: torch.Size([2])

box_head.conv1_size.0.weight 	len: torch.Size([256, 768, 3, 3])
box_head.conv1_size.0.bias 	len: torch.Size([256])
box_head.conv1_size.1.weight 	len: torch.Size([256])
box_head.conv1_size.1.bias 	len: torch.Size([256])
box_head.conv1_size.1.running_mean 	len: torch.Size([256])
box_head.conv1_size.1.running_var 	len: torch.Size([256])
box_head.conv1_size.1.num_batches_tracked 	len: torch.Size([])
box_head.conv2_size.0.weight 	len: torch.Size([128, 256, 3, 3])
box_head.conv2_size.0.bias 	len: torch.Size([128])
box_head.conv2_size.1.weight 	len: torch.Size([128])
box_head.conv2_size.1.bias 	len: torch.Size([128])
box_head.conv2_size.1.running_mean 	len: torch.Size([128])
box_head.conv2_size.1.running_var 	len: torch.Size([128])
box_head.conv2_size.1.num_batches_tracked 	len: torch.Size([])
box_head.conv3_size.0.weight 	len: torch.Size([64, 128, 3, 3])
box_head.conv3_size.0.bias 	len: torch.Size([64])
box_head.conv3_size.1.weight 	len: torch.Size([64])
box_head.conv3_size.1.bias 	len: torch.Size([64])
box_head.conv3_size.1.running_mean 	len: torch.Size([64])
box_head.conv3_size.1.running_var 	len: torch.Size([64])
box_head.conv3_size.1.num_batches_tracked 	len: torch.Size([])
box_head.conv4_size.0.weight 	len: torch.Size([32, 64, 3, 3])
box_head.conv4_size.0.bias 	len: torch.Size([32])
box_head.conv4_size.1.weight 	len: torch.Size([32])
box_head.conv4_size.1.bias 	len: torch.Size([32])
box_head.conv4_size.1.running_mean 	len: torch.Size([32])
box_head.conv4_size.1.running_var 	len: torch.Size([32])
box_head.conv4_size.1.num_batches_tracked 	len: torch.Size([])

box_head.conv5_size.weight 	len: torch.Size([2, 32, 1, 1])
box_head.conv5_size.bias 	len: torch.Size([2])
'''