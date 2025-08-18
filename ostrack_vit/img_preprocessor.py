import numpy as np
import torch

from utils.image_parse_lib import ImageParseLib


class Preprocessor(object):
    def __init__(self, img_lib=ImageParseLib.TORCHVISION):

        self.img_lib = img_lib
        self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view((1, 3, 1, 1)) # .to('cuda:0')
        self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view((1, 3, 1, 1))  # .to('cuda:0')

    def process(self, img_arr, amask_arr=None):
        # Deal with the image patch
        if self.img_lib is ImageParseLib.TORCHVISION:
            img_tensor = img_arr.to(device='cuda').requires_grad_(False)
        elif self.img_lib is ImageParseLib.OPENCV:
            img_tensor = torch.tensor(img_arr, device='cuda').permute(2, 0, 1).requires_grad_(False)

        img_tensor = img_tensor.float().unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)

        # Deal with the attention mask
        if amask_arr is not None:
            from lib.utils.misc import NestedTensor
            amask_tensor = torch.from_numpy(amask_arr).to(torch.bool, device='cuda').unsqueeze(dim=0) # (1,H,W)
            return NestedTensor(img_tensor_norm, amask_tensor)
        else:
            return img_tensor_norm
