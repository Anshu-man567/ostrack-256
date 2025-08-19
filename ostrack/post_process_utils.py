import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import v2
import math
import cv2 as cv
import sys

from ostrack.hann import hann2d

from utils.image_parse_lib import ImageParseLib
from utils.image_utils import ImageUtils


class PostProcessUtils(nn.Module):
    def __init__(self, side_length=256, patch_size=16, img_lib=ImageParseLib.TORCHVISION, print_stats=0, apply_hann_window=False):
        super().__init__()

        self.state = [64, 64, 128, 128]
        self.print_stats = print_stats
        self.img_lib = img_lib
        self.side_len = side_length
        self.search_size_N = int(self.side_len * self.side_len) / (patch_size ** 2)
        self.feat_sz = patch_size
        self.apply_hann_window = apply_hann_window

        if self.apply_hann_window:
            self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz], device='cuda').long(), centered=True)

        self.img_utils = ImageUtils()
        self.resize_factor = 1

    def get_output_coords(self, classifier_score_map, size_map, offset_map):
        if self.print_stats:
            self.show_classification_scores(classifier_score_map)

        pred_score_map = classifier_score_map
        if self.print_stats:
            self.show_classification_scores(self.output_window)
            print(self.output_window.shape, pred_score_map.shape)

        if self.apply_hann_window:
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map

        if self.print_stats:
            self.show_classification_scores(response)

        pred_boxes = self.cal_bbox(response, size_map, offset_map)
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(
            dim=0) * self.side_len ).tolist()  # (cx, cy, w, h) [0,1]

        bbox = pred_box

        processed_bbox = [int(bbox[0] - (0.5 * bbox[2])), int(bbox[1] - (0.5 * bbox[3])),
                          int(bbox[0] + (0.5 * bbox[2])), int(bbox[1] + (0.5 * bbox[3]))]

        state = processed_bbox

        op_coord = torch.tensor(state).reshape(1, 4)

        return op_coord

    def cal_bbox(self, classifier_score, pred_bb_size, offset_values, return_score=False):
        max_score, idx = torch.max(classifier_score.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = pred_bb_size.flatten(2).gather(dim=2, index=idx)
        offset = offset_values.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def clip_box(self, box: list, H, W, margin=0):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        x1 = min(max(0, x1), W - margin)
        x2 = min(max(margin, x2), W)
        y1 = min(max(0, y1), H - margin)
        y2 = min(max(margin, y2), H)
        w = max(margin, x2 - x1)
        h = max(margin, y2 - y1)
        return [x1, y1, w, h]

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size_N / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def show_classification_scores(self, classifier_score_map):
        # TODO(Anshu-man567): Fix this
        a = classifier_score_map.to('cpu').squeeze(0).permute(1, 2, 0).detach().numpy()
        fig, ax = plt.subplots()
        im = plt.imshow(a, cmap='hot', interpolation='nearest')
        cbar = fig.colorbar(im)
        plt.show()

        # a = size_map.squeeze(0).permute(1,2,0).detach().numpy()
        # fig, ax = plt.subplots()
        # im = plt.imshow(a, cmap='hot', interpolation='nearest')
        # cbar = fig.colorbar(im)
        # plt.show()

        # a = offset_map.squeeze(0).permute(1,2,0).detach().numpy()
        # fig, ax = plt.subplots()
        # im = plt.imshow(a, cmap='hot', interpolation='nearest')
        # cbar = fig.colorbar(im)
        # plt.show()