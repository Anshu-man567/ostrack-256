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
    def __init__(self, side_length=256, patch_size=16, img_lib=ImageParseLib.TORCHVISION, print_stats=0):
        super().__init__()

        self.state = [64, 64, 128, 128]
        self.print_stats = print_stats
        self.img_lib = img_lib
        self.side_len = side_length
        self.search_size_N = int(self.side_len * self.side_len) / (patch_size ** 2)

        self.feat_sz = patch_size

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

        # TODO (Anshu-man567): Why hann windows and option for it?
        # response = self.output_window * pred_score_map
        response = pred_score_map

        if self.print_stats:
            self.show_classification_scores(response)

        pred_boxes = self.cal_bbox(response, size_map, offset_map)
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(
            dim=0) * self.side_len ).tolist()  # (cx, cy, w, h) [0,1]

        bbox = pred_box

        # state = self.search_patch_maker.clip_box(box=self.map_box_back(bbox.reshape(4, 1), self.resize_factor), H=256, W=256, margin=10)

        # state = self.clip_box(box=self.map_box_back(bbox, self.resize_factor), H=256, W=256, margin=10)

        processed_bbox = [int(bbox[0] - (0.5 * bbox[2])), int(bbox[1] - (0.5 * bbox[3])),
                          int(bbox[0] + (0.5 * bbox[2])), int(bbox[1] + (0.5 * bbox[3]))]

        # print("A: processed bb", processed_bbox)

        state = processed_bbox # self.clip_box(box=processed_bbox, H=256, W=256, margin=0)
        # state = int(state)
        # state = self.search_patch_maker.clip_box(box=bbox.reshape(4, 1), H=256, W=256, margin=10)

        # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        # print("A:",state)

        # print(bbox, state)

        # return

        # op_coord = torch.tensor([outputs_coord[0,0] - outputs_coord[0,2]/2,
        #                         outputs_coord[0,1] - outputs_coord[0,3]/2,
        #                         outputs_coord[0,0] + outputs_coord[0,2]/2,
        #                         outputs_coord[0,1] + outputs_coord[0,3]/2]).reshape(1, 4)
        #
        #
        # print(op_coord)

        # op_coord = torch.tensor([state[0] ,
        #                          state[1],
        #                          state[0] + state[2],
        #                          state[1] + state[3]]).reshape(1, 4)

        op_coord = torch.tensor(state).reshape(1, 4)
        # print("A:op_coord", op_coord)

        # op_coord = torchvision.ops.box_convert(op_coord)


        return op_coord

    def cal_bbox(self, classifier_score, pred_bb_size, offset_values, return_score=False):
        max_score, idx = torch.max(classifier_score.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = pred_bb_size.flatten(2).gather(dim=2, index=idx)
        offset = offset_values.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz

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

    # def show_bbox_on_img(self, image_file_path, bbox_coords):
    #     image = self.load_image_and_params(image_file_path)
    #     image = torchvision.utils.draw_bounding_boxes(image, bbox_coords)
    #     self.image_viewer(image, "bbox",True)
    #
    # def load_image_and_params(self, image_file_path):
    #     image = None
    #     if self.img_lib is ImageParseLib.TORCHVISION:
    #         image = torchvision.io.read_image(image_file_path)
    #     elif self.img_lib is ImageParseLib.OPENCV:
    #         image = cv.imread(image_file_path)
    #     elif self.img_lib is ImageParseLib.PIL:
    #         with Image.open(image_file_path) as image:
    #             image.load()
    #
    #     if image is None:
    #         sys.exit("Could not read the image at path", image_file_path)
    #
    #     # WORKS for all
    #     self.height, self.width, self.n_channels = image.shape
    #     if self.print_stats:
    #         print("Image file", image_file_path, "of h:", self.height, "w:", self.width, "c:", self.n_channels)
    #
    #     return image
    #
    # def image_viewer(self, image, img_str="",dump_img_flag=1):
    #     if dump_img_flag != 1:
    #         return
    #
    #     if self.img_lib is ImageParseLib.TORCHVISION:
    #         plt.imshow(image.permute(1, 2, 0))  # matplotlib expects h, w, c; but tensors are c, h, w
    #         h, w, c = self.get_image_size(image)
    #         plt.text(0.5, 0.05, img_str+f" Image size: {h} x {w}",
    #                  horizontalalignment='center',
    #                  verticalalignment='center',
    #                  transform=plt.gca().transAxes,
    #                  color='white',
    #                  fontsize=12,
    #                  bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
    #         # Remove axis ticks
    #         # plt.xticks([])
    #         # plt.yticks([])
    #         plt.show()
    #     elif self.img_lib is ImageParseLib.OPENCV:
    #         cv.imshow("Display Image", image)
    #     elif self.img_lib is ImageParseLib.PIL:
    #         image.show()
    #     else:
    #         print("Did not set img lib :(")

    # def get_image_size(self, image):
    #     if self.img_lib is ImageParseLib.TORCHVISION:
    #         c, h, w = image.shape
    #     elif self.img_lib is ImageParseLib.OPENCV or self.img_lib is ImageParseLib.PIL:
    #         h, w, c = image.shape
    #
    #     return h, w, c

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