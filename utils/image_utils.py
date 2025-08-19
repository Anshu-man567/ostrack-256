import torch
import torchvision
import cv2 as cv
from PIL import Image
from torchvision.transforms.v2.functional import crop_image

from utils.image_parse_lib import ImageParseLib
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import torch.nn.functional as nnf


class ImageUtils():
    def __init__(self, side_length=256, patch_size=16, img_lib=ImageParseLib.TORCHVISION, print_stats=0):
        self.img_lib = img_lib
        self.debug_flag = print_stats
        self.patch_size = patch_size

    def step_size(self):
        return self.patch_size

    def load_image_and_params(self, image_file_path):
        image = None
        try:
            if self.img_lib is ImageParseLib.TORCHVISION:
                image = torchvision.io.read_image(image_file_path)
            elif self.img_lib is ImageParseLib.OPENCV:
                image = cv.imread(image_file_path)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            elif self.img_lib is ImageParseLib.PIL:
                with Image.open(image_file_path) as image:
                    image.load()
        except (IOError, SyntaxError):
            print("Could not read the image at path", image_file_path)
            return

        # image = cv.imread(image_file_path)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if image is None:
            sys.exit("Could not read the image at path", image_file_path)

        if self.debug_flag:
            print("Image file", image_file_path, "assumed to be of h:", self.height, "w:", self.width, "c:",
                  self.n_channels)
        return image

    '''
    x, y => top, left corner coords
    w, h => draw a rectangle from the top left corner with these height and widths
    '''
    def crop_image(self, image, top, left, w, h):
        x = left
        y = top
        cropped_img = None
        if self.img_lib is ImageParseLib.TORCHVISION:
            # wAS USED EARLIER BUT CRUDELY DOES IT ONLY FOR 16X16 PATCHES
            # cropped_img = v2.functional.crop(image, y, x, self.step_size(), self.step_size())
            cropped_img = v2.functional.crop(image, x, y, h, w)
        elif self.img_lib is ImageParseLib.OPENCV:
            end_x = x+w
            end_y = y+h
            cropped_img = image[x:end_x, y:end_y]
        elif self.img_lib is ImageParseLib.PIL:
            end_x = x+w
            end_y = y+h
            cropped_img = image.crop((x, y, end_x, end_y))
        else:
            print("Did not set img lib :(")
        return cropped_img

    def get_image_size(self, image):
        if self.img_lib is ImageParseLib.TORCHVISION:
            c, h, w = image.shape
        elif self.img_lib is ImageParseLib.OPENCV or self.img_lib is ImageParseLib.PIL:
            h, w, c = image.shape

        return h, w, c

    def image_viewer(self, image, img_str="",dump_img_flag=1):
        if dump_img_flag != 1:
            return

        if self.img_lib is ImageParseLib.TORCHVISION:
            image = image.detach()
            plt.imshow(image.permute(1, 2, 0))  # matplotlib expects h, w, c; but tensors are c, h, w
            h, w, c = self.get_image_size(image)
            plt.text(0.5, 0.05, img_str+f" Image size: {h} x {w}",
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=plt.gca().transAxes,
                     color='white',
                     fontsize=12,
                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
            # Remove axis ticks
            plt.xticks([])
            plt.yticks([])
            plt.show()
        elif self.img_lib is ImageParseLib.OPENCV:
            cv.imshow("Display Image", image)
        elif self.img_lib is ImageParseLib.PIL:
            image.show()
        else:
            print("Did not set img lib :(")

    def show_bbox_on_img(self, image_file_path, bbox_coords, req_resz=False, des_img_sz=256, str="", width=1):
        image = self.load_image_and_params(image_file_path)
        if req_resz is True:
            image = self.resize_image(image, des_img_sz=des_img_sz)

        bbox_coords = bbox_coords.detach().reshape(1, 4)

        image = torchvision.utils.draw_bounding_boxes(image, bbox_coords, width=width)
        self.image_viewer(image, str+"bbox", True)
        return image


    def resize_image(self, image, des_img_sz=256):
        if self.img_lib is ImageParseLib.TORCHVISION:
            # max_size should be strictly greater than des_img_sz
            resz_img_op = torchvision.transforms.Resize(size=des_img_sz-1, max_size=des_img_sz)
            re_sized_img = resz_img_op(image)
            _resz_c, resz_h, resz_w = re_sized_img.shape
            pad_h = des_img_sz - resz_h if resz_h < des_img_sz else 0
            pad_w = des_img_sz - resz_w if resz_w < des_img_sz else 0
            # Center aligns with padding
            re_sized_img = F.pad(re_sized_img, (pad_w//2,(pad_w+1)//2,pad_h//2,(pad_h+1)//2), "constant", 0)
        elif self.img_lib is ImageParseLib.OPENCV:
            print("To be implemented")
        elif self.img_lib is ImageParseLib.PIL:
            print("To be implemented")
        else:
            print("Did not set img lib :(")

        return re_sized_img

    def show_ce_out_img_fm_indices(self, layer_idx, search_img, topk_indices_at_layer, img_size):
        print("top k indices for layer: ", layer_idx, topk_indices_at_layer.shape, topk_indices_at_layer)
        filter = torch.zeros(size=(self.patch_size, self.patch_size), dtype=torch.uint8)
        for idx in topk_indices_at_layer:
            # The below stuff depends on the pytorch conv2d operator projects the patches
            idx_x = idx // self.patch_size
            idx_y = idx % self.patch_size
            filter[idx_x, idx_y] = 1

        # interpolate functions needs [n_batch, n_channels, x_0, x_1] as the input & size as (resz_dim_0 (for x_0), resz_dim_1 (for x_1))
        resz_filter = nnf.interpolate(filter.unsqueeze(0).unsqueeze(0),
                                      size=(img_size, img_size)).squeeze(0)
        print(search_img.shape, resz_filter.shape)
        self.image_viewer(search_img * resz_filter, "ce_out_" + str(layer_idx+1) + " ct: " + str(len(topk_indices_at_layer)), 1)

# The below functions are only used for testing, no need to fix anything!
def test_img_utils():
    img_utils = ImageUtils()
    search_image_file_path = "/home/anshu-man567/PycharmProjects/ProjectIdeas/OSTrack/data/got10k/test/GOT-10k_Test_000001/00000091.jpg"
    image = img_utils.load_image_and_params(search_image_file_path)
    re_size_img = img_utils.resize_image(image)
    # re_size_img = img_utils.resize_image(re_size_img)
    img_utils.image_viewer(re_size_img)

def parse_gt_bbox(txt_file_path, image_shape, pad=0):
    with open(txt_file_path) as file:
        lines = [line.rstrip() for line in file]

    raw_gt_box =  lines[0].split(',')

    gt_box = [int(float(raw_gt_box[0])),
              int(float(raw_gt_box[1])),
              int(float(raw_gt_box[2])),
              int(float(raw_gt_box[3]))]

    bbox_params = [gt_box[0],
                   gt_box[1],
                   gt_box[0]+gt_box[2],
                   gt_box[1]+gt_box[3]]

    # TODO (OPT): Handle padding going out of bounds
    crop_params = [gt_box[1],
                   gt_box[0],
                   gt_box[2]+pad,
                   gt_box[3]+pad]

    import torch
    bbox_params = torch.tensor(bbox_params).reshape(1, 4)
    print(gt_box, bbox_params, crop_params)

    return bbox_params, crop_params

def test_img_crop_and_resz():
    img_utils = ImageUtils(img_lib=ImageParseLib.TORCHVISION)
    search_image_file_path = "/home/anshu-man567/PycharmProjects/ProjectIdeas/OSTrack/data/got10k/test/GOT-10k_Test_000001/00000001.jpg"
    image = img_utils.load_image_and_params(search_image_file_path)
    txt_path = "/home/anshu-man567/PycharmProjects/ProjectIdeas/OSTrack/data/got10k/test/GOT-10k_Test_000001/groundtruth.txt"
    # bbox, crop = parse_gt_bbox(txt_path, image.shape)
    import torch

    bbox, crop = torch.tensor([50, 50, 60, 80]).reshape(1, 4), [50, 50, 10, 30]
    img_utils.show_bbox_on_img(search_image_file_path, bbox_coords=bbox)
    image = img_utils.crop_image(image, crop[0], crop[1], crop[2], crop[3])
    # image = img_utils.crop_image(image, bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3])
    img_utils.image_viewer(image, "crop")

    re_size_img = img_utils.resize_image(image, des_img_sz=100)
    img_utils.image_viewer(re_size_img, "resz")

if __name__ == "__main__":
    # test_img_utils()
    test_img_crop_and_resz()