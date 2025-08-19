import os
from collections import OrderedDict

import torch
from enum import Enum

import torchvision
from torchvision.transforms import v2
from mpmath.identification import transforms
from torchvision.tv_tensors import BoundingBoxes
from requests.utils import iter_slices

from ostrack.ostrack_model import OSTrackModel
from utils.image_parse_lib import ImageParseLib
from utils.image_utils import ImageUtils
from utils.get_trained_features import GetTrainedFeatures

import time

class TrackExecutionMode(Enum):
    TEST = 1
    TRAIN = 2

class SeqOSTrack:
    def __init__(self,
                 vit_img_dim=224,
                 search_img_dim=256,
                 tmpl_img_dim=128,
                 exec_mode=TrackExecutionMode.TEST,
                 img_lib=ImageParseLib.TORCHVISION,
                 show_dumps=0,
                 print_stats=0,
                 en_early_cand_elimn=1):

        self.exec_mode = exec_mode
        self.img_utils = ImageUtils(img_lib)
        self.show_dumps = show_dumps
        self.print_stats = print_stats
        self.en_early_cand_elimn = en_early_cand_elimn
        self.search_img_dim = search_img_dim
        self.tmpl_img_dim = tmpl_img_dim

        self.template_patch = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ostrack = OSTrackModel(vit_img_dim=vit_img_dim,
                                    search_img_dim=self.search_img_dim,
                                    tmpl_img_dim=self.tmpl_img_dim,
                                    print_stats=self.print_stats,
                                    show_dumps=self.show_dumps,
                                    en_early_cand_elimn=en_early_cand_elimn)
        self.ostrack = self.ostrack.to(self.device)

        if exec_mode is TrackExecutionMode.TEST:

            if self.en_early_cand_elimn:
                ostrack_trained_wts_file = '../weights/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar'

                # THE BELOW ONE ACTUALLY GAVE WORSE RESULT IN ONE GOT-10k test with kangaroo and racoon, towards the end identifies the 2nd racoon as the kangaroo
                # FROM THE HEATMAPS can confirm that it recognizes the racoon and kangaroo as the object of interest, probably something to do with less training causing transformer to achieve subpar results
                # Try to figure out why???
                # ostrack_trained_wts_file = '../weights/vitb_256_mae_ce_32x4_got10k_ep100/OSTrack_ep0100.pth.tar'
            else:
                ostrack_trained_wts_file = '../weights/vitb_256_mae_32x4_ep300/OSTrack_ep0300.pth.tar'

            print("Picked weights from path: ", ostrack_trained_wts_file, "EARLY CAND ELIMN is", self.en_early_cand_elimn)

            self.gtf_ostrack = GetTrainedFeatures(ostrack_trained_wts_file)

            if self.print_stats:
                self.gtf_ostrack.print_model_info(self.ostrack.backbone)
                self.gtf_ostrack.print_model_info(self.ostrack.box_head)

            if self.print_stats:
                self.gtf_ostrack.print_loaded_model_info()

            self.ostrack = self.ostrack.eval()
            ostrack_state_dict, is_exact_copy = self.gtf_ostrack.create_new_state_dict(self.ostrack)
            print("Is it an exact copy?", is_exact_copy)
            missing_keys, unexpected_keys = self.ostrack.load_state_dict(ostrack_state_dict, strict=False)
        #
        # elif exec_mode is TrackExecutionMode.TRAIN:
        #     TODO : Implement optimizers, dropout rates and stuff
    
    def init_new_seq(self, tmpl_img_file_path=None, req_resz=False, crop_box=None):
        if tmpl_img_file_path is None:
            print("ERROR no template image file path provided", tmpl_img_file_path)
        else:
            self.template_patch = self.ostrack.create_tmpl_patches(tmpl_img_file_path, req_resz, crop_box)

    def track_seq_test_internal(self, search_image_folder=None, search_image_paths=None, req_resz=False):
        start = time.time()

        if search_image_folder is not None:
            search_image_paths = []
            for f in sorted(os.listdir(search_image_folder)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    full_path = os.path.join(search_image_folder, f)
                    search_image_paths.append(full_path)
        else:
            print("Continuing with the search_image_path provided", search_image_paths)

        print("It took", time.time() - start, "seconds to prepare search image paths")

        output_data = OrderedDict()
        output_data['folder'] = search_image_folder
        iter = 0
        for search_image_path in search_image_paths:
            print("Testing image", search_image_path)
            img_start = time.time()
            search_patches = self.ostrack.create_search_patches(search_image_path, req_resz)
            patches = self.ostrack.combine_patches(search_patches, tmpl_patches=self.template_patch)

            op_coord, classifier_score_map, _1, _2 = self.ostrack.forward(patches)

            output_data[str(iter)] = OrderedDict()
            output_data[str(iter)]['search_image_path'] = search_image_path
            output_data[str(iter)]['op_coord'] = op_coord
            # transfer to cpu for conversion to numpy
            output_data[str(iter)]['classifier_score'] = classifier_score_map.squeeze(0).detach().cpu()
            iter += 1

            if self.show_dumps:
               if self.en_early_cand_elimn:
                    search_img = self.img_utils.resize_image(self.img_utils.load_image_and_params(search_image_path), des_img_sz=self.search_img_dim)

                    for lim in range(len(self.ostrack.backbone.layer_idx_to_en_early_cand_elimn)):
                        topk_indices_at_prev_layer = self.ostrack.get_global_topk_indices_fm_layer(lim)
                        self.img_utils.show_ce_out_img_fm_indices(layer_idx=self.ostrack.backbone.layer_idx_to_en_early_cand_elimn[lim],
                                                        search_img=search_img,
                                                        topk_indices_at_layer=topk_indices_at_prev_layer.squeeze(0),
                                                        img_size=self.ostrack.search_size_N)

               # transfer to cpu for conversion to numpy
               self.img_utils.image_viewer(classifier_score_map.squeeze(0).cpu(), "classifier score")
               self.img_utils.show_bbox_on_img(search_image_path, op_coord, req_resz=True,
                                                        des_img_sz=self.search_img_dim)

            print("Done with iteration", iter, "took", time.time() - img_start, "seconds")

        output_data['num_iters'] = iter

        dur = time.time() - start
        fps_rate = len(search_image_paths) / dur
        print("It took a total", dur, "seconds to prepare and track", len(search_image_paths), "search images, rate",
             fps_rate)

        output_data['time_to_execute_in_sec'] = dur
        output_data['fps_rate'] = fps_rate

        return output_data

    def track_seq_test(self, search_image_folder=None, search_image_paths=None, req_resz=False):
        with torch.no_grad():
            return self.track_seq_test_internal(search_image_folder=search_image_folder,
                                                search_image_paths=search_image_paths,
                                                req_resz=req_resz)

    def parse_gt_bbox_got10k(self, txt_file_path):
        with open(txt_file_path) as file:
            lines = [line.rstrip() for line in file]

        raw_gt_box = lines[0].split(',')

        gt_box = [int(float(raw_gt_box[0])),
                  int(float(raw_gt_box[1])),
                  int(float(raw_gt_box[2])),
                  int(float(raw_gt_box[3]))]

        bbox_params = torchvision.ops.box_convert(torch.Tensor(gt_box), 'xywh', 'xyxy')

        crop_params = gt_box

        print(gt_box, bbox_params, crop_params)

        return bbox_params, crop_params

def show_outputs_from_data(save_path, patch_img_dim):
    img_utils = ImageUtils(side_length=patch_img_dim)
    full_out_data = torch.load(save_path)
    folder_ct = 0
    for _, folder in full_out_data['folders'].items():
        print("Showing outputs for folder: ", folder, full_out_data[folder]['folder'])
        for curr_iter in range(full_out_data[folder]['num_iters']):
            # View classifier scores
            classifier_score_map = full_out_data[folder][str(curr_iter)]['classifier_score']
            img_utils.image_viewer(classifier_score_map, "classifer score")

            # View output on patch images
            search_image_path = full_out_data[folder][str(curr_iter)]['search_image_path']
            op_coord = full_out_data[folder][str(curr_iter)]['op_coord']
            bb_img = img_utils.show_bbox_on_img(search_image_path, op_coord, req_resz=True, des_img_sz=patch_img_dim, str=str(folder_ct))

            # View BB on full size image, only partially works for now
            # TODO (Anshu-man567): Fix it completely
            torch_image = img_utils.load_image_and_params(search_image_path)
            short_w, long_h = torch_image.shape[1], torch_image.shape[2]
            op_coord = torch.tensor(op_coord).squeeze(0).detach()
            pad = 10
            bb_resz = torch.tensor([
                (op_coord[0] - 0) * long_h / 256 - pad,
                (op_coord[1] - 56) * long_h / 256 - pad,
                (op_coord[2] - 0) * long_h / 256 + pad,
                (op_coord[3] - 56) * long_h / 256 + pad,
            ])
            img_utils.show_bbox_on_img(search_image_path, bb_resz.reshape(1,4), req_resz=False, str=str(folder_ct), width=5)

            # break
            # if curr_iter == 20:
            #     break
        folder_ct += 1
        # break


# def check_generated_outputs

def test_seq_ostrack_got10k(iter_lim=-1, show_dumps=0):

    seq_ostrack = SeqOSTrack(exec_mode=TrackExecutionMode.TEST,
                             img_lib=ImageParseLib.TORCHVISION,
                             show_dumps=show_dumps)

    got10k_test_dir = "/home/anshu-man567/PycharmProjects/ProjectIdeas/OSTrack/data/got10k/test/"
    full_out_data = OrderedDict()
    full_out_data['folders'] = OrderedDict()
    iter = 1
    running_avg_fps = 0.0
    for folder in sorted(os.listdir(got10k_test_dir)):
        if os.path.isdir(os.path.join(got10k_test_dir, folder)) is False:
            print("ERROR, this path is not a folder, skipping using it", os.path.join(got10k_test_dir, folder), folder)
            continue
        tmpl_img_file_path = os.path.join(got10k_test_dir, folder, "00000001.jpg")
        txt_path = os.path.join(got10k_test_dir, folder, "groundtruth.txt")
        search_image_folder_path = os.path.join(got10k_test_dir, folder)
        print(tmpl_img_file_path, txt_path, search_image_folder_path)
        image = seq_ostrack.img_utils.load_image_and_params(tmpl_img_file_path)
        bbox, crop = seq_ostrack.parse_gt_bbox_got10k(txt_path)
        seq_ostrack.img_utils.show_bbox_on_img(tmpl_img_file_path, bbox_coords=bbox)
        seq_ostrack.init_new_seq(tmpl_img_file_path=tmpl_img_file_path, req_resz=True, crop_box=crop)

        full_out_data['folders'][str(iter)] = folder
        full_out_data[folder] = seq_ostrack.track_seq_test(search_image_folder=search_image_folder_path, req_resz=True)
        running_avg_fps += full_out_data[folder]['fps_rate']
        print("Last FPS:", full_out_data[folder]['fps_rate'], "Current avg FPS:", running_avg_fps/iter)

        if iter == iter_lim:
            break
        else:
            iter += 1

    save_path = 'got10k_test_run_results_'+str(iter)+'.pth'
    torch.save(full_out_data, save_path)

    # show_outputs_from_data(save_path, img_dim=seq_ostrack.search_img_dim)


def test_seq_ostrack_got1_dynamic():
    # search_image_file_path = "../test_image/search/angry_cat.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/angry_cat_crop.jpg"
    # search_image_file_path = "../test_image/search/shapes.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/shapes_crop.jpg"
    # search_image_file_path = "../test_image/search/dk_search.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/dk_tmpl.jpg"
    # search_image_file_path = "../test_image/search/doggo.png"
    # tmpl_img_file_path = "../test_image/tmpl/doggo_crop.png"
    # search_image_file_path = "../test_image/search/x_patch_arr.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/template.jpg"

    # search_image_file_path = "../test_image/search/bb_search.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/bb_tmpl.jpg"

    # search_image_file_path = "../test_image/search/whale_sample.jpg"

    '''
    GOT 10k details:
    01 => dolphin
    02 => black boat in city river
    03 => blue boat
    23 => driffffffffting
    38 => some angel fish?
    52 => bullock cart battles
    90 => is the weirdest I cant comprehend
    62 => kangaroo fights with another animal (forgot its name) c for combat :)
    43 => another kangaroo running, but couldnt track it due to occlusion of the same color
    45 => template so smol, even I cant detect it
    '''

    seq_ostrack = SeqOSTrack(exec_mode=TrackExecutionMode.TEST,
                             img_lib=ImageParseLib.TORCHVISION,
                             show_dumps=0,
                             print_stats=0,
                             en_early_cand_elimn=1)
    input_img_folder = "/home/anshu-man567/PycharmProjects/ProjectIdeas/OSTrack/data/got10k/test/GOT-10k_Test_000062/"

    tmpl_img_file_path = os.path.join(input_img_folder, "00000001.jpg")
    txt_path = os.path.join(input_img_folder, "groundtruth.txt")
    search_image_folder_path = input_img_folder
    print(tmpl_img_file_path, txt_path, search_image_folder_path)
    image = seq_ostrack.img_utils.load_image_and_params(tmpl_img_file_path)
    bbox, crop = seq_ostrack.parse_gt_bbox_got10k(txt_path)
    seq_ostrack.img_utils.show_bbox_on_img(tmpl_img_file_path, bbox_coords=bbox)
    seq_ostrack.init_new_seq(tmpl_img_file_path=tmpl_img_file_path, req_resz=True, crop_box=crop)
    seq_ostrack.track_seq_test(search_image_folder=search_image_folder_path, req_resz=True)


def test_seq_ostrack_single():
    # search_image_file_path = "../test_image/search/angry_cat.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/angry_cat_crop.jpg"
    # search_image_file_path = "../test_image/search/shapes.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/shapes_crop.jpg"
    # search_image_file_path = "../test_image/search/dk_search.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/dk_tmpl.jpg"
    # search_image_file_path = "../test_image/search/doggo.png"
    # tmpl_img_file_path = "../test_image/tmpl/doggo_crop.png"
    # search_image_file_path = "../test_image/search/x_patch_arr.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/template.jpg"

    search_image_file_path = "../test_image/search/bb_search.jpg"
    tmpl_img_file_path = "../test_image/tmpl/bb_tmpl.jpg"

    # search_image_file_path = "../test_image/search/whale_sample.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/whale_tmpl.jpg"

    seq_ostrack = SeqOSTrack(exec_mode=TrackExecutionMode.TEST,
                             img_lib=ImageParseLib.TORCHVISION,
                             show_dumps=1)

    seq_ostrack.init_new_seq(tmpl_img_file_path=tmpl_img_file_path)
    seq_ostrack.track_seq_test(search_image_folder=None, search_image_paths=[search_image_file_path])


def test_ostrack_blocks():
    search_image_file_path = "../test_image/search/angry_cat.jpg"
    tmpl_img_file_path = "../test_image/tmpl/angry_cat_crop.jpg"
    # search_image_file_path = "../test_image/search/shapes.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/shapes_crop.jpg"
    # search_image_file_path = "../test_image/search/dk_search.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/dk_tmpl.jpg"
    # search_image_file_path = "../test_image/search/doggo.png"
    # tmpl_img_file_path = "../test_image/tmpl/doggo_crop.png"
    # search_image_file_path = "../test_image/search/x_patch_arr.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/template.jpg"

    # search_image_file_path = "../test_image/search/bb_search.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/bb_tmpl.jpg"

    # search_image_file_path = "../test_image/search/whale_sample.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/whale_tmpl.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ostrack = OSTrackModel(print_stats=0, en_early_cand_elimn=True)
    ostrack = ostrack.eval()
    ostrack = ostrack.to(device)

    with torch.no_grad():
        ostrack_state_dict, is_exact_copy = ostrack.gtf_ostrack.create_new_state_dict(ostrack)
        print("Is it an exact copy?", is_exact_copy)

        missing_keys, unexpected_keys = ostrack.load_state_dict(ostrack_state_dict, strict=False)

        # ostrack.gtf_ostrack.print_model_info(ostrack)

        ostrack.create_tmpl_patches(template_img_file_path=tmpl_img_file_path)

        x = ostrack.try_on_input_tokens(search_img_file_path=search_image_file_path)

if __name__ == "__main__":
    # test_seq_ostrack_single()
    test_seq_ostrack_got1_dynamic()
    # test_seq_ostrack_got10k(iter_lim=30, show_dumps=1)
    # show_outputs_from_data("got10k_test_run_results_180.pth", 256)