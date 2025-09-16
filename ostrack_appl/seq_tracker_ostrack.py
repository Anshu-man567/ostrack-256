import os
import time
import torch
import torchvision
from enum import Enum
from pathlib import Path
from collections import OrderedDict

from ostrack.ostrack_model import OSTrackModel
from utils.image_parse_lib import ImageParseLib
from utils.image_utils import ImageUtils
from utils.get_trained_features import GetTrainedFeatures


class TrackExecutionMode(Enum):
    TEST = 1
    TRAIN = 2

class SeqOSTrack:
    def __init__(self,
                 size_D=768,
                 vit_img_dim=224,
                 search_img_dim=256,
                 tmpl_img_dim=128,
                 exec_mode=TrackExecutionMode.TEST,
                 img_lib=ImageParseLib.TORCHVISION,
                 show_dumps=0,
                 save_outputs=0,
                 print_stats=0,
                 en_early_cand_elimn=1,
                 pretrained_weights=''):

        self.exec_mode = exec_mode
        self.img_utils = ImageUtils(img_lib)
        self.save_outputs = save_outputs
        self.show_dumps = show_dumps
        self.print_stats = print_stats
        self.en_early_cand_elimn = en_early_cand_elimn
        self.search_img_dim = search_img_dim
        self.tmpl_img_dim = tmpl_img_dim
        self.size_D = size_D
        self.pretrained_weights = pretrained_weights

        # Used to cache the template patches across a single tracking iteration
        self.template_patch = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ostrack = OSTrackModel(size_D=self.size_D,
                                    vit_img_dim=vit_img_dim,
                                    search_img_dim=self.search_img_dim,
                                    tmpl_img_dim=self.tmpl_img_dim,
                                    print_stats=self.print_stats,
                                    show_dumps=self.show_dumps,
                                    en_early_cand_elimn=en_early_cand_elimn)
        self.ostrack = self.ostrack.to(self.device)

        if exec_mode is TrackExecutionMode.TEST:
            if self.pretrained_weights == '':
                # Get the folder path to project
                script_dir = str(Path(__file__).parent.parent)
                if self.en_early_cand_elimn:
                    # self.pretrained_weights = script_dir +'/weights/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar'

                    # THE BELOW ONE ACTUALLY GAVE WORSE RESULT IN ONE GOT-10k test with kangaroo and racoon, towards the end identifies the 2nd racoon as the kangaroo
                    # FROM THE HEATMAPS can confirm that it recognizes the racoon and kangaroo as the object of interest, probably something to do with less training causing transformer to achieve subpar results
                    # Try to figure out why???
                    self.pretrained_weights = script_dir+'/weights/vitb_256_mae_ce_32x4_got10k_ep100/OSTrack_ep0100.pth.tar'
                else:
                    self.pretrained_weights = script_dir+'/weights/vitb_256_mae_32x4_ep300/OSTrack_ep0300.pth.tar'

            print("Picked weights from path: ", self.pretrained_weights, "EARLY CAND ELIMN is", self.en_early_cand_elimn)

            self.gtf_ostrack = GetTrainedFeatures(self.pretrained_weights)

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

        if self.save_outputs:
            output_data = OrderedDict()
            output_data['folder'] = search_image_folder

        iter = 0
        start2 = time.time()
        for search_image_path in search_image_paths:
            if self.print_stats:
                print("Testing image", search_image_path)
            img_start = time.time()
            search_patches = self.ostrack.create_search_patches(search_image_path, req_resz)
            patches = self.ostrack.combine_patches(search_patches, tmpl_patches=self.template_patch)

            op_coord, classifier_score_map, _1, _2 = self.ostrack.forward(patches)

            if self.save_outputs:
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
            if self.print_stats:
                print("Done with iteration", iter, "took", time.time() - img_start, "seconds")

        dur = time.time() - start2
        fps_rate = len(search_image_paths) / dur
        print("It took a total", dur, "seconds to prepare and track", len(search_image_paths), "search images, rate", fps_rate)

        if self.save_outputs:
            output_data['num_iters'] = iter
            output_data['time_to_execute_in_sec'] = dur
            output_data['fps_rate'] = fps_rate

        if self.save_outputs:
            return output_data
        return None

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
