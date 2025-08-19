from collections import OrderedDict

import numpy
import torch
import torch.nn as nn

from ostrack.post_process_utils import PostProcessUtils
from ostrack.ostrack_decoder import OSTrackDecoder

from utils.img_preprocessor import Preprocessor
from ostrack_vit.vit import VisionTransformerModel
from utils.image_parse_lib import ImageParseLib
from utils.image_utils import ImageUtils

"""OSTrack Model implementation

This class implements the end-to-end OSTrack model architecture, including patch embedding,
positional encoding, the transformer encoder layers, and the decoder for classification scores,
offsets, and bounding box sizes.
"""
class OSTrackModel(nn.Module):

    """Initialize the OSTrack model

    Args:
        patch_size: Size of each image patch
        vit_img_dim: Dimension of the input image for ViT (H and W are assumed to be equal)
        search_img_dim: Dimension of the search image for ViT (H and W are assumed to be equal)
        tmpl_img_dim: Dimension of the template image for ViT (H and W are assumed to be equal)
        size_D: Dimension of the input embeddings
        num_heads: Number of attention heads in MHSA
        hidden_layer_multiplier: Multiplier for hidden layer size in MLP
        vit_encoder_depth: Number of transformer encoder layers
        print_stats: Enable printing of debug stats
        save_outputs: Enable saving outputs to a dictionary
        show_dumps: Enable showing dumps of intermediate outputs
        en_early_cand_elimn: Enable early candidate elimination if True
        img_lib: Image library to use for image processing
    """
    def __init__(self,
                 patch_size=16,
                 vit_img_dim=224,
                 search_img_dim=256,
                 tmpl_img_dim=128,
                 size_D=768,
                 num_heads=12,
                 hidden_layer_multiplier=4,
                 vit_encoder_depth=12,
                 print_stats=0,
                 save_outputs=0,
                 show_dumps=0,
                 en_early_cand_elimn=False,
                 img_lib=ImageParseLib.TORCHVISION):
        super().__init__()
        self.print_stats = print_stats
        self.save_outputs = save_outputs
        self.show_dumps = show_dumps
        self.en_early_cand_elimn = en_early_cand_elimn

        self.patch_size = patch_size
        self.vit_img_dim = vit_img_dim
        self.search_img_dim = search_img_dim
        self.tmpl_img_dim = tmpl_img_dim
        self.n_channels = 3
        self.size_D = size_D
        self.num_heads = num_heads
        self.search_size_N = self.calc_num_search_patches()
        self.tmpl_size_N = self.calc_num_template_patches()
        self.size_N = self.search_size_N + self.tmpl_size_N
        self.hidden_layer_multiplier = hidden_layer_multiplier
        self.vit_encoder_depth = vit_encoder_depth

        if self.save_outputs:
            self.out_dict = OrderedDict()

        self.img_lib = img_lib
        self.img_utils = ImageUtils(img_lib=self.img_lib)
        self.preprocessor = Preprocessor(img_lib=self.img_lib)

        self.backbone = VisionTransformerModel(patch_size=self.patch_size,
                                               size_D=self.size_D,
                                               size_N=self.size_N,  # size of N for ViT-B/16 for 224x224 img
                                               search_size_N=self.search_size_N,
                                               tmpl_size_N=self.tmpl_size_N,
                                               num_heads=self.num_heads,
                                               hidden_layer_multiplier=self.hidden_layer_multiplier,
                                               vit_encoder_depth=self.vit_encoder_depth,
                                               split_pos_embed=True,
                                               vit_img_dim=self.vit_img_dim,
                                               tmpl_img_dim=self.tmpl_img_dim,
                                               search_img_dim=self.search_img_dim,
                                               en_early_cand_elimn=self.en_early_cand_elimn,
                                               print_stats=self.print_stats)

        self.box_head = OSTrackDecoder(print_stats=0)

        self.ostrack_post_process = PostProcessUtils(apply_hann_window=False)

        print("Initializing OSTrack model with these stats:")
        print("\ten_early_cand_elimn:", self.en_early_cand_elimn)
        print("\tpatch_size:", self.patch_size)
        print("\tvit_img_dim:", self.vit_img_dim)
        print("\tsearch_img_dim", self.search_img_dim)
        print("\ttmpl_img_dim", self.tmpl_img_dim)
        print("\thidden dimension", self.size_D)

    def forward(self, patches):

        # PART 1 - ViT Backbone
        vit_output = self.backbone(patches)
        if self.print_stats:
            print("ViT Output shape:", vit_output.shape)

        if self.save_outputs:
            self.out_dict['x_patch_arr'] = patches
            self.out_dict['backbone_feat'] = vit_output
            for i, blk in enumerate(self.backbone.blocks):
                self.out_dict['attn_out_' + str(i)] = blk.msa_output
                self.out_dict['mlp_out_' + str(i)] = blk.mlp_output
                if blk.topk_indices is not None:
                    self.out_dict['topk_indices_' + str(i)] = blk.topk_indices

        # Extract search patches from the ViT output
        # This is done to get the remaining search patches from the last layer
        # if en_early_cand_elimn is True, else all search patches are returned
        search_patches_fm_vit_out = self.get_search_patches_fm_vit_out(vit_output)
        if self.print_stats:
            print("Only Search Patches from ViT Output shape:", search_patches_fm_vit_out.shape, self.en_early_cand_elimn)

        # TODO : DEPRECATE THIS LOGIC
        # Fix this contiguous thing
        # opt = ((search_patches_fm_vit_out.unsqueeze(-1)).permute((0, 3, 2, 1))).contiguous()
        # if self.print_stats:
        #     print("search_patches_fm_vit_out", search_patches_fm_vit_out.shape)
        #
        # bs, Nq, C, HW = opt.size()
        # if self.print_stats:
        #     print("bs, Nq, C, HW", bs, Nq, C, HW)

        opt = search_patches_fm_vit_out.permute(0, 2, 1)
        if self.print_stats:
            print("opt", opt.shape)

        opt_feat = opt.view(-1, self.size_D, self.patch_size, self.patch_size)
        if self.print_stats:
            print("opt_feat", opt_feat.shape)

        # PART 2 - Decoder
        # Get the classifier score map, size map, and offset map from the FCN heads
        classifier_score_map, size_map, offset_map = self.box_head(
            opt_feat)  # = self.box_head(search_patches_fm_vit_out.unsqueeze(0).permute(2, 3, 1, 0))

        if self.save_outputs:
            self.out_dict['score_map'] = classifier_score_map

        # Get the output coordinates from the post-processing step
        # The coordinates are in the format (x1, y1, x2, y2)
        op_coord = self.ostrack_post_process.get_output_coords(classifier_score_map=classifier_score_map,
                                                               size_map=size_map,
                                                               offset_map=offset_map)

        if self.save_outputs:
            print("Dumped the data for this ")
            torch.save(self.out_dict, 'my_ostrack_out.pt')

        return op_coord, classifier_score_map, size_map, offset_map

    """Get the global indices of the top-k search tokens from the last layer

    Returns the top-k indices from a given layer (idx in the layer_idx_to_en_early_cand_elimn list)
    The returned indices are global indices ie indices belong to the position they would have been
     amongst the original #search_size_N input vectors
    """
    def get_global_topk_indices_fm_layer(self, lim):
        topk_indices_at_prev_layer = self.get_topk_ind_fm_layer(self.backbone.layer_idx_to_en_early_cand_elimn[0])
        for idx in range(0, lim + 1):  # Assumes that the order of indices from low to high
            if idx == 0:
                continue
            layer_idx = self.backbone.layer_idx_to_en_early_cand_elimn[idx]
            topk_indices_at_curr_layer = self.get_topk_ind_fm_layer(layer_idx)
            topk_indices_at_prev_layer = torch.gather(topk_indices_at_prev_layer,
                                                      dim=1, 
                                                      index=topk_indices_at_curr_layer)
        return topk_indices_at_prev_layer

    """"Calculate the number of search patches based on the image dimension and patch size

    Returns:
        int: Number of search patches
    """
    def calc_num_search_patches(self):
        return int((self.search_img_dim * self.search_img_dim) / (self.patch_size ** 2))

    """Calculate the number of template patches based on the image dimension and patch size

    Returns:
        int: Number of template patches
    """
    def calc_num_template_patches(self):
        return int((self.tmpl_img_dim * self.tmpl_img_dim) / (self.patch_size ** 2))

    """Get the top-k indices from a specific layer

    Args:
        layer_idx (int): Index of the layer to get the top-k indices from

    Returns:
        torch.Tensor: Top-k indices from the specified layer
    """
    def get_topk_ind_fm_layer(self, layer_idx):
        return self.backbone.blocks[layer_idx].topk_indices

    """Combine search and template patches into a single tensor

    Args:
        search_patches (torch.Tensor): Search patches tensor
        tmpl_patches (torch.Tensor, optional): Template patches tensor. Defaults to None.
        template_img_file_path (str, optional): Path to the template image file. Defaults to None.

    Returns:
        torch.Tensor: Combined patches tensor
    """
    def combine_patches(self, search_patches, tmpl_patches:torch.tensor =None, template_img_file_path=None):
        template_patches = tmpl_patches
        if template_img_file_path is not None:
            template_patches = self.create_tmpl_patches(template_img_file_path)
        patches = torch.concat((template_patches, search_patches), 1)
        return patches

    """Get the search patches from the ViT output

    Args:
        vit_output (torch.Tensor): Output from the ViT backbone

    Returns:
        torch.Tensor: Search patches extracted from the ViT output
    """
    def get_search_patches_fm_vit_out(self, vit_output):
        search_patches_fm_vit_out = None

        if self.en_early_cand_elimn:
            search_patches_fm_vit_out = torch.zeros(size=(vit_output.shape[0], self.search_size_N, vit_output.shape[-1]), device='cuda')
            last_layer_idx_fm_list = len(self.backbone.layer_idx_to_en_early_cand_elimn) - 1
            topk_indices = self.get_global_topk_indices_fm_layer(last_layer_idx_fm_list)
            search_patches_fm_vit_out[:,topk_indices,:] = vit_output[:, self.tmpl_size_N:,:]
        else:
            search_patches_fm_vit_out = vit_output[:, self.tmpl_size_N:,:]

        return search_patches_fm_vit_out

    """Create template patches from the template image file

    Args:
        template_img_file_path (str): Path to the template image file
        req_resz (bool, optional): Whether to resize the image. Defaults to False.
        crop_box (tuple, optional): Coordinates for cropping the image. Defaults to None.

    Returns:
        torch.Tensor: Processed template patches tensor
    """
    def create_tmpl_patches(self, template_img_file_path, req_resz=False, crop_box=None):
        tmpl_img = self.img_utils.load_image_and_params(template_img_file_path)
        print("tmpl img", template_img_file_path)
        self.img_utils.image_viewer(tmpl_img, "tmpl_img ")
        if req_resz is not False:
            print("crop", crop_box)
            if crop_box is not None:
                tmpl_img = self.img_utils.crop_image(tmpl_img, crop_box[0], crop_box[1], crop_box[2], crop_box[3])
            tmpl_img = self.img_utils.resize_image(tmpl_img, self.tmpl_img_dim)
        self.img_utils.image_viewer(tmpl_img, "tmpl_img 222 ")
        template_patches = self.preprocessor.process(tmpl_img)
        if self.print_stats:
            print("template patches", template_patches.shape)
        if self.save_outputs:
            self.out_dict['in_z'] = template_patches

        template_patches = self.backbone.patch_embed(template_patches)
        if self.save_outputs:
            self.out_dict['patch_z'] = template_patches

        template_patches = template_patches + self.backbone.pos_embed_z
        if self.print_stats:
            print("Template Patches shape:", template_patches.shape)
        if self.save_outputs:
            self.out_dict['patch_pos_z'] = template_patches

        return template_patches

    """Create search patches from the search image file

    Args:
        search_img_file_path (str): Path to the search image file
        req_resz (bool, optional): Whether to resize the image. Defaults to False.

    Returns:
        torch.Tensor: Processed search patches tensor
    """
    def create_search_patches(self, search_img_file_path, req_resz=False):
        # TODO (Anshu-man567): Apply this https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/3
        search_img = self.img_utils.load_image_and_params(search_img_file_path)
        if req_resz is not False:
            search_img = self.img_utils.resize_image(search_img, self.search_img_dim)

        search_patches = self.preprocessor.process(search_img)
        if self.print_stats:
            print("search patches", search_patches.shape)
        if self.save_outputs:
            self.out_dict['in_x'] = search_patches

        search_patches = self.backbone.patch_embed(search_patches)
        if self.save_outputs:
            self.out_dict['patch_x'] = search_patches

        search_patches = search_patches + self.backbone.pos_embed_x
        if self.print_stats:
            print("Search Patches shape:", search_patches.shape)
        if self.save_outputs:
            self.out_dict['patch_pos_x'] = search_patches

        return search_patches

    """Show the output image with bounding boxes for a specific layer

    Args:
        layer_idx (int): Index of the layer to show the output for
        search_img_file_path (str): Path to the search image file
    """
    def show_ce_out_img(self, layer_idx, search_img_file_path):
        search_img = self.img_utils.load_image_and_params(search_img_file_path)
        # patch size is also the number of tokens in each direction so...
        topk_indices_at_layer = self.get_topk_ind_fm_layer(layer_idx)
        self.show_ce_out_img_fm_indices(layer_idx, search_img, topk_indices_at_layer)

    """Show the output image with bounding boxes for a specific layer and indices

    Args:
        layer_idx (int): Index of the layer to show the output for
        search_img (torch.Tensor): Search image tensor
        topk_indices_at_layer (torch.Tensor): Indices of the top-k search tokens at the specified layer
    """
    def try_on_input_tokens(self, search_img_file_path, template_img_file_path=None):
        search_patches = self.create_search_patches(search_img_file_path)

        patches = self.combine_patches(search_patches, template_img_file_path=template_img_file_path)
        if self.print_stats:
            print("Combined Patches shape:", patches.shape)

        # TODO (Anshu-man567): Add dropout here!
        # x = self.pos_drop(x)
        op_coord, classifier_score_map, _1, _2 = self.forward(patches)


        # transfer to cpu for conversion to numpy
        self.img_utils.image_viewer(classifier_score_map.squeeze(0).cpu(), "classifier score", self.show_dumps)

        # if self.en_early_cand_elimn:
        #     for layer_idx in self.backbone.layer_idx_to_en_early_cand_elimn:
        #         self.show_ce_out_img(layer_idx, search_img_file_path)

        if self.en_early_cand_elimn:
            search_img = self.img_utils.load_image_and_params(search_img_file_path)

            for lim in range(len(self.backbone.layer_idx_to_en_early_cand_elimn)):
                topk_indices_at_prev_layer = self.get_global_topk_indices_fm_layer(lim)
                self.show_ce_out_img_fm_indices(layer_idx=self.backbone.layer_idx_to_en_early_cand_elimn[lim],
                                                search_img=search_img,
                                                topk_indices_at_layer=topk_indices_at_prev_layer.squeeze(0))

        if self.save_outputs:
            print("Dumped the data for this ")
            torch.save(self.out_dict, 'my_ostrack_out.pt')

        self.img_utils.show_bbox_on_img(search_img_file_path, op_coord)

        return op_coord


def test_ostrack_blocks():
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

    search_image_file_path = "../test_image/search/whale_sample.jpg"
    tmpl_img_file_path = "../test_image/tmpl/whale_tmpl.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ostrack = OSTrackModel(print_stats=1, show_dumps=1)
    ostrack = ostrack.eval()
    ostrack = ostrack.to(device)

    with torch.no_grad():
        ostrack_state_dict, is_exact_copy = ostrack.gtf_ostrack.create_new_state_dict(ostrack)
        print("Is it an exact copy?", is_exact_copy)

        missing_keys, unexpected_keys = ostrack.load_state_dict(ostrack_state_dict, strict=False)

        # ostrack.gtf_ostrack.print_model_info(ostrack)

        x = ostrack.try_on_input_tokens(search_img_file_path=search_image_file_path,
                                        template_img_file_path=tmpl_img_file_path)


if __name__ == "__main__":
    test_ostrack_blocks()
