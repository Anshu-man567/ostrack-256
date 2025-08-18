import torch
import torch.nn as nn

from collections import OrderedDict

# from keras.src.ops import top_k

from ostrack_vit.input_layer import InputLayer
from ostrack_vit.transformer import ViTEncoder

class VisionTransformerModel(nn.Module):
    def __init__(self,
                 patch_size=16,
                 size_D=768,
                 size_N=196,
                 search_size_N=256,
                 tmpl_size_N=64,
                 num_heads=12,
                 hidden_layer_multiplier=4,
                 vit_encoder_depth=12,
                 split_pos_embed=False,
                 vit_img_dim=224,
                 tmpl_img_dim=128,
                 search_img_dim=256,
                 en_early_cand_elimn=False,
                 print_stats=0):
        super().__init__()

        self.en_early_cand_elimn = en_early_cand_elimn
        self.patch_size = patch_size
        self.n_channels = 3
        self.size_D = size_D
        self.size_N = size_N
        self.search_size_N = search_size_N
        self.tmpl_size_N = tmpl_size_N
        self.num_heads = num_heads
        self.hidden_layer_multiplier = hidden_layer_multiplier
        self.print_stats = print_stats
        self.vit_encoder_depth = vit_encoder_depth
        self.vit_img_dim = vit_img_dim

        self.cls_token = nn.Parameter(torch.empty(size_D))

        self.patch_embed = InputLayer(patch_size=self.patch_size,
                                      n_channels=self.n_channels,
                                      output_size=self.size_D,
                                      print_stats=self.print_stats)

        # directly has the embedding mappings
        self.pos_embed = nn.Parameter(torch.empty([1, 197, self.size_D]))

        self.pos_embed_z = None
        self.pos_embed_x = None

        # for patch embedding
        patch_start_index = 1 # idx 0 is for CLS
        patch_pos_embed = self.extract_patch_pos_embed(patch_start_index)

        if split_pos_embed is True:
            # for search region
            search_patch_pos_embed = self.interpolate_patch_pos_embed(patch_pos_embed, patch_size, search_img_dim)

            # for template region
            template_patch_pos_embed = self.interpolate_patch_pos_embed(patch_pos_embed, patch_size, tmpl_img_dim)

            self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
            self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # TODO (Anshu-man567): Add Drop Out layer for training value = 0.1
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule


        if self.en_early_cand_elimn is True:
            print("EARLY CAND ELIMN is:", self.en_early_cand_elimn)
            self.layer_idx_to_en_early_cand_elimn = [3, 6, 9]    # indices for layers 4, 7, 10
            self.top_k_ratio_for_early_cand_elimn = 0.7          # 0.7 is the keeping ratio in paper
        else:
            self.layer_idx_to_en_early_cand_elimn = []
            self.top_k_ratio_for_early_cand_elimn = 1.0

        self.blocks = nn.Sequential(OrderedDict(
            [ (str(layer_idx), ViTEncoder(layer_idx=layer_idx,
                                          size_D=self.size_D,
                                          search_size_N=self.search_size_N,
                                          tmpl_size_N=self.tmpl_size_N,
                                          num_heads=self.num_heads,
                                          hidden_layer_multiplier=self.hidden_layer_multiplier,
                                          size_N=self.size_N,
                                          en_early_cand_elimn=(1 if layer_idx in self.layer_idx_to_en_early_cand_elimn else 0),
                                          early_cand_elimn_ratio=(self.top_k_ratio_for_early_cand_elimn if layer_idx in self.layer_idx_to_en_early_cand_elimn else 1.0)))
            for layer_idx in range(self.vit_encoder_depth)]))

        self.norm = nn.LayerNorm([self.size_D])

    def extract_patch_pos_embed(self, patch_start_index):
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.vit_img_dim // self.patch_size, self.vit_img_dim // self.patch_size
        print(patch_pos_embed.shape)
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        return patch_pos_embed

    def interpolate_patch_pos_embed(self, patch_pos_embed, patch_size, img_dim):
        H, W = img_dim, img_dim
        new_P_H, new_P_W = H // patch_size, W // patch_size
        print("P H W search", new_P_H, new_P_W)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed,
                                                    size=(new_P_H, new_P_W),
                                                    mode='bicubic',
                                                    align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        return patch_pos_embed

    # TODO : Fix this and add support for proper inputs, check out DataLoader and Dataset
    def forward(self, patches):

        # B, _v1 = patches.shape
        # print("I am here : ", patches.shape)

        ## NEED THE BELOW FOR VIT:
        # B = 1
        # patches = patches.reshape([B, self.size_N,  self.size_D])
        # # print(self.pos_embed[0,1:,:].shape)
        # patches = patches + self.pos_embed[0,1:,:]

        # The OG implementation use this, idk what diff
        # TODO (anshu-man567): Figure out why this
        # for i, blk in enumerate(self.blocks):
        #     patches = blk(patches)
        #
        # vit_output = self.norm(patches)

        vit_output = self.norm(self.blocks(patches))

        return vit_output


if __name__ == "__main__":
    hello = OrderedDict((['1', "big"], ['2', "hero"]))

    print(hello['1'])
