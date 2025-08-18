import sys
import math
import torch
import torch.nn as nn
from torch.onnx.symbolic_opset9 import unsqueeze

from ostrack_vit.mlp import MultiLayerPerceptron
from ostrack_vit.msa import MultiHeadSelfAttention


class ViTEncoder(nn.Module):
    def __init__(self,
                 layer_idx=sys.maxsize,
                 size_D=768,
                 num_heads=12,
                 hidden_layer_multiplier=4,
                 size_N=196,
                 search_size_N=256,
                 tmpl_size_N=64,
                 en_early_cand_elimn=False,
                 early_cand_elimn_ratio=1.0, # 1.0 => keep all, 0.0 => keep none
                 print_stats=False):
        super().__init__()
        self.print_stats = print_stats
        self.en_early_cand_elimn = en_early_cand_elimn

        self.layer_idx = layer_idx
        self.search_size_N = search_size_N
        self.tmpl_size_N = tmpl_size_N
        self.size_D = size_D
        self.norm1 = nn.LayerNorm([size_D])

        # TODO : Figure out how to pass norm1
        self.attn = MultiHeadSelfAttention(en_early_cand_elimn=self.en_early_cand_elimn,
                                           size_D=self.size_D,
                                           num_heads=num_heads,
                                           qkv_bias=True)

        self.norm2 = nn.LayerNorm([self.size_D])

        # TODO (Anshu-man567): Figure out integrating your MLP or using timm's
        self.mlp = MultiLayerPerceptron(size_D=self.size_D,
                                        hidden_layer_multiplier=hidden_layer_multiplier,
                                        size_N=size_N)

        # from timm.models.layers import Mlp
        # self.mlp = Mlp(in_features=self.size_D,
        #                hidden_features=self.size_D*hidden_layer_multiplier,
        #                act_layer=nn.GELU)
                        # drop=drop

        self.en_early_cand_elimn = en_early_cand_elimn
        self.early_cand_elimn_token_mask = nn.Identity()
        self.early_cand_elimn_ratio = early_cand_elimn_ratio  # 1.0 => keep all, 0.0 => keep none

        self.topk_indices = None
        self.msa_output = None
        self.mlp_output = None

        print("ViT Encoder initalized with",
              "layer_idx", self.layer_idx,
              "en_early_cand_elimn", self.en_early_cand_elimn,
              "early_cand_elimn_ratio", self.early_cand_elimn_ratio)

    def forward(self, input):

        # Includes residual connection after MSA
        msa_output = self.attn(self.norm1(input)) + input

        if self.en_early_cand_elimn == 1:
            # TODO (Anshu-man567): Perform the padding here ???? OR NOT FiGURE OUT
            #    in case of CE, it would be the remaining patches from search side
            msa_output = self.perform_early_cand_elimn(attn_mat=self.attn.scaled_qk_out, msa_output=msa_output)

        self.msa_output = msa_output
        if self.print_stats:
            print("size of output aft msa layer", msa_output.size())

        # Residual connection after MLP, with values after MSA
        mlp_output = self.mlp(self.norm2(msa_output)) + msa_output

        if self.print_stats:
            print("size of output aft residual conn", mlp_output.size())

        self.mlp_output = mlp_output

        return mlp_output

    def perform_early_cand_elimn(self, attn_mat, msa_output):

        # msa_output has the shape size_B, size_N, size_D
        tmpl_tokens = msa_output[:,:self.tmpl_size_N]
        search_tokens = msa_output[:,self.tmpl_size_N:]

        # print("INPUT TO CE MODULE SHAPES", attn_mat.shape, msa_output.shape, tmpl_tokens.shape, search_tokens.shape)

        # extract the search tokens corresponding to the template image (has values from Q_x * K_q)
        extracted_attn_mat = attn_mat[:,:,:self.tmpl_size_N:,self.tmpl_size_N:]

        # apply masking
        extracted_attn_mat = self.early_cand_elimn_token_mask(extracted_attn_mat) # TODO : Make it a mat mul

        # calculate mean across all the template tokens (dim=2) and then take mean across the heads of MHSA (dim=1)
        # Output shape is [1, 256] ie [1, search_size_N]
        extracted_attn_mat = extracted_attn_mat.mean(dim=2).mean(dim=1)
        
        # sort and get the top k idx for the mean values
        _, sorted_indices = torch.sort(extracted_attn_mat, dim=1, descending=True)
        ct_early_cand_elimn = self.search_size_N
        if self.early_cand_elimn_ratio < 1.0:
            ct_early_cand_elimn = math.floor(self.early_cand_elimn_ratio * search_tokens.shape[1])

        # Do not try to combine this line with the below one, torch doenst likes it for some reason
        topk_indices = sorted_indices[:,:ct_early_cand_elimn]

        self.topk_indices = topk_indices
        # Need to do this to match the dimension of the index and input arrays for gather op
        topk_indices = topk_indices.unsqueeze(-1).expand(-1,-1,self.size_D)

        # store the top k idx locally
        # return after appending only the tmpl + top k

        # Get the topK tokens from the search_size_N dim
        # TODO (Anshu-man567): Add this to the words of wisdom
        # NOTE: Also notice that since the topk_indices are in (ascending) order of importance,
        #   the tokens that the token thinks most likely is the template is now placed first
        topk_search_tokens = torch.gather(search_tokens, dim=1, index=topk_indices)

        filtered_attn_out = torch.cat([tmpl_tokens, topk_search_tokens], dim=1)

        return filtered_attn_out

def print_model_state_dict():
    model = ViTEncoder()

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # x = torch.randn(2, 3, 4)
    #
    # print(x)
    #
    # print(torch.sort(x, dim=2))
    #
    # print(torch.sort(x, dim=1))
    #
    # print(torch.sort(x, dim=0))

    # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])


if __name__ == "__main__":
    print_model_state_dict()






'''
Available weights per ViT Encoder Block

LN:
blocks.0.norm1.weight 	len: torch.Size([768])
blocks.0.norm1.bias 	len: torch.Size([768])

Proj:
blocks.0.attn.proj.weight 	len: torch.Size([768, 768])
blocks.0.attn.proj.bias 	len: torch.Size([768])

Attention Weights aka W_q, W_k, W_v; hopefully in that order:
blocks.0.attn.qkv.bias 	len: torch.Size([2304])
blocks.0.attn.qkv.weight 	len: torch.Size([2304, 768])

LN:
blocks.0.norm2.weight 	len: torch.Size([768])
blocks.0.norm2.bias 	len: torch.Size([768])

MLP l1:
blocks.0.mlp.fc1.weight 	len: torch.Size([3072, 768])
blocks.0.mlp.fc1.bias 	len: torch.Size([3072])

MLP l2:
blocks.0.mlp.fc2.weight 	len: torch.Size([768, 3072])
blocks.0.mlp.fc2.bias 	len: torch.Size([768])

Maybe at the end???
norm.weight 	len: torch.Size([768])
norm.bias 	len: torch.Size([768])


blocks.1.attn.qkv.bias 	len: torch.Size([2304])
blocks.2.attn.qkv.bias 	len: torch.Size([2304])
blocks.3.attn.qkv.bias 	len: torch.Size([2304])
blocks.4.attn.qkv.bias 	len: torch.Size([2304])
blocks.5.attn.qkv.bias 	len: torch.Size([2304])
blocks.6.attn.qkv.bias 	len: torch.Size([2304])
blocks.7.attn.qkv.bias 	len: torch.Size([2304])
blocks.8.attn.qkv.bias 	len: torch.Size([2304])
blocks.9.attn.qkv.bias 	len: torch.Size([2304])
blocks.10.attn.qkv.bias 	len: torch.Size([2304])
blocks.11.attn.qkv.bias 	len: torch.Size([2304])



'''
