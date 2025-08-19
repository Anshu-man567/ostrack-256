import torch
import torch.nn as nn

"""Multi Head Self Attention implementation for Transformers

This class implements the Multi Head Self Attention

This module consists of the following layers:
1. Linear projection for Q, K, V matrices
2. Scaled Dot Product Attention (SDPA) calculations
3. Linear projection for SDPA output
"""
class MultiHeadSelfAttention(nn.Module):

    """Multi-head self-attention module for transformer models.

    Args:
        en_early_cand_elimn: Enables early candidate elimination if True.
        size_D: Dimension of the input embeddings
        num_heads: Number of attention heads
        attn_dropout_rate: Dropout probability for attention scores
        proj_dropout_rate: Dropout probability for output projection
        qkv_bias: If True, adds bias to query, key, value projections
    """
    def __init__(self,
                 en_early_cand_elimn=False,
                 size_D=768,
                 num_heads=12,
                 attn_dropout_rate=0.,
                 proj_dropout_rate=0.,
                 qkv_bias=False):
        super().__init__()

        self.num_heads = num_heads
        self.size_d = size_D // num_heads        # D = d * h for MULTI HEAD
        self.attn_scale = self.size_d ** -0.5    # 1 / sqrt(size_d) for the scaling factor of qkv

        # Since the weights are aligned such that we get a combined QKV in the implementation
        self.qkv = nn.Linear(in_features=size_D,
                             out_features=size_D*3,
                             bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_dropout_rate)

        self.proj = nn.Linear(in_features=size_D,
                              out_features=size_D)

        self.proj_drop = nn.Dropout(proj_dropout_rate)

        self.attn_mat = None
        
        self.en_early_cand_elimn = en_early_cand_elimn
        self.scaled_qk_out = None

    def forward(self, input):
        # TODO : Figure out the motivation behind doing a linear norm after
        #        linear projection and splitting the QKV matrices, only done for Q and K and final output of sdpa

        # Note that for Self Attention we have the same inputs
        size_B, size_N, size_D = input.shape
        size_QKV = 3

        # Now first we have to create are Q, K, V (3) matrices, which is
        # created after passing the inputs through linear projector
        # Each of Q, K, V will have the shape B, N, (D) => B, N, (h, d)
        # B, N, size_QKV, D
        qkv_output = self.qkv(input)

        # Showing this just for clarity: qkv_output = qkv_output.reshape(size_B, size_N, size_QKV, size_D ---> num_heads * size_d)
        qkv_output = qkv_output.reshape(size_B, size_N, size_QKV, self.num_heads, self.size_d)

        # Reorders them on the basis of size_QKV, size_B, num_heads, size_N, size_d
        qkv_output = qkv_output.permute(2, 0, 3, 1, 4)

        # So that you can extract q, k, v from first index itself
        # Each of shape (B, N, d, h)
        q_output, k_output, v_output = qkv_output.unbind(0)

        # timm's implementation has this here, though OG ViT doesn't
        # q_output = self.after_qkv_split_ln(q_output)

        scaled_q_output = self.attn_scale * q_output

        # timm's implementation has this here, though OG ViT doesn't
        # k_T_output = self.after_qkv_split_ln(k_output)

        # Swap size_h with size_d dimension
        # So the output shape would be (B, N, h, d)
        k_T_output = k_output.transpose(-2, -1)

        # So shapes of the input matrix are: (B, N, d, h) @ (B, N, h, d)
        # And the output shape would be : (B, N, d, d)
        scaled_qk_output = ( scaled_q_output @ k_T_output )

        # self.attn_mat = scaled_qk_output

        # Apply softmax along the last dimension
        scaled_qk_output = scaled_qk_output.softmax(dim=-1)
        scaled_qk_output = self.attn_drop(scaled_qk_output)
        
        self.scaled_qk_out = scaled_qk_output

        # Finally get the Scaled Dot Product Attention output
        # Inputs : (B, N, d, d) @ (B, N, d, h)
        # Outputs : (B, N, d, h)
        sdpa_output = scaled_qk_output @ v_output

        # Transpose the output to make it (B, N, h, d)
        # Then reshape (opposite of before) to get back the correctly positioned values in (B, N, D)
        sdpa_output = sdpa_output.transpose(1, 2).reshape(size_B, size_N, size_D)

        # timm's implementation has this here, though OG ViT doesn't
        # sdpa_output = self.after_qkv_split_ln(sdpa_output)

        # Linear projection after concatenating the results
        sdpa_output = self.proj(sdpa_output)
        sdpa_output = self.proj_drop(sdpa_output)

        return sdpa_output
