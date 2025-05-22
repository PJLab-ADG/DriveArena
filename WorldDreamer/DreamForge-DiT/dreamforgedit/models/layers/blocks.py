# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# PixArt: https://github.com/PixArt-alpha/PixArt-alpha
# Latte:  https://github.com/Vchitect/Latte
# DiT:    https://github.com/facebookresearch/DiT/tree/main
# GLIDE:  https://github.com/openai/glide-text2im
# MAE:    https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import os
import functools
import math
from typing import Optional
import logging

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")
USE_XFORMERS = eval(os.environ.get("USE_XFORMERS", "True"))

import numpy as np
import torch
if not torch.cuda.is_available() or DEVICE_TYPE == "npu":
    import torch_npu
    USE_NPU = True
else:
    from flash_attn import flash_attn_func
    USE_NPU = False
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
if USE_XFORMERS:
    try:
        import xformers.ops
    except BaseException as e:
        logging.warning(f"Import xformers got {e}, will disable xformers")
        USE_XFORMERS = False
else:
    logging.info(f"USE_XFORMERS={USE_XFORMERS}")
from einops import rearrange
from timm.models.vision_transformer import Mlp

from dreamforgedit.acceleration.communications import all_to_all, split_forward_gather_backward
from dreamforgedit.acceleration.parallel_states import get_sequence_parallel_group
from dreamforgedit.utils.misc import warn_once


verbose = False

def approx_gelu(): return nn.GELU(approximate="tanh")


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


def modulate(norm_func, x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    dtype = x.dtype
    x = norm_func(x.to(torch.float32)).to(dtype)
    x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
    x = x.to(dtype)
    return x


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


# ===============================================
# General-purpose Layers
# ===============================================


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        if np.prod(x.shape[-3:]) > np.prod([33, 112, 200]) and USE_NPU:
            # NOTE: conv3d on NPU cannot take too large batch sizes.
            x = torch.cat([self.proj(_x) for _x in x.chunk(2, dim=0)], dim=0)
        else:
            x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        enable_xformers: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
        is_cross_attention=False,  # useless
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn
        self.enable_xformers = enable_xformers and USE_XFORMERS

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope
        
        self.is_causal = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # NOTE: rotary_emb need -2 as seq_dim
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if enable_flash_attn or self.enable_xformers:
            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

        if enable_flash_attn:
            if verbose:
                logging.debug(f"[{self.__class__}] use flash_attn")
            if USE_NPU:
                x = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    self.num_heads,
                    "BSND",
                    keep_prob=(1.0 - self.attn_drop.p) if self.training else 1.0,
                    scale=self.scale
                )[0]
            else:
                x = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=self.is_causal,
                )
        elif self.enable_xformers:
            if verbose:
                logging.debug(f"[{self.__class__}] use xformers")
            attn_bias = None
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        else:
            if verbose:
                logging.debug(f"[{self.__class__}] use torch")
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if self.is_causal:
                causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
                causal_mask = torch.where(causal_mask.bool(), 0, float('-inf'))
                attn += causal_mask
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).contiguous()

        x_output_shape = (B, N, C)
        if eval(os.environ.get('USE_WRONG', "False")) == True:
            if self.enable_xformers:
                warn_once(f"use_wrong in {self.__class__}")
                x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqParallelAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
    ) -> None:
        assert rope is None, "Rope is not supported in SeqParallelAttention"
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            enable_flash_attn=enable_flash_attn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # for sequence parallel here, the N is a local sequence length
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape)

        sp_group = get_sequence_parallel_group()
        sp_size = dist.get_world_size(sp_group)

        # apply all_to_all to gather sequence and split attention heads
        # [B, SUB_N, 3, NUM_HEAD, HEAD_DIM] -> [B, N, 3, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        qkv = all_to_all(qkv, sp_group, scatter_dim=3, gather_dim=1)

        if self.enable_flash_attn:
            qkv_permute_shape = (
                2,
                0,
                1,
                3,
                4,
            )  # [3, B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        else:
            qkv_permute_shape = (
                2,
                0,
                3,
                1,
                4,
            )  # [3, B, NUM_HEAD_PER_DEVICE, N, HEAD_DIM]
        qkv = qkv.permute(qkv_permute_shape)

        # ERROR: Should qk_norm first
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.enable_flash_attn:
            if USE_NPU:
                x = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    self.num_heads // sp_size,
                    "BSND",
                    keep_prob=(1.0 - self.attn_drop.p) if self.training else 1.0,
                    scale=self.scale
                )[0]
            else:
                x = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    softmax_scale=self.scale,
                )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        if not self.enable_flash_attn:
            x = x.transpose(1, 2)

        # apply all to all to gather back attention heads and split sequence
        # [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]  -> [B, SUB_N, NUM_HEAD, HEAD_DIM]
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # reshape outputs back to [B, N, C]
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        enable_xformers: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
        is_cross_attention=False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn
        self.enable_xformers = enable_xformers and USE_XFORMERS

    
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_cross_attention = is_cross_attention

        self.rope = False
        if rope is not None and not is_cross_attention:
            self.rope = True
            self.rotary_emb = rope
        assert self.rope == False, "rope is not correct here!"

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)

        if self.is_cross_attention:
            assert cond is not None
        else:
            assert cond is None
            cond = x
        Bc, Nc, Cc = cond.shape
        assert B == Bc and C == Cc

        # (B, N, #head, #dim)
        bias = None if self.qkv.bias is None else self.qkv.bias[:self.dim]
        q = F.linear(x, self.qkv.weight[:self.dim, :], bias)
        q = q.view(B, N, self.num_heads, self.head_dim)

        bias = None if self.qkv.bias is None else self.qkv.bias[self.dim:]
        kv = F.linear(cond, self.qkv.weight[self.dim:, :], bias)
        kv = kv.view(B, Nc, 2, self.num_heads, self.head_dim)

        k, v = kv.unbind(2)

        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if not (enable_flash_attn or self.enable_xformers):
            # (B, N, #heads, #dim) -> (B, #heads, N, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

        if enable_flash_attn:
            if verbose:
                logging.debug(f"[{self.__class__}] use flash_attn")
            if USE_NPU:
                x = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    self.num_heads,
                    "BSND",
                    keep_prob=(1.0 - self.attn_drop.p) if self.training else 1.0,
                    scale=self.scale
                )[0]
            else:
                x = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    softmax_scale=self.scale,
                )
        elif self.enable_xformers and USE_XFORMERS:
            if verbose:
                logging.debug(f"[{self.__class__}] use xformers")
            attn_bias = None
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        else:
            if verbose:
                logging.debug(f"[{self.__class__}] use torch")
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).contiguous()

        x_output_shape = (B, N, C)
        if eval(os.environ.get('USE_WRONG', "False")) == True:
            if self.enable_xformers and USE_XFORMERS:
                warn_once(f"use_wrong in {self.__class__}")
                x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqParallelMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        enable_xformers: bool = False,
        rope=None,
        is_cross_attention=False,
    ) -> None:
        assert rope is None, "Rope is not supported in SeqParallelAttention"
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            enable_flash_attn=enable_flash_attn,
            enable_xformers=enable_xformers,
            is_cross_attention=is_cross_attention,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape  # for sequence parallel here, the N is a local sequence length

        if self.is_cross_attention:
            assert cond is not None
        else:
            assert cond is None
            cond = x
        Bc, Nc, Cc = cond.shape  # for sequence parallel here, the N1 is a local sequence length
        assert B == Bc and C == Cc

        # (B, N, #head, #dim)
        bias = None if self.qkv.bias is None else self.qkv.bias[:self.dim]
        q = F.linear(x, self.qkv.weight[:self.dim, :], bias)
        q = q.view(B, N, self.num_heads, self.head_dim)

        bias = None if self.qkv.bias is None else self.qkv.bias[self.dim:]
        kv = F.linear(cond, self.qkv.weight[self.dim:, :], bias)
        kv = kv.view(B, Nc, 2, self.num_heads, self.head_dim)

        sp_group = get_sequence_parallel_group()
        sp_size = dist.get_world_size(sp_group)

        # apply all_to_all to gather sequence and split attention heads
        # [B, SUB_N, NUM_HEAD, HEAD_DIM] -> [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        q = all_to_all(q, sp_group, scatter_dim=2, gather_dim=1)
        # [B, SUB_N, 3, NUM_HEAD, HEAD_DIM] -> [B, N, 3, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        kv = all_to_all(kv, sp_group, scatter_dim=3, gather_dim=1)

        if self.enable_flash_attn or self.enable_xformers:
            qkv_permute_shape = (
                0,
                1,
                2,
                3,
            )  # [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        else:
            qkv_permute_shape = (
                0,
                2,
                1,
                3,
            )  # [B, NUM_HEAD_PER_DEVICE, N, HEAD_DIM]
        k, v = kv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.permute(qkv_permute_shape)
        k = k.permute(qkv_permute_shape)
        v = v.permute(qkv_permute_shape)
        if self.enable_flash_attn:
            if USE_NPU:
                x = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    self.num_heads // sp_size,  # NOTE: otherwise will report error!
                    "BSND",
                    keep_prob=(1.0 - self.attn_drop.p) if self.training else 1.0,
                    scale=self.scale
                )[0]
            else:
                x = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    softmax_scale=self.scale,
                )
        elif self.enable_xformers:
            attn_bias = None
            x = xformers.ops.memory_efficient_attention(
                q, k, v, p=self.attn_drop.p, attn_bias=attn_bias,
                # NOTE: head_num is not real, we should set scale!
                scale=self.scale)
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).contiguous()

        if eval(os.environ.get('USE_WRONG', "False")) == True:
            if self.enable_xformers:
                warn_once(f"use_wrong in {self.__class__}")
                x = x.transpose(1, 2)

        # apply all to all to gather back attention heads and split sequence
        # [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]  -> [B, SUB_N, NUM_HEAD, HEAD_DIM]
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # reshape outputs back to [B, N, C]
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def head_to_batch_dim(self, tensor, out_dim=3, num_heads=None):
        # NOTE: in seq_parallel case, num_heads may not equal to self.num_heads
        batch_size, seq_len, dim = tensor.shape
        num_heads = self.num_heads if num_heads is None else num_heads
        # reshape then transpose, B S (N D) -> B N S D
        tensor = torch_npu.npu_confusion_transpose(
            tensor, [0, 2, 1, 3],
            (batch_size, seq_len, num_heads, dim // num_heads), False)

        if out_dim == 3:
           tensor = tensor.reshape(batch_size * num_heads, seq_len, dim // num_heads)

        return tensor

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape
        Bc, Nc, Cc = cond.shape  # [B, TS/p, C]
        assert Bc == B

        q = self.q_linear(x)
        kv = self.kv_linear(cond)

        # if N > B:  # xformers
        if USE_XFORMERS:  # xformers
            if USE_NPU:
                assert mask is None
                q = q.view(B, N, self.num_heads * self.head_dim)
                kv = kv.view(Bc, Nc, 2, self.num_heads * self.head_dim)
                k, v = kv.unbind(2)

                q = self.head_to_batch_dim(q, out_dim=4)
                k = self.head_to_batch_dim(k, out_dim=4)
                v = self.head_to_batch_dim(v, out_dim=4)
                x = torch_npu.npu_fusion_attention(
                    q, k, v, self.num_heads, input_layout="BNSD",
                    pse=None,
                    atten_mask=mask,
                    scale=self.scale,
                    pre_tockens=2147483647,
                    next_tockens=2147483647,
                    keep_prob=(1 - self.attn_drop.p) if self.training else 1.0,
                    sync=False
                )[0]
                x = x.transpose(1, 2).contiguous()
            else:
                if mask is None:  # for cond
                    mask = [Nc] * B

                q = q.view(1, -1, self.num_heads, self.head_dim)
                kv = kv.view(1, -1, 2, self.num_heads, self.head_dim)
                k, v = kv.unbind(2)

                if verbose:
                    logging.debug(f"[{self.__class__}] use xformers")
                attn_bias = None
                if mask is not None:
                    attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens([N] * B, mask)

                x = xformers.ops.memory_efficient_attention(
                    q, k, v, p=self.attn_drop.p if self.training else 0.0,
                    attn_bias=attn_bias, scale=self.scale)
        else:
            q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = kv.view(B, Nc, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(2)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            dtype = q.dtype

            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).contiguous()

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqParallelMultiHeadCrossAttention(MultiHeadCrossAttention):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        sp_group = get_sequence_parallel_group()
        sp_size = dist.get_world_size(sp_group)
        B, SUB_N, C = x.shape  # [B, TS/p, C]
        Bc, Nc, Cc = cond.shape  # [B, TS, C]
        N = SUB_N * sp_size

        # shape:
        # q: [B, SUB_N, NUM_HEADS, HEAD_DIM] -> [B, N, NUM_HEADS/p, HEAD_DIM]
        # kv: [B, Nc, NUM_HEADS, HEAD_DIM] -> [B, Nc, NUM_HEADS/p, HEAD_DIM]
        q = self.q_linear(x).view(B, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        kv = split_forward_gather_backward(kv, get_sequence_parallel_group(), dim=3, grad_scale="down")

        # apply all_to_all to gather sequence and split attention heads
        q = all_to_all(q, sp_group, scatter_dim=2, gather_dim=1)

        # compute attention
        if USE_XFORMERS:
            if USE_NPU:
                assert mask is None
                q = q.view(B, N, self.num_heads // sp_size * self.head_dim)
                kv = kv.view(Bc, Nc, 2, self.num_heads // sp_size * self.head_dim)
                k, v = kv.unbind(2)

                q = self.head_to_batch_dim(q, out_dim=4, num_heads=self.num_heads // sp_size)
                k = self.head_to_batch_dim(k, out_dim=4, num_heads=self.num_heads // sp_size)
                v = self.head_to_batch_dim(v, out_dim=4, num_heads=self.num_heads // sp_size)
                x = torch_npu.npu_fusion_attention(
                    q, k, v, self.num_heads // sp_size, input_layout="BNSD",
                    pse=None,
                    atten_mask=mask,
                    scale=self.scale,
                    pre_tockens=2147483647,
                    next_tockens=2147483647,
                    keep_prob=(1 - self.attn_drop.p) if self.training else 1.0,
                    sync=False
                )[0]
                x = x.transpose(1, 2).contiguous()
            else:
                q = q.view(1, -1, self.num_heads // sp_size, self.head_dim)
                kv = kv.view(1, -1, 2, self.num_heads // sp_size, self.head_dim)
                k, v = kv.unbind(2)

                # for cond
                if mask is None:
                    assert Bc == B
                    mask = [Nc] * B
                if mask is not None:
                    attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens([N] * B, mask)
                else:
                    attn_bias = None
                x = xformers.ops.memory_efficient_attention(
                    q, k, v, p=self.attn_drop.p, attn_bias=attn_bias,
                    # NOTE: head_num is not real, we should set scale!
                    scale=self.scale)
        else:
            q = q.view(B, N, self.num_heads // sp_size, self.head_dim).permute(0, 2, 1, 3)
            kv = kv.view(B, Nc, 2, self.num_heads // sp_size, self.head_dim)
            k, v = kv.unbind(2)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            dtype = q.dtype

            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).contiguous()
           
        # apply all to all to gather back attention heads and scatter sequence
        x = x.view(B, -1, self.num_heads // sp_size, self.head_dim)
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # apply output projection
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, num_patch, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final, x, shift, scale)
        x = self.linear(x)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels, d_t=None, d_s=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self, x, t, x_mask=None, t0=None, T=None, S=None):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        if x_mask is not None:
            shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).chunk(2, dim=1)
            x_zero = t2i_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)
        x = self.linear(x)
        return x


# ===============================================
# Embedding Layers for Timesteps and Class Labels
# ===============================================


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]).cuda() < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs // s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        self.register_buffer(
            "y_embedding",
            torch.randn(token_num, in_channels) / in_channels**0.5,
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class PositionEmbedding2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4"
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_sin_cos_emb(self, t: torch.Tensor):
        out = torch.einsum("i,d->id", t, self.inv_freq)
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        return torch.cat((emb_sin, emb_cos), dim=-1)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
        self,
        device: torch.device,
        dtype: torch.dtype,
        h: int,
        w: int,
        scale: float = 1.0,
        base_size: Optional[int] = None,
    ):
        grid_h = torch.arange(h, device=device) / scale
        grid_w = torch.arange(w, device=device) / scale
        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w
        grid_h, grid_w = torch.meshgrid(
            grid_w,
            grid_h,
            indexing="ij",
        )  # here w goes first
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return torch.concat([emb_h, emb_w], dim=-1).unsqueeze(0).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        scale: Optional[float] = 1.0,
        base_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self._get_cached_emb(x.device, x.dtype, h, w, scale, base_size)


# ===============================================
# Sine/Cosine Positional Embedding Functions
# ===============================================
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
