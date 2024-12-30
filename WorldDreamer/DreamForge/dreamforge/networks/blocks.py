from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    XFormersAttnProcessor,
    XFormersAttnAddedKVProcessor,
    CustomDiffusionXFormersAttnProcessor,
    LoRAXFormersAttnProcessor,
)
from diffusers.models.attention import (
    BasicTransformerBlock, AdaLayerNorm, AdaLayerNormZero
)
from diffusers.models.controlnet import zero_module


def is_xformers(module):
    return isinstance(module.processor, (
        XFormersAttnProcessor,
        XFormersAttnAddedKVProcessor,
        CustomDiffusionXFormersAttnProcessor,
        LoRAXFormersAttnProcessor,
    ))


def _ensure_kv_is_int(view_pair: dict):
    """yaml key can be int, while json cannot. We convert here.
    """
    new_dict = {}
    for k, v in view_pair.items():
        new_value = [int(vi) for vi in v]
        new_dict[int(k)] = new_value
    return new_dict


def get_zero_module(zero_module_type, dim):
    if zero_module_type == "zero_linear":
        # NOTE: zero_module cannot apply to successive layers.
        connector = zero_module(nn.Linear(dim, dim))
    elif zero_module_type == "gated":
        connector = GatedConnector(dim)
    elif zero_module_type == "none":
        # TODO: if this block is in controlnet, we may not need zero here.
        def connector(x): return x
    else:
        raise TypeError(f"Unknown zero module type: {zero_module_type}")
    return connector


class GatedConnector(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        data = torch.zeros(dim)
        self.alpha = nn.parameter.Parameter(data)

    def forward(self, inx):
        # as long as last dim of input == dim, pytorch can auto-broad
        return F.tanh(self.alpha) * inx


class BasicMultiviewTransformerBlock(BasicTransformerBlock):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        # multi_view
        neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
        neighboring_attn_type: Optional[str] = "add",
        zero_module_type="zero_linear",
        attn1_q_trainable=False,
    ):
        super().__init__(
            dim, num_attention_heads, attention_head_dim, dropout,
            cross_attention_dim, activation_fn, num_embeds_ada_norm,
            attention_bias, only_cross_attention, double_self_attention,
            upcast_attention, norm_elementwise_affine, norm_type, final_dropout)

        self._args = {
            k: v for k, v in locals().items()
            if k != "self" and not k.startswith("_")}
        self.neighboring_view_pair = _ensure_kv_is_int(neighboring_view_pair)
        self.neighboring_attn_type = neighboring_attn_type
        # multiview attention
        self.norm4 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        )
        self.attn4 = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.connector = get_zero_module(zero_module_type, dim)
        self.attn1_q_trainable = attn1_q_trainable

    @property
    def new_module(self):
        ret = {
            "norm4": self.norm4,
            "attn4": self.attn4,
        }
        if isinstance(self.connector, nn.Module):
            ret["connector"] = self.connector

        if self.attn1_q_trainable:
            ret['attn1.to_q'] = self.attn1.to_q

        return ret

    @property
    def n_cam(self):
        return len(self.neighboring_view_pair)

    def _construct_attn_input(self, norm_hidden_states):
        B = len(norm_hidden_states)
        # reshape, key for origin view, value for ref view
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        if self.neighboring_attn_type == "add":
            for key, values in self.neighboring_view_pair.items():
                for value in values:
                    hidden_states_in1.append(norm_hidden_states[:, key])
                    hidden_states_in2.append(norm_hidden_states[:, value])
                    cam_order += [key] * B
            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "concat":
            for key, values in self.neighboring_view_pair.items():
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(torch.cat([
                    norm_hidden_states[:, value] for value in values
                ], dim=1))
                cam_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 2*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "self":
            hidden_states_in1 = rearrange(
                norm_hidden_states, "b n l ... -> b (n l) ...")
            hidden_states_in2 = None
            cam_order = None
        else:
            raise NotImplementedError(
                f"Unknown type: {self.neighboring_attn_type}")
        return hidden_states_in1, hidden_states_in2, cam_order

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        embedding=None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if embedding is not None:
            norm_hidden_states = norm_hidden_states + embedding
        attn_output = self.attn1(
            norm_hidden_states, encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention else None,
            attention_mask=attention_mask, **cross_attention_kwargs,)
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm2(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # multi-view cross attention
        norm_hidden_states = (
            self.norm4(hidden_states, timestep) if self.use_ada_layer_norm else
            self.norm4(hidden_states)
        )
        # batch dim first, cam dim second
        norm_hidden_states = rearrange(
            norm_hidden_states, '(b n) ... -> b n ...', n=self.n_cam)
        B = len(norm_hidden_states)
        # key is query in attention; value is key-value in attention
        hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
            norm_hidden_states, )
        # attention
        bs1, dim1 = hidden_states_in1.shape[:2]
        grpn = 6  # TODO: hard-coded to use bs=6, avoiding numerical error.
        if bs1 > grpn and dim1 > 1400 and not is_xformers(self.attn4):
            hidden_states_in1s = torch.split(hidden_states_in1, grpn)
            hidden_states_in2s = torch.split(hidden_states_in2, grpn)
            grps = len(hidden_states_in1s)
            attn_raw_output = [None for _ in range(grps)]
            for i in range(grps):
                attn_raw_output[i] = self.attn4(
                    hidden_states_in1s[i],
                    encoder_hidden_states=hidden_states_in2s[i],
                    **cross_attention_kwargs,
                )
            attn_raw_output = torch.cat(attn_raw_output, dim=0)
        else:
            attn_raw_output = self.attn4(
                hidden_states_in1,
                encoder_hidden_states=hidden_states_in2,
                **cross_attention_kwargs,
            )
        # final output
        if self.neighboring_attn_type == "self":
            attn_output = rearrange(
                attn_raw_output, 'b (n l) ... -> b n l ...', n=self.n_cam)
        else:
            attn_output = torch.zeros_like(norm_hidden_states)
            for cam_i in range(self.n_cam):
                attn_out_mv = rearrange(attn_raw_output[cam_order == cam_i],
                                        '(n b) ... -> b n ...', b=B)
                attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
        attn_output = rearrange(attn_output, 'b n ... -> (b n) ...')
        # apply zero init connector (one layer)
        attn_output = self.connector(attn_output)
        # short-cut
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (
                1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class LearnablePosEmb(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))

    def forward(self, inx):
        return self.param + inx
    

class MotionAttention(nn.Module):
    def __init__(self, in_channels, attnetion_dim, out_channels) -> None:
        super().__init__()

        self.to_qkv = nn.Conv2d(in_channels, attnetion_dim*3, kernel_size=1)
        self.forward_block = nn.Sequential(
            nn.Conv2d(attnetion_dim, attnetion_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(attnetion_dim, attnetion_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.backward_block = nn.Sequential(
            nn.Conv2d(attnetion_dim, attnetion_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(attnetion_dim, attnetion_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.learnable_param = nn.Parameter(torch.ones(2)/2)

    def forward(self, hidden_states):
        B, T = hidden_states.shape[:2] # b, t, c, h, w
        hidden_states_in = self.to_qkv(hidden_states.flatten(0, 1)) # b*t, c, h, w
        hs_q, hs_k, hs_v = torch.chunk(hidden_states_in, 3, dim=1)
        hs_q = hs_q.reshape(B, T, hs_q.shape[1], hs_q.shape[2], hs_q.shape[3])
        hs_k = hs_k.reshape(B, T, hs_k.shape[1], hs_k.shape[2], hs_k.shape[3]) # b, t, c, h, w

        motion_forward = torch.cat([torch.zeros_like(hs_q[:, :1]), hs_q[:, 1:]-hs_k[:, :-1]], dim=1) # b, t, c, h, w
        attn_forward = self.forward_block(motion_forward.flatten(0, 1))

        motion_backward = torch.cat([hs_q[:, :-1]-hs_k[:, 1:], torch.zeros_like(hs_q[:, -1:]), ], dim=1) # b, t, c, h, w
        attn_backward = self.backward_block(motion_backward.flatten(0, 1))

        attn = self.learnable_param[0] * attn_forward + self.learnable_param[1] * attn_backward
        
        outputs = attn * hs_v

        return outputs


class LongShortTemporalMultiviewTransformerBlock(BasicMultiviewTransformerBlock):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            video_length,  # temporal
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
            # multi_view
            attn1_q_trainable=False,
            neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
            neighboring_attn_type: Optional[str] = "add",
            zero_module_type="zero_linear",
            # temporal
            pos_emb="learnable",
            zero_module_type2="zero_linear",
            spatial_trainable=False,
            # ref_bank
            with_ref=False,
            ref_length=2,
            # can_bus
            with_can_bus=False,
            # motioin
            with_motion=False,
            # attn type
            transformer_type='ff_last',
    ):
        super().__init__(
            dim, num_attention_heads, attention_head_dim,
            dropout, cross_attention_dim, activation_fn, num_embeds_ada_norm,
            attention_bias, only_cross_attention, double_self_attention,
            upcast_attention, norm_elementwise_affine, norm_type, final_dropout,
            neighboring_view_pair, neighboring_attn_type, zero_module_type)

        self._args = {
            k: v for k, v in locals().items()
            if k != "self" and not k.startswith("_")}
        self.spatial_trainable = spatial_trainable
        self.video_length = video_length
        self.ref_length = ref_length
        self.with_ref = with_ref
        self.with_can_bus = with_can_bus
        self.with_motion = with_motion

        # temporal attn
        if pos_emb == "learnable":
            temp_length = video_length+ref_length if with_ref else video_length
            self.pos_emb = LearnablePosEmb(size=(1, temp_length, dim))
        elif pos_emb == "none":
            self.pos_emb = None
        else:
            raise NotImplementedError(f"Unknown type {pos_emb}")

        if self.use_ada_layer_norm:
            self.temp_norm = AdaLayerNorm(dim, num_embeds_ada_norm)
        else:
            self.temp_norm = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine)
        self.temp_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.temp_connector = get_zero_module(zero_module_type2, dim)

        if with_ref:
            self.ref_linear = nn.Linear(dim, dim, bias=True)
        if with_can_bus:
            self.can_bus_linear = nn.Linear(cross_attention_dim, dim, bias=True)
        if with_motion:
            self.motion_attn = MotionAttention(dim, dim, dim)
            self.motion_connector = get_zero_module(zero_module_type2, dim)
        self.transformer_type = transformer_type
        self._sc_attn_index = None

    @property
    def new_module(self):
        if self.spatial_trainable:
            ret = {
                "norm4": self.norm4,
                "attn4": self.attn4,
            }
            if isinstance(self.connector, nn.Module):
                ret["connector"] = self.connector
        else:
            ret = {}
        ret = {
            "temp_attn": self.temp_attn,
            "temp_norm": self.temp_norm,
            **ret,
        }

        if self.with_ref:
            ret['ref_linear'] = self.ref_linear

        if self.with_can_bus:
            ret['can_bus_linear'] = self.can_bus_linear
        
        if self.with_motion:
            ret['motion_attn'] = self.motion_attn
            if isinstance(self.motion_connector, nn.Module):
                ret['motion_connector'] = self.motion_connector

        if isinstance(self.temp_connector, nn.Module):
            ret["temp_connector"] = self.temp_connector
        if isinstance(self.pos_emb, nn.Module):
            ret["temp_pos_emb"] = self.pos_emb
        
        if self.transformer_type.startswith("_"):
            ret['attn1.to_q'] = self.attn1.to_q
        
        return ret

    @property
    def sc_attn_index(self):
        # one can set `self._sc_attn_index` to a function for convenient changes
        # among batches.
        if callable(self._sc_attn_index):
            return self._sc_attn_index()
        else:
            return self._sc_attn_index

    def _construct_sc_attn_input(
        self, norm_hidden_states, sc_attn_index, type="add"
    ):
        # assume data has form (b, frame, c), frame == len(sc_attn_index)
        # return two sets of hidden_states and an order list.
        B = len(norm_hidden_states)
        hidden_states_in1 = []
        hidden_states_in2 = []
        back_order = []

        if type == "add":
            for key, values in zip(range(len(sc_attn_index)), sc_attn_index):
                for value in values:
                    hidden_states_in1.append(norm_hidden_states[:, key])
                    hidden_states_in2.append(norm_hidden_states[:, value])
                    back_order += [key] * B
            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            back_order = torch.LongTensor(back_order)
        elif type == "concat":
            for key, values in zip(range(len(sc_attn_index)), sc_attn_index):
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(torch.cat([
                    norm_hidden_states[:, value] for value in values
                ], dim=1))
                back_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 3*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            back_order = torch.LongTensor(back_order)
        else:
            raise NotImplementedError(f"Unknown type: {type}")
        return hidden_states_in1, hidden_states_in2, back_order
    
    def forward_old_attns(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        embedding=None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        shift_mlp, scale_mlp, gate_mlp = None, None, None
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if embedding is not None:
            norm_hidden_states = norm_hidden_states + embedding
        if self.transformer_type.startswith("_"):
            norm_hidden_states = rearrange(
                norm_hidden_states, "(b f n) d c -> (b n) f d c",
                f=self.video_length, n=self.n_cam)
            B = len(norm_hidden_states)

            # this index is for kv pair, your dataloader should make it consistent.
            norm_hidden_states_q, norm_hidden_states_kv, back_order = self._construct_sc_attn_input(
                norm_hidden_states, self.sc_attn_index, type="concat")

            attn_raw_output = self.attn1(
                norm_hidden_states_q,
                encoder_hidden_states=norm_hidden_states_kv,
                attention_mask=attention_mask, **cross_attention_kwargs)
            attn_output = torch.zeros_like(norm_hidden_states)
            for frame_i in range(self.video_length):
                # TODO: any problem here? n should == 1
                attn_out_mt = rearrange(attn_raw_output[back_order == frame_i],
                                        '(n b) ... -> b n ...', b=B)
                attn_output[:, frame_i] = torch.sum(attn_out_mt, dim=1)
            attn_output = rearrange(
                attn_output, "(b n) f d c -> (b f n) d c", n=self.n_cam)
        else:
            attn_output = self.attn1(
                norm_hidden_states, encoder_hidden_states=encoder_hidden_states
                if self.only_cross_attention else None,
                attention_mask=attention_mask, **cross_attention_kwargs,)
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm2(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # multi-view cross attention
        norm_hidden_states = (
            self.norm4(hidden_states, timestep) if self.use_ada_layer_norm else
            self.norm4(hidden_states)
        )
        # batch dim first, cam dim second
        norm_hidden_states = rearrange(
            norm_hidden_states, '(b n) ... -> b n ...', n=self.n_cam)
        B = len(norm_hidden_states)
        # key is query in attention; value is key-value in attention
        hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
            norm_hidden_states, )
        # attention
        bs1, dim1 = hidden_states_in1.shape[:2]
        grpn = 6  # TODO: hard-coded to use bs=6, avoiding numerical error.
        if bs1 > grpn and dim1 > 1400 and not is_xformers(self.attn4):
            hidden_states_in1s = torch.split(hidden_states_in1, grpn)
            hidden_states_in2s = torch.split(hidden_states_in2, grpn)
            grps = len(hidden_states_in1s)
            attn_raw_output = [None for _ in range(grps)]
            for i in range(grps):
                attn_raw_output[i] = self.attn4(
                    hidden_states_in1s[i],
                    encoder_hidden_states=hidden_states_in2s[i],
                    **cross_attention_kwargs,
                )
            attn_raw_output = torch.cat(attn_raw_output, dim=0)
        else:
            attn_raw_output = self.attn4(
                hidden_states_in1,
                encoder_hidden_states=hidden_states_in2,
                **cross_attention_kwargs,
            )
        # final output
        if self.neighboring_attn_type == "self":
            attn_output = rearrange(
                attn_raw_output, 'b (n l) ... -> b n l ...', n=self.n_cam)
        else:
            attn_output = torch.zeros_like(norm_hidden_states)
            for cam_i in range(self.n_cam):
                attn_out_mv = rearrange(attn_raw_output[cam_order == cam_i],
                                        '(n b) ... -> b n ...', b=B)
                attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
        attn_output = rearrange(attn_output, 'b n ... -> (b n) ...')
        # apply zero init connector (one layer)
        attn_output = self.connector(attn_output)
        # short-cut
        hidden_states = attn_output + hidden_states

        return hidden_states, shift_mlp, scale_mlp, gate_mlp
    
    def add_temp_pos_emb(self, hidden_state):
        if self.pos_emb is None:
            return hidden_state
        return self.pos_emb(hidden_state)

    def forward_temporal(self, hidden_states, timestep):       
        if self.with_ref:
            bank_fea = [self.ref_linear(rearrange(self.ref_hidden_states.clone(), "(b f n) d c -> (b d n) f c",
                f=self.ref_length, n=self.n_cam))]
        else:
            bank_fea = []

        # Temporal-Attention
        d = hidden_states.shape[1]
        hidden_states_in = rearrange(
            hidden_states, "(b f n) d c -> (b d n) f c", f=self.video_length,
            n=self.n_cam)
        hidden_states_in = torch.cat(bank_fea + [hidden_states_in], dim=1)
        hidden_states_in = self.add_temp_pos_emb(hidden_states_in)
        if self.with_can_bus:
            can_bus_embedding = self.can_bus_embedding.repeat(1, d, 1)
            can_bus_embedding = rearrange(
                can_bus_embedding, "(b f n) d c -> (b d n) f c",
                f=self.ref_length+self.video_length, n=self.n_cam
            )
            hidden_states_in = hidden_states_in + can_bus_embedding
        norm_hidden_states = (
            self.temp_norm(hidden_states_in, timestep)
            if self.use_ada_layer_norm else self.temp_norm(hidden_states_in))
        norm_hidden_states_ = norm_hidden_states
        # NOTE: xformers cannot take bs larger than 8192
        if len(norm_hidden_states) >= 8192:
            chunk_num = math.ceil(len(norm_hidden_states) / 4096.)
            norm_hidden_states = norm_hidden_states.chunk(chunk_num)
            attn_output = torch.cat([
                self.temp_attn(norm_hidden_states[i]) for i in range(chunk_num)
            ], dim=0)
        else:
            attn_output = self.temp_attn(norm_hidden_states)
        attn_output = attn_output[:, self.ref_length:]
        # apply zero init connector (one layer)
        attn_output = self.temp_connector(attn_output)
        attn_output = rearrange(
            attn_output, "(b d n) f c -> (b f n) d c", d=d, n=self.n_cam)
        
        # short-term motion attn
        if self.with_motion:
            h, w = self.size
            norm_hidden_states = rearrange(
                norm_hidden_states_[:, self.ref_length:], "(b d n) f c -> (b n) f c d", d=d, n=self.n_cam)
            norm_hidden_states = rearrange(
                norm_hidden_states, "b f c (h w) -> b f c h w", h=h, w=w
            )
            motion_output = self.motion_attn(norm_hidden_states)
            motion_output = self.motion_connector(rearrange(
                motion_output, "(b n f) c h w -> (b f n) (h w) c", f=self.video_length, n=self.n_cam, h=h, w=w))

            # short-cut
            hidden_states = attn_output + hidden_states + motion_output
        else:
            # short-cut
            hidden_states = attn_output + hidden_states

        return hidden_states

    def forward_ff_last(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        embedding=None,
    ):
        hidden_states, shift_mlp, scale_mlp, gate_mlp = self.forward_old_attns(
            hidden_states, attention_mask, encoder_hidden_states,
            encoder_attention_mask, timestep, cross_attention_kwargs,
            class_labels, embedding=embedding
        )

        hidden_states = self.forward_temporal(hidden_states, timestep)

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (
                1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states
        return hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        size=None,
        scene_embedding=None,
        can_bus_embedding=None,
    ):   
        self.size = size
        if self.with_can_bus and can_bus_embedding is not None:
            self.can_bus_embedding = self.can_bus_linear(can_bus_embedding)
        if self.with_ref:
            hidden_states = rearrange(
                    hidden_states, "(b f n) d c -> (b n) f d c",
                    f=self.ref_length+self.video_length, n=self.n_cam
                )
            self.ref_hidden_states = rearrange(
                    hidden_states[:, :self.ref_length].clone(), "(b n) f d c -> (b f n) d c",
                    f=self.ref_length, n=self.n_cam
                )
            hidden_states = rearrange(
                    hidden_states[:, self.ref_length:].clone(), "(b n) f d c -> (b f n) d c",
                    f=self.video_length, n=self.n_cam
                )

        hidden_states = self.forward_ff_last(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep,
            cross_attention_kwargs,
            class_labels,
            embedding=scene_embedding
        )

        if self.with_ref:
            ref_hidden_states = rearrange(
                    self.ref_hidden_states, "(b f n) d c -> (b n) f d c",
                    f=self.ref_length, n=self.n_cam
                )
            hidden_states = rearrange(
                    hidden_states, "(b f n) d c -> (b n) f d c",
                    f=self.video_length, n=self.n_cam
                )
            hidden_states = torch.cat([ref_hidden_states, hidden_states], dim=1)
            hidden_states = rearrange(
                    hidden_states, "(b n) f d c -> (b f n) d c",
                    f=self.ref_length+self.video_length, n=self.n_cam
                )

        return hidden_states

