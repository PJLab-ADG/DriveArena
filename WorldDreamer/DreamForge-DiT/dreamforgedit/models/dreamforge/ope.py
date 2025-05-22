import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.utils.transformer import inverse_sigmoid
import mmcv.ops
from mmdet_plugin.ops.points_in_boxes import points_in_boxes_gpu
from diffusers.models.controlnet import zero_module


class ObjectPositionEmbedding(nn.Module):
    """
    Use 3d position embedding
    """

    def __init__(
        self,
        depth_num=64,
        depth_start=1,
        position_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        embed_dims=320,
        LID=False,
        ori_shape=[1600, 900],
        gen_shape=[400, 224],
        block_out_channels=(256, 256),
        scale=4,
        output_size=None
    ):
        super().__init__()

        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.embed_dims = embed_dims
        self.LID = LID
        self.position_range = position_range
        self.depth_start = depth_start
        self.ori_shape = ori_shape
        self.gen_shape = gen_shape
        self.scale = scale
        assert self.scale * len(block_out_channels) == 8

        self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, depth_num*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(depth_num*4, depth_num*2, kernel_size=1, stride=1, padding=0),
            )

        # downsampling to be consistent with the shape of latents extracted by vae
        self.conv_in = nn.Conv2d(depth_num*2, block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.output_size = output_size        
        if output_size is not None:
            channel_out = block_out_channels[-1]
            self.up_conv = nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1)
            self.connector = zero_module(nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], embed_dims, kernel_size=1)
        )

    def load_param(self, weight):
        s = self.state_dict()
        for key, val in weight.items():

            # process ckpt from parallel module
            if key[:6] == 'module':
                key = key[7:]

            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mismached shape in key {}'.format(key))

        self.load_state_dict(s)

    def forward(self, x, img_metas):
        # 28,  50
        eps = 1e-5
        ori_w, ori_h = self.ori_shape
        W, H = int(self.gen_shape[0]/self.scale), int(self.gen_shape[1]/self.scale)
        B, T, N = x.shape[:3]
        if T != 1:
            assert B == 1 # for temporal version, the batchsize is awalys set to 1
            B = T
        device = x.device

        coords_h = torch.arange(H, device=device).float() * ori_h / H
        coords_w = torch.arange(W, device=device).float() * ori_w / W

        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        img2lidars = []
        for lidar2image in img_metas['lidar2image'] if T == 1 else img_metas['lidar2image'][0]:
            img2lidar = []
            for i in range(len(lidar2image.data)):
                img2lidar.append(torch.inverse(lidar2image.data[i]))
            img2lidars.append(torch.stack(img2lidar))
        img2lidars = torch.stack(img2lidars).to(device) # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars.double(), coords.double()).squeeze(-1)[..., :3].float() # TODO: why output float16 by default

        norm_coords3d = coords3d.clone()
        norm_coords3d[..., 0:1] = (norm_coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        norm_coords3d[..., 1:2] = (norm_coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        norm_coords3d[..., 2:3] = (norm_coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])

        coords_mask = (norm_coords3d < 1.0).all(-1) & (norm_coords3d > 0.0).all(-1)  # B, N, W, H, D 
        
        # get box mask
        object_masks = []
        for bi, (gt_bboxes_3d, points) in enumerate(zip(img_metas['gt_bboxes_3d'] if T == 1 else img_metas['gt_bboxes_3d'][0], coords3d)):
            valid_mask = coords_mask[bi]
            boxes = gt_bboxes_3d.data.tensor[:, :7].to(device).float()
            points = points[valid_mask].float()
            chunk_num = 6
            points = points.chunk(chunk_num)
            mask = torch.cat([points_in_boxes_gpu(points[i].unsqueeze(0), boxes.unsqueeze(0)).squeeze(0) != -1 for i in range(chunk_num)])

            object_mask = valid_mask.float() 
            object_mask[valid_mask] = mask.float()
            object_masks.append(object_mask)
        object_masks = torch.stack(object_masks).permute(0, 1, 4, 3, 2).contiguous().view(B*N, D, H, W)

        norm_coords3d = norm_coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
        norm_coords3d = inverse_sigmoid(norm_coords3d).view(B*N, D, 3, H, W)

        with torch.cuda.amp.autocast(dtype=self.conv_in.weight.dtype):
            embedding = self.position_encoder((norm_coords3d * object_masks[:, :, None]).flatten(1, 2))
        embedding = self.conv_in(embedding)
        embedding = F.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        if self.output_size is not None:
            embedding = F.interpolate(embedding, self.output_size, mode='bilinear', align_corners=False)
            embedding = embedding + self.connector(F.silu(self.up_conv(embedding)))

        embedding = self.conv_out(embedding)

        object_masks = object_masks.view(B, N, D, H, W).bool().any(dim=2).float()
        object_masks = F.interpolate(object_masks, embedding.shape[-2:])
        
        return embedding.view(B, N, embedding.shape[1], embedding.shape[2], embedding.shape[3]), object_masks