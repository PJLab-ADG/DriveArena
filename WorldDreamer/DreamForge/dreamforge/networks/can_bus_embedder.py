import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class CanbusEmbedding(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        embed_dims: int = 768,
        can_bus_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.can_bus_norm = can_bus_norm

        self.can_bus_mlp = nn.Sequential(
            nn.Linear(input_channels, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def forward(self, can_bus: torch.Tensor, **kwargs):
        emb = self.can_bus_mlp(can_bus)
        return emb
