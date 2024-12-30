# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional

import torch
from einops import rearrange

from .transformer_2d import MultiviewTransformer2DModel, TemporalMultiviewTransformer2DModel

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

class ReferenceTransformerControl:
    def __init__(
        self,
        unet,
        fusion_blocks="midup",
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert fusion_blocks in ["midup", "full"]
        self.fusion_blocks = fusion_blocks


    def update(self, bank_fea, dtype=torch.float16):
        if self.fusion_blocks == "midup":
            modules = [
                module
                for module in (
                    torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                )
                if isinstance(module, MultiviewTransformer2DModel) or isinstance(module, TemporalMultiviewTransformer2DModel)
            ]
        elif self.fusion_blocks == "full":
            modules = [
                module
                for module in torch_dfs(self.unet)
                if isinstance(module, MultiviewTransformer2DModel) or isinstance(module, TemporalMultiviewTransformer2DModel)
            ]
        for r in modules:
            r.bank = [v.clone().to(dtype) for v in bank_fea]

    def clear(self):
        if self.fusion_blocks == "midup":
            modules = [
                module
                for module in (
                    torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                )
                if isinstance(module, MultiviewTransformer2DModel) or isinstance(module, TemporalMultiviewTransformer2DModel)
            ]
        elif self.fusion_blocks == "full":
            modules = [
                module
                for module in torch_dfs(self.unet)
                if isinstance(module, MultiviewTransformer2DModel) or isinstance(module, TemporalMultiviewTransformer2DModel)
            ]
        for r in modules:
            r.bank.clear()