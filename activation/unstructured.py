from typing import Any
from .base import ActivationPrune

import torch
import torch.nn as nn


class OutputActivationPrune(ActivationPrune):
    
    def __init__(self, sparsity) -> None:
        super().__init__(sparsity)


    @torch.no_grad()
    def compute_mask(self, output: torch.Tensor):
        
        B, C, *shape = output.shape

        mask = torch.ones_like(output)

        n_activation = output.nelement() // B
        n_prune = round(n_activation * self.sparsity)

        indices = torch.topk(output.abs().flatten(1), n_prune, dim=-1, largest=False).indices
        strides = torch.arange(0, B, device=mask.device).reshape(-1, 1) * n_activation

        prune_indices = (indices + strides).view(-1)

        mask.view(-1)[prune_indices] = 0

        return mask