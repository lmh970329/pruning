from typing import Any
from .base import ActivationPrune

import torch
import torch.nn as nn


class OutputFeaturemapPrune(ActivationPrune):

    def __init__(self, sparsity, score_type='max') -> None:
        super().__init__(sparsity)
        self.score_type = score_type

        if not score_type in ['max', 'l2', 'l1']:
            raise ValueError(f"'{score_type}' is not supported!!!")
        

    def _get_score(self, output: torch.Tensor):
        if self.score_type == 'max':
            score = torch.max(output.abs().flatten(start_dim=2), dim=-1).values
        elif self.score_type == 'l2':
            score = torch.norm(output.flatten(start_dim=2), p=2, dim=-1).values
        else:
            score = torch.norm(output.flatten(start_dim=2), p=1, dim=-1).values

        return score


    @torch.no_grad()
    def compute_mask(self, output: torch.Tensor):
        B, C, *featuremap_shape = output.shape

        score = self._get_score(output)
        mask = torch.ones_like(score)
        
        n_prune = int(C * self.sparsity)
        indices = torch.topk(score, n_prune, largest=False).indices
        strides = torch.arange(0, B, device=mask.device).reshape(-1, 1) * C
        
        mask.view(-1)[indices + strides] = 0
        
        new_dims = [1] * len(featuremap_shape)
        mask = mask.reshape(B, C, *new_dims)
        
        return mask