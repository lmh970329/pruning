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
    

class AttentionHeadPrune(ActivationPrune):

    def __init__(self, sparsity, num_heads, score_type='max') -> None:
        super().__init__(sparsity)
        self.num_heads = num_heads
        self.score_type = score_type

        if not score_type in ['max', 'l2', 'l1']:
            raise ValueError(f"'{score_type}' is not supported!!!")

    def _get_score(self, output: torch.Tensor):
        # output shape : B, H, N, C
        if self.score_type == 'max':
            score = torch.max(torch.flatten(output.abs(), -2), dim=-1).values
        elif self.score_type == 'l2':
            score = torch.norm(torch.flatten(output, -2), p=2, dim=-1).values
        elif self.score_type == 'l1':
            score = torch.norm(torch.flatten(output, -2), p=1, dim=-1).values

        return score
    
    @torch.no_grad()
    def compute_mask(self, output: torch.Tensor):
        B, H, N, C = output.shape

        score = self._get_score(output) # shape : [B, H]
        mask = torch.ones_like(score) # shape : [B, H]

        n_prune = H * self.sparsity
        indices = torch.topk(score, n_prune, largest=False).indices
        strides = torch.arange(0, B, device=mask.device).reshape(-1, 1) * H

        mask.view(-1)[indices + strides] = 0
        mask = mask.reshape(B, H, 1)

        return mask


class OutputEmbeddingPrune(OutputFeaturemapPrune):

    def __init__(self, sparsity, score_type='max') -> None:
        super().__init__(sparsity, score_type)


    @torch.no_grad()
    def compute_mask(self, output: torch.Tensor):
        output = output.transpose(-2, -1)
        mask =  super().compute_mask(output)
        return mask.transpose(-2, -1)
    

class OutputTokenPrune(OutputFeaturemapPrune):

    def __init__(self, sparsity, cls_token_idx=0, score_type='max') -> None:
        super().__init__(sparsity, score_type)
        self.cls_token_idx = cls_token_idx


    @torch.no_grad()
    def compute_mask(self, output: torch.Tensor):
        n_samples = output.size(0)
        n_tokens = output.size(1)
        token_indices = [idx for idx in range(n_tokens) if not idx == self.cls_token_idx]
        exclude_cls_token = output[:, token_indices, :]
        cls_token_mask = torch.ones(n_samples, 1, 1, device=output.device)
        mask = super().compute_mask(exclude_cls_token)
        
        if self.cls_token_idx == 0:
            mask = torch.concat([cls_token_mask, mask], dim=-2)
        elif self.cls_token_idx == n_tokens - 1:
            mask = torch.concat([mask, cls_token_mask], dim=-2)
        elif self.cls_token_idx < n_tokens - 1:
            mask = torch.concat([mask[:, :self.cls_token_idx, :], cls_token_mask, mask[:, self.cls_token_idx:, :]], dim=-2)

        return mask
    


class RandomFeaturemapPrune(ActivationPrune):

    def __init__(self, sparsity, d) -> None:
        super().__init__(sparsity)
        self.d = d


    def compute_mask(self, output):
        return NotImplementedError("Not implemented!!!")


    def __call__(self, module: nn.Module, input, output: torch.Tensor):
        if self.d == 1:
            return nn.functional.dropout1d(output, self.sparsity, module.training)
        elif self.d == 2:
            return nn.functional.dropout2d(output, self.sparsity, module.training)
        


class RandomEmbeddingPrune(ActivationPrune):

    def __init__(self, sparsity) -> None:
        super().__init__(sparsity)


    def compute_mask(self, output):
        return NotImplementedError("Not implemented!!!")


    def __call__(self, module: nn.Module, input, output: torch.Tensor):
        out = nn.functional.dropout1d(output.transpose(-2, -1), self.sparsity, module.training)
        return out.transpose(-2, -1)