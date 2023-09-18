from typing import Any

import torch

class OutputFeaturemapPrune():

    def __init__(self, sparsity, score_type='max') -> None:
        self.sparsity = sparsity
        self.score_type = score_type

    def __call__(self, module, input, output) -> Any:
        mask = self.compute_mask(output)
        return output * mask
        

    def compute_mask(self, output):
        with torch.no_grad():
            B, C, *featuremap_shape = output.shape

            score = torch.max(output.abs().flatten(start_dim=2), dim=-1).values

            mask = torch.ones_like(score)

            n_prune = int(C * self.sparsity)

            indices = torch.topk(score, n_prune, largest=False).indices

            strides = torch.arange(0, B, device=mask.device).reshape(-1, 1) * C

            mask.view(-1)[indices + strides] = 0

            new_dims = [1] * len(featuremap_shape)

            mask = mask.reshape(B, C, *new_dims)
        
        return mask