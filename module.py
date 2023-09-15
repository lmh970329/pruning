from typing import Union
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.common_types import _size_1_t
from typing import Union
    

class Conv1d(nn.Sequential):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            padding: Union[_size_1_t, str] = 0,
            dilation: _size_1_t = 1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            norm_layer=None,
            act_layer=None
        ) -> None:
        super().__init__()

        self.add_module(
            'conv',
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
                device=device,
                dtype=dtype
            )
        )

        if norm_layer:
            self.add_module(
                'norm',
                norm_layer(out_channels)
            )

        if act_layer:
            self.add_module(
                'act',
                act_layer()
            )



class Conv1dChannelDrop(Conv1d):
    
    def __init__(
            self,
            in_channels:int,
            out_channels: int,
            kernel_size: _size_1_t,
            information_ratio,
            stride: _size_1_t = 1,
            padding: Union[_size_1_t, str] = 0,
            dilation: _size_1_t = 1,
            groups: int = 1, 
            bias: bool = False,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            norm_layer=None,
            act_layer=None
        ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            norm_layer,
            act_layer
        )

        self.information_ratio = information_ratio


    @torch.no_grad()
    def get_binary_mask(self, out):
        gmp = torch.max(out, dim=-1).values

        mask = torch.ones_like(gmp)

        score_val, indices = torch.div(
            gmp,
            torch.sum(gmp, dim=-1, keepdim=True)
        ).sort(dim=-1, descending=True)

        cumulative = torch.cumsum(score_val, dim=-1)
        
        for sample_idx in range(out.shape[0]):
            prune_idx = indices[sample_idx][(cumulative > self.information_ratio)[sample_idx]]
            mask[sample_idx][prune_idx] = 0

        return mask.unsqueeze(-1)


    def forward(self, input: Tensor) -> Tensor:
        out = super().forward(input)
        mask = self.get_binary_mask(out)
        return out * mask