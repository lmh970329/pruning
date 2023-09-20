from typing import Any, Dict
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn


class ActivationPrune(object, metaclass=ABCMeta):

    def __init__(self, sparsity) -> None:
        self.sparsity = sparsity
        self._handles = dict()


    def __call__(self, module: nn.Module, input, output: torch.Tensor):
        mask = self.compute_mask(output)
        return output * mask


    def apply(self, module: nn.Module):
        self._handles[module] = module.register_forward_hook(self)


    def remove(self, module: nn.Module):
        self._handles.pop(module).remove()


    @abstractmethod
    def compute_mask(self, output):
        raise NotImplementedError("You should implement this abstract method!!!")