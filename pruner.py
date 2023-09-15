import torch
import torch.nn as nn
from torch.nn.utils import prune
from .custom_prune import ln_attention_head_structured, l1_unstructured_masked, ln_structured_masked, l1_unstructured_partitioned, l1_unstructured_inversed
from typing import Iterable, Union, Tuple, List, MutableSet, Sequence, Dict
from collections import defaultdict


Parameter = Tuple[nn.Module, str]



class ParameterRegistrator(object):
    
    def __init__(self) -> None:
        self.__parameters: Dict[str, MutableSet[Parameter]] = defaultdict(set)
    

    def register(self, parameters: Sequence, group_name: str='default'):
        param_group = self.__parameters[group_name]
        if isinstance(parameters, (list, tuple)):
            for parameter in parameters:
                if isinstance(parameter, (list, tuple)):
                    self._register(parameter, param_group)
                else:
                    self._register(parameters, param_group)
        else:
            raise TypeError("Type of 'parameters' must be one of list or tuple")


    def _register(self, parameter: Parameter, group: MutableSet[Parameter]):
        
        if not len(parameter) == 2:
            raise ValueError("Shape of single parameter must be (module, name)")
        
        module, name = parameter
        
        if hasattr(module, name):
            group.add(parameter)
        else:
            raise ValueError(f"{module} has no attribute named {name}")
        
    
    def registrations(self):
        all_params = [param for group in self.__parameters.values() for param in group]
        return all_params
    

    def get_group(self, group_name):
        return [param for param in self.__parameters[group_name]]
    

    def group_names(self):
        return self.__parameters.keys()





class BasePruner(ParameterRegistrator):

    TYPE = 'base'

    def prune(self, amount, group_name=None, importance_suffix=None, **kwargs):

        param_group = self.get_group(group_name) if group_name is not None else self.registrations()
        
        for module, name in param_group:
            importance_scores = getattr(module, f'{name}_{importance_suffix}', None)
            self.prune_parameter(module, name, amount, importance_scores=importance_scores, **kwargs)
        
    
    def prune_parameter(self, module, name, amount, importance_scores=None, **kwargs):
        raise NotImplementedError("Not implemented!!!")


    def remove(self):
        for module, name in self.registrations():
            prune.remove(module, name)

    def get_module_sparsities(self):
        sparsities = [
            calculate_sparsity(module.get_parameter(name)) for module, name in self.registrations()
        ]
        return sparsities



class GlobalPruner(BasePruner):

    TYPE = 'global'
    
    def prune(self, amount, group_name=None, importance_suffix=None, **kwargs):

        param_group = self.get_group(group_name) if group_name is not None else self.registrations()
        prune.global_unstructured(param_group, prune.L1Unstructured, amount=amount, importance_scores=None, **kwargs)



class L1UnstructuredPruner(BasePruner):

    TYPE = 'l1'
    
    def prune_parameter(self, module, name, amount, importance_scores=None, **kwargs):
        prune.l1_unstructured(module, name, amount, importance_scores=importance_scores)



class L1UnstructuredMaskedPruner(BasePruner):

    TYPE = 'l1'
    masked_pruning = True

    def prune_parameter(self, module, name, amount, importance_scores=None, **kwargs):
        init_candidate_mask = getattr(module, f'{name}_grad_mask', None)
        prev_mask = getattr(module, f'{name}_mask', None)
        l1_unstructured_masked(module, name, amount, init_candidate_mask=init_candidate_mask, prev_mask=prev_mask, importance_scores=importance_scores)


    def remove(self):
        for module, name in self.__parameters:
            mask = module.get_buffer(name+'_mask')
            create_grad_mask(module, name, (mask == 0))
        return super().remove()



class L1UnstructuredPartitionedPruner(BasePruner):

    TYPE = 'l1_partitioned'

    def __init__(self, npartitions) -> None:
        super().__init__()
        self.npartitions = npartitions


    def prune_parameter(self, module, name, amount, importance_scores=None, **kwargs):
        l1_unstructured_partitioned(module, name, amount, self.npartitions, importance_scores=importance_scores)



class L1UnstructuredInversedPruner(BasePruner):

    TYPE = 'l1_inversed'

    def prune_parameter(self, module, name, amount, importance_scores=None, **kwargs):
        l1_unstructured_inversed(module, name, amount, importance_scores=importance_scores)



class LnStructuredPruner(BasePruner):
    
    TYPE = 'ln'

    def __init__(self, n=2, dim=0) -> None:
        super().__init__()
        self.n = n
        self.dim = dim


    def prune_parameter(self, module, name, amount, importance_scores=None, **kwargs):
        prune.ln_structured(module, name, amount, n=self.n, dim=self.dim, importance_scores=importance_scores)



class LnStructuredMaskedPruner(BasePruner):

    TYPE = 'ln'

    masked_pruning = True

    def __init__(self, n, dim) -> None:
        super().__init__()
        self.n = n
        self.dim = dim


    def prune_parameter(self, module, name, amount, importance_scores=None, **kwargs):
        init_candidate_mask = getattr(module, f'{name}_grad_mask', None)
        prev_mask = getattr(module, f'{name}_mask', None)
        ln_structured_masked(module, name, amount, self.n, self.dim, init_candidate_mask, prev_mask, importance_scores=importance_scores)


    def remove(self):
        for module, name in self.__parameters:
            mask = module.get_buffer(name+'_mask')
            create_grad_mask(module, name, mask == 0)
        return super().remove()



class LnAttentionHeadPruner(BasePruner):
    
    TYPE = 'ln_attention'

    def __init__(self, n):
        super().__init__()
        self.n = n

    def _register(self, parameter: Union[List, Tuple]):
        if not len(parameter) == 3:
            raise ValueError("Shape of single parameter must be (module, name, head_dim)")
        
        module, name, head_dim = parameter
        if hasattr(module, name):
            self.__parameters.add(parameter)
        else:
            raise ValueError(f"{module} has no attribute named {name}")
        
    def prune(self, sparsity, **kwargs):
        ln_attention_head_structured(self.__parameters, sparsity, self.n, **kwargs)



class CustumPruner(BasePruner):
    
    def __init__(self, prune_func):
        super().__init__()
        self.prune_func = prune_func


    def prune_parameter(self, module, name, amount, *args, importance_scores=None, **kwargs):
        self.prune_func(module, name, amount, *args, importance_scores=importance_scores, **kwargs)



class PrunerContainer(BasePruner):

    def __init__(self, pruners: Iterable[BasePruner]) -> None:
        self.pruners = pruners


    def prune(self, amount, *args, group_name=None, importance_suffix=None, **kwargs):
        for pruner in self.pruners:
            pruner.prune(amount, *args, group_name=group_name, importance_suffix=importance_suffix, **kwargs)


    def registrations(self):
        return [params for pruner in self.pruners for params in pruner.registrations()]
    


def calculate_sparsity(tensor: torch.Tensor):
    tensor_size = tensor.nelement()
    zeros = (tensor == 0).sum().item()
    return zeros / tensor_size



def create_grad_mask(module: nn.Module, name: str, mask: torch.Tensor):
    module.register_buffer(name+'_grad_mask', mask)



def apply_grad_mask(module: nn.Module, name: str):
    if hasattr(module, name+'_grad_mask'):
        grad_mask = module.get_buffer(name+'_grad_mask')
        
        if prune.is_pruned(module):
            name = name+'_orig'
        
        getattr(module, name).grad.mul_(grad_mask)