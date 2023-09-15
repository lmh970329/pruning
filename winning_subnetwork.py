import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from pruner import L1UnstructuredPruner
from torch.optim import SGD, Adam
from torch.autograd import Function
from typing import Any, Callable, List, Optional
from utils import ParameterRegistrator


class GetSubnetFaster(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, scores: torch.Tensor, sparsity: float) -> Any:

        nparams_to_survive = round(scores.nelement() * (1 - sparsity))

        mask = torch.zeros_like(scores.view(-1))
        mask[scores.view(-1).topk(nparams_to_survive).indices] = 1
        return mask.view_as(scores)
    

    @staticmethod
    def backward(ctx, g):
        return g, None



class WSNForwardHook():

    def __init__(self, name, sparsity) -> None:
        self.name = name
        self.sparsity = sparsity
    

    def __call__(self, module: nn.Module, inputs) -> Any:
        
        orig = module.get_parameter(f'{self.name}_orig')

        if module.training:    
            score = module.get_parameter(f'{self.name}_score')
            mask = GetSubnetFaster.apply(score.abs(), self.sparsity)

            module.register_buffer(f'{self.name}_mask', mask)
        else:
            # For evalutation, the mask buffer must be assigned with mask for specified task id
            mask = getattr(module, f'{self.name}_mask', torch.ones_like(orig))
        
        setattr(module, self.name, orig * mask)



class WinningSubNetwork(ParameterRegistrator):

    accum_suffix = 'mask_accum'
    score_suffix = 'score'
    mask_suffix = 'mask'
    orig_suffix = 'orig'

    def __init__(self, model, c) -> None:
        super(WinningSubNetwork, self).__init__()
        self.model = model
        self.c = c
        self.sparsity = 1. - c

        self.current_task = None

        self.is_initialized = False
    
        self.bn_layers: List[nn.BatchNorm2d] = []
        self.bn_states = dict()

        self.prev_bn_states = None

    
    def register_bn(self, layers):

        if isinstance(layers, (list, tuple)):
            for bn in layers:
                self.bn_layers.append(bn)
        elif isinstance(layers, nn.BatchNorm2d):
            self.bn_layers.append(bn)


    def update_accumulate_mask(self):
        
        for module, name in self.registrations():

            mask_accum = self.get_mask_accum(module, name)
            current_mask = self.get_current_mask(module, name)

            module.register_buffer(f'{name}_{self.mask_suffix}_{self.current_task}', current_mask.clone().detach())

            mask_accum.data = torch.bitwise_or(mask_accum.to(torch.int32), current_mask.to(torch.int32)).to(torch.float32)
        
        current_bn_states = [bn.state_dict() for bn in self.bn_layers]
        self.bn_states[f'task_{self.current_task}'] = current_bn_states

        self.current_task += 1


    def mask_gradients(self):

        for module, name in self.registrations():

            orig = module.get_parameter(f'{name}_{self.orig_suffix}')

            mask_accum = self.get_mask_accum(module, name)

            orig.grad.mul_((1 - mask_accum))


    def step(self, optimizer: torch.optim.Optimizer, closure: Optional[Callable[[], float]]=None):

        self.mask_gradients()

        return optimizer.step(closure)


    def initialize(self):

        assert not self.is_initialized
        
        self.current_task = 0

        for module, name in self.registrations():

            orig = module.get_parameter(name)

            # Add weight score matrices used to select important weights
            module.register_parameter(
                f'{name}_{self.score_suffix}',
                nn.Parameter(torch.randn_like(orig))
            )

            module.register_parameter(
                f'{name}_{self.orig_suffix}',
                orig
            )
            del module._parameters[name]

            # Add forward pre-hook for generating mask
            method = WSNForwardHook(name, self.sparsity)
            method(module, None)

            module.register_forward_pre_hook(method)

            # Add accumulate mask used to update gradient selectively
            module.register_buffer(
                f'{name}_{self.accum_suffix}',
                torch.zeros_like(orig)
            )

        self.is_initialized = True


    def get_sparsities(self):
        sparsities = [(self.get_current_mask(module, name) == 0.).sum().item() / self.get_current_mask(module, name).nelement() for module, name in self.registrations()]
        return sparsities
    

    def _switch_to_mask_for_task(self, task_id):

        for module, name in self.registrations():
            target_mask = self.get_mask_for_task(module, name, task_id)
            
            mask = self.get_current_mask(module, name)
            mask.data = target_mask.data


    def _set_state(self, module: nn.Module, state):
        prev_state = module.state_dict()
        module.load_state_dict(state)

        return prev_state
    

    def _set_bn_states(self, states):
        for bn, state in zip(self.bn_layers, states):
            self._set_state(bn, state)


    def _switch_to_bn_states_for_task(self, task_id):

        if not task_id < len(self.bn_states):
            raise ValueError(f"Only {len(self.bn_states)} tasks has been learned!!!")
        
        target_bn_states = self.bn_states[f'task_{task_id}']

        self._set_bn_states(target_bn_states)


    def adapt_to_subnetwork_for_task(self, task_id):
        self._switch_to_mask_for_task(task_id)
        self._switch_to_bn_states_for_task(task_id)

    
    def train(self):
        if self.prev_bn_states:
            for bn, state in zip(self.bn_layers, self.prev_bn_states):
                self._set_state(bn, state)
            self.prev_bn_states = None
            self.model.train()


    def eval(self):
        if self.prev_bn_states is None:
            prev_bn_states = [bn.state_dict() for bn in self.bn_layers]
            self.prev_bn_states = prev_bn_states
            self.model.eval()


    @property
    def scores(self):
        return [module.get_parameter(f'{name}_{self.score_suffix}') for module, name in self.registrations()]


    @property
    def has_bn_layers(self):
        return self.bn_layers is not None

    
    def get_current_mask(self, module: nn.Module, name: str):
        return module.get_buffer(f'{name}_{self.mask_suffix}')
    

    def get_mask_for_task(self, module: nn.Module, name: str, task_id: int):
        return module.get_buffer(f'{name}_{self.mask_suffix}_{task_id}')


    def get_score(self, module: nn.Module, name: str):
        return module.get_parameter(f'{name}_{self.score_suffix}')
    

    def get_mask_accum(self, module: nn.Module, name: str):
        return module.get_buffer(f'{name}_{self.accum_suffix}')