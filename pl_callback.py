import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Any, List, Tuple, Union, Iterable
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn.utils.convert_parameters import parameters_to_vector
from .pruner import BasePruner, calculate_sparsity, apply_grad_mask
from .scheduler import BaseSparsityScheduler, DefaultSparsityScheduler
from .manager import IterativePruningManager


class PruningManagerCallback(pl.Callback):

    def __init__(self, manager: IterativePruningManager, on_epoch: bool=True) -> None:
        super().__init__()
        
        self.manager = manager
        self.on_epoch = on_epoch


    def make_sparsity_metrics(self, model: nn.Module):

        total_params = parameters_to_vector(model.parameters()).nelement()
        total_zeros = 0

        for module, name in self.manager.registrations():
            module_zeros = (getattr(module, name) == 0).sum().item()
            total_zeros += module_zeros
        
        sparsity_model = total_zeros / total_params

        metrics = dict()
        for group_name in self.manager.group_names():
            metrics[f'sparsity/{group_name}/scheduler'] = self.manager.get_scheduler(group_name).get_sparsity()

        for group_name in self.manager.group_names():
            for idx, (module, name) in enumerate(self.manager.get_group(group_name)):
                pruned_tensor = getattr(module, name)
                metrics[f'sparsity/{group_name}/parameter_{idx}'] = calculate_sparsity(pruned_tensor)

        metrics['sparsity/model'] = sparsity_model

        return metrics


    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:

        if not self.on_epoch:
            prune = self.manager.step()

            if prune:
                device = pl_module.device
                pl_module.to('cpu')

                self.manager.prune()

                pl_module.to(device)

                sparsity_metrics = self.make_sparsity_metrics(pl_module)
                pl_module.log_dict(sparsity_metrics)

            
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        
        if self.on_epoch:
            prune = self.manager.step()

            if prune:
                device = pl_module.device
                pl_module.to('cpu')

                self.manager.prune()

                pl_module.to(device)

                sparsity_metrics = self.make_sparsity_metrics(pl_module)
                pl_module.log_dict(sparsity_metrics)




class IterativePruningCallback(pl.Callback):
    
    def __init__(self, pruner: BasePruner, sparsity_scheduler: BaseSparsityScheduler=None, frequency: int=100, warmup: int=0) -> None:
        self._pruner = pruner
        self._sparsity_scheduler = sparsity_scheduler
        
        self._frequency = frequency
        self._warmup = warmup

        self._is_running = False
        self.__cnt = 0


    def reschedule(self, scheduler=None, frequency=100, final_sparsity=0.5, pruning_steps=50, init_sparsity=0., warmup=None):
        old_scheduler = self.sparsity_scheduler

        if scheduler and isinstance(scheduler, BaseSparsityScheduler):
            self.sparsity_scheduler = scheduler
        else:
            self.sparsity_scheduler.reschedule(final_sparsity, pruning_steps, init_sparsity)

        self.frequency = frequency
        self.is_running = False
        self._warmup = warmup if warmup else 0
        self.__cnt = 0

        del old_scheduler


    def should_be_pruned(self):
        self.__cnt += 1
        return not self.__cnt % self.frequency


    def make_sparsity_metrics(self, model: nn.Module):
        sparsity_sched = self.sparsity_scheduler.get_sparsity()

        total_params = parameters_to_vector(model.parameters()).nelement()
        total_zeros = 0

        for module, name in self.pruner.registrations():
            module_zeros = (getattr(module, name) == 0).sum().item()
            total_zeros += module_zeros
        
        sparsity_model = total_zeros / total_params

        sparsities_module = [
            calculate_sparsity(getattr(module, name)) for module, name in self._pruner.registrations()
        ]

        metrics = dict()
        metrics['sparsity/sched'] = sparsity_sched
        metrics['sparsity/model'] = sparsity_model
        for idx, sparsity in enumerate(sparsities_module):
            metrics['sparsity/module_{}'.format(idx)] = sparsity

        return metrics


    def on_sanity_check_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.sparsity_scheduler:
            raise RuntimeError("Sparsity scheduler was not specified!!! "
                               "Specify sparsity scheduler or use PruningPhaseManager")


    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # Check warm-up stage, pruning process won't be start
        # until the warm-up stage finishes.
        # (Internal counter does not increase when the pruning process is not progressed)
        if not (self.is_done or self.is_running):
            # if warmup ends, start to prune the model
            if not self._warmup:
                self.is_running = True
            self._warmup -= 1

        # Pruning process is running and pruning frequency is reached
        if self.is_running and self.should_be_pruned():

            # Update sparsity if scheduling is not completed
            if not self.sparsity_scheduler.is_completed():
                self.sparsity_scheduler.step()

                # By internal implementation of Pytorch's pruning methods,
                # new memory space is allocated in GPU repeatedly if GPU is being used.
                # So the model must be moved to CPU before being pruned, which prevent memory leakage in GPU
                device = pl_module.device
            
                pl_module.to('cpu')

                self.pruner.prune(self.sparsity_scheduler.get_amount())

                metrics = self.make_sparsity_metrics(pl_module)
                pl_module.log_dict(metrics)

                pl_module.to(device)

        # If sparsity scheduling is completed and recovery steps after last pruning is finished,
        # the pruning process will be done. (Not running state)
        if self.is_done:
            self.is_running = False


    @property
    def frequency(self):
        return self._frequency
    

    @property
    def pruner(self):
        return self._pruner


    @property
    def sparsity_scheduler(self):
        return self._sparsity_scheduler


    @property
    def current_sparsity(self):
        return self.sparsity_scheduler.get_sparsity()


    @property
    def is_running(self):
        return self._is_running


    @property
    def total_steps(self):
        if self.sparsity_scheduler:
            return self.frequency * (self.sparsity_scheduler.pruning_steps + 1)
        else:
            raise AttributeError("sparsity scheduler was not specified!!!")


    @property
    def is_done(self):
        return self.sparsity_scheduler.is_completed() and self.__cnt >= self.total_steps


    @is_running.setter
    def is_running(self, value):
        if isinstance(value, bool):
            self._is_running = value


    @sparsity_scheduler.setter
    def sparsity_scheduler(self, obj):
        if isinstance(obj, BaseSparsityScheduler):
            self._sparsity_scheduler = obj
        else:
            raise TypeError("Sparsity scheduler must be 'BaseSparsityScheduler' type")


    @pruner.setter
    def pruner(self, obj):
        if isinstance(obj, BasePruner):
            self._pruner = obj
        else:
            raise TypeError("Pruner must be 'BasePruner' type")


    @frequency.setter
    def frequency(self, value):
        if isinstance(value, int):
            if not value > 0:
                raise ValueError("Not positive frequency is not possible")
            self._frequency = value
        else:
            raise TypeError("Frequency must be int")



class PruningPhaseManager(pl.Callback):

    def __init__(self,
                 n_phases: int,
                 target_sparsities: Union[List[float], Tuple[float], float],
                 pruning_steps: Union[List[int], Tuple[int], int],
                 pruning_callback: IterativePruningCallback,
                 frequencies: Union[List[int], Tuple[int], int]=None,
                 scheduler_types: Union[List, Tuple, Any]=DefaultSparsityScheduler,
                 init_sparsity=0.,
                 weight_freeze=True,
                 weight_reinit=False,
                 ) -> None:
        
        self._pruning_callback = pruning_callback

        self.weight_freeze = weight_freeze
        self.weight_reinit = weight_reinit

        self._n_phases = n_phases

        self._is_running = False
        self._is_done = False

        if isinstance(target_sparsities, (list, tuple)):
            assert n_phases == len(target_sparsities)
            self._target_sparsitis = target_sparsities
        elif isinstance(target_sparsities, float):
            self._target_sparsitis = [target_sparsities] * n_phases
        
        if isinstance(pruning_steps, (list, tuple)):
            assert n_phases == len(pruning_steps)
            self._pruning_steps = pruning_steps
        elif isinstance(pruning_steps, int):
            self._pruning_steps = [pruning_steps] * n_phases

        if isinstance(frequencies, (list, tuple)):
            assert n_phases == len(frequencies)
            self._frequencies = frequencies
        elif isinstance(frequencies, int):
            self._frequencies = [frequencies] * n_phases

        if isinstance(scheduler_types, (list, tuple)):
            assert n_phases == len(scheduler_types)
            self._scheduler_types = scheduler_types
        elif issubclass(scheduler_types, BaseSparsityScheduler):
            self._scheduler_types = [scheduler_types] * n_phases

        self.__current_phase = 0

        self.pruning_callback.reschedule(
            self.scheduler_type(self.target_sparsity, self.pruning_steps, init_sparsity),
            frequency=self.frequency
        )


    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:

        # Check whether current pruning phase is done.
        # If true, pruning phase will be updated or whole pruning phase managing process will be terminated.
        if not self.is_done and self.current_phase_is_done:

            current_sparsity = self.pruning_callback.current_sparsity            
            self.pruning_callback.pruner.remove()
            current_sparsity = 0.
                # for g in pl_module.optimizers(False).param_groups:
                #     g['lr'] = 0.1

            if self.is_last_phase:
                self.is_done = True

            if not self.is_done:

                self.next_phase()

                if self.weight_reinit:
                    for module, name in self.pruning_callback.pruner.registrations():

                        module.requires_grad_(False)

                        param = module.get_parameter(name)
                        grad_mask = module.get_buffer(name+'_grad_mask')

                        mean, std = torch.mean(param), torch.std(param)

                        param[grad_mask] = torch.normal(mean, std, param[grad_mask].shape, device=param.device)

                        module.requires_grad_(True)

                self.pruning_callback.reschedule(
                    self.scheduler_type(
                    final_sparsity=self.target_sparsity,
                    pruning_steps=self.pruning_steps,
                    init_sparsity=current_sparsity
                    ),
                    frequency=self.frequency
                )


    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.is_done and self.weight_freeze:
            for module, name in self.pruning_callback.pruner.registrations():
                apply_grad_mask(module, name)
        elif self.is_done:
            pl_module.zero_grad()


    def next_phase(self):
        self.__current_phase += 1
        return self.__current_phase


    @property
    def target_sparsities(self):
        return self._target_sparsitis


    @property
    def n_phases(self):
        return self._n_phases


    @property
    def pruning_callback(self):
        return self._pruning_callback
    

    @property
    def target_sparsity(self):
        return self._target_sparsitis[self.__current_phase]


    @property
    def scheduler_type(self):
        return self._scheduler_types[self.__current_phase]


    @property
    def frequency(self):
        return self._frequencies[self.__current_phase]


    @property
    def pruning_steps(self):
        return self._pruning_steps[self.__current_phase]
    

    @property
    def is_running(self):
        return self._is_running


    @property
    def is_done(self):
        return self._is_done
    

    @property
    def current_phase_is_done(self):
        return self.pruning_callback.is_done
    

    @property
    def is_last_phase(self):
        return self.__current_phase == (self._n_phases - 1)


    @is_running.setter
    def is_running(self, value):
        self._is_running = value


    @is_done.setter
    def is_done(self, value):
        self._is_done = value


    @n_phases.setter
    def n_phases(self, value):
        raise RuntimeError("Change of 'n_phases' is not allowed")


    @target_sparsities.setter
    def target_sparsities(self, sparsities):
        raise RuntimeError("Change of 'target_sparsities' is not allowed")


    @pruning_callback.setter
    def pruning_callback(self, callback):
        raise RuntimeError("Change of 'pruning_callback' is not allowed")