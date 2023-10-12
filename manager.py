from .pruner import BasePruner
from .scheduler import BaseSparsityScheduler, LinearSparsityScheduler
from typing import Type, Sequence, List, Dict, MutableSet


class IterativePruningManager(object):
    
    def __init__(self, steps, frequency: int=None, pruning_points: List[int]=None) -> None:

        assert (frequency is not None and pruning_points is None) or (pruning_points is not None and frequency is None)

        assert (pruning_points is not None) and (len(pruning_points) == steps)

        self.__methods: MutableSet[BasePruner] = set()
        self.__schedulers: MutableSet[BaseSparsityScheduler] = set()
        self.__schedules: Dict[str, BaseSparsityScheduler] = dict()

        self.frequency = frequency
        self.steps = steps
        self.pruning_points = pruning_points

        self.__iters = 0
        self.__current_pruning_step = 0

    def register(self, parameters: Sequence, prune_type: Type[BasePruner], target_sparsity, scheduler_type: Type[BaseSparsityScheduler], **pruner_kwargs):
        pruner = None
        for exist_pruner in self.__methods:
            if isinstance(exist_pruner, prune_type):
                pruner = exist_pruner

        if pruner is None:
            pruner = prune_type(**pruner_kwargs)
            self.__methods.add(pruner)

        scheduler = None
        for exist_scheduler in self.__schedulers:
            if isinstance(exist_scheduler, scheduler_type) and target_sparsity == exist_scheduler.final_sparsity:
                scheduler = exist_scheduler

        if scheduler is None:
            scheduler = scheduler_type(target_sparsity, self.steps)
            self.__schedulers.add(scheduler)

        group_name = '/'.join([pruner.TYPE, scheduler.TYPE, str(target_sparsity * 100)])
        
        self.__schedules[group_name] = scheduler

        pruner.register(parameters, group_name=group_name)
        



    def registrations(self):
        return [parameter for method in self.__methods for parameter in method.registrations()]


    def group_names(self):
        return [group_name for method in self.__methods for group_name in method.group_names()]
    

    def get_group(self, group_name):
        for method in self.__methods:
            if group_name in method.group_names():
                return method.get_group(group_name)


    def get_scheduler(self, group_name):
        try:
            scheduler = self.__schedules[group_name]
        except KeyError as e:
            raise e
        
        return scheduler
    
    def _scheduler_step(self):
        for scheduler in self.__schedulers:
            scheduler.step()
    

    def _prune(self):
        for method in self.__methods:
            for group_name in method.group_names():
                amount = self.__schedules[group_name].get_amount()
                method.prune(amount, group_name=group_name)


    def need_prune(self):
        if self.frequency:
            return self.__iters % self.frequency == 0
        else:
            return self.__iters == self.pruning_points[self.__current_pruning_step]

    
    def step(self):
        '''
        Update sparsity schedulers and prune modules with the amount specified by schedulers.
        Only return 'True' when pruning frequency has been reached and pruning has occured, otherwise return 'False'
        '''
        self.__iters += 1

        if not self.is_completed:
            return self.need_prune()

        return False
    

    def prune(self):
        self._scheduler_step()
        self._prune()

        self.__current_pruning_step += 1


    @property
    def num_methods(self):
        return len(self.__methods)
    
    @property
    def num_schedulers(self):
        return len(self.__schedulers)
    
    @property
    def num_groups(self):
        return len(self.__schedules.keys())
    
    @property
    def current_iteration(self):
        return self.__iters
    
    @property
    def current_pruning_step(self):
        return self.__current_pruning_step
    
    @property
    def is_completed(self):
        return self.__current_pruning_step == self.steps