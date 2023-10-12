from abc import ABCMeta, abstractmethod



class BaseSparsityScheduler(metaclass=ABCMeta):
    
    TYPE = 'base'

    def __init__(self, final_sparsity, pruning_steps, init_sparsity=0.) -> None:
        self.init_sparsity = init_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_steps = pruning_steps

        self.current_pruning_step = 0
        self.current_sparsity = init_sparsity if init_sparsity else 0.
        self.prev_sparsity = 0


    def __hash__(self) -> int:
        return hash((self.__class__, self.init_sparsity, self.final_sparsity, self.pruning_steps))


    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, BaseSparsityScheduler):
            return self.__class__ == __value.__class__ and \
                   self.init_sparsity == __value.init_sparsity and \
                   self.final_sparsity == __value.final_sparsity and \
                   self.pruning_steps == __value.pruning_steps
        else:
            raise TypeError(f"Only {self.__class__} is supported for this operation!!!")


    def __repr__(self) -> str:
        return f'{self.__class__}(target_sparsity={self.final_sparsity}, pruning_steps={self.pruning_steps})'


    @abstractmethod
    def update_sparsity(self):
        raise NotImplementedError("Do not use this class directly!!!")


    def get_amount(self):
        return (self.current_sparsity - self.prev_sparsity) / (1 - self.prev_sparsity)


    def get_sparsity(self):
        return self.current_sparsity


    def is_completed(self):
        return self.current_pruning_step == self.pruning_steps


    def reschedule(self, final_sparsity, pruning_steps, init_sparsity=0.):
        self.init_sparsity = init_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_steps = pruning_steps
        self.current_pruning_step = 0

        self.current_sparsity = init_sparsity if init_sparsity else 0.


    def step(self):
        self.prev_sparsity = self.current_sparsity
        if not self.is_completed():
            self.current_pruning_step += 1
            self.update_sparsity()
    


class DefaultSparsityScheduler(BaseSparsityScheduler):

    TYPE = 'gradual'

    def __init__(self, final_sparsity=0.5, pruning_steps=50, init_sparsity=0.) -> None:
        super().__init__(final_sparsity, pruning_steps, init_sparsity)


    def update_sparsity(self):
        self.current_sparsity = self.final_sparsity + (self.init_sparsity - self.final_sparsity) \
        * pow((1 - self.current_pruning_step / self.pruning_steps), 3)



class ExponentialSparsityScheduler(BaseSparsityScheduler):

    TYPE = 'exponential'

    def __init__(self, final_sparsity, pruning_steps, init_sparsity=0) -> None:
        super().__init__(final_sparsity, pruning_steps, init_sparsity)
        self.factor = ((1 - final_sparsity) / (1 - init_sparsity)) ** (1 / pruning_steps)


    def update_sparsity(self):
        self.current_sparsity = 1 - (1 - self.prev_sparsity) * self.factor



class LinearSparsityScheduler(BaseSparsityScheduler):
    
    TYPE = 'linear'

    def __init__(self, final_sparsity, pruning_steps, init_sparsity=0.) -> None:
        super().__init__(final_sparsity, pruning_steps, init_sparsity)
        self.delta = (final_sparsity - init_sparsity) / pruning_steps


    def update_sparsity(self):
        self.current_sparsity += self.delta