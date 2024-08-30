# learning rate schedulers

class LRScheduler:
    """
    Base class for learning rate schedulers.
    """
    def __init__(self, optimizer, initial_lr):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def set_lr(self, lr):
        for p_group in self.optimizer.param_groups:
            p_group['lr'] = lr

    def step(self, epoch):
        raise NotImplementedError("Implement the step method in the derived class")
    

class StepLR(LRScheduler):
    """
    Decays the learning rate by gamma every step_size epochs.
    """
    def __init__(self, optimizer, initial_lr, step_size, gamma=0.1):
        super(StepLR, self).__init__(optimizer, initial_lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def step(self, epoch):
        if epoch % self.step_size == 0 and epoch != 0:
            new_lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))
            self.set_lr(new_lr)
        return self.get_lr()

class ExponentialLR(LRScheduler):
    """
    Decays the learning rate by gamma every epoch.
    """
    def __init__(self, optimizer, initial_lr, gamma=0.1):
        super(ExponentialLR, self).__init__(optimizer, initial_lr)
        self.gamma = gamma
    
    def step(self, epoch):
        new_lr = self.initial_lr * (self.gamma ** epoch)
        self.set_lr(new_lr)
        return self.get_lr()
    

class CosineAnnealingLRScheduler(LRScheduler):
    """
    Cosine annealing LR scheduler. Reduces the learning rate following a cosine curve.
    """
    def __init__(self, optimizer, initial_lr, T_max, eta_min=0):
        super().__init__(optimizer, initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, epoch):
        import math
        new_lr = self.eta_min + (self.initial_lr - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        self.set_lr(new_lr)