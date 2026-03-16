import torch

from ModelRunner import ModelRunner
from train_with_early_stopping import train_with_early_stopping


class OneCycleModelRunner(ModelRunner):
    def __init__(self, model, accuracy, optimizer=None, criterion=None, device=None, max_learning_rate = 1e-2):
        super().__init__(model, accuracy, optimizer, criterion, device)
        self.max_learning_rate = max_learning_rate

    def train_model(self, train_loader, valid_loader, n_epochs = 100):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, epochs=n_epochs, max_lr=self.max_learning_rate,
                                                        steps_per_epoch=len(train_loader))
        return train_with_early_stopping(self.model, self.optimizer, self.criterion, self.accuracy, train_loader,
                                         valid_loader, n_epochs, scheduler=scheduler)