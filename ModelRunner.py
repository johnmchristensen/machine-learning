import torch
from torch import nn

from get_device import get_device
from train_with_early_stopping import train_with_early_stopping


class ModelRunner():
    def __init__(self, model, accuracy, optimizer=None, criterion=None, device=None):
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        self.accuracy = accuracy.to(self.device)
        self.optimizer = optimizer if optimizer is not None else torch.optim.NAdam(model.parameters(), lr=2e-3)
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def train_model(self, train_loader, valid_loader, n_epochs = 100):
        return train_with_early_stopping(self.model, self.optimizer, self.criterion, self.accuracy, train_loader,
                                            valid_loader, n_epochs)

    def run_model(self, data_loader):
        self.model.eval()
        self.accuracy.reset()
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                self.accuracy.update(y_pred, y_batch)
        return self.accuracy.compute()