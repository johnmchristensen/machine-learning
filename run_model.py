import torch
import torchmetrics
from torch import nn

from model_runner.ModelRunner import train_with_early_stopping
from get_device import get_device


def run_model(model, train_loader, valid_loader, learning_rate = 2e-3, n_epocs = 100):
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    criterion=nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(get_device())

    history = train_with_early_stopping(model, optimizer, criterion, accuracy, train_loader, valid_loader, n_epocs)
    return history

def run_model_1cycle_scheduling(model, train_loader, valid_loader, learning_rate = 2e-3, n_epocs = 100, max_learning_rate = 1e-2):
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=n_epocs, max_lr=max_learning_rate, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(get_device())

    history = train_with_early_stopping(model, optimizer, criterion, accuracy, train_loader, valid_loader, n_epocs,
                                        scheduler=scheduler)
    return history