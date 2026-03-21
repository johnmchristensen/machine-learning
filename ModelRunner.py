import torch
from torch import nn

from evaluate_tm import evaluate_tm
from get_device import get_device
from history import History


class ModelRunner:
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


class OneCycleModelRunner(ModelRunner):
    def __init__(self, model, accuracy, optimizer=None, criterion=None, device=None, max_learning_rate=1e-2):
        super().__init__(model, accuracy, optimizer, criterion, device)
        self.max_learning_rate = max_learning_rate

    def train_model(self, train_loader, valid_loader, n_epochs=100):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, epochs=n_epochs, max_lr=self.max_learning_rate,
                                                        steps_per_epoch=len(train_loader))
        return train_with_early_stopping(self.model, self.optimizer, self.criterion, self.accuracy, train_loader,
                                         valid_loader, n_epochs, scheduler=scheduler)


def train_with_early_stopping(model, optimizer, loss_fn, metric, train_loader, valid_loader, n_epochs,
                              patience=10, checkpoint_path=None, scheduler=None):
    def best_callback():
        torch.save(model.state_dict(), checkpoint_path)

    checkpoint_path = checkpoint_path or "my_checkpoint.pt"
    history = History(n_epochs, patience)
    for epoch in range(n_epochs):
        metric.reset()
        model.train()
        with history.start():
            total_loss = __train_batch(model, train_loader, loss_fn, optimizer, metric)
            train_metric = metric.compute().item()
            valid_metric = evaluate_tm(model, valid_loader, metric).item()

        history.append(train_loss=total_loss / len(train_loader), train_metric=train_metric, valid_metric=valid_metric,
                       best_callback=best_callback)
        history.print()

        if scheduler is not None:
            scheduler.step()
        if history.is_out_of_patience():
            print("Early stopping!")
            break
    model.load_state_dict(torch.load(checkpoint_path))
    return history


def __train_batch(model, loader, loss_fn, optimizer, metric):
    device = get_device()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        metric.update(y_pred, y_batch)
    return total_loss