import torch

from get_device import get_device
from history import History
from torch import nn


class ModelRunner:
    def __init__(self, model, metric, optimizer=None, loss_fn=None, device=None):
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        self.metric = metric.to(self.device)
        self.optimizer = optimizer if optimizer is not None else torch.optim.NAdam(model.parameters(), lr=2e-3)
        self.criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def train_model(self, train_loader, valid_loader, n_epochs = 50, patience=50, scheduler_step_callback = None):
        def best_callback():
            torch.save(self.model.state_dict(), checkpoint_path)

        checkpoint_path = "my_checkpoint.pt"
        history = self._create_history(n_epochs, patience)
        for epoch in range(n_epochs):
            self.metric.reset()
            self.model.train()
            with history.start():
                total_loss = self.__train_batch(train_loader)
                train_metric = self.metric.compute().item()
                valid_metric = self.__evaluate_tm(valid_loader).item()

            history.append(train_loss=total_loss / len(train_loader), train_metric=train_metric,
                           valid_metric=valid_metric,
                           best_callback=best_callback)
            history.print()

            if not scheduler_step_callback is None:
                scheduler_step_callback(history)
            if history.is_out_of_patience():
                print("Early stopping!")
                break
        self.model.load_state_dict(torch.load(checkpoint_path))
        return history

    def test_model(self, data_loader, metric=None):
        metric = metric.to(self.device) if metric is not None else self.metric

        self.model.eval()
        metric.reset()
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                metric.update(y_pred, y_batch)

        return metric.compute()
    
    def run_model(self, data_loader):
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(self.device)
                y_pred.append(self.model(X_batch))
        return torch.cat(y_pred)

    def __train_batch(self, loader):
        device = get_device()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.metric.update(y_pred, y_batch)
        return total_loss

    def __evaluate_tm(self, data_loader):
        self.model.eval()
        self.metric.reset()  # reset the metric at the beginning
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                self.metric.update(y_pred, y_batch)  # update it at each iteration
        return self.metric.compute()  #

    def _create_history(self, n_epochs, patience):
        return History(n_epochs, patience)
