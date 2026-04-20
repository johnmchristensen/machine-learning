import torch

from history import History
from model_runner.ModelRunner import ModelRunner


class LRReducingModelRunner(ModelRunner):
    def __init__(self, model, metric, optimizer=None, loss_fn=None, device=None, mode="min", factor=0.1):
        super().__init__(model, metric, optimizer, loss_fn, device)
        self.mode = mode
        self.factor = factor

    def train_model(self, train_loader, valid_loader, n_epochs=100, patience=10):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=self.mode, factor=self.factor, patience=patience)

        def scheduler_step(history):
            scheduler.step(history.valid_metrics[-1])

        return super().train_model(train_loader, valid_loader, n_epochs=n_epochs, patience=patience,
                                   scheduler_step_callback=scheduler_step)

    def _create_history(self, n_epochs, patience):
        return History(n_epochs, patience, is_higher_metric_better=False)