import torch

from model_runner.ModelRunner import ModelRunner, train_with_early_stopping


class OneCycleModelRunner(ModelRunner):
    def __init__(self, model, metric, optimizer=None, loss_fn=None, device=None, max_learning_rate=1e-2):
        super().__init__(model, metric, optimizer, loss_fn, device)
        self.max_learning_rate = max_learning_rate

    def train_model(self, train_loader, valid_loader, n_epochs=100, patience=10, scheduler_step_callback=None):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, epochs=n_epochs, max_lr=self.max_learning_rate,
                                                        steps_per_epoch=len(train_loader))

        def scheduler_step():
            scheduler.step()

        return super().train_model(train_loader, valid_loader, n_epochs=n_epochs, patience=patience, scheduler_step_callback=scheduler_step)
