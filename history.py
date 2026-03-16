import time
from contextlib import contextmanager

class History:
    def __init__(self, n_epochs, patience):
        self.epoch = 0
        self.n_epochs = n_epochs
        self.train_losses = []
        self.train_metrics = []
        self.valid_metrics = []
        self.times = []
        self.best_valid_metric = 0.0
        self.patience = patience
        self.patience_counter = 0
        self.startTime = None

    def is_valid_metric_best(self, valid_metric):
        return valid_metric > self.best_valid_metric

    @contextmanager
    def start(self):
        if not self.startTime is None:
            raise Exception("History already started")
        self.startTime = time.time()
        yield
        self.times.append(time.time() - self.startTime)
        self.startTime = None

    def append(self, train_loss, train_metric, valid_metric, best_callback = None):
        self.epoch += 1
        self.train_losses.append(train_loss)
        self.train_metrics.append(train_metric)
        self.valid_metrics.append(valid_metric)

        if (valid_metric > self.best_valid_metric):
            self.patience_counter = 0
            self.best_valid_metric = valid_metric
            if best_callback is not None:
                best_callback()
        else:
            self.patience_counter += 1

    def is_out_of_patience(self):
        return self.patience_counter >= self.patience

    def print(self):
        print(f"Epoch:{self.epoch} / {self.n_epochs}, "
              f"train loss: {self.train_losses[-1]:.4f}, "
              f"train metric: {self.train_metrics[-1]:.4f}, "
              f"valid metric: {self.valid_metrics[-1]:.4f}{" (best)" if self.valid_metrics[-1] == self.best_valid_metric else ""}"
              f" in {self.times[-1]:.1f}s")