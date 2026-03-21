from experiment_runner import run
from MnistModel import MnistEnsembleAveragingModel, MnistEnsembleModel, CLASS_COUNT
from torch import nn

import mnist_loading
import torch
import torchmetrics

train_loader, valid_loader, test_loader = mnist_loading.with_image_augmentation()
print("Data loaded")

torch.manual_seed(42)

xentropy = nn.CrossEntropyLoss()
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=CLASS_COUNT)

def model_selecting_builder(p):
    if p == "Ensemble":
        return MnistEnsembleModel(3)

    if p == "Averaging":
        return MnistEnsembleAveragingModel(3)

    raise ValueError(f"Unknown model type: {p}")

def model_count_builder(p):
    return MnistEnsembleModel(p)

def optimizer_generator(m):
    return torch.optim.Adam(m.parameters())

best_model, best_valid_metric, best_parameter_value = run(model_count_builder, optimizer_generator, [3, 5, 7],
                                                          accuracy, train_loader, valid_loader, xentropy, n_epochs=5)
print(f"Best parameters: {best_parameter_value}")
print(f"Best validation metric: {best_valid_metric}")