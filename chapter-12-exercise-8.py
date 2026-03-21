from MnistModel import MnistEnsembleAveragingModel, CLASS_COUNT
from ModelRunner import OneCycleModelRunner
from graph_history import graph_history
from torch import nn

import mnist_loading
import torch
import torchmetrics

train_loader, valid_loader, test_loader = mnist_loading.with_image_augmentation()
print("Data loaded")

torch.manual_seed(42)

xentropy = nn.CrossEntropyLoss()
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=CLASS_COUNT)

mnist_model = MnistEnsembleAveragingModel(3)

def hook_fun(module, grad_input, grad_output):
    print(f"{module.__class__.__name__} moving backwards")
for l in mnist_model.ensemble_model:
    l.register_backward_hook(hook_fun)

optimizer = torch.optim.Adam(mnist_model.parameters())

print("Beginning Training")
runner = OneCycleModelRunner(mnist_model, accuracy, optimizer, xentropy)
history = runner.train_model(train_loader, valid_loader, n_epochs=1)
graph_history(history)
print("Finished Training")

print(runner.run_model(test_loader))