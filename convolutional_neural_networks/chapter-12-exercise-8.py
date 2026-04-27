from models.MnistModel import CLASS_COUNT, MnistModel
from model_runner.ModelRunner import OneCycleModelRunner
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

mnist_model = MnistModel()
optimizer = torch.optim.Adam(mnist_model.parameters())

print("Beginning Training")
runner = OneCycleModelRunner(mnist_model, accuracy, optimizer, xentropy)
history = runner.train_model(train_loader, valid_loader, n_epochs=10)
graph_history(history)
print("Finished Training")

print(runner.test_model(test_loader))