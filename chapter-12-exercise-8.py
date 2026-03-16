import torch
import torchmetrics
import torchvision
import torchvision.transforms.v2 as T
from torch import nn
from torch.utils.data import DataLoader
from MnistModel import MnistModel
from OneCycleModelRunner import OneCycleModelRunner
from graph_history import graph_history

toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32)])

train_and_valid_data = torchvision.datasets.MNIST(root="datasets", train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.MNIST(root="datasets", train=False, download=True, transform=toTensor)

torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(train_and_valid_data, [55_000, 5_000])
print("Data Loaded")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

torch.manual_seed(42)

mnist_model = MnistModel()
optimizer = torch.optim.Adam(mnist_model.parameters())
xentropy = nn.CrossEntropyLoss()
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

print("Beginning Training")
runner = OneCycleModelRunner (mnist_model, accuracy, optimizer, xentropy)
history = runner.train_model(train_loader, valid_loader, n_epochs=10)
graph_history(history)
print("Finished Training")

print(runner.run_model(test_loader))
