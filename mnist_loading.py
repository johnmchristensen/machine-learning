import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

def standard():
    return load(T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]))


def with_image_augmentation():
    return load(T.Compose([T.ToImage(),
                           T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
                           T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                           T.ToDtype(torch.float32, scale=True)]))

def load(transformer):
    train_and_valid_data = torchvision.datasets.MNIST(root="datasets", train=True, download=True, transform=transformer)
    test_data = torchvision.datasets.MNIST(root="datasets", train=False, download=True, transform=transformer)

    torch.manual_seed(42)
    train_data, valid_data = torch.utils.data.random_split(train_and_valid_data, [55_000, 5_000])

    return (DataLoader(train_data, batch_size=32, shuffle=True),
            DataLoader(valid_data, batch_size=32),
            DataLoader(test_data, batch_size=32))
