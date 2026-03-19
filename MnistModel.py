from torch import nn

CLASS_COUNT = 10

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(1, 32, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2),
                                     nn.Flatten(),
                                     nn.Dropout(0.25),
                                     nn.Linear(1 * 28 * 28 * 32 // 2, 128),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(128, CLASS_COUNT)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
