from torch import nn
import torch

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

class MnistEnsembleModel(nn.Module):
    def __init__(self, n_models, dropout_rate = 0, inner_layer_connections = 64, inner_layer_count = 1):
        super().__init__()
        self.models = nn.ModuleList([MnistModel() for _ in range(n_models)])

        final_model = [nn.Flatten(),
                        nn.Linear(CLASS_COUNT * n_models, inner_layer_connections),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)]
        for i in range(inner_layer_count):
            final_model.append(nn.Linear(inner_layer_connections, inner_layer_connections))
            final_model.append(nn.ReLU())
            final_model.append(nn.Dropout(dropout_rate))

        final_model = final_model + [nn.Linear(inner_layer_connections, CLASS_COUNT)]

        self.ensemble_model = nn.Sequential(*final_model)
        
        for m in self.ensemble_model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        model_out = [m(x) for m in self.models]
        return self.ensemble_model(torch.cat(model_out, dim=1))

class MnistEnsembleAveragingModel(nn.Module):
    def __init__(self, n_models):
        super().__init__()
        self.models = nn.ModuleList([MnistModel() for _ in range(n_models)])

    def forward(self, x):
        model_out = [m(x) for m in self.models]
        return torch.stack(model_out).mean(dim=0)
