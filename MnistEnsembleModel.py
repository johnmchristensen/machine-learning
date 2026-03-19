from MnistModel import CLASS_COUNT, MnistModel
import torch
from torch import nn


class MnistEnsembleModel(nn.Module):
    def __init__(self, n_models):
        super().__init__()
        self.models = [MnistModel() for _ in range(n_models)]
        self.ensemble_model = nn.Sequential(nn.Flatten(),
                                            nn.Linear(CLASS_COUNT * n_models, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, CLASS_COUNT))
        for m in self.ensemble_model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        model_out = [m(x) for m in self.models]
        return self.ensemble_model(torch.cat(model_out, dim=1))