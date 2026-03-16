import torch
from torch import nn

from get_device import get_device
from use_he_init import use_he_init
from use_lecun_init import use_lecun_init


def build_deep_model(n_hidden, n_neurons, n_inputs, n_outputs):
    layers = [nn.Flatten(), nn.Linear(n_inputs, n_neurons), nn.SiLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(n_neurons, n_neurons), nn.SiLU()]

    layers += [nn.Linear(n_neurons, n_outputs)]
    model = torch.nn.Sequential(*layers)
    model.apply(use_he_init)
    return model.to(get_device())

def build_deep_batch_normalized_model(n_hidden = 20, n_neurons = 100, n_inputs = 3*32*32, n_outputs=10):
    layers = [nn.Flatten(), nn.Linear(n_inputs, n_neurons), nn.BatchNorm1d(n_neurons), nn.SiLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(n_neurons, n_neurons),
                   nn.BatchNorm1d(n_neurons),
                   nn.SiLU()]

    layers += [nn.Linear(n_neurons, n_outputs)]
    model = torch.nn.Sequential(*layers)
    model.apply(use_he_init)
    return model.to(get_device())

def build_deep_model_with_selu(standardize, n_hidden = 20, n_neurons = 100, n_inputs = 3*32*32, n_outputs=10):
    layers = [nn.Flatten(), standardize, nn.Linear(n_inputs, n_neurons), nn.SELU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(n_neurons, n_neurons), nn.SELU()]

    layers += [nn.Linear(n_neurons, n_outputs)]
    model = torch.nn.Sequential(*layers)
    model.apply(use_lecun_init)
    return model.to(get_device())

def build_deep_model_with_alpha_dropout(standardize, n_hidden=20, n_neurons=100, n_inputs=3*32*32, n_outputs=10, dropout_rate=0.1):
    layers = [nn.Flatten(), standardize, nn.Linear(n_inputs, n_neurons), nn.SELU(), nn.AlphaDropout(dropout_rate)]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(n_neurons, n_neurons), nn.SELU(), nn.AlphaDropout(dropout_rate)]

    layers += [nn.Linear(n_neurons, n_outputs)]
    model = torch.nn.Sequential(*layers)
    model.apply(use_lecun_init)
    return model.to(get_device())