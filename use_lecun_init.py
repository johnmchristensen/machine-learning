from torch import nn


def use_lecun_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(module.bias)