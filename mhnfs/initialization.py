from torch import nn
import torch


def init_lecun(m):
    nn.init.normal_(m.weight, mean=0.0, std=torch.sqrt(torch.tensor([1.0]) / m.in_features).numpy()[0])
    nn.init.zeros_(m.bias)


def init_kaiming(m, nonlinearity):
    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nonlinearity)
    nn.init.zeros_(m.bias)


@torch.no_grad()
def init_weights(m, activation_function='linear'):
    if activation_function == 'relu':
        if type(m) == nn.Linear:
            init_kaiming(m, nonlinearity='relu')
    elif activation_function == "selu":
        if type(m) == nn.Linear:
            init_lecun(m)
    elif activation_function == 'linear':
        if type(m) == nn.Linear:
            init_lecun(m)
