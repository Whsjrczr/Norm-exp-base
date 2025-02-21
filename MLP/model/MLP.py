import torch
import torch.nn as nn

import extension as ext
from extension.my_modules.normalization import *


class MLP(nn.Module):
    def __init__(self, depth=4, width=100, input_size=28*28, output_size=10, **kwargs):
        super(MLP, self).__init__()
        layers = [ext.View(input_size), nn.Linear(input_size, width), ext.Norm(width), ext.Activation(width)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(ext.Norm(width))
            layers.append(ext.Activation(width))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class PreNormMLP(nn.Module):
    def __init__(self, depth=4, width=100, input_size=28*28, output_size=10, **kwargs):
        super(PreNormMLP, self).__init__()
        layers = [ext.View(input_size), nn.Linear(input_size, width), ext.Activation(width), ext.Norm(width)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(ext.Activation(width))
            layers.append(ext.Norm(width))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class CenDropScalingMLP(nn.Module):
    # LNCentering -> Dropout -> Scaling -> Activation
    def __init__(self, depth=4, width=100, input_size=28 * 28, output_size=10, dropout_prob=0, **kwargs):
        super(CenDropScalingMLP, self).__init__()
        layers = [ext.View(input_size), nn.Linear(input_size, width), ext.Activation(width), ext.Norm(width)]
        for index in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(LayerNormCentering(width, elementwise_affine=False))  # centering
            if (dropout_prob > 0):
                layers.append(nn.Dropout(dropout_prob))  # dropout
            layers.append(LayerNormScaling(width, elementwise_affine=True))  # scaling
            layers.append(ext.Activation(width))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)
