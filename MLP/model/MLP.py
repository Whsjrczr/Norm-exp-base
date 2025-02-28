import torch
import torch.nn as nn

import extension as ext
from extension.my_modules.normalization import *


class MLP(nn.Module):
    def __init__(self, depth=4, width=100, input_size=28*28, output_size=10, dropout_prob=0, **kwargs):
        super(MLP, self).__init__()
        layers = [ext.View(input_size), nn.Linear(input_size, width), ext.Norm(width), ext.Activation(width)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))  # dropout
            layers.append(ext.Norm(width))
            layers.append(ext.Activation(width))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class PreNormMLP(nn.Module):
    def __init__(self, depth=4, width=100, input_size=28*28, output_size=10, dropout_prob=0, **kwargs):
        super(PreNormMLP, self).__init__()
        layers = [ext.View(input_size), nn.Linear(input_size, width), ext.Activation(width), ext.Norm(width)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(ext.Activation(width))
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))  # dropout
            layers.append(ext.Norm(width))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class CenDropScalingMLP(nn.Module):
    # LNCentering -> Dropout -> Scaling -> Activation
    def __init__(self, depth=4, width=100, input_size=28 * 28, output_size=10, dropout_prob=0, **kwargs):
        super(CenDropScalingMLP, self).__init__()
        layers = [ext.View(input_size), nn.Linear(input_size, width), ext.Activation(width), ext.Norm(width,dropout_prob=dropout_prob)]
        for index in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(ext.Norm(width, dropout_prob=dropout_prob))
            # layers.append(LayerNormCentering(width, elementwise_affine=False))  # centering
            # if (dropout_prob > 0):
            #     layers.append(nn.Dropout(dropout_prob))  # dropout
            # layers.append(LayerNormScaling(width, elementwise_affine=True))  # scaling
            layers.append(ext.Activation(width))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class CenDropScalingPreNormMLP(nn.Module):
    # Activation -> LNCentering -> Dropout -> Scaling 
    def __init__(self, depth=4, width=100, input_size=28 * 28, output_size=10, dropout_prob=0, **kwargs):
        super(CenDropScalingPreNormMLP, self).__init__()
        layers = [ext.View(input_size), nn.Linear(input_size, width), ext.Activation(width), ext.Norm(width, dropout_prob=dropout_prob)]
        for index in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(ext.Activation(width))
            layers.append(ext.Norm(width,dropout_prob=dropout_prob))
            # layers.append(LayerNormCentering(width, elementwise_affine=True))  # centering
            # if (dropout_prob > 0):
            #     layers.append(nn.Dropout(dropout_prob))  # dropout
            # layers.append(LayerNormScaling(width, elementwise_affine=True))  # scaling
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)

# ResMLP
# - ┬ -> linear -> activation -> centering -> dropout -┬ -> scaling
#   └ ->   -------------------------------------   -> -┘
# 残差只能用forward写，所以需要单独建一个block


class ResBlockDropout(nn.Module):

    def __init__(self, width=100, dropout_prob=0, **kwargs):
        super(ResBlockDropout, self).__init__()
        self.fc1 = nn.Linear(width, width)
        self.activation = nn.ReLU()
        self.centering = LayerNormCentering(width, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.scaling = LayerNormScaling(width, elementwise_affine=True)

    def forward(self, x):
        identity = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.centering(x)
        x = self.dropout(x)
        x += identity
        return self.scaling(x)


class ResCenDropScalingMLP(nn.Module):
    def __init__(self, depth=4, width=100, input_size=28*28, output_size=10, dropout_prob=0,  **kwargs):
        super(ResCenDropScalingMLP, self).__init__()
        layers = [ext.View(input_size), nn.Linear(input_size, width), ext.Activation(width), ext.Norm(width, dropout_prob=dropout_prob)]
        for index in range(depth - 1):
            layers.append(ResBlockDropout(width,dropout_prob=dropout_prob))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)
    


class ResBlock(nn.Module):
    def __init__(self, width=100, **kwargs):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(width, width)
        self.activation = nn.ReLU()
        self.norm = ext.norm(width)

    def forward(self, x):
        identity = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.norm(x)
        x += identity
        return 


class ResMLP(nn.Module):
    def __init__(self, depth=4, width=100, input_size=28*28, output_size=10, **kwargs):
        super(ResMLP, self).__init__()
        layers = [ext.View(input_size), nn.Linear(input_size, width), ext.Activation(width), ext.Norm(width)]
        for index in range(depth - 1):
            layers.append(ResBlock(width))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)