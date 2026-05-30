import torch
import torch.nn as nn


class SinArctan(nn.Module):
    def __init__(self, num_features):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(x ** 2 + 1)