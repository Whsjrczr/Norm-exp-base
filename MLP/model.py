import torch
import torch.nn as nn

import extension as ext


class MLP(nn.Module):
    def __init__(self, depth=4, width=100, **kwargs):
        super(MLP, self).__init__()
        layers = [ext.View(32 * 32), nn.Linear(32 * 32, width), ext.Norm(width), nn.ReLU(True)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(ext.Norm(width))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)
    
    
class MLPReLU(nn.Module):
    def __init__(self, depth=4, width=100, **kwargs):
        super(MLPReLU, self).__init__()
        layers = [ext.View(32 * 32), nn.Linear(32 * 32, width),nn.ReLU(True)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)
    
    
class MLPM(nn.Module):
    def __init__(self, depth=4, width=100, **kwargs):
        super(MLPM, self).__init__()
        layers = [ext.View(28 * 28), nn.Linear(28*28, width), ext.Norm(width), nn.ReLU(True)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(ext.Norm(width))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class LinearModel(nn.Module):
    def __init__(self, depth=4, width=100, **kwargs):
        super(LinearModel, self).__init__()
        layers = [ext.View(32 * 32),nn.Linear(32 * 32, width), ext.Norm(width)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(ext.Norm(width))
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)
    def forward(self, input):
        return self.net(input)


class Linear(nn.Module):
    def __init__(self, depth=4, width=100, **kwargs):
        super(Linear, self).__init__()
        layers = [ext.View(32 * 32),nn.Linear(32 * 32, width)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)
    def forward(self, input):
        return self.net(input)

