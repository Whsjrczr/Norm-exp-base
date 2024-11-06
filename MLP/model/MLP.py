import torch
import torch.nn as nn

import extension as ext


class MLP(nn.Module):
    def __init__(self, depth=4, width=100, input_size=28*28, output_size=10, **kwargs):
        super(MLP, self).__init__()
        layers = [ext.View(input_size), nn.Linear(input_size, width), ext.Norm(width), nn.ReLU(True)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(ext.Norm(width))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)
    

class PreNormMLP(nn.Module):
    def __init__(self, depth=4, width=100, input_size=28*28, output_size=10, **kwargs):
        super(PreNormMLP, self).__init__()
        layers = [ext.View(input_size), nn.Linear(input_size, width), nn.ReLU(True), ext.Norm(width)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU(True))
            layers.append(ext.Norm(width))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)
    
    
class MLPReLU(nn.Module):
    def __init__(self, depth=4, width=100, input_size=28 * 28, output_size=10, **kwargs):
        super(MLPReLU, self).__init__()
        layers = [ext.View(input_size), nn.Linear(input_size, width),nn.ReLU(True)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


class LinearModel(nn.Module):
    def __init__(self, depth=4, width=100, input_size=28*28, output_size=10, **kwargs):
        super(LinearModel, self).__init__()
        layers = [ext.View(input_size),nn.Linear(input_size, width), ext.Norm(width)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(ext.Norm(width))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.net(input)


class Linear(nn.Module):
    def __init__(self, depth=4, width=100, input_size=28*28, output_size=10, **kwargs):
        super(Linear, self).__init__()
        layers = [ext.View(input_size),nn.Linear(input_size, width)]
        for index in range(depth-1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, output_size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, input):
        return self.net(input)

