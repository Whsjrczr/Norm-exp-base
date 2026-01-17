import torch
import torch.nn as nn

class MLP(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            activation=torch.nn.ReLU,
            norm=torch.nn.BatchNorm1d,
    ):
        super(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-2], layers_hidden[1:-1]):
            self.layers.append(
                torch.nn.Linear(in_features,out_features)
            )
            self.layers.append(
                norm(out_features)
            )
            self.layers.append(
                activation()
            )
        self.layers.append(
            torch.nn.Linear(layers_hidden[-2], layers_hidden[-1])
        )

    def forward(self, x:torch.Tensor):
        for layers in self.layers:
            x = layers(x)
        return x
    

class MLP_drop(nn.Module):
    def __init__(
            self,
            layers_hidden,
            activation=torch.nn.ReLU,
            norm=torch.nn.BatchNorm1d,
            dropout_rate=0.05
    ):
        super(MLP_drop, self).__init__()

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-2], layers_hidden[1:-1]):
            self.layers.append(
                torch.nn.Linear(in_features,out_features)
            )
            self.layers.append(
                norm(out_features)
            )
            self.layers.append(
                activation()
            )
            self.layers.append(torch.nn.Dropout(p=dropout_rate))
        self.layers.append(
            torch.nn.Linear(layers_hidden[-2], layers_hidden[-1])
        )

    def forward(self, x:torch.Tensor):
        for layers in self.layers:
            x = layers(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, activation=nn.ReLU, norm=nn.BatchNorm1d):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = norm(hidden_dim)
        self.activation = activation()

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = x + residual  # Residual connection
        return self.activation(x)



class MLP_res(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            activation=torch.nn.ReLU,
            norm=torch.nn.BatchNorm1d,
    ):
        super(MLP_res, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(
            torch.nn.Linear(layers_hidden[0],layers_hidden[1])
        )
        self.layers.append(
            norm(layers_hidden[1])
        )
        self.layers.append(
            activation()
        )
        for in_features, out_features in zip(layers_hidden[1:-2], layers_hidden[2:-1]):
            self.layers.append(ResidualBlock(in_features, activation=activation, norm=norm))

        self.layers.append(
            torch.nn.Linear(layers_hidden[-2], layers_hidden[-1])
        )

    def forward(self, x:torch.Tensor):
        for layers in self.layers:
            x = layers(x)
        return x