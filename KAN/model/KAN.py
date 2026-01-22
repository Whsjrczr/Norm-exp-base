import torch
import torch.nn as nn
import torch.nn.functional as F
from .KAN_layer import KANLinear


class KAN_norm(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=0.3,
        scale_spline=0.1,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        upgrade_grid=False,
        norm=torch.nn.BatchNorm1d,
        weight_norm=False,
        init = 'origin'
    ):
        super(KAN_norm, self).__init__()
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数
        self.update_grid = upgrade_grid
        self.norm = norm
        self.weight_norm = weight_norm

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-2], layers_hidden[1:-1]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    init=init,

                )
            )
            self.layers.append(
                norm(out_features)
            )
        self.layers.append(
            KANLinear(
                layers_hidden[-2],
                layers_hidden[-1],
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                init=init,
            )
        )



    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否在前向传播过程中更新网格。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if layer._get_name() == "KANLinear":
                if self.update_grid and self.train():
                    layer.update_grid(x)
                if self.weight_norm:
                    layer.weight_norm()
            # print(4)
            x = layer(x)
            # print(5)
        # print(6)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_activation (float): 激活正则化系数。
            regularize_entropy (float): 熵正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    
