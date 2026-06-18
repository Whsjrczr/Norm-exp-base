import torch
import torch.nn as nn

from .KAN_layer import KANLinear


class KANNetwork(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=0.3,
        scale_spline=0.1,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=None,
        upgrade_grid=False,
        norm=nn.Identity,
        weight_norm=False,
        init="origin",
        dropout_rate=0.0,
        use_base_branch=True,
    ):
        super().__init__()
        if len(layers_hidden) < 2:
            raise ValueError("layers_hidden must contain at least input and output dimensions")
        if grid_range is None:
            grid_range = [-1, 1]

        self.update_grid = upgrade_grid
        self.weight_norm = weight_norm
        self.layers_hidden = list(layers_hidden)

        self.layers = nn.ModuleList()
        hidden_pairs = list(zip(layers_hidden[:-2], layers_hidden[1:-1]))
        for in_features, out_features in hidden_pairs:
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
                    use_base_branch=use_base_branch,
                )
            )
            self.layers.append(norm(out_features))
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(p=dropout_rate))

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
                use_base_branch=use_base_branch,
            )
        )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            if isinstance(layer, KANLinear):
                flat_x = x.reshape(-1, x.size(-1))
                if self.update_grid and self.training:
                    layer.update_grid(flat_x)
                if self.weight_norm:
                    layer.normalize_weights()
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
            if isinstance(layer, KANLinear)
        )


class KAN_norm(KANNetwork):
    pass
