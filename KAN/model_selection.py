import torch
from model.KAN_layer import KAN_norm
from model.MLP import MLP

def get_model(model_name, layers_hidden, norm, update_grid):
    if model_name == "KAN":
        return KAN_norm(
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            norm=norm,
            upgrade_grid=update_grid
        )
    elif model_name == "MLP":
        return MLP(layers_hidden, norm=norm)
    else:
        raise ValueError("Invalid model name. Choose 'KAN' or 'MLP'.")