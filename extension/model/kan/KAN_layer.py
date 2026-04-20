import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.01,
        scale_spline: float = 0.3,
        scale_base: float = 0.3,
        enable_spline_scale: bool = False,
        base_activation=nn.SiLU,
        grid_range=None,
        grid_eps: float = 1e-4,
        eps: float = 1e-5,
        init: str = "origin",
        use_base_branch: bool = True,
        output_activation=None,
    ):
        super().__init__()
        if grid_range is None:
            grid_range = [-1, 1]

        self.in_feature = in_feature
        self.out_feature = out_feature
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_spline = scale_spline
        self.scale_base = scale_base
        self.enable_spline_scale = enable_spline_scale
        self.grid_eps = grid_eps
        self.eps = eps
        self.use_base_branch = use_base_branch

        self.base_activation = base_activation()
        self.output_activation = output_activation() if output_activation is not None else None

        grid_gap = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
            * grid_gap
            + grid_range[0]
        ).expand(in_feature, -1)
        self.register_buffer("grid", grid.contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_feature, in_feature))
        self.spline_weight = nn.Parameter(
            torch.empty(out_feature, in_feature, grid_size + spline_order)
        )
        if enable_spline_scale:
            self.spline_scaler = nn.Parameter(torch.empty(out_feature, in_feature))
        else:
            self.register_parameter("spline_scaler", None)

        self.reset_parameters(init=init)

    def reset_parameters(self, init: str = "origin"):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_feature, self.out_feature) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            coeff = self.curve2coeff(
                self.grid.T[self.spline_order : -self.spline_order],
                noise,
            )
            self.spline_weight.copy_(
                (self.scale_spline if not self.enable_spline_scale else 1.0) * coeff
            )
            if self.enable_spline_scale:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

        if init == "xavier":
            nn.init.xavier_uniform_(self.spline_weight, gain=nn.init.calculate_gain("relu"))
        elif init == "kaiming":
            nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    @property
    def scaled_spline_weight(self):
        if not self.enable_spline_scale:
            return self.spline_weight
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor):
        if x.size(-1) != self.in_feature:
            raise ValueError(f"Expected input dim {self.in_feature}, got {x.size(-1)}")

        original_shape = x.shape
        x = x.reshape(-1, self.in_feature)

        base_output = torch.zeros(
            x.size(0),
            self.out_feature,
            device=x.device,
            dtype=x.dtype,
        )
        if self.use_base_branch:
            base_output = F.linear(self.base_activation(x), self.base_weight)

        spline_output = F.linear(
            self.b_splines(x).reshape(x.size(0), -1),
            self.scaled_spline_weight.reshape(self.out_feature, -1),
        )

        combined_output = base_output + spline_output
        output = combined_output
        if self.output_activation is not None:
            output = self.output_activation(output)
        self.residual_states = {
            "default": {
                "stream": base_output.detach(),
                "branch": spline_output.detach(),
                "output": combined_output.detach(),
            }
        }
        return output.reshape(*original_shape[:-1], self.out_feature)

    @torch.no_grad()
    def normalize_weights(self):
        centered_base = self.base_weight - self.base_weight.mean(dim=1, keepdim=True)
        base_var = centered_base.var(dim=1, keepdim=True, unbiased=False)
        self.base_weight.copy_(centered_base / torch.sqrt(base_var + self.eps))

        spline_view = self.spline_weight.reshape(self.out_feature, -1)
        centered_spline = spline_view - spline_view.mean(dim=1, keepdim=True)
        spline_var = centered_spline.var(dim=1, keepdim=True, unbiased=False)
        normalized = centered_spline / torch.sqrt(spline_var + self.eps)
        self.spline_weight.copy_(normalized.reshape_as(self.spline_weight))

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        if x.dim() != 2 or x.size(1) != self.in_feature:
            raise ValueError("update_grid expects a 2D tensor with matching input dimension")

        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=x.dtype, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.copy_(self.curve2coeff(x, spline_output))

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ):
        l1_fake = self.spline_weight.abs().mean(-1)
        activation_loss = l1_fake.sum()
        prob = l1_fake / activation_loss.clamp_min(self.eps)
        entropy_loss = -(prob * prob.clamp_min(self.eps).log()).sum()
        return regularize_activation * activation_loss + regularize_entropy * entropy_loss

    def b_splines(self, x: torch.Tensor):
        if x.dim() != 2 or x.size(1) != self.in_feature:
            raise ValueError("b_splines expects a 2D tensor with matching input dimension")

        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, : -(k + 1)]
            left_den = grid[:, k:-1] - grid[:, : -(k + 1)]
            right_num = grid[:, k + 1 :] - x
            right_den = grid[:, k + 1 :] - grid[:, 1:(-k)]

            bases = (
                left_num / left_den.clamp_min(self.eps) * bases[:, :, :-1]
                + right_num / right_den.clamp_min(self.eps) * bases[:, :, 1:]
            )

        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        if x.dim() != 2 or x.size(1) != self.in_feature:
            raise ValueError("curve2coeff expects x with shape [N, in_feature]")
        if y.size() != (x.size(0), self.in_feature, self.out_feature):
            raise ValueError("curve2coeff expects y with shape [N, in_feature, out_feature]")

        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1)
        solution = torch.linalg.lstsq(a, b).solution
        return solution.permute(2, 0, 1).contiguous()


class KANLinearReLU(KANLinear):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("output_activation", nn.ReLU)
        super().__init__(*args, **kwargs)


class KANLinearNoRes(KANLinear):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_base_branch", False)
        super().__init__(*args, **kwargs)


class KANLinearWN(KANLinear):
    pass
