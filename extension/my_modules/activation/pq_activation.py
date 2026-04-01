import torch
import torch.nn as nn


class PQActivation(nn.Module):
    def __init__(self, num_features=None, p=2, q=2):
        super().__init__()
        if p < 1 or q < 1:
            raise ValueError(f"PQActivation expects p >= 1 and q >= 1, but got p={p}, q={q}.")
        self.p = p
        self.q = q

    def forward(self, x):
        ratio = self.p / self.q
        numerator = torch.sign(x) * torch.abs(x).pow(ratio)
        denominator = (torch.abs(x).pow(self.p) + 1.0).pow(1.0 / self.q)
        return numerator / denominator

    def extra_repr(self):
        return f"p={self.p}, q={self.q}"
