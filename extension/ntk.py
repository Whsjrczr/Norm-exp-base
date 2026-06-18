import os

import numpy as np
import torch


class EmpiricalNTK:
    def __init__(self, result_path=None, metrics=None, save_eigvals=False):
        self.result_path = result_path
        self.metrics = metrics
        self.save_eigvals = save_eigvals

    @staticmethod
    def sample_loader_batch(loader, ntk_batch_size):
        if loader is None or ntk_batch_size <= 0:
            return None
        try:
            batch = next(iter(loader))
        except StopIteration:
            return None
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        inputs = inputs[: min(int(ntk_batch_size), int(inputs.shape[0]))]
        return inputs.detach().cpu()

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    def infer_output_dim(self, model, inputs):
        model = self._unwrap_model(model)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            y = model(inputs[:1])
        if was_training:
            model.train()
        return int(y.reshape(1, -1).shape[1])

    def compute_empirical_jacobian(self, model, inputs):
        model = self._unwrap_model(model)
        params = [param for param in model.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters were found for NTK analysis.")

        was_training = model.training
        model.eval()
        rows = []
        for i in range(inputs.shape[0]):
            xi = inputs[i : i + 1]
            yi = model(xi).reshape(-1)
            grads_per_output = []
            for out_idx in range(yi.numel()):
                grads = torch.autograd.grad(
                    yi[out_idx],
                    params,
                    retain_graph=out_idx + 1 < yi.numel(),
                    create_graph=False,
                    allow_unused=False,
                )
                grads_per_output.append(torch.cat([grad.reshape(-1) for grad in grads]))
            rows.append(torch.cat(grads_per_output).detach())
        if was_training:
            model.train()
        return torch.stack(rows, dim=0)

    @staticmethod
    def compute_ntk_spectrum(jacobian):
        with torch.no_grad():
            theta = jacobian @ jacobian.T
            theta = 0.5 * (theta + theta.T)
        eigvals = torch.linalg.eigvalsh(theta).detach().cpu().numpy()
        eigvals = np.clip(np.sort(eigvals)[::-1], a_min=0.0, a_max=None)
        trace = float(eigvals.sum())
        lambda_max = float(eigvals[0]) if eigvals.size else 0.0
        lambda_min = float(eigvals[-1]) if eigvals.size else 0.0
        denom = max(lambda_min, 1e-12)
        cond = lambda_max / denom if eigvals.size else float("inf")
        energy = trace if trace > 0 else 1.0
        cumsum = np.cumsum(eigvals) / energy
        eff_rank_90 = int(np.searchsorted(cumsum, 0.9) + 1) if eigvals.size else 0
        tol = max(lambda_max, 1.0) * 1e-12
        numerical_rank = int(np.sum(eigvals > tol))
        mean_self_kernel = trace / max(len(eigvals), 1)
        stable_rank = trace / max(lambda_max, 1e-12) if eigvals.size else 0.0
        return {
            "eigvals": eigvals,
            "cond": cond,
            "eff_rank_90": eff_rank_90,
            "numerical_rank": numerical_rank,
            "mean_self_kernel": mean_self_kernel,
            "trace": trace,
            "lambda_max": lambda_max,
            "lambda_min": lambda_min,
            "stable_rank": stable_rank,
        }

    def append_record(self, record, eigvals=None):
        if self.metrics is not None:
            self.metrics.log_ntk(record, eigvals=eigvals, save_eigvals=self.save_eigvals)
        elif self.save_eigvals and eigvals is not None and self.result_path is not None:
            eig_path = os.path.join(self.result_path, f"ntk_eigvals_{record['phase']}_{record['point_set']}.npy")
            np.save(eig_path, eigvals)

    def save_records(self):
        if self.metrics is not None:
            self.metrics.save_ntk_records()
