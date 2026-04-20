#!/usr/bin/env python3
import argparse
import os
import sys

os.environ["DDE_BACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
MLP_DIR = os.path.join(PROJECT_ROOT, "MLP")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if MLP_DIR not in sys.path:
    sys.path.append(MLP_DIR)

import extension as ext
from extension.utils import str2list
from pde_dataset import PDEBuilder


class PDELandscape:
    def __init__(self):
        self.cfg = self.add_arguments()
        self.checkpoint_cfg = self._checkpoint_cfg()
        self._merge_checkpoint_cfg()
        self.logger = ext.logger.setting("landscape.log", self.cfg.output, only_print=True)

        ext.normalization.setting(self.cfg)
        ext.activation.setting(self.cfg)
        self._set_precision()
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        os.makedirs(self.cfg.output, exist_ok=True)

        self._build_runtime()

    def add_arguments(self):
        parser = argparse.ArgumentParser("PDE Landscape")
        parser.add_argument("--resume", required=True, type=ext.utils.path, help="checkpoint.pth path from PDE training")
        parser.add_argument("--best-model", default="", type=ext.utils.path, help="optional best.pth path; defaults to best.pth next to checkpoint")
        parser.add_argument("--output", default="./results/pde-landscape", help="directory to save landscape outputs")
        parser.add_argument("--modes", type=str2list, default="train_loss,val_error", help="comma-separated landscape targets")
        parser.add_argument("--grid-size", type=int, default=41, help="grid resolution for landscape slice")
        parser.add_argument("--s-range", type=ext.utils.str2list, default=[-1.0 / 3.0, 4.0 / 3.0], help="comma-separated s range")
        parser.add_argument("--t-range", type=ext.utils.str2list, default=[-1.0 / 3.0, 4.0 / 3.0], help="comma-separated t range")
        parser.add_argument("--direction-seed", type=int, default=42, help="seed for orthogonal direction sampling")
        parser.add_argument("--plot-3d", action="store_true", help="also save a 3d surface plot")
        return parser.parse_args()

    def _checkpoint_cfg(self):
        checkpoint = torch.load(self.cfg.resume, map_location="cpu")
        cfg = checkpoint.get("cfg")
        if cfg is None:
            raise ValueError("Checkpoint does not contain cfg; use checkpoint.pth from PDE training.")
        return cfg

    def _merge_checkpoint_cfg(self):
        preserved = {
            "resume",
            "best_model",
            "output",
            "modes",
            "grid_size",
            "s_range",
            "t_range",
            "direction_seed",
            "plot_3d",
        }
        cli_output = os.path.abspath(self.cfg.output)
        cli_modes = list(self.cfg.modes)
        cli_grid = self.cfg.grid_size
        cli_s_range = list(self.cfg.s_range)
        cli_t_range = list(self.cfg.t_range)
        cli_direction_seed = self.cfg.direction_seed
        cli_plot_3d = self.cfg.plot_3d
        cli_best_model = self.cfg.best_model
        cli_resume = self.cfg.resume
        for key, value in vars(self.checkpoint_cfg).items():
            if key not in preserved:
                setattr(self.cfg, key, value)
        self.cfg.resume = cli_resume
        self.cfg.best_model = cli_best_model
        self.cfg.output = cli_output
        self.cfg.modes = cli_modes
        self.cfg.grid_size = cli_grid
        self.cfg.s_range = cli_s_range
        self.cfg.t_range = cli_t_range
        self.cfg.direction_seed = cli_direction_seed
        self.cfg.plot_3d = cli_plot_3d
        self.cfg.modes = [str(mode) for mode in self.cfg.modes]
        if len(self.cfg.s_range) != 2 or len(self.cfg.t_range) != 2:
            raise ValueError("--s-range and --t-range must each contain exactly two values.")

    def _set_precision(self):
        if getattr(self.cfg, "float64", False):
            dde.config.set_default_float("float64")
            torch.set_default_dtype(torch.float64)
            dde.backend.torch.torch.set_default_dtype(torch.float64)
        else:
            dde.config.set_default_float("float32")
            torch.set_default_dtype(torch.float32)
            dde.backend.torch.torch.set_default_dtype(torch.float32)

    def _build_runtime(self):
        ext.landscape.set_seed(getattr(self.checkpoint_cfg, "seed", None))
        self.model_net = ext.model.get_model(self.cfg).to(self.device)
        self.optimizer = ext.optimizer.setting(self.model_net, self.cfg)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.cfg)
        self.saver = ext.checkpoint.Checkpoint(
            self.model_net,
            self.cfg,
            self.optimizer,
            self.scheduler,
            save_dir="",
            save_to_disk=False,
        )

        builder = PDEBuilder(self.cfg, self.model_net, self.optimizer)
        self.data, self.net, self.model = builder.build()
        if hasattr(self.net, "to"):
            self.net = self.net.to(self.device)

        self.theta0 = ext.landscape.flatten_parameters(self.net)
        best_state = torch.load(self._best_model_path(), map_location="cpu", weights_only=True)
        self.net.load_state_dict(best_state, strict=True)
        self.theta_v = ext.landscape.flatten_parameters(self.net)

    def _best_model_path(self):
        if self.cfg.best_model:
            return self.cfg.best_model
        return os.path.join(os.path.dirname(self.cfg.resume), "best.pth")

    def _reload_theta_pair(self):
        ext.landscape.set_seed(getattr(self.checkpoint_cfg, "seed", None))
        self.model_net = ext.model.get_model(self.cfg).to(self.device)
        self.optimizer = ext.optimizer.setting(self.model_net, self.cfg)
        builder = PDEBuilder(self.cfg, self.model_net, self.optimizer)
        self.data, self.net, self.model = builder.build()
        if hasattr(self.net, "to"):
            self.net = self.net.to(self.device)
        self.theta0 = ext.landscape.flatten_parameters(self.net)
        best_state = torch.load(self._best_model_path(), map_location="cpu", weights_only=True)
        self.net.load_state_dict(best_state, strict=True)
        self.theta_v = ext.landscape.flatten_parameters(self.net)

    def _evaluate_train_loss(self):
        _outputs, losses = self.model.outputs_losses_train(
            self.data.train_x,
            self.data.train_y,
            self.data.train_aux_vars,
        )
        if torch.is_tensor(losses):
            return float(torch.sum(losses).detach().cpu().item())
        losses = np.asarray(losses)
        return float(np.sum(losses))

    def _reference_solution(self, x):
        if self.cfg.pde_type in {"poisson", "poisson_new"}:
            return (x**2 - 1) / 2
        if self.cfg.pde_type in {"helmholtz", "helmholtz_new", "helmholtz_learnable_2"}:
            return np.sin(np.pi * x[:, 0:1])
        if self.cfg.pde_type in {"allen_cahn", "allen_cahn_new"}:
            epsilon = 0.01
            return np.tanh(x[:, 0:1] / np.sqrt(2 * epsilon))
        return None

    def _predict_validation_points(self):
        if self.cfg.pde_type in {"poisson", "poisson_new", "helmholtz", "helmholtz_new", "helmholtz_learnable_2", "allen_cahn", "allen_cahn_new"}:
            x = np.linspace(-1, 1, 100).reshape(-1, 1)
            return x, self.model.predict(x), self._reference_solution(x)
        if self.cfg.pde_type in {"helmholtz2d", "helmholtz_2d"}:
            axis = np.linspace(-1, 1, 80)
            xx, yy = np.meshgrid(axis, axis)
            xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
            y_pred = self.model.predict(xy)
            y_true = np.sin(np.pi * xy[:, 0:1]) * np.sin(4 * np.pi * xy[:, 1:2])
            return xy, y_pred, y_true
        return None, None, None

    def _evaluate_val_error(self):
        _x, y_pred, y_true = self._predict_validation_points()
        if y_true is None:
            raise ValueError(f"val_error landscape is not available for pde_type={self.cfg.pde_type}")
        return float(np.mean((y_pred - y_true) ** 2))

    def _evaluate_scalar(self, mode):
        if mode == "train_loss":
            return self._evaluate_train_loss()
        if mode == "val_error":
            return self._evaluate_val_error()
        raise ValueError(f"Unsupported PDE landscape mode: {mode}")

    def _slice_mode(self, mode):
        self._reload_theta_pair()
        u, v = ext.landscape.random_isosceles_right_directions(
            self.theta0,
            self.theta_v,
            seed=self.cfg.direction_seed,
        )
        s_values = torch.linspace(float(self.cfg.s_range[0]), float(self.cfg.s_range[1]), steps=int(self.cfg.grid_size))
        t_values = torch.linspace(float(self.cfg.t_range[0]), float(self.cfg.t_range[1]), steps=int(self.cfg.grid_size))
        values = torch.zeros((len(t_values), len(s_values)), dtype=torch.double)
        original = ext.landscape.flatten_parameters(self.net)
        try:
            for j, t in enumerate(t_values):
                for i, s in enumerate(s_values):
                    theta = self.theta0 + float(s.item()) * u + float(t.item()) * v
                    ext.landscape.assign_parameters(self.net, theta)
                    values[j, i] = self._evaluate_scalar(mode)
        finally:
            ext.landscape.assign_parameters(self.net, original)

        s_grid, t_grid = np.meshgrid(s_values.cpu().numpy(), t_values.cpu().numpy())
        values_np = values.cpu().numpy()
        stem = f"landscape_{mode}"
        ext.landscape.save_landscape_arrays(self.cfg.output, s_grid, t_grid, values_np, stem=stem)
        ext.landscape.plot_landscape_2d(
            s_grid,
            t_grid,
            values_np,
            save_path=os.path.join(self.cfg.output, f"{stem}_2d.png"),
        )
        if self.cfg.plot_3d:
            ext.landscape.plot_landscape_3d(
                s_grid,
                t_grid,
                values_np,
                save_path=os.path.join(self.cfg.output, f"{stem}_3d.png"),
            )

    def run(self):
        for mode in self.cfg.modes:
            self.logger(f"==> building PDE landscape for {mode}")
            self._slice_mode(mode)


if __name__ == "__main__":
    runner = PDELandscape()
    runner.run()
