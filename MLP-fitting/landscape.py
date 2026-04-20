#!/usr/bin/env python3
import argparse
import os
import sys

import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

import extension as ext
from fitting import OneLayerNN


class FittingLandscape:
    def __init__(self):
        self.cfg = self.add_arguments()
        self.checkpoint_cfg = self._checkpoint_cfg()
        self._merge_checkpoint_cfg()
        ext.normalization.setting(self.cfg)
        ext.activation.setting(self.cfg)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.loss_fn = nn.MSELoss()

        self.output_dir = self.cfg.output
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = OneLayerNN(self.cfg.in_dim, self.cfg.width).to(self.device)
        self.theta0 = ext.landscape.flatten_parameters(self.model)
        self.theta_v = self._load_target_state()
        self.loader = self._build_loader()

    def add_arguments(self):
        parser = argparse.ArgumentParser("MLP Fitting Landscape")
        parser.add_argument("--resume", required=True, type=ext.utils.path, help="checkpoint.pth path from fitting training")
        parser.add_argument("--best-model", default="", type=ext.utils.path, help="optional best.pth path; defaults to best.pth next to checkpoint")
        parser.add_argument("--output", default="./results/landscape", help="directory to save landscape outputs")
        parser.add_argument("--in-dim", type=int, default=None, help="override input dim")
        parser.add_argument("--num-samples", type=int, default=None, help="override total synthetic samples")
        parser.add_argument("--train-ratio", type=float, default=None, help="override train split ratio")
        parser.add_argument("--val-ratio", type=float, default=None, help="override val split ratio")
        parser.add_argument("--width", type=int, default=None, help="override hidden width")
        parser.add_argument("--batch-size", type=int, default=256, help="batch size for loss evaluation")
        parser.add_argument("--grid-size", type=int, default=41, help="grid resolution for landscape slice")
        parser.add_argument("--s-range", type=ext.utils.str2list, default=[-1.0 / 3.0, 4.0 / 3.0], help="comma-separated s range")
        parser.add_argument("--t-range", type=ext.utils.str2list, default=[-1.0 / 3.0, 4.0 / 3.0], help="comma-separated t range")
        parser.add_argument("--direction-seed", type=int, default=42, help="seed for orthogonal direction sampling")
        parser.add_argument("--max-batches", type=int, default=None, help="optional cap on evaluation batches")
        parser.add_argument("--plot-3d", action="store_true", help="also save a 3d surface plot")
        ext.normalization.add_arguments(parser)
        ext.activation.add_arguments(parser)
        args = parser.parse_args()
        if len(args.s_range) != 2 or len(args.t_range) != 2:
            raise ValueError("--s-range and --t-range must each contain exactly two values.")
        return args

    def _checkpoint_cfg(self):
        checkpoint = torch.load(self.cfg.resume, map_location="cpu")
        cfg = checkpoint.get("cfg")
        if cfg is None:
            raise ValueError("Checkpoint does not contain cfg; use checkpoint.pth from fitting training.")
        return cfg

    def _merge_checkpoint_cfg(self):
        for key in (
            "in_dim",
            "num_samples",
            "train_ratio",
            "val_ratio",
            "width",
            "norm",
            "norm_cfg",
            "activation",
            "activation_cfg",
        ):
            setattr(self.cfg, key, getattr(self.checkpoint_cfg, key, getattr(self.cfg, key, None)))

    def _resolved_cfg_value(self, key, default=None):
        override = getattr(self.cfg, key)
        if override is not None:
            return override
        return getattr(self.checkpoint_cfg, key, default)

    def _best_model_path(self):
        if self.cfg.best_model:
            return self.cfg.best_model
        return os.path.join(os.path.dirname(self.cfg.resume), "best.pth")

    def _load_target_state(self):
        best_model_path = self._best_model_path()
        state_dict = torch.load(best_model_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        return ext.landscape.flatten_parameters(self.model)

    def _build_loader(self):
        self.cfg.in_dim = self._resolved_cfg_value("in_dim", getattr(self.checkpoint_cfg, "in_dim", 8))
        self.cfg.num_samples = self._resolved_cfg_value("num_samples", getattr(self.checkpoint_cfg, "num_samples", 512))
        self.cfg.train_ratio = self._resolved_cfg_value("train_ratio", getattr(self.checkpoint_cfg, "train_ratio", 0.8))
        self.cfg.val_ratio = self._resolved_cfg_value("val_ratio", getattr(self.checkpoint_cfg, "val_ratio", 0.1))
        self.cfg.width = self._resolved_cfg_value("width", getattr(self.checkpoint_cfg, "width", 512))

        generator = torch.Generator().manual_seed(int(getattr(self.checkpoint_cfg, "seed", 0)))
        x = 2 * torch.rand(self.cfg.num_samples, self.cfg.in_dim, generator=generator) - 1
        y = 2 * torch.rand(self.cfg.num_samples, 1, generator=generator) - 1
        dataset = torch.utils.data.TensorDataset(x, y)

        train_size = int(self.cfg.num_samples * self.cfg.train_ratio)
        val_size = int(self.cfg.num_samples * self.cfg.val_ratio)
        test_size = self.cfg.num_samples - train_size - val_size
        split_generator = torch.Generator().manual_seed(int(getattr(self.checkpoint_cfg, "seed", 0)))
        train_set, _, _ = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=split_generator,
        )
        return torch.utils.data.DataLoader(
            train_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def run(self):
        self.model = OneLayerNN(self.cfg.in_dim, self.cfg.width).to(self.device)
        self.theta0 = ext.landscape.flatten_parameters(self.model)

        state_dict = torch.load(self._best_model_path(), map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.theta_v = ext.landscape.flatten_parameters(self.model)

        u, v = ext.landscape.random_isosceles_right_directions(
            self.theta0,
            self.theta_v,
            seed=self.cfg.direction_seed,
        )
        s_grid, t_grid, losses = ext.landscape.loss_landscape_slice(
            model=self.model,
            theta0=self.theta0,
            u=u,
            v=v,
            loss_fn=self.loss_fn,
            loader=self.loader,
            device=self.device,
            grid_size=self.cfg.grid_size,
            s_range=self.cfg.s_range,
            t_range=self.cfg.t_range,
            max_batches=self.cfg.max_batches,
        )

        ext.landscape.save_landscape_arrays(self.output_dir, s_grid, t_grid, losses, stem="loss_landscape")
        ext.landscape.plot_landscape_2d(
            s_grid,
            t_grid,
            losses,
            save_path=os.path.join(self.output_dir, "loss_landscape_2d.png"),
        )
        if self.cfg.plot_3d:
            ext.landscape.plot_landscape_3d(
                s_grid,
                t_grid,
                losses,
                save_path=os.path.join(self.output_dir, "loss_landscape_3d.png"),
            )


if __name__ == "__main__":
    runner = FittingLandscape()
    runner.run()
