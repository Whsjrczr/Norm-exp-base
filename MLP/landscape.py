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

import extension as ext


class MLPLandscape:
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
        self.loss_fn = nn.CrossEntropyLoss()
        self.output_dir = self.cfg.output
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = ext.logger.setting("landscape.log", self.output_dir, only_print=True)

        ext.landscape.set_seed(getattr(self.checkpoint_cfg, "seed", None))
        self.model = ext.model.get_model(self.cfg).to(self.device)
        self.theta0 = ext.landscape.flatten_parameters(self.model)
        self.theta_v = self._load_target_state()
        self.loader = self._build_loader()

    def add_arguments(self):
        parser = argparse.ArgumentParser("MLP Landscape")
        parser.add_argument("--resume", required=True, type=ext.utils.path, help="checkpoint.pth path from MLP training")
        parser.add_argument("--best-model", default="", type=ext.utils.path, help="optional best.pth path; defaults to best.pth next to checkpoint")
        parser.add_argument("--output", default="./results/mlp-landscape", help="directory to save landscape outputs")
        parser.add_argument("--split", choices=["train", "val"], default="train", help="dataset split used for loss evaluation")
        parser.add_argument("--batch-size", type=int, default=256, help="batch size for loss evaluation")
        parser.add_argument("--grid-size", type=int, default=41, help="grid resolution for landscape slice")
        parser.add_argument("--s-range", type=ext.utils.str2list, default=[-1.0 / 3.0, 4.0 / 3.0], help="comma-separated s range")
        parser.add_argument("--t-range", type=ext.utils.str2list, default=[-1.0 / 3.0, 4.0 / 3.0], help="comma-separated t range")
        parser.add_argument("--direction-seed", type=int, default=42, help="seed for orthogonal direction sampling")
        parser.add_argument("--max-batches", type=int, default=None, help="optional cap on evaluation batches")
        parser.add_argument("--plot-3d", action="store_true", help="also save a 3d surface plot")
        args = parser.parse_args()
        if len(args.s_range) != 2 or len(args.t_range) != 2:
            raise ValueError("--s-range and --t-range must each contain exactly two values.")
        return args

    def _checkpoint_cfg(self):
        checkpoint = torch.load(self.cfg.resume, map_location="cpu")
        cfg = checkpoint.get("cfg")
        if cfg is None:
            raise ValueError("Checkpoint does not contain cfg; use checkpoint.pth from MLP training.")
        return cfg

    def _merge_checkpoint_cfg(self):
        for key, value in vars(self.checkpoint_cfg).items():
            if not hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
        # force dataset loader to use a simple fixed batch size for analysis
        self.cfg.batch_size = [self.cfg.batch_size, self.cfg.batch_size]

    def _best_model_path(self):
        if self.cfg.best_model:
            return self.cfg.best_model
        return os.path.join(os.path.dirname(self.cfg.resume), "best.pth")

    def _load_target_state(self):
        state_dict = torch.load(self._best_model_path(), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        return ext.landscape.flatten_parameters(self.model)

    def _build_loader(self):
        train = self.cfg.split == "train"
        return ext.dataset.get_dataset_loader(self.cfg, train=train, use_cuda=False)

    def run(self):
        ext.landscape.set_seed(getattr(self.checkpoint_cfg, "seed", None))
        self.model = ext.model.get_model(self.cfg).to(self.device)
        self.theta0 = ext.landscape.flatten_parameters(self.model)

        state_dict = torch.load(self._best_model_path(), map_location="cpu", weights_only=True)
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
    runner = MLPLandscape()
    runner.run()
