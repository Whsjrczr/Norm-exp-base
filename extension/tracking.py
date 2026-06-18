import argparse
import os

import numpy as np
import torch


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Visualization Options")
    group.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    group.add_argument("--visualize", action="store_true", help="legacy alias of --wandb")
    group.add_argument("--wandb_project", type=str, default="test", help="wandb project name")
    group.add_argument("--visdom", action="store_true", help="enable Visdom visualization")
    group.add_argument("--vis", action="store_true", help="legacy alias of --visdom")
    group.add_argument("--visdom-port", "--vis-port", dest="vis_port", default=6006, type=int, help="Visdom port")
    group.add_argument("--visdom-env", "--vis-env", dest="vis_env", default=None, help="Visdom environment name")


def normalize_config(cfg: argparse.Namespace):
    cfg.wandb = bool(getattr(cfg, "wandb", False) or getattr(cfg, "visualize", False))
    cfg.visualize = cfg.wandb
    cfg.visdom = bool(getattr(cfg, "visdom", False) or getattr(cfg, "vis", False))
    cfg.vis = cfg.visdom
    if not hasattr(cfg, "wandb_project") or getattr(cfg, "wandb_project", None) is None:
        cfg.wandb_project = getattr(cfg, "subject_name", "test")
    return cfg


class VisdomVisualizer:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.viz = None
        self.env = None
        self.names = {}
        self.values = {}
        self.windows = {}
        self.cnt = {}
        self.num = {}

    @property
    def enabled(self):
        return bool(getattr(self.cfg, "visdom", False))

    def set(self, env_name, names: dict):
        if not self.enabled:
            return
        try:
            import visdom

            self.env = getattr(self.cfg, "vis_env", None) or env_name
            self.viz = visdom.Visdom(env=self.env, port=self.cfg.vis_port)
        except ImportError:
            print("You do not install visdom!!!!")
            self.cfg.visdom = False
            self.cfg.vis = False
            return
        self.names = names
        self.values = {}
        self.windows = {}
        self.cnt = {}
        self.num = {}
        for name, label in self.names.items():
            self.values[name] = 0
            self.cnt[label] = 0
            self.num[label] = 0
            self.windows.setdefault(label, [])
            self.windows[label].append(name)

        for label, window_names in self.windows.items():
            opts = dict(title=label, legend=window_names, showlegend=True)
            zero = np.ones((1, len(window_names)))
            self.viz.line(zero, zero, win=label, opts=opts)

    def add_value(self, name, value):
        if not self.enabled or self.viz is None:
            return
        if isinstance(value, torch.Tensor):
            assert value.numel() == 1
            value = value.item()
        self.values[name] = value
        label = self.names[name]
        self.cnt[label] += 1
        if self.cnt[label] == len(self.windows[label]):
            y = np.array([[self.values[window_name] for window_name in self.windows[label]]])
            x = np.ones_like(y) * self.num[label]
            opts = dict(
                title=label,
                legend=self.windows[label],
                showlegend=True,
                layoutopts={"plotly": {"legend": {"x": 0.05, "y": 1}}},
            )
            self.viz.line(y, x, update="append" if self.num[label] else "new", win=label, opts=opts)
            self.cnt[label] = 0
            self.num[label] += 1

    def clear(self, label):
        if not self.enabled:
            return
        self.num[label] = 0

    def add_images(self, images, title="images", win="images", nrow=8):
        if self.enabled and self.viz is not None:
            self.viz.images(images, win=win, nrow=nrow, opts={"title": title})

    def close(self):
        if self.viz:
            self.viz.save([self.env])

    def __del__(self):
        self.close()


class Visualizer:
    def __init__(
        self,
        cfg: argparse.Namespace,
        env_name: str = None,
        vis_names: dict = None,
        wandb_kwargs: dict = None,
    ):
        self.cfg = normalize_config(cfg)
        self.visdom = VisdomVisualizer(self.cfg)
        self.wandb = None
        self.run_dir = None

        if vis_names:
            self.visdom.set(env_name, vis_names)

        if not self.cfg.wandb:
            return

        try:
            import wandb as wandb_module
        except ImportError:
            print("You do not install wandb!!!!")
            self.cfg.wandb = False
            self.cfg.visualize = False
            return

        self.wandb = wandb_module
        if getattr(self.cfg, "offline", False):
            os.environ["WANDB_MODE"] = "offline"

        if wandb_kwargs is None:
            wandb_kwargs = {}
        self.wandb.init(**wandb_kwargs)
        self.run_dir = os.path.dirname(self.wandb.run.dir)

    @property
    def wandb_enabled(self):
        return self.wandb is not None

    @property
    def visdom_enabled(self):
        return self.visdom.enabled

    @property
    def run_id(self):
        if self.wandb_enabled and self.wandb.run is not None:
            return self.wandb.run.id
        return None

    def log(self, data: dict):
        if self.wandb_enabled:
            self.wandb.log(data)

    def update_wandb_config(self, data: dict):
        if self.wandb_enabled and self.wandb.run is not None and data:
            self.wandb.config.update(data, allow_val_change=True)

    def update_wandb_summary(self, data: dict):
        if self.wandb_enabled and self.wandb.run is not None and data:
            for key, value in data.items():
                self.wandb.run.summary[key] = value

    def add_value(self, name, value):
        self.visdom.add_value(name, value)

    def add_images(self, images, title="images", win="images", nrow=8):
        self.visdom.add_images(images, title=title, win=win, nrow=nrow)

    def clear(self, label):
        self.visdom.clear(label)

    def close(self):
        self.visdom.close()

    def finish(self, sync_offline=False):
        info = {"synced": False, "wandb_finished": False}
        self.close()
        wandb_enabled = self.wandb_enabled
        run_dir = self.run_dir
        if wandb_enabled:
            try:
                self.wandb.finish()
                info["wandb_finished"] = True
            except Exception as exc:
                print(f"WandB finish failed: {exc}")
            finally:
                self.wandb = None
        if sync_offline and wandb_enabled and run_dir:
            os.system(f"wandb sync {run_dir}")
            info["synced"] = True
        return info


def setting(
    cfg: argparse.Namespace,
    env_name: str = None,
    vis_names: dict = None,
    wandb_kwargs: dict = None,
):
    return Visualizer(
        cfg,
        env_name=env_name,
        vis_names=vis_names,
        wandb_kwargs=wandb_kwargs,
    )


def setting_visdom(cfg: argparse.Namespace, env_name: str, names: dict):
    normalize_config(cfg)
    vis = VisdomVisualizer(cfg)
    vis.set(env_name, names)
    return vis
