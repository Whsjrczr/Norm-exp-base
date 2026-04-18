import argparse
import os

import torch


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Taiyi Options")
    group.add_argument("--taiyi", action="store_true", help="enable Taiyi monitor bridge")


class TaiyiTracker:
    def __init__(self, cfg, model=None, monitor_config=None, wandb=None, output_dir=None):
        self.cfg = cfg
        self.model = model
        self.monitor_config = monitor_config or {}
        self.wandb = wandb
        self.output_dir = output_dir
        self.monitor = None
        self.visualization = None

        if not getattr(self.cfg, "taiyi", False):
            return

        try:
            from Taiyi.taiyi.monitor import Monitor
        except ImportError:
            print("You do not install Taiyi!!!!")
            self.cfg.taiyi = False
            return

        self.monitor = Monitor(self.model, self.monitor_config)
        if self.wandb is not None and getattr(self.wandb, "run", None) is not None:
            from Taiyi.visualize import Visualization
            self.visualization = Visualization(self.monitor, self.wandb)

    @property
    def enabled(self):
        return self.monitor is not None

    def track(self, step):
        if not self.enabled:
            return
        self.monitor.track(step)
        if self.visualization is not None:
            self.visualization.show(step)

    def close(self):
        if self.visualization is not None:
            self.visualization.close()

    def finish(self):
        info = {"taiyi_output": False, "taiyi_output_path": None}
        self.close()
        output = None
        if self.monitor is not None:
            output = self.monitor.get_output()
        if output is not None:
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                output_path = os.path.join(self.output_dir, "taiyi_output.pt")
                torch.save(output, output_path)
                info["taiyi_output_path"] = output_path
            info["taiyi_output"] = True
        return info


def setting(cfg, model=None, monitor_config=None, wandb=None, output_dir=None):
    return TaiyiTracker(cfg, model=model, monitor_config=monitor_config, wandb=wandb, output_dir=output_dir)
