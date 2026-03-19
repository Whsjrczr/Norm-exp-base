import argparse


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Taiyi Options")
    group.add_argument("--taiyi", action="store_true", help="enable Taiyi monitor bridge")


class TaiyiTracker:
    def __init__(self, cfg, model=None, monitor_config=None, wandb=None):
        self.cfg = cfg
        self.model = model
        self.monitor_config = monitor_config or {}
        self.wandb = wandb
        self.monitor = None
        self.visualization = None

        if not getattr(self.cfg, "taiyi", False):
            return
        if self.wandb is None or getattr(self.wandb, "run", None) is None:
            return

        try:
            from Taiyi.taiyi.monitor import Monitor
            from Taiyi.visualize import Visualization
        except ImportError:
            print("You do not install Taiyi!!!!")
            self.cfg.taiyi = False
            return

        self.monitor = Monitor(self.model, self.monitor_config)
        self.visualization = Visualization(self.monitor, self.wandb)

    @property
    def enabled(self):
        return self.monitor is not None and self.visualization is not None

    def track(self, step):
        if not self.enabled:
            return
        self.monitor.track(step)
        self.visualization.show(step)

    def close(self):
        if self.visualization is not None:
            self.visualization.close()

    def finish(self):
        info = {"taiyi_output": False}
        self.close()
        if self.monitor is not None:
            self.monitor.get_output()
            info["taiyi_output"] = True
        return info


def setting(cfg, model=None, monitor_config=None, wandb=None):
    return TaiyiTracker(cfg, model=model, monitor_config=monitor_config, wandb=wandb)
