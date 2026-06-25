import argparse
import os
import re
import sys
import tempfile

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
            local_taiyi = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Taiyi")
            if os.path.isdir(local_taiyi) and local_taiyi not in sys.path:
                sys.path.append(local_taiyi)
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

    def log_ext(self, scalars, step=None):
        if not self.enabled or not scalars:
            return
        payload = {}
        if step is not None:
            payload["steps"] = int(step)
        payload.update(scalars)
        if self.visualization is not None:
            self.visualization.log_ext(payload)
        elif self.wandb is not None and getattr(self.wandb, "run", None) is not None:
            self.wandb.log(payload)

    def close(self):
        if self.visualization is not None:
            self.visualization.close()

    def _artifact_name(self):
        run = getattr(self.wandb, "run", None) if self.wandb is not None else None
        if run is not None and getattr(run, "id", None):
            base = f"taiyi-output-{run.id}"
        else:
            base = "taiyi-output"
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", base)

    def _upload_artifact(self, output):
        if self.wandb is None or getattr(self.wandb, "run", None) is None:
            return None
        with tempfile.TemporaryDirectory(prefix="taiyi_wandb_") as tmp_dir:
            output_path = os.path.join(tmp_dir, "taiyi_output.pt")
            torch.save(output, output_path)
            artifact = self.wandb.Artifact(
                self._artifact_name(),
                type="taiyi-output",
                metadata={
                    "monitor_config": str(self.monitor_config),
                    "output_format": "torch.save",
                },
            )
            artifact.add_file(output_path, name="taiyi_output.pt")
            logged_artifact = self.wandb.run.log_artifact(artifact)
            if hasattr(logged_artifact, "wait"):
                logged_artifact.wait()
            return getattr(logged_artifact, "name", artifact.name)

    def finish(self):
        info = {"taiyi_output": False, "taiyi_artifact": None}
        output = None
        if self.monitor is not None:
            output = self.monitor.get_output()
        if output is not None:
            info["taiyi_output"] = True
            info["taiyi_artifact"] = self._upload_artifact(output)
        self.close()
        return info


def setting(cfg, model=None, monitor_config=None, wandb=None, output_dir=None):
    return TaiyiTracker(cfg, model=model, monitor_config=monitor_config, wandb=wandb, output_dir=output_dir)
