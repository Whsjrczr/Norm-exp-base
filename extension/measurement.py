import csv
import json
import os

import numpy as np
import torch


class MeasurementTracker:
    def __init__(self, result_path=None, visualizer=None, logger=None):
        self.result_path = result_path
        self.visualizer = visualizer
        self.logger = logger
        self.records = {}

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.item()
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(scalar):
            return None
        return scalar

    def log_scalars(self, scalars, step=None, epoch=None, prefix=None, vis_scalars=None, step_key="steps"):
        payload = {}
        if step is not None:
            payload[step_key] = int(step)
        if epoch is not None:
            payload["epochs"] = int(epoch)

        if scalars:
            for key, value in scalars.items():
                scalar = self._to_scalar(value)
                if scalar is None:
                    continue
                payload[f"{prefix}/{key}" if prefix else str(key)] = scalar

        if payload and self.visualizer is not None and getattr(self.visualizer, "wandb_enabled", False):
            self.visualizer.log(payload)

        if vis_scalars and self.visualizer is not None and getattr(self.visualizer, "visdom_enabled", False):
            for key, value in vis_scalars.items():
                scalar = self._to_scalar(value)
                if scalar is None:
                    continue
                self.visualizer.add_value(str(key), scalar)

    def log_classification(self, split, loss, accuracy=None, step=None, epoch=None, extra_scalars=None):
        scalars = {f"{split}_loss": loss}
        vis_scalars = {f"{split} loss": loss}
        if accuracy is not None:
            scalars[f"{split}_acc"] = accuracy
            vis_scalars[f"{split} accuracy"] = accuracy
        if extra_scalars:
            scalars.update(extra_scalars)
        self.log_scalars(scalars, step=step, epoch=epoch, vis_scalars=vis_scalars)

    def log_validation(self, name, value, step=None, epoch=None, wandb_key=None, vis_name=None, extra_scalars=None):
        scalars = {wandb_key or name: value}
        if extra_scalars:
            scalars.update(extra_scalars)
        self.log_scalars(
            scalars,
            step=step,
            epoch=epoch,
            vis_scalars={vis_name or name: value},
        )

    def append_record(self, group, record):
        self.records.setdefault(group, []).append(record)

    def save_records(self, group, stem=None):
        if self.result_path is None:
            return
        records = self.records.get(group, [])
        if not records:
            return
        stem = stem or group
        csv_path = os.path.join(self.result_path, f"{stem}.csv")
        fieldnames = []
        for record in records:
            for key in record.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        json_path = os.path.join(self.result_path, f"{stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def log_ntk(self, record, eigvals=None, save_eigvals=False):
        self.append_record("ntk", record)
        self.log_scalars(
            {
                "cond": record["cond"],
                "eff_rank_90": record["eff_rank_90"],
                "numerical_rank": record["numerical_rank"],
                "mean_self_kernel": record["mean_self_kernel"],
                "trace": record["trace"],
                "lambda_max": record["lambda_max"],
                "lambda_min": record["lambda_min"],
                "stable_rank": record["stable_rank"],
            },
            step=record.get("step"),
            prefix=f"ntk/{record['phase']}/{record['point_set']}",
            vis_scalars={
                f"ntk {record['phase']} {record['point_set']} cond": record["cond"],
                f"ntk {record['phase']} {record['point_set']} eff_rank_90": record["eff_rank_90"],
                f"ntk {record['phase']} {record['point_set']} numerical_rank": record["numerical_rank"],
                f"ntk {record['phase']} {record['point_set']} mean_self_kernel": record["mean_self_kernel"],
                f"ntk {record['phase']} {record['point_set']} trace": record["trace"],
                f"ntk {record['phase']} {record['point_set']} stable_rank": record["stable_rank"],
            },
        )
        if save_eigvals and eigvals is not None and self.result_path is not None:
            eig_path = os.path.join(self.result_path, f"ntk_eigvals_{record['phase']}_{record['point_set']}.npy")
            np.save(eig_path, eigvals)

    def save_ntk_records(self):
        self.save_records("ntk", stem="ntk_stats")


def setting(result_path=None, visualizer=None, logger=None):
    return MeasurementTracker(result_path=result_path, visualizer=visualizer, logger=logger)
