#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import extension as ext
from cifar10 import MNIST


class MLPNTKTrainer(MNIST):
    def __init__(self):
        super().__init__()
        self.step = int(getattr(self, "step", 0))
        self.ntk = ext.ntk.EmpiricalNTK(
            result_path=self.result_path,
            metrics=self.metrics,
            save_eigvals=self.cfg.ntk_save_eigvals,
        )
        self._ntk_point_sets = self._collect_ntk_point_sets()

    def add_arguments(self):
        parser = argparse.ArgumentParser("MLP Classification with NTK Analysis")
        ext.model.add_model_arguments(parser, task="classification", default_family="mlp")
        parser.add_argument("--offline", "-offline", action="store_true", help="offline mode")

        ext.trainer.add_arguments(parser)
        parser.set_defaults(epochs=500)
        ext.dataset.add_arguments(parser)
        parser.set_defaults(dataset="cifar10", workers=1, batch_size=[64, 1000])
        ext.scheduler.add_arguments(parser)
        parser.set_defaults(lr_method="fix", lr=1e-3)
        ext.optimizer.add_arguments(parser)
        parser.set_defaults(optimizer="adam", weight_decay=1e-5)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.normalization.add_arguments(parser)
        ext.activation.add_arguments(parser)
        ext.tracking.add_arguments(parser)
        ext.taiyi.add_arguments(parser)

        parser.add_argument(
            "--ntk-batch-size",
            type=int,
            default=8,
            help="number of samples taken from each loader for NTK analysis",
        )
        parser.add_argument(
            "--ntk-when",
            type=ext.utils.str2list,
            default="init,final",
            help="when to run NTK analysis: init,final,train",
        )
        parser.add_argument(
            "--ntk-track-every",
            type=int,
            default=0,
            help="track NTK every N optimizer steps during training; <=0 disables periodic tracking",
        )
        parser.add_argument(
            "--ntk-save-eigvals",
            action="store_true",
            help="save NTK eigenvalues to .npy files",
        )

        args = parser.parse_args()
        if args.resume:
            args = parser.parse_args(namespace=ext.checkpoint.Checkpoint.load_config(args.resume))
        stages = getattr(ext.optimizer, "get_stages", lambda _cfg: None)(args)
        stage_total_epochs = getattr(ext.optimizer, "infer_total_epochs", lambda _stages: None)(stages)
        if stage_total_epochs is not None:
            args.epochs = stage_total_epochs
        return ext.tracking.normalize_config(args)

    def train(self):
        ntk_when = {str(item).lower() for item in getattr(self.cfg, "ntk_when", [])}
        if "init" in ntk_when:
            self.run_ntk_analysis("init", step=0)

        if self.cfg.test:
            self.validate()
            if "final" in ntk_when:
                self.run_ntk_analysis("final", step=getattr(self, "step", 0))
            self._finish_and_copy_log()
            return

        if not hasattr(self, "step"):
            self.step = 0

        stage_plan = self._epoch_stage_plan()
        multistage_enabled = (len(stage_plan) > 1) or (stage_plan and stage_plan[0][1] is not None)
        if multistage_enabled:
            self.logger(f"==> Multi-stage training enabled: {len(stage_plan)} segments")

        for si, st, epoch_begin, epoch_end in stage_plan:
            self._apply_stage_to_cfg(st)
            self._rebuild_optim_sched_and_sync_saver()

            self.logger(
                f"==> Stage {si}: optimizer={self.cfg.optimizer}, lr={getattr(self.cfg, 'lr', None)}, "
                f"wd={getattr(self.cfg, 'weight_decay', None)}, epochs=[{epoch_begin},{epoch_end})"
            )

            for epoch in range(epoch_begin, epoch_end):
                if self.cfg.lr_method != "auto":
                    self.scheduler.step()

                self.train_epoch(epoch)

                self.metrics.log_scalars(
                    {"learning_rate": self.scheduler.get_last_lr()[0], "stage": si},
                    step=self.step,
                    epoch=epoch,
                )

                accuracy, val_loss = self.validate(epoch)

                if self.visualizer.wandb_enabled:
                    self.saver.save_checkpoint(epoch=epoch, best_acc=self.best_acc, wandb_id=self.visualizer.run_id, step=self.step)
                else:
                    self.saver.save_checkpoint(epoch=epoch, best_acc=self.best_acc)

                if self.cfg.lr_method == "auto":
                    self.scheduler.step(val_loss)

        if "final" in ntk_when:
            self.run_ntk_analysis("final", step=self.step)
        self._finish_and_copy_log()

    def train_epoch(self, epoch):
        self.logger(
            "\nEpoch: {}, lr: {:.2g}, weight decay: {:.2g} on model {}".format(
                epoch,
                self.optimizer.param_groups[0]["lr"],
                self.optimizer.param_groups[0]["weight_decay"],
                self.model_name,
            )
        )
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        progress_bar = ext.ProgressBar(len(self.train_loader))
        ntk_when = {str(item).lower() for item in getattr(self.cfg, "ntk_when", [])}

        for i, (inputs, targets) in enumerate(self.train_loader, 1):
            inputs = inputs.to(self.device)
            targets = inputs if self.cfg.arch == "AE" else targets.to(self.device)

            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            self.taiyi.track(self.step)
            self.step += 1

            if "train" in ntk_when and self.cfg.ntk_track_every > 0 and self.step % self.cfg.ntk_track_every == 0:
                self.run_ntk_analysis("train", step=self.step)

            train_loss += losses.item() * targets.size(0)
            if self.cfg.arch == "AE":
                correct = -train_loss
            else:
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(targets.view_as(pred)).sum().item()
            total += targets.size(0)
            if i % 10 == 0 or i == len(self.train_loader):
                progress_bar.step("Loss: {:.5g} | Accuracy: {:.2f}%".format(train_loss / total, 100.0 * correct / total), 10)

        train_loss /= total
        accuracy = 100.0 * correct / total

        self.metrics.log_classification("train", train_loss, accuracy=accuracy, step=self.step, epoch=epoch)

        self.logger(
            "Train on epoch {}: average loss={:.5g}, accuracy={:.2f}% ({}/{}), time: {}".format(
                epoch, train_loss, accuracy, correct, total, progress_bar.time_used()
            )
        )

    def run_ntk_analysis(self, phase, step=None):
        point_sets = self._ntk_point_sets or {}
        if not point_sets:
            self.logger(f"==> NTK [{phase}] skipped: no available sampling batches.")
            return

        if step is None:
            step = int(getattr(self, "step", -1))

        self.logger(f"==> NTK [{phase}] analyzing {len(point_sets)} batch sets.")
        for set_name, batch in point_sets.items():
            inputs = batch.to(self.device)
            jacobian = self._compute_empirical_jacobian(inputs)
            stats = self._compute_ntk_spectrum(jacobian)
            output_dim = self._infer_output_dim(inputs)
            record = {
                "phase": phase,
                "step": int(step),
                "point_set": set_name,
                "arch": str(self.cfg.arch),
                "norm": str(self.cfg.norm),
                "depth": int(self.cfg.depth),
                "width": int(self.cfg.width),
                "n_points": int(inputs.shape[0]),
                "input_dim": int(inputs[0].numel()),
                "output_dim": int(output_dim),
                "cond": float(stats["cond"]),
                "eff_rank_90": int(stats["eff_rank_90"]),
                "numerical_rank": int(stats["numerical_rank"]),
                "mean_self_kernel": float(stats["mean_self_kernel"]),
                "trace": float(stats["trace"]),
                "lambda_max": float(stats["lambda_max"]),
                "lambda_min": float(stats["lambda_min"]),
                "stable_rank": float(stats["stable_rank"]),
            }
            self.ntk.append_record(record, eigvals=stats["eigvals"])
            self.logger(
                f"==> NTK [{phase}/{set_name}] cond={record['cond']:.3e}, "
                f"eff_rank_90={record['eff_rank_90']}, numerical_rank={record['numerical_rank']}, "
                f"trace={record['trace']:.3e}, stable_rank={record['stable_rank']:.3e}"
            )
        self.ntk.save_records()

    def _collect_ntk_point_sets(self):
        point_sets = {}
        train_batch = self.ntk.sample_loader_batch(self.train_loader, self.cfg.ntk_batch_size)
        val_batch = self.ntk.sample_loader_batch(self.val_loader, self.cfg.ntk_batch_size)
        if train_batch is not None:
            point_sets["train"] = train_batch
        if val_batch is not None:
            point_sets["val"] = val_batch
        return point_sets

    def _infer_output_dim(self, inputs):
        return self.ntk.infer_output_dim(self.model, inputs)

    def _compute_empirical_jacobian(self, inputs):
        return self.ntk.compute_empirical_jacobian(self.model, inputs)

    @staticmethod
    def _compute_ntk_spectrum(jacobian):
        return ext.ntk.EmpiricalNTK.compute_ntk_spectrum(jacobian)

    def _finish_and_copy_log(self):
        now_date = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.logger("==> end time: {}".format(now_date))

        taiyi_info = self.taiyi.finish()
        finish_info = self.visualizer.finish(sync_offline=self.cfg.offline)
        if taiyi_info["taiyi_output"]:
            self.logger("==> Taiyi monitor collected output.")
        if finish_info.get("synced"):
            self.logger("==> WandB offline logs synced.")

        new_log_filename = r"{}_{}_{:5.2f}%%.txt".format(self.model_name, now_date, self.best_acc)
        self.logger("==> Network training completed. Copy log file to {}".format(new_log_filename))
        new_log_path = os.path.join(self.result_path, new_log_filename)
        if getattr(self.logger, "file", None) is not None and os.path.exists(self.logger.filename):
            shutil.copy(self.logger.filename, new_log_path)


if __name__ == "__main__":
    trainer = MLPNTKTrainer()
    torch.set_num_threads(1)
    trainer.train()
