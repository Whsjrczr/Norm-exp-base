#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import r2_score

KAN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(KAN_DIR, ".."))
for path in (KAN_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.append(path)

import extension as ext
from extension.utils import str2list

from kan_dataset import KANDatasetBuilder, add_dataset_arguments
from model.select_kan import add_model_arguments, get_model


class KANTrainer:
    def __init__(self):
        self.cfg = self.add_arguments()
        ext.normalization.setting(self.cfg)
        ext.activation.setting(self.cfg)

        self.model_name = self._build_model_name()
        self.result_path = os.path.join(self.cfg.output, self.model_name, self.cfg.log_suffix)
        os.makedirs(self.result_path, exist_ok=True)

        self.logger = ext.logger.setting("log.txt", self.result_path, self.cfg.test, self.cfg.resume is not None)
        ext.trainer.setting(self.cfg)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.num_gpu = torch.cuda.device_count()
        self.logger("==> use {:d} GPUs".format(self.num_gpu))

        self.dataset_builder = KANDatasetBuilder(self.cfg)
        dataset = self.dataset_builder.build()
        self.train_loader = dataset["train_loader"]
        self.val_loader = dataset["val_loader"]
        self.test_loader = dataset["test_loader"]
        self.x_test = dataset["x_test"]
        self.y_test = dataset["y_test"]
        self.y_true_val = dataset["y_true_val"]
        self.y_true_test = dataset["y_true_test"]

        self.model = get_model(self.cfg)
        self.logger("==> model [{}]: {}".format(self.model_name, self.model))

        self.optimizer = ext.optimizer.setting(self.model, self.cfg)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.cfg)

        self.saver = ext.checkpoint.Checkpoint(
            self.model,
            self.cfg,
            self.optimizer,
            self.scheduler,
            self.result_path,
            not self.cfg.test,
        )
        self.saver.load(self.cfg.load)

        if self.num_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        self.best_r2 = float("-inf")
        if self.cfg.visualize and self.cfg.offline:
            os.environ["WANDB_MODE"] = "offline"
        if self.cfg.resume:
            saved = self.saver.resume(self.cfg.resume)
            self.cfg.start_epoch = saved["epoch"]
            if self.cfg.visualize and "wandb_id" in saved:
                self.wandb_id = saved["wandb_id"]
                self.step = saved["step"]
            self.best_r2 = saved.get("best_r2", self.best_r2)
            self.cfg.seed = saved.get("seed", self.cfg.seed)

        self.criterion = nn.MSELoss()
        ext.trainer.set_seed(self.cfg)

        if self.cfg.visualize:
            wandb_kwargs = dict(
                project=self.cfg.wandb_project,
                entity="whsjrc-buaa",
                name=self.model_name,
                notes=str(self.cfg),
                config={
                    "model": self.cfg.arch,
                    "layers_hidden": self.cfg.layers_hidden,
                    "normalization": ext.normalization.setting(self.cfg),
                    "activation": ext.activation.setting(self.cfg),
                    "optimizer": self.cfg.optimizer,
                    "learning_rate": self.cfg.lr,
                    "batch_size": self.cfg.batch_size[0],
                    "weight_decay": self.cfg.weight_decay,
                    "dataset": self.cfg.function,
                    "epochs": self.cfg.epochs,
                    "seed": self.cfg.seed,
                    "scheduler": getattr(self.cfg, "lr_method", None),
                    "scheduler_cfg": f"step{getattr(self.cfg, 'lr_step', None)}_gamma{getattr(self.cfg, 'lr_gamma', None)}",
                    "error": self.cfg.error,
                    "grid_size": self.cfg.grid_size,
                    "spline_order": self.cfg.spline_order,
                    "update_grid": self.cfg.update_grid,
                },
            )
            if self.cfg.resume and hasattr(self, "wandb_id") and self.wandb_id:
                wandb_kwargs.update(dict(id=self.wandb_id, resume="must"))
            wandb.init(**wandb_kwargs)
            self.run_dir = os.path.dirname(wandb.run.dir)

        if self.cfg.vis:
            self.vis = ext.visualization.setting(
                self.cfg,
                self.model_name,
                {
                    "train loss": "loss",
                    "val loss": "loss",
                    "train r2": "metric",
                    "val r2": "metric",
                    "test loss": "loss",
                    "test r2": "metric",
                    "true test loss": "loss",
                    "true test r2": "metric",
                },
            )

    def add_arguments(self):
        parser = argparse.ArgumentParser("KAN Regression")
        add_model_arguments(parser)
        add_dataset_arguments(parser)
        parser.add_argument("--batch-size", dest="batch_size", type=str2list, default="256,1024")
        parser.add_argument("--offline", "-offline", action="store_true")

        ext.trainer.add_arguments(parser)
        parser.set_defaults(epochs=500)
        ext.scheduler.add_arguments(parser)
        parser.set_defaults(lr_method="fix", lr=1e-3)
        ext.optimizer.add_arguments(parser)
        parser.set_defaults(optimizer="adam", weight_decay=0.0)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.normalization.add_arguments(parser)
        ext.activation.add_arguments(parser)
        ext.vis_taiyi.add_arguments(parser)
        ext.visualization.add_arguments(parser)

        args = parser.parse_args()
        if args.resume:
            args = parser.parse_args(namespace=ext.checkpoint.Checkpoint.load_config(args.resume))
        if args.layers_hidden is None:
            args.layers_hidden = [args.input_dim] + [args.width] * args.depth + [args.output_dim]
        else:
            args.layers_hidden = list(args.layers_hidden)
            args.input_dim = int(args.layers_hidden[0])
            args.output_dim = int(args.layers_hidden[-1])
        if len(args.batch_size) == 1:
            args.batch_size = [args.batch_size[0], args.batch_size[0]]
        stages = getattr(ext.optimizer, "get_stages", lambda _cfg: None)(args)
        stage_total_epochs = getattr(ext.optimizer, "infer_total_epochs", lambda _stages: None)(stages)
        if stage_total_epochs is not None:
            args.epochs = stage_total_epochs
        args.im_size = [args.input_dim]
        args.dataset_classes = args.output_dim
        return args

    def _build_model_name(self):
        hidden = "-".join(str(v) for v in self.cfg.layers_hidden) if self.cfg.layers_hidden else f"{self.cfg.depth}x{self.cfg.width}"
        return (
            f"KAN_{self.cfg.arch}_{self.cfg.function}"
            f"_h{hidden}"
            f"_{ext.normalization.setting(self.cfg)}_{ext.activation.setting(self.cfg)}"
            f"_lr{self.cfg.lr}_bs{self.cfg.batch_size[0]}"
            f"_wd{self.cfg.weight_decay}_noise{self.cfg.error}_seed{self.cfg.seed}"
        )

    def _rebuild_optim_sched_and_sync_saver(self):
        self.optimizer = ext.optimizer.setting(self.model, self.cfg)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.cfg)
        if hasattr(self.saver, "optimizer"):
            self.saver.optimizer = self.optimizer
        if hasattr(self.saver, "scheduler"):
            self.saver.scheduler = self.scheduler

    def _epoch_stage_plan(self):
        get_stages = getattr(ext.optimizer, "get_stages", None)
        stages = get_stages(self.cfg) if callable(get_stages) else None
        explicit_total_epochs = getattr(ext.optimizer, "infer_total_epochs", lambda _stages: None)(stages)

        epoch_begin = self.cfg.start_epoch + 1
        epoch_end = explicit_total_epochs if explicit_total_epochs is not None else self.cfg.epochs
        if not stages:
            return [(1, None, epoch_begin, epoch_end)]

        plan = []
        cur = epoch_begin
        stage_idx = 1
        for stage in stages:
            if cur >= epoch_end:
                break
            if stage is None:
                continue

            if "end_epoch" in stage:
                end = int(stage["end_epoch"])
            else:
                duration = stage.get("epochs", stage.get("epoch", None))
                end = cur + int(duration) if duration is not None else epoch_end

            end = max(cur, min(end, epoch_end))
            plan.append((stage_idx, stage, cur, end))
            cur = end
            stage_idx += 1

        if explicit_total_epochs is None and cur < epoch_end:
            plan.append((stage_idx, stages[-1], cur, epoch_end))
        return plan

    def _apply_stage_to_cfg(self, stage_cfg):
        if not stage_cfg:
            return
        if "optimizer" in stage_cfg:
            self.cfg.optimizer = stage_cfg["optimizer"]
        elif "name" in stage_cfg:
            self.cfg.optimizer = stage_cfg["name"]

        if "lr" in stage_cfg:
            self.cfg.lr = stage_cfg["lr"]
        if "weight_decay" in stage_cfg:
            self.cfg.weight_decay = stage_cfg["weight_decay"]
        if "optimizer_config" in stage_cfg:
            self.cfg.optimizer_config = stage_cfg["optimizer_config"]
        if "lr_method" in stage_cfg:
            self.cfg.lr_method = stage_cfg["lr_method"]
        if "lr_step" in stage_cfg:
            self.cfg.lr_step = stage_cfg["lr_step"]
        if "lr_gamma" in stage_cfg:
            self.cfg.lr_gamma = stage_cfg["lr_gamma"]

    def train(self):
        if self.cfg.test:
            self.validate()
            self.test()
            return

        if not hasattr(self, "step"):
            self.step = 0

        stage_plan = self._epoch_stage_plan()
        multistage_enabled = (len(stage_plan) > 1) or (stage_plan and stage_plan[0][1] is not None)
        if multistage_enabled:
            self.logger(f"==> Multi-stage training enabled: {len(stage_plan)} segments")

        for stage_idx, stage_cfg, epoch_begin, epoch_end in stage_plan:
            self._apply_stage_to_cfg(stage_cfg)
            self._rebuild_optim_sched_and_sync_saver()
            self.logger(
                f"==> Stage {stage_idx}: optimizer={self.cfg.optimizer}, lr={getattr(self.cfg, 'lr', None)}, "
                f"wd={getattr(self.cfg, 'weight_decay', None)}, epochs=[{epoch_begin},{epoch_end})"
            )

            for epoch in range(epoch_begin, epoch_end):
                train_loss, train_r2 = self.train_epoch(epoch)
                val_loss, val_r2, true_val_loss, true_val_r2 = self.validate(epoch)

                if self.cfg.visualize:
                    wandb.log(
                        {
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "epochs": epoch,
                            "steps": self.step,
                            "stage": stage_idx,
                            "train_loss": train_loss,
                            "train_r2": train_r2,
                            "val_loss": val_loss,
                            "val_r2": val_r2,
                            "true_val_loss": true_val_loss,
                            "true_val_r2": true_val_r2,
                        }
                    )

                if self.cfg.visualize:
                    self.saver.save_checkpoint(
                        epoch=epoch,
                        best_r2=self.best_r2,
                        wandb_id=wandb.run.id,
                        step=self.step,
                        seed=self.cfg.seed,
                    )
                else:
                    self.saver.save_checkpoint(epoch=epoch, best_r2=self.best_r2, step=self.step, seed=self.cfg.seed)

                if self.cfg.lr_method == "auto":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        self.test()
        self._save_prediction_plot()

        now_date = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.logger("==> end time: {}".format(now_date))

        new_log_filename = f"{self.model_name}_{now_date}.txt"
        self.logger("==> Network training completed. Copy log file to {}".format(new_log_filename))
        if self.cfg.offline and self.cfg.visualize:
            os.system(f"wandb sync {self.run_dir}")
        shutil.copy(self.logger.filename, os.path.join(self.result_path, new_log_filename))

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
        total_loss = 0.0
        total_count = 0
        outputs_list = []
        targets_list = []

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            if self.cfg.arch == "KAN" and self.cfg.kan_regularization > 0:
                loss = loss + self.cfg.kan_regularization * self._regularization_loss()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.step += 1

            total_loss += loss.item() * targets.size(0)
            total_count += targets.size(0)
            outputs_list.append(outputs.detach().cpu())
            targets_list.append(targets.detach().cpu())

        avg_loss = total_loss / max(total_count, 1)
        train_r2 = self._compute_r2(torch.cat(outputs_list), torch.cat(targets_list))

        if self.cfg.vis:
            self.vis.add_value("train loss", avg_loss)
            self.vis.add_value("train r2", train_r2)
        self.logger(
            "Train on epoch {}: average loss={:.5g}, r2={:.4f}".format(epoch, avg_loss, train_r2)
        )
        return avg_loss, train_r2

    def validate(self, epoch=-1):
        val_loss, val_r2 = self._evaluate_loader(self.val_loader)
        true_loss, true_r2 = self._evaluate_true_targets(self.val_loader, split="val")

        if self.cfg.vis:
            self.vis.add_value("val loss", val_loss)
            self.vis.add_value("val r2", val_r2)
        self.logger(
            "Validate on epoch {}: loss={:.5g}, r2={:.4f}, true_loss={:.5g}, true_r2={:.4f}".format(
                epoch, val_loss, val_r2, true_loss, true_r2
            )
        )

        if not self.cfg.test and true_r2 > self.best_r2:
            self.best_r2 = true_r2
            self.saver.save_model("best.pth")
            self.logger("==> best true r2: {:.4f}".format(self.best_r2))
        return val_loss, val_r2, true_loss, true_r2

    def test(self):
        test_loss, test_r2 = self._evaluate_loader(self.test_loader)
        true_loss, true_r2 = self._evaluate_true_targets(self.test_loader, split="test")
        if self.cfg.vis:
            self.vis.add_value("test loss", test_loss)
            self.vis.add_value("test r2", test_r2)
            self.vis.add_value("true test loss", true_loss)
            self.vis.add_value("true test r2", true_r2)
        if self.cfg.visualize:
            wandb.log(
                {
                    "test_loss": test_loss,
                    "test_r2": test_r2,
                    "true_test_loss": true_loss,
                    "true_test_r2": true_r2,
                }
            )
        self.logger(
            "Test: loss={:.5g}, r2={:.4f}, true_loss={:.5g}, true_r2={:.4f}".format(
                test_loss, test_r2, true_loss, true_r2
            )
        )
        return test_loss, test_r2, true_loss, true_r2

    def _evaluate_loader(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        outputs_list = []
        targets_list = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                total_count += targets.size(0)
                outputs_list.append(outputs.detach().cpu())
                targets_list.append(targets.detach().cpu())
        avg_loss = total_loss / max(total_count, 1)
        avg_r2 = self._compute_r2(torch.cat(outputs_list), torch.cat(targets_list))
        return avg_loss, avg_r2

    def _evaluate_true_targets(self, loader, split="test"):
        if split == "val":
            true_targets = self.y_true_val
        else:
            true_targets = self.y_true_test
        self.model.eval()
        outputs_list = []
        with torch.no_grad():
            for inputs, _targets in loader:
                outputs_list.append(self.model(inputs.to(self.device)).detach().cpu())
        predictions = torch.cat(outputs_list)
        true_loss = self.criterion(predictions, true_targets.cpu()).item()
        true_r2 = self._compute_r2(predictions, true_targets.cpu())
        return true_loss, true_r2

    def _regularization_loss(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model, "regularization_loss"):
            return model.regularization_loss()
        return torch.tensor(0.0, device=self.device)

    def _compute_r2(self, prediction, target):
        return r2_score(target.numpy(), prediction.numpy())

    def _save_prediction_plot(self):
        self.model.eval()
        x_curve, y_curve = self.dataset_builder.build_curve(self.cfg.curve_points)
        with torch.no_grad():
            prediction = self.model(x_curve.to(self.device)).detach().cpu()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        with torch.no_grad():
            test_prediction = self.model(self.x_test.to(self.device)).detach().cpu()

        axes[0].scatter(
            self.y_test.numpy(),
            test_prediction.numpy(),
            alpha=0.6,
            facecolor="none",
            edgecolor="tab:blue",
            linewidth=1,
        )
        axes[0].plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            "k--",
            linewidth=2,
        )
        axes[0].set_title("Prediction vs label")
        axes[0].set_xlabel("label")
        axes[0].set_ylabel("prediction")

        axes[1].plot(x_curve[:, 0].numpy(), y_curve.numpy(), label="true", linewidth=2)
        axes[1].plot(x_curve[:, 0].numpy(), prediction.numpy(), label="prediction", linewidth=2)
        axes[1].set_title("Curve slice")
        axes[1].set_xlabel("x0")
        axes[1].set_ylabel("y")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        figure_path = os.path.join(self.result_path, "prediction_curve.png")
        fig.tight_layout()
        fig.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)
        self.logger("==> Saved prediction plot to {}".format(figure_path))


if __name__ == "__main__":
    trainer = KANTrainer()
    torch.set_num_threads(1)
    trainer.train()
