#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import extension as ext


class FittingTrainer:
    def __init__(self):
        self.cfg = self.add_arguments()
        self.cfg.im_size = [self.cfg.in_dim]
        self.cfg.dataset_classes = 1
        ext.normalization.setting(self.cfg)
        ext.activation.setting(self.cfg)

        self.model_name = self._build_model_name()
        self.result_path = os.path.join(
            self.cfg.output, self.model_name, self.cfg.log_suffix
        )
        os.makedirs(self.result_path, exist_ok=True)

        self.logger = ext.logger.setting(
            "log.txt",
            self.result_path,
            self.cfg.test,
            self.cfg.resume is not None,
        )
        ext.trainer.setting(self.cfg)

        self.device = self._get_device()
        self.num_gpu = torch.cuda.device_count()
        self.train_loader, self.val_loader, self.test_loader = self.build_dataloaders()

        self.model = OneLayerNN(self.cfg.in_dim, self.cfg.width)
        if self.num_gpu > 1 and self.device.type == "cuda":
            self.model = torch.nn.DataParallel(self.model)
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

        self.best_loss = float("inf")
        self.step = 0
        self.wandb_id = None
        if self.cfg.resume:
            saved = self.saver.resume(self.cfg.resume)
            self.cfg.start_epoch = saved.get("epoch", self.cfg.start_epoch)
            self.step = saved.get("step", self.step)
            self.best_loss = saved.get("best_loss", self.best_loss)
            self.wandb_id = saved.get("wandb_id", self.wandb_id)

        self.model.to(self.device)
        self.criterion = nn.MSELoss()

        wandb_kwargs = {
            "project": self.cfg.wandb_project,
            "entity": "whsjrc-buaa",
            "name": self.model_name,
            "notes": str(self.cfg),
            "config": {
                "task": "synthetic_mlp_fitting",
                "model": "OneLayerNN",
                "in_dim": self.cfg.in_dim,
                "width": self.cfg.width,
                "normalization": ext.normalization.setting(self.cfg),
                "activation": ext.activation.setting(self.cfg),
                "num_samples": self.cfg.num_samples,
                "train_ratio": self.cfg.train_ratio,
                "val_ratio": self.cfg.val_ratio,
                "optimizer": self.cfg.optimizer,
                "learning_rate": self.cfg.lr,
                "batch_size": self.cfg.batch_size[0],
                "weight_decay": self.cfg.weight_decay,
                "epochs": self.cfg.epochs,
                "seed": self.cfg.seed,
                "track_geometry": self.cfg.track_geometry,
                "track_geometry_every": self.cfg.track_geometry_every,
            },
        }
        if self.cfg.resume and self.wandb_id:
            wandb_kwargs.update({"id": self.wandb_id, "resume": "must"})

        self.visualizer = ext.tracking.setting(
            self.cfg,
            env_name=self.model_name,
            vis_names={
                "train loss": "loss",
                "val loss": "loss",
                "test loss": "loss",
                "train r2": "metric",
                "val r2": "metric",
                "test r2": "metric",
                "distance_from_init": "geometry",
                "update_rate": "geometry",
                "curve_rate": "geometry",
                "cosine_similarity": "geometry",
            },
            wandb_kwargs=wandb_kwargs,
        )
        self.metrics = ext.measurement.setting(
            result_path=self.result_path,
            visualizer=self.visualizer,
            logger=self.logger,
        )
        self.taiyi = ext.taiyi.setting(
            self.cfg,
            model=self.model,
            monitor_config={},
            wandb=self.visualizer.wandb,
            output_dir=self.result_path,
        )
        self.geometry = ext.landscape.GeometryTracker(
            enabled=self.cfg.track_geometry,
            every=self.cfg.track_geometry_every,
            metrics=self.metrics,
        )
        self.geometry.capture_start(self.model)

    def add_arguments(self):
        parser = argparse.ArgumentParser("MLP Fitting")
        parser.add_argument("--in-dim", type=int, default=8, help="input dimension")
        parser.add_argument(
            "--num-samples", type=int, default=512, help="total synthetic samples"
        )
        parser.add_argument(
            "--train-ratio", type=float, default=0.8, help="train split ratio"
        )
        parser.add_argument(
            "--val-ratio", type=float, default=0.1, help="validation split ratio"
        )
        parser.add_argument("--width", type=int, default=512, help="hidden width")
        parser.add_argument(
            "-b",
            "--batch-size",
            type=ext.utils.str2list,
            default=[64, 256, 256],
            metavar="NUMs",
            help="mini-batch sizes for train/val/test",
        )
        parser.add_argument("-j", "--workers", default=0, type=int, metavar="N")
        parser.add_argument(
            "--track-geometry",
            action="store_true",
            help="log distance/update geometry metrics from mlpfitting.py",
        )
        parser.add_argument(
            "--track-geometry-every",
            type=int,
            default=1,
            help="track geometry every N optimizer steps",
        )
        parser.add_argument("--offline", action="store_true", help="offline wandb mode")

        ext.trainer.add_arguments(parser)
        parser.set_defaults(epochs=200)
        ext.scheduler.add_arguments(parser)
        parser.set_defaults(lr_method="fix", lr=1e-3)
        ext.optimizer.add_arguments(parser)
        parser.set_defaults(optimizer="adam", weight_decay=0.0)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.normalization.add_arguments(parser)
        ext.activation.add_arguments(parser)
        ext.tracking.add_arguments(parser)
        ext.taiyi.add_arguments(parser)

        args = parser.parse_args()
        if args.resume:
            args = parser.parse_args(
                namespace=ext.checkpoint.Checkpoint.load_config(args.resume)
            )
        if len(args.batch_size) == 0:
            args.batch_size = [64, 256, 256]
        elif len(args.batch_size) == 1:
            args.batch_size = [args.batch_size[0]] * 3
        elif len(args.batch_size) == 2:
            args.batch_size.append(args.batch_size[1])
        return ext.tracking.normalize_config(args)

    def _build_model_name(self):
        return (
            "Fitting"
            + "_d"
            + str(self.cfg.in_dim)
            + "_w"
            + str(self.cfg.width)
            + "_"
            + ext.normalization.setting(self.cfg)
            + "_"
            + ext.activation.setting(self.cfg)
            + "_lr"
            + str(self.cfg.lr)
            + "_bs"
            + str(self.cfg.batch_size[0])
            + "_wd"
            + str(self.cfg.weight_decay)
            + "_seed"
            + str(self.cfg.seed)
        )

    def _get_device(self):
        if self.cfg.gpu is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            device = torch.device("cuda:{}".format(self.cfg.gpu))
        num_gpu = torch.cuda.device_count()
        self.logger("==> use {:d} GPUs".format(num_gpu))
        return device

    @staticmethod
    def _r2_score(targets, predictions):
        target_mean = targets.mean()
        ss_tot = torch.sum((targets - target_mean) ** 2)
        ss_res = torch.sum((targets - predictions) ** 2)
        if ss_tot.abs().item() < 1e-12:
            return 0.0
        return (1.0 - ss_res / ss_tot).item()

    def build_dataloaders(self):
        total = int(self.cfg.num_samples)
        train_size = int(total * self.cfg.train_ratio)
        val_size = int(total * self.cfg.val_ratio)
        test_size = total - train_size - val_size
        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            raise ValueError(
                "Invalid split ratios: train/val/test splits must all be positive."
            )

        generator = torch.Generator().manual_seed(self.cfg.seed)
        x = 2 * torch.rand(self.cfg.num_samples, self.cfg.in_dim, generator=generator) - 1
        y = 2 * torch.rand(self.cfg.num_samples, 1, generator=generator) - 1
        dataset = TensorDataset(x, y)

        split_generator = torch.Generator().manual_seed(self.cfg.seed)
        train_set, val_set, test_set = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=split_generator,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.batch_size[0],
            shuffle=True,
            drop_last=False,
            num_workers=self.cfg.workers,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.batch_size[1],
            shuffle=False,
            num_workers=self.cfg.workers,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.cfg.batch_size[2],
            shuffle=False,
            num_workers=self.cfg.workers,
        )
        return train_loader, val_loader, test_loader

    def train(self):
        if self.cfg.test:
            self.evaluate(self.val_loader, "val", epoch=-1)
            self.evaluate(self.test_loader, "test", epoch=-1)
            self.finish()
            return

        for epoch in range(self.cfg.start_epoch + 1, self.cfg.epochs):
            self.train_epoch(epoch)
            val_loss, _ = self.evaluate(self.val_loader, "val", epoch=epoch)
            self.saver.save_checkpoint(
                epoch=epoch,
                best_loss=self.best_loss,
                wandb_id=self.visualizer.run_id,
                step=self.step,
            )
            if self.cfg.lr_method == "auto":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

        self.evaluate(self.test_loader, "test", epoch=self.cfg.epochs - 1)
        self.finish()

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
        progress_bar = ext.ProgressBar(len(self.train_loader))
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

            self.taiyi.track(self.step)
            self.geometry.track(self.model, step=self.step, epoch=epoch)
            self.step += 1

            progress_bar.step(
                "Loss: {:.6f}".format(total_loss / max(total_samples, 1))
            )

        avg_loss = total_loss / max(total_samples, 1)
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        r2 = self._r2_score(targets, predictions)

        self.metrics.log_scalars(
            {
                "train_loss": avg_loss,
                "train_r2": r2,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            },
            step=self.step,
            epoch=epoch,
            vis_scalars={
                "train loss": avg_loss,
                "train r2": r2,
            },
        )
        self.logger(
            "Train on epoch {}: average loss={:.6f}, r2={:.4f}, time: {}".format(
                epoch,
                avg_loss,
                r2,
                progress_bar.time_used(),
            )
        )

    def evaluate(self, loader, split, epoch=-1):
        self.model.eval()
        progress_bar = ext.ProgressBar(len(loader))
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * targets.size(0)
                total_samples += targets.size(0)
                all_predictions.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())

                progress_bar.step(
                    "Loss: {:.6f}".format(total_loss / max(total_samples, 1))
                )

        avg_loss = total_loss / max(total_samples, 1)
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        r2 = self._r2_score(targets, predictions)

        self.metrics.log_scalars(
            {
                f"{split}_loss": avg_loss,
                f"{split}_r2": r2,
            },
            step=self.step,
            epoch=epoch,
            vis_scalars={
                f"{split} loss": avg_loss,
                f"{split} r2": r2,
            },
        )
        self.logger(
            "{} on epoch {}: average loss={:.6f}, r2={:.4f}, time: {}".format(
                split.capitalize(),
                epoch,
                avg_loss,
                r2,
                progress_bar.time_used(),
            )
        )

        if split == "val" and not self.cfg.test and avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.saver.save_model("best.pth")
            self.logger("==> best loss: {:.6f}".format(self.best_loss))
        return avg_loss, r2

    def finish(self):
        self.geometry.finish()
        taiyi_info = self.taiyi.finish()
        tracking_info = self.visualizer.finish(sync_offline=self.cfg.offline)

        now_date = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.logger("==> end time: {}".format(now_date))
        if taiyi_info.get("taiyi_output_path"):
            self.logger(
                "==> taiyi output saved to {}".format(taiyi_info["taiyi_output_path"])
            )
        if tracking_info.get("synced"):
            self.logger("==> synced offline wandb run")

        new_log_filename = "{}_{}_loss{:.6f}.txt".format(
            self.model_name, now_date, self.best_loss
        )
        if self.logger.filename and os.path.exists(self.logger.filename):
            new_log_path = os.path.join(self.result_path, new_log_filename)
            shutil.copy(self.logger.filename, new_log_path)
            self.logger("==> copied log file to {}".format(new_log_filename))


class OneLayerNN(nn.Module):
    def __init__(self, in_dim, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            ext.Norm(width),
            ext.Activation(width),
            nn.Linear(width, 1),
        )

    def forward(self, inputs):
        return self.net(inputs)


if __name__ == "__main__":
    if hasattr(ext, "MagnitudeDebug") and hasattr(ext.MagnitudeDebug, "reset"):
        ext.MagnitudeDebug.reset()
    torch.set_num_threads(4)
    runner = FittingTrainer()
    runner.train()
