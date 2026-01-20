#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb

from Taiyi.taiyi.monitor import Monitor
from Taiyi.visualize import Visualization

sys.path.append("..")
import extension as ext


class Fitting:
    def __init__(self):
        self.cfg = self.add_arguments()
        norm_flag = ext.normalization.setting(self.cfg)
        act_flag = ext.activation.setting(self.cfg)
        self.model_name = (
            "fitting"
            + "_w"
            + str(self.cfg.width)
            + "_"
            + norm_flag
            + "_"
            + act_flag
            + "_lr"
            + str(self.cfg.lr)
            + "_bs"
            + str(self.cfg.batch_size[0])
            + "_seed"
            + str(self.cfg.seed)
            + "_wd"
            + str(self.cfg.weight_decay)
        )
        self.result_path = os.path.join(self.cfg.output, self.model_name, self.cfg.log_suffix)
        os.makedirs(self.result_path, exist_ok=True)
        self.logger = ext.logger.setting(
            "log.txt", self.result_path, self.cfg.test, self.cfg.resume is not None
        )
        ext.trainer.setting(self.cfg)

        self.model = OneLayerNN(self.cfg.in_dim, self.cfg.width)
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

        self.train_loader, self.val_loader = self.build_dataloaders()

        self.device = self.get_device()
        self.model.to(self.device)

        self.best_loss = float("inf")
        self.step = 0

        if self.cfg.offline:
            os.environ["WANDB_MODE"] = "offline"

        if self.cfg.resume:
            saved = self.saver.resume(self.cfg.resume)
            if "epoch" in saved:
                self.cfg.start_epoch = saved["epoch"]
            if "wandb_id" in saved:
                self.wandb_id = saved["wandb_id"]
            if "step" in saved:
                self.step = saved["step"]
            if "best_loss" in saved:
                self.best_loss = saved["best_loss"]

        self.criterion = nn.MSELoss()
        ext.trainer.set_seed(self.cfg)

        taiyi_config = {}
        self.monitor = Monitor(self.model, taiyi_config)

        self.init_wandb()
        self.vis_wandb = Visualization(self.monitor, wandb)

    def add_arguments(self):
        parser = argparse.ArgumentParser("Fitting Task")
        parser.add_argument("--in-dim", type=int, default=8, help="Input dimension.")
        parser.add_argument("--num-samples", type=int, default=512, help="Total samples.")
        parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio.")
        parser.add_argument("--width", type=int, default=512, help="Hidden width.")
        parser.add_argument("--offline", action="store_true", help="Offline wandb mode.")
        parser.add_argument(
            "-b",
            "--batch-size",
            type=ext.utils.str2list,
            default=[64, 256],
            metavar="NUMs",
            help="Mini-batch sizes for train/val.",
        )
        parser.add_argument("-j", "--workers", default=0, type=int, metavar="N")
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
        args = parser.parse_args()
        if args.resume:
            args = parser.parse_args(
                namespace=ext.checkpoint.Checkpoint.load_config(args.resume)
            )
        return args

    def init_wandb(self):
        wandb_config = {
            "model": "OneLayerNN",
            "width": self.cfg.width,
            "norm": ext.normalization.setting(self.cfg),
            "activation": self.cfg.activation,
            "learning_rate": self.cfg.lr,
            "batch_size": self.cfg.batch_size[0],
            "weight_decay": self.cfg.weight_decay,
            "seed": self.cfg.seed,
            "optimizer": self.cfg.optimizer,
            "in_dim": self.cfg.in_dim,
            "num_samples": self.cfg.num_samples,
            "epochs": self.cfg.epochs,
        }
        if self.cfg.resume and hasattr(self, "wandb_id"):
            self.logger("resume wandb from id {}".format(self.wandb_id))
            wandb.init(
                project="MLP Fitting",
                name=self.model_name,
                id=self.wandb_id,
                resume="must",
                notes=str(self.cfg),
                config=wandb_config,
            )
        else:
            wandb.init(
                project="MLP Fitting",
                name=self.model_name,
                notes=str(self.cfg),
                config=wandb_config,
            )
        self.run_dir = os.path.dirname(wandb.run.dir)

    def get_device(self):
        if self.cfg.gpu is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda:{}".format(self.cfg.gpu))
        num_gpu = torch.cuda.device_count()
        self.logger("==> use {:d} GPUs".format(num_gpu))
        if num_gpu > 1 and device.type == "cuda":
            self.model = torch.nn.DataParallel(self.model)
        return device

    def build_dataloaders(self):
        if len(self.cfg.batch_size) == 0:
            self.cfg.batch_size = [64, 64]
        elif len(self.cfg.batch_size) == 1:
            self.cfg.batch_size.append(self.cfg.batch_size[0])

        torch.manual_seed(self.cfg.seed)
        x = 2 * torch.rand(self.cfg.num_samples, self.cfg.in_dim) - 1
        y = 2 * torch.rand(self.cfg.num_samples, 1) - 1
        dataset = TensorDataset(x, y)

        train_size = int(self.cfg.num_samples * self.cfg.train_ratio)
        val_size = self.cfg.num_samples - train_size
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.batch_size[0],
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.workers,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.batch_size[1],
            shuffle=False,
            num_workers=self.cfg.workers,
        )
        return train_loader, val_loader

    def train(self):
        if self.cfg.test:
            self.validate()
            return

        if self.step is None:
            self.step = 0

        for epoch in range(self.cfg.start_epoch + 1, self.cfg.epochs):
            if self.cfg.lr_method != "auto":
                self.scheduler.step()
            self.train_epoch(epoch)
            wandb.log(
                {"learning_rate": self.scheduler.get_last_lr()[0], "steps": self.step}
            )

            val_loss = self.validate(epoch)
            self.saver.save_checkpoint(
                epoch=epoch,
                best_loss=self.best_loss,
                wandb_id=wandb.run.id,
                step=self.step,
            )
            if self.cfg.lr_method == "auto":
                self.scheduler.step(val_loss)

        now_date = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.logger("==> end time: {}".format(now_date))

        self.vis_wandb.close()
        self.monitor.get_output()
        self.logger("==> Wandb successfully get output.")

        new_log_filename = "{}_{}_loss{:.6f}.txt".format(
            self.model_name, now_date, self.best_loss
        )
        self.logger("==> Network training completed. Copy log file to {}".format(new_log_filename))
        if self.cfg.offline:
            self.logger("syncing wandb...{}".format(self.run_dir))
            os.system("wandb sync {}".format(self.run_dir))
        new_log_path = os.path.join(self.result_path, new_log_filename)
        shutil.copy(self.logger.filename, new_log_path)

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
        total = 0
        progress_bar = ext.ProgressBar(len(self.train_loader))

        for i, (inputs, targets) in enumerate(self.train_loader, 1):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor.track(self.step)
            self.vis_wandb.show(self.step)
            self.step += 1

            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            if i % 10 == 0 or i == len(self.train_loader):
                progress_bar.step("Loss: {:.6f}".format(total_loss / total), 10)

        avg_loss = total_loss / total
        wandb.log({"train_loss": avg_loss, "epochs": epoch, "steps": self.step})
        self.logger(
            "Train on epoch {}: average loss={:.6f}, time: {}".format(
                epoch, avg_loss, progress_bar.time_used()
            )
        )

    def validate(self, epoch=-1):
        test_loss = 0.0
        total = 0
        progress_bar = ext.ProgressBar(len(self.val_loader))
        self.model.eval()

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets).item() * targets.size(0)
                total += targets.size(0)
                progress_bar.step("Loss: {:.6f}".format(test_loss / total))

        avg_loss = test_loss / max(total, 1)
        wandb.log({"val_loss": avg_loss, "epochs": epoch, "steps": self.step})
        self.logger(
            "Val on epoch {}: average loss={:.6f}, time: {}".format(
                epoch, avg_loss, progress_bar.time_used()
            )
        )

        if not self.cfg.test and avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.saver.save_model("best.pth")
            self.logger("==> best loss: {:.6f}".format(self.best_loss))
        return avg_loss


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
    ext.MagnitudeDebug.reset()
    torch.set_num_threads(4)
    runner = Fitting()
    runner.train()
