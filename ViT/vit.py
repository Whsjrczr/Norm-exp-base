#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time

import torch
import torch.nn as nn

sys.path.append("../")
import extension as ext

from model_vit.select_vit import add_model_arguments, get_model


class ViTTrainer:
    def __init__(self):
        self.cfg = self.add_arguments()
        self.image_size = int(self.cfg.im_size[-1]) if isinstance(self.cfg.im_size, (list, tuple)) else int(self.cfg.im_size)
        train_batch_size = self.cfg.batch_size[0] if len(self.cfg.batch_size) > 0 else None
        val_batch_size = self.cfg.batch_size[1] if len(self.cfg.batch_size) > 1 else train_batch_size
        self.model_name = (
            f"ViT_{self.cfg.arch}_{ext.dataset.setting(self.cfg)}"
            f"_img{self.image_size}_patch{self.cfg.patch_size}"
            f"_{ext.normalization.setting(self.cfg)}_{ext.activation.setting(self.cfg)}"
            f"_lr{self.cfg.lr}_bs{train_batch_size}_dropout{self.cfg.dropout}"
            f"_droppath{self.cfg.drop_path_rate}_wd{self.cfg.weight_decay}_seed{self.cfg.seed}"
        )
        self.result_path = os.path.join(self.cfg.output, self.model_name, self.cfg.log_suffix)
        os.makedirs(self.result_path, exist_ok=True)

        self.logger = ext.logger.setting("log.txt", self.result_path, self.cfg.test, self.cfg.resume is not None)
        ext.trainer.setting(self.cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.num_gpu = torch.cuda.device_count()
        self.logger("==> use {:d} GPUs".format(self.num_gpu))

        self.train_loader = ext.dataset.get_dataset_loader(self.cfg, train=True, use_cuda=self.device.type == "cuda")
        self.val_loader = ext.dataset.get_dataset_loader(self.cfg, train=False, use_cuda=self.device.type == "cuda")

        self.model = get_model(self.cfg)
        self.logger("==> model [{}]: {}".format(self.model_name, self.model))

        self.optimizer = ext.optimizer.setting(self.model, self.cfg)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.cfg)

        self.saver = ext.checkpoint.Checkpoint(self.model, self.cfg, self.optimizer, self.scheduler, self.result_path, not self.cfg.test)
        self.saver.load(self.cfg.load)

        if self.num_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        self.best_acc1 = 0.0
        if self.cfg.resume:
            saved = self.saver.resume(self.cfg.resume)
            self.cfg.start_epoch = saved["epoch"]
            if self.cfg.wandb and "wandb_id" in saved.keys():
                self.wandb_id = saved["wandb_id"]
                self.step = saved["step"]
            self.best_acc1 = saved.get("best_acc1", saved.get("best_acc", 0.0))
            self.cfg.seed = saved.get("seed", self.cfg.seed)

        self.criterion = nn.CrossEntropyLoss()
        ext.trainer.set_seed(self.cfg)

        taiyi_config = {
            ext.LayerNormScaling: [["InputSndNorm", "linear(5,0)"], ["OutputGradSndNorm", "linear(5,0)"]],
            nn.LayerNorm: [["InputSndNorm", "linear(5,0)"], ["OutputGradSndNorm", "linear(5,0)"]],
            "Block": [
                ["ResidualInputAngleMean", "linear(5,0)"],
                ["ResidualStreamOutputAngleMean", "linear(5,0)"],
            ],
        }
        wandb_kwargs = dict(
            project=self.cfg.wandb_project if hasattr(self.cfg, "wandb_project") else "test",
            entity="whsjrc-buaa",
            name=self.model_name,
            notes=str(self.cfg) + " ---- " + str(taiyi_config),
            config={
                "model": self.cfg.arch,
                "image_size": self.image_size,
                "patch_size": self.cfg.patch_size,
                "normalization": ext.normalization.setting(self.cfg),
                "activation": ext.activation.setting(self.cfg),
                "dropout_prob": self.cfg.dropout,
                "drop_path_rate": self.cfg.drop_path_rate,
                "optimizer": self.cfg.optimizer,
                "learning_rate": self.cfg.lr,
                "batch_size": train_batch_size,
                "val_batch_size": val_batch_size,
                "weight_decay": self.cfg.weight_decay,
                "dataset": self.cfg.dataset,
                "dataset_root": self.cfg.dataset_root,
                "epochs": self.cfg.epochs,
                "seed": self.cfg.seed,
                "scheduler": getattr(self.cfg, "lr_method", None),
                "scheduler_cfg": f"step{getattr(self.cfg, 'lr_step', None)}_gamma{getattr(self.cfg, 'lr_gamma', None)}",
            },
        )

        if self.cfg.resume and hasattr(self, "wandb_id") and self.wandb_id:
            print("resume wandb from id " + str(self.wandb_id))
            wandb_kwargs.update(dict(id=self.wandb_id, resume="must"))

        self.visualizer = ext.tracking.setting(
            self.cfg,
            env_name=self.model_name,
            vis_names={
                "train loss": "loss",
                "test loss": "loss",
                "train accuracy": "accuracy",
                "test accuracy": "accuracy",
                "train acc@5": "top5",
                "test acc@5": "top5",
            },
            wandb_kwargs=wandb_kwargs,
        )
        self.taiyi = ext.taiyi.setting(
            self.cfg,
            model=self.model,
            monitor_config=taiyi_config,
            wandb=self.visualizer.wandb,
        )
        if self.taiyi.enabled:
            self.logger("==> taiyi config: {}".format(taiyi_config))

    def add_arguments(self):
        argv = ["--batch-size" if arg == "--batch_size" else arg for arg in sys.argv[1:]]
        parser = argparse.ArgumentParser("ViT Classification")
        add_model_arguments(parser)
        parser.add_argument("--data_path", type=str, default="", help="alias of --dataset-root for ImageFolder datasets")
        parser.add_argument("--val-resize-size", dest="val_resize_size", type=int, default=None)
        parser.add_argument("--disable-train-shuffle", action="store_true")
        parser.add_argument("--offline", "-offline", action="store_true", help="offline mode")
        parser.add_argument("--val-batch-size", dest="val_batch_size", type=int, default=None)
        ext.trainer.add_arguments(parser)
        parser.set_defaults(epochs=100)
        ext.dataset.add_arguments(parser)
        parser.set_defaults(dataset="ImageNet", workers=4, batch_size=[256, 256], im_size=(224, 224), dataset_cfg={"loader": "vit"})
        ext.scheduler.add_arguments(parser)
        parser.set_defaults(lr_method="cos", lr=1e-4)
        ext.optimizer.add_arguments(parser)
        parser.set_defaults(optimizer="adamw", weight_decay=0.1, seed=0)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.normalization.add_arguments(parser)
        ext.activation.add_arguments(parser)
        ext.tracking.add_arguments(parser)
        ext.taiyi.add_arguments(parser)

        args = parser.parse_args(argv)
        if args.resume:
            args = parser.parse_args(argv, namespace=ext.checkpoint.Checkpoint.load_config(args.resume))

        if args.data_path:
            args.dataset_root = args.data_path
            if args.dataset == "ImageNet":
                args.dataset = "folder"
        if args.batch_size is None:
            args.batch_size = [256, 256]
        elif isinstance(args.batch_size, int):
            args.batch_size = [args.batch_size]
        elif not isinstance(args.batch_size, list):
            args.batch_size = list(args.batch_size)
        if args.val_batch_size is not None:
            if len(args.batch_size) == 0:
                args.batch_size = [256, args.val_batch_size]
            else:
                args.batch_size = list(args.batch_size)
                if len(args.batch_size) == 1:
                    args.batch_size.append(args.val_batch_size)
                else:
                    args.batch_size[1] = args.val_batch_size
        image_size = int(args.im_size[-1]) if isinstance(args.im_size, (list, tuple)) else int(args.im_size)
        if args.val_resize_size is None:
            args.val_resize_size = int(image_size * 256 / 224)
        if args.num_classes is not None:
            args.dataset_classes = args.num_classes
        if not isinstance(args.dataset_cfg, dict):
            args.dataset_cfg = {}
        args.dataset_cfg.setdefault("loader", "vit")
        args.dataset_cfg.setdefault("image_size", image_size)
        args.dataset_cfg.setdefault("val_resize_size", args.val_resize_size)
        if args.lr_method == "cos" and args.lr_step == 30:
            args.lr_step = args.epochs
            args.lr_gamma = 0.0
        stages = getattr(ext.optimizer, "get_stages", lambda _cfg: None)(args)
        stage_total_epochs = getattr(ext.optimizer, "infer_total_epochs", lambda _stages: None)(stages)
        if stage_total_epochs is not None:
            args.epochs = stage_total_epochs
        return ext.tracking.normalize_config(args)

    def train(self):
        if self.cfg.test:
            self.validate()
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
                self.train_epoch(epoch)
                self.visualizer.log(
                    {
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "steps": self.step,
                        "stage": stage_idx,
                        "epochs": epoch,
                    }
                )

                accuracy1, accuracy5, val_loss = self.validate(epoch)
                if self.visualizer.wandb_enabled:
                    self.saver.save_checkpoint(
                        epoch=epoch,
                        best_acc1=self.best_acc1,
                        wandb_id=self.visualizer.run_id,
                        step=self.step,
                        seed=self.cfg.seed,
                        acc5=accuracy5,
                    )
                else:
                    self.saver.save_checkpoint(epoch=epoch, best_acc1=self.best_acc1, step=self.step, seed=self.cfg.seed, acc5=accuracy5)
                if self.cfg.lr_method == "auto":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        now_date = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.logger("==> end time: {}".format(now_date))

        taiyi_info = self.taiyi.finish()
        finish_info = self.visualizer.finish(sync_offline=self.cfg.offline)
        if taiyi_info["taiyi_output"]:
            self.logger("==> Taiyi monitor collected output.")

        new_log_filename = r"{}_{}_{:5.2f}%%.txt".format(self.model_name, now_date, self.best_acc1)
        self.logger("==> Network training completed. Copy log file to {}".format(new_log_filename))
        new_log_path = os.path.join(self.result_path, new_log_filename)
        shutil.copy(self.logger.filename, new_log_path)

    def _rebuild_optim_sched_and_sync_saver(self):
        self.optimizer = ext.optimizer.setting(self.model, self.cfg)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.cfg)
        if hasattr(self, "saver") and self.saver is not None:
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

    def train_epoch(self, epoch):
        self.logger("\nEpoch: {}, lr: {:.2g}, weight decay: {:.2g} on model {}".format(
            epoch, self.optimizer.param_groups[0]["lr"], self.optimizer.param_groups[0]["weight_decay"], self.model_name))
        self.model.train()
        train_loss = AverageMeter("loss")
        train_acc1 = AverageMeter("acc1")
        train_acc5 = AverageMeter("acc5")
        progress_bar = ext.ProgressBar(len(self.train_loader))

        for i, (inputs, targets) in enumerate(self.train_loader, 1):
            inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
            targets = targets.to(self.device, non_blocking=self.device.type == "cuda")

            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            self.taiyi.track(self.step)
            self.step += 1

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            train_loss.update(losses.item(), targets.size(0))
            train_acc1.update(acc1.item(), targets.size(0))
            train_acc5.update(acc5.item(), targets.size(0))

            if i % 10 == 0 or i == len(self.train_loader):
                progress_bar.step(
                    "Loss: {:.5g} | Acc@1: {:.2f}% | Acc@5: {:.2f}%".format(
                        train_loss.avg, train_acc1.avg, train_acc5.avg
                    ),
                    10,
                )

        self.visualizer.log(
            {"train_acc": train_acc1.avg, "train_acc5": train_acc5.avg, "train_loss": train_loss.avg, "epochs": epoch, "steps": self.step}
        )
        self.visualizer.add_value("train loss", train_loss.avg)
        self.visualizer.add_value("train accuracy", train_acc1.avg)
        self.visualizer.add_value("train acc@5", train_acc5.avg)
        self.logger(
            "Train on epoch {}: average loss={:.5g}, acc@1={:.2f}%, acc@5={:.2f}%, time: {}".format(
                epoch, train_loss.avg, train_acc1.avg, train_acc5.avg, progress_bar.time_used()
            )
        )

    def validate(self, epoch=-1):
        test_loss = AverageMeter("loss")
        test_acc1 = AverageMeter("acc1")
        test_acc5 = AverageMeter("acc5")
        progress_bar = ext.ProgressBar(len(self.val_loader))
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=self.device.type == "cuda")
                targets = targets.to(self.device, non_blocking=self.device.type == "cuda")

                outputs = self.model(inputs)
                losses = self.criterion(outputs, targets)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

                test_loss.update(losses.item(), targets.size(0))
                test_acc1.update(acc1.item(), targets.size(0))
                test_acc5.update(acc5.item(), targets.size(0))
                progress_bar.step(
                    "Loss: {:.5g} | Acc@1: {:.2f}% | Acc@5: {:.2f}%".format(
                        test_loss.avg, test_acc1.avg, test_acc5.avg
                    )
                )

        self.visualizer.add_value("test loss", test_loss.avg)
        self.visualizer.add_value("test accuracy", test_acc1.avg)
        self.visualizer.add_value("test acc@5", test_acc5.avg)
        self.visualizer.log({"test_acc": test_acc1.avg, "test_acc5": test_acc5.avg, "test_loss": test_loss.avg, "epochs": epoch, "steps": self.step})
        self.logger(
            "Test on epoch {}: average loss={:.5g}, acc@1={:.2f}%, acc@5={:.2f}%, time: {}".format(
                epoch, test_loss.avg, test_acc1.avg, test_acc5.avg, progress_bar.time_used()
            )
        )
        if not self.cfg.test and test_acc1.avg > self.best_acc1:
            self.best_acc1 = test_acc1.avg
            self.saver.save_model("best.pth")
            self.logger("==> best acc@1: {:.2f}%".format(self.best_acc1))
        return test_acc1.avg, test_acc5.avg, test_loss.avg


class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            kk = min(k, output.size(1))
            correct_k = correct[:kk].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    trainer = ViTTrainer()
    torch.set_num_threads(1)
    trainer.train()
