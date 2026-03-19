#!/usr/bin/env python3
import argparse
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.append("../")
import extension as ext

from model_vit.select_vit import get_model


class ViTTrainer:
    def __init__(self):
        self.cfg = self.add_arguments()
        self.cfg.im_size = [self.cfg.image_size]

        self.model_name = (
            f"ViT_{self.cfg.arch}_img{self.cfg.image_size}_patch{self.cfg.patch_size}"
            f"_bs{self.cfg.batch_size}_lr{self.cfg.lr}_epochs{self.cfg.epochs}_seed{self.cfg.seed}"
        )
        self.result_path = os.path.join(self.cfg.output, self.model_name, self.cfg.log_suffix)
        os.makedirs(self.result_path, exist_ok=True)

        self.logger = ext.logger.setting(
            "log.txt", self.result_path, self.cfg.test, self.cfg.resume is not None
        )
        ext.trainer.setting(self.cfg)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.num_gpu = torch.cuda.device_count() if self.device.type == "cuda" else 0
        self.logger(f"==> use {self.num_gpu} GPUs")

        self.train_loader, self.val_loader = self.build_dataloader()
        self.cfg.dataset_classes = self.infer_num_classes()

        self.model = get_model(self.cfg).to(self.device)
        if self.num_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.logger(f"==> model [{self.model_name}]: {self.model}")

        self.criterion = nn.CrossEntropyLoss().to(self.device)
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

        self.best_acc1 = 0.0
        if self.cfg.resume:
            resume_state = self.saver.resume(self.cfg.resume)
            self.cfg.start_epoch = resume_state.get("epoch", self.cfg.start_epoch)
            self.best_acc1 = resume_state.get("best_acc1", 0.0)
        elif self.cfg.load:
            self.saver.load(self.cfg.load)

        cudnn.benchmark = self.device.type == "cuda"

    def add_arguments(self):
        parser = argparse.ArgumentParser("ViT Training")
        model_names = ["vit_tiny", "vit_small", "vit_base"]
        parser.add_argument(
            "-a",
            "--arch",
            metavar="ARCH",
            default="vit_small",
            choices=model_names,
            help="model architecture: " + " | ".join(model_names),
        )
        parser.add_argument("--data_path", type=str, default="/path/to/imagenet")
        parser.add_argument("--image-size", dest="image_size", type=int, default=224)
        parser.add_argument("--patch-size", dest="patch_size", type=int, default=16)
        parser.add_argument("--in-chans", dest="in_chans", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--val-batch-size", dest="val_batch_size", type=int, default=256)
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--num_classes", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--drop-path-rate", dest="drop_path_rate", type=float, default=0.1)
        parser.add_argument("--disable-train-shuffle", action="store_true")

        ext.trainer.add_arguments(parser)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.optimizer.add_arguments(parser)
        ext.scheduler.add_arguments(parser)

        parser.set_defaults(
            epochs=100,
            optimizer="adamw",
            lr=1e-4,
            weight_decay=0.1,
            lr_method="cos",
            seed=0,
        )

        args = parser.parse_args()
        if args.resume:
            args = parser.parse_args(namespace=ext.checkpoint.Checkpoint.load_config(args.resume))
        if args.lr_method == "cos" and args.lr_step == 30:
            args.lr_step = args.epochs
            args.lr_gamma = 0.0
        return args

    def infer_num_classes(self):
        if self.cfg.num_classes is not None:
            return self.cfg.num_classes
        if hasattr(self.train_loader.dataset, "classes"):
            return len(self.train_loader.dataset.classes)
        raise ValueError("Unable to infer num_classes from dataset. Please set --num_classes.")

    def build_dataloader(self):
        traindir = os.path.join(self.cfg.data_path, "train")
        valdir = os.path.join(self.cfg.data_path, "val")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.cfg.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(int(self.cfg.image_size * 256 / 224)),
                    transforms.CenterCrop(self.cfg.image_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        pin_memory = self.device.type == "cuda"
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=not self.cfg.disable_train_shuffle,
            num_workers=self.cfg.workers,
            pin_memory=pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.workers,
            pin_memory=pin_memory,
        )

        self.logger(f"==> train images: {len(train_dataset)}")
        self.logger(f"==> val images: {len(val_dataset)}")
        return train_loader, val_loader

    def train(self):
        if self.cfg.test:
            self.validate(self.cfg.start_epoch if self.cfg.start_epoch >= 0 else 0)
            return

        start_epoch = max(self.cfg.start_epoch, 0)
        for epoch in range(start_epoch, self.cfg.epochs):
            self.logger(f"============ Starting epoch {epoch} ============")
            self.train_epoch(epoch)
            acc1, acc5, val_loss = self.validate(epoch)
            self.scheduler.step()

            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(self.best_acc1, acc1)
            self.saver.save_checkpoint(
                "checkpoint.pth",
                epoch=epoch + 1,
                best_acc1=self.best_acc1,
                acc1=acc1,
                acc5=acc5,
                val_loss=val_loss,
            )
            if is_best:
                self.saver.save_model("best.pth")
                self.logger(f"==> best acc@1: {self.best_acc1:.3f}")

        now_date = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.logger(f"==> end time: {now_date}")

    def train_epoch(self, epoch):
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix=f"Epoch: [{epoch}]",
            logger=self.logger,
        )

        self.model.train()
        end = time.time()
        for i, (images, target) in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            images = images.to(self.device, non_blocking=self.device.type == "cuda")
            target = target.to(self.device, non_blocking=self.device.type == "cuda")

            output = self.model(images)
            loss = self.criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.cfg.print_f == 0:
                progress.display(i)

    def validate(self, epoch):
        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(
            len(self.val_loader),
            [batch_time, losses, top1, top5],
            prefix="Test: ",
            logger=self.logger,
        )

        self.model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(self.val_loader):
                images = images.to(self.device, non_blocking=self.device.type == "cuda")
                target = target.to(self.device, non_blocking=self.device.type == "cuda")

                output = self.model(images)
                loss = self.criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.cfg.print_f == 0:
                    progress.display(i)

        self.logger(
            f" * Epoch {epoch} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {losses.avg:.4f}"
        )
        return top1.avg, top5.avg, losses.avg


class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix="", logger=print):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


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
    trainer.train()
