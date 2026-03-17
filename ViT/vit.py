#!/usr/bin/env python3
import time
import os
import shutil
import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torchvision import datasets
import sys

from Taiyi.taiyi.monitor import Monitor
import wandb
from Taiyi.visualize import Visualization

sys.path.append('../')
import extension as ext
from model_vit.select_vit import vit_tiny, vit_small, vit_base


class ViT:
    def __init__(self):
        self.cfg = self.add_arguments()
        self.model_name = self.cfg.arch + '_' + ext.dataset.setting(self.cfg) + '_' + ext.normalization.setting(self.cfg) + '_' + ext.activation.setting(self.cfg)
        self.model_name = self.model_name + '_lr' + str(self.cfg.lr) + '_bs' + str(
            self.cfg.batch_size[0]) + '_wd' + str(self.cfg.weight_decay) + '_seed' + str(self.cfg.seed)
        
        self.result_path = os.path.join(self.cfg.output, self.model_name, self.cfg.log_suffix)
        os.makedirs(self.result_path, exist_ok=True)
        self.logger = ext.logger.setting('log.txt', self.result_path, self.cfg.test, self.cfg.resume is not None)
        ext.trainer.setting(self.cfg)
        
        # dataset loader
        self.train_loader = ext.dataset.get_dataset_loader(self.cfg, train=True, use_cuda=False)
        self.val_loader = ext.dataset.get_dataset_loader(self.cfg, train=False, use_cuda=False)

        self.model = self.get_model(self.cfg)
        self.logger('==> model [{}]: {}'.format(self.model_name, self.model))

        self.optimizer = ext.optimizer.setting(self.model, self.cfg)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.cfg)

        self.saver = ext.checkpoint.Checkpoint(self.model, self.cfg, self.optimizer, self.scheduler, self.result_path, not self.cfg.test)
        self.saver.load(self.cfg.load)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.num_gpu = torch.cuda.device_count()
        self.logger('==> use {:d} GPUs'.format(self.num_gpu))
        if self.num_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        self.best_acc = 0
        if self.cfg.visualize and self.cfg.offline:
            os.environ['WANDB_MODE'] = 'offline'
        if self.cfg.resume:
            saved = self.saver.resume(self.cfg.resume)
            self.cfg.start_epoch = saved['epoch']
            if self.cfg.visualize and 'wandb_id' in saved.keys():
                self.wandb_id = saved['wandb_id']
                self.step = saved['step']
            self.best_acc = saved['best_acc']
            self.cfg.seed = saved['seed']

        self.criterion = nn.CrossEntropyLoss()
        ext.trainer.set_seed(self.cfg)

        taiyi_config = {
            ext.BatchNorm2dScaling: [['InputSndNorm','linear(5,0)'],['OutputGradSndNorm','linear(5,0)']],
            nn.BatchNorm2d: [['InputSndNorm','linear(5,0)'],['OutputGradSndNorm','linear(5,0)']],
            ext.LayerNormScaling: [['InputSndNorm','linear(5,0)'],['OutputGradSndNorm','linear(5,0)']],
            nn.LayerNorm: [['InputSndNorm','linear(5,0)'],['OutputGradSndNorm','linear(5,0)']],
        }
        if self.cfg.visualize:
            if self.cfg.taiyi:
                self.monitor = Monitor(self.model, taiyi_config)

            wandb_kwargs = dict(
                project=self.cfg.wandb_project if hasattr(self.cfg, "wandb_project") else "test",
                entity="whsjrc-buaa",
                name=self.model_name,
                notes=str(self.cfg) + " ---- " + str(taiyi_config),
                config={
                    "model": self.cfg.arch,
                    "normalization": ext.normalization.setting(self.cfg),
                    "activation": self.cfg.activation,

                    "optimizer": self.cfg.optimizer,
                    "learning_rate": self.cfg.lr,
                    "batch_size": self.cfg.batch_size[0],
                    "weight_decay": self.cfg.weight_decay,

                    "dataset": self.cfg.dataset,
                    "epochs": self.cfg.epochs,
                    "seed": self.cfg.seed,

                    "scheduler": getattr(self.cfg, "lr_method", None),
                    "scheduler_cfg": f"step{getattr(self.cfg,'lr_step',None)}_gamma{getattr(self.cfg,'lr_gamma',None)}",
                },
            )

            if self.cfg.resume and self.wandb_id:
                print("resume wandb from id " + str(self.wandb_id))
                wandb_kwargs.update(dict(id=self.wandb_id, resume="must"))

            wandb.init(**wandb_kwargs)
            self.run_dir = os.path.dirname(wandb.run.dir)

            if self.cfg.taiyi:
                self.vis_wandb = Visualization(self.monitor, wandb)
                self.logger('==> taiyi config: {}'.format(taiyi_config))
            if self.cfg.vis:
                self.vis = ext.visualization.setting(self.cfg, self.model_name,
                                                {'train loss': 'loss', 'test loss': 'loss', 'train accuracy': 'accuracy',
                                                'test accuracy': 'accuracy'})
        return

    def get_model(self, cfg):
        """获取ViT模型"""
        model_dict = {
            'vit_tiny': vit_tiny,
            'vit_small': vit_small,
            'vit_base': vit_base,
        }
        
        if cfg.arch not in model_dict:
            raise ValueError(f"Model {cfg.arch} not found. Available: {list(model_dict.keys())}")
        
        model = model_dict[cfg.arch](num_classes=cfg.num_classes)
        return model

    def add_arguments(self):
        parser = argparse.ArgumentParser('ViT Classification')
        model_names = ['vit_tiny', 'vit_small', 'vit_base']
        parser.add_argument('-a', '--arch', metavar='ARCH', default=model_names[1], choices=model_names,
                            help='model architecture: ' + ' | '.join(model_names))
        parser.add_argument('--num_classes', default=1000, type=int, help='number of classes')
        
        # Common training arguments
        ext.trainer.add_arguments(parser)
        parser.set_defaults(epochs=500)
        
        ext.dataset.add_arguments(parser)
        parser.set_defaults(dataset='cifar10', workers=1, batch_size=[64, 1000])
        
        ext.scheduler.add_arguments(parser)
        parser.set_defaults(lr_method='fix', lr=1e-3)
        
        ext.optimizer.add_arguments(parser)
        parser.set_defaults(optimizer='adam', weight_decay=1e-5)
        
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.normalization.add_arguments(parser)
        ext.activation.add_arguments(parser)
        ext.vis_taiyi.add_arguments(parser)
        ext.visualization.add_arguments(parser)
        
        args = parser.parse_args()
        if args.resume:
            args = parser.parse_args(namespace=ext.checkpoint.Checkpoint.load_config(args.resume))
        return args

    def train(self):
        if self.cfg.test:
            self.validate()
            return
        
        if not hasattr(self, 'step'):
            self.step = 0
            
        for epoch in range(self.cfg.start_epoch + 1, self.cfg.epochs):
            if self.cfg.lr_method != 'auto':
                self.scheduler.step()

            self.train_epoch(epoch)
            if self.cfg.visualize:
                wandb.log({"learning_rate": self.scheduler.get_last_lr()[0], "steps": self.step})

            accuracy, val_loss = self.validate(epoch)
            if self.cfg.visualize:
                self.saver.save_checkpoint(epoch=epoch, best_acc=self.best_acc, wandb_id=wandb.run.id, step=self.step)
            else:
                self.saver.save_checkpoint(epoch=epoch, best_acc=self.best_acc)
            if self.cfg.lr_method == 'auto':
                self.scheduler.step(val_loss)
        
        # finish train
        now_date = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.logger('==> end time: {}'.format(now_date))
        
        if self.cfg.visualize and self.cfg.taiyi:
            self.vis_wandb.close()
            self.monitor.get_output()
            self.logger("==> Wandb successfully get output.")

        new_log_filename = r'{}_{}_{:5.2f}%%.txt'.format(self.model_name, now_date, self.best_acc)
        self.logger('==> Network training completed. Copy log file to {}'.format(new_log_filename))
        if self.cfg.offline and self.cfg.visualize:
            print(f"syncing wandb...{self.run_dir}")
            os.system(f"wandb sync {self.run_dir}")
        new_log_path = os.path.join(self.result_path, new_log_filename)
        shutil.copy(self.logger.filename, new_log_path)
        return

    def train_epoch(self, epoch):
        self.logger('\nEpoch: {}, lr: {:.2g}, weight decay: {:.2g} on model {}'.format(epoch,
            self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['weight_decay'], self.model_name))
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        progress_bar = ext.ProgressBar(len(self.train_loader))
        for i, (inputs, targets) in enumerate(self.train_loader, 1):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # compute output
            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            if self.cfg.visualize and self.cfg.taiyi:
                self.monitor.track(self.step)
                self.vis_wandb.show(self.step)
            self.step += 1

            # measure accuracy and record loss
            train_loss += losses.item() * targets.size(0)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total += targets.size(0)
            if i % 10 == 0 or i == len(self.train_loader):
                progress_bar.step('Loss: {:.5g} | Accuracy: {:.2f}%'.format(train_loss / total, 100. * correct / total), 10)
        train_loss /= total
        accuracy = 100. * correct / total
        if self.cfg.visualize:
            wandb.log({"train_acc": accuracy, "train_loss": train_loss, "epochs": epoch, "steps": self.step})
        if self.cfg.vis:
            self.vis.add_value('train loss', train_loss)
            self.vis.add_value('train accuracy', accuracy)
        self.logger(
            'Train on epoch {}: average loss={:.5g}, accuracy={:.2f}% ({}/{}), time: {}'.format(epoch, train_loss,
                accuracy, correct, total, progress_bar.time_used()))
        return

    def validate(self, epoch=-1):
        test_loss = 0
        correct = 0
        total = 0
        progress_bar = ext.ProgressBar(len(self.val_loader))
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets).item() * targets.size(0)
                prediction = outputs.max(1, keepdim=True)[1]
                correct += prediction.eq(targets.view_as(prediction)).sum().item()
                total += targets.size(0)
                progress_bar.step('Loss: {:.5g} | Accuracy: {:.2f}%'.format(test_loss / total, 100. * correct / total))
        test_loss /= total
        accuracy = correct * 100. / total
        if self.cfg.vis:
            self.vis.add_value('test loss', test_loss)
            self.vis.add_value('test accuracy', accuracy)
        if self.cfg.visualize:
            wandb.log({"test_acc": accuracy, "test_loss": test_loss, "epochs": epoch, "steps": self.step})
        self.logger('Test on epoch {}: average loss={:.5g}, accuracy={:.2f}% ({}/{}), time: {}'.format(epoch, test_loss,
            accuracy, correct, total, progress_bar.time_used()))
        if not self.cfg.test and accuracy > self.best_acc:
            self.best_acc = accuracy
            self.saver.save_model('best.pth')
            self.logger('==> best accuracy: {:.2f}%'.format(self.best_acc))
        return accuracy, test_loss


if __name__ == '__main__':
    vit = ViT()
    torch.set_num_threads(1)
    vit.train()
