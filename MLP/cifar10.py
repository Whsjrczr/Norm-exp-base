#!/usr/bin/env python3
import time
import os
import shutil
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image
import sys

from Taiyi.taiyi.monitor import Monitor
import wandb
from Taiyi.visualize import Visualization

sys.path.append('../')
import extension as ext
from model.selection_tool import add_model_arguments, get_model


def to_img(x):
    return x.clamp(0, 1)


class MNIST:
    def __init__(self):
        self.cfg = self.add_arguments()
        self.model_name = self.cfg.arch + '_' + ext.dataset.setting(self.cfg) + '_d' + str(self.cfg.depth) + '_w' + str(self.cfg.width) + '_' + ext.normalization.setting(self.cfg) + '_' + ext.activation.setting(self.cfg)
        self.model_name = self.model_name + '_lr' + str(self.cfg.lr) + '_bs' + str(
            self.cfg.batch_size[0]) + '_dropout' + str(self.cfg.dropout) + \
                           '_wd' + str(self.cfg.weight_decay) + '_seed' + str(self.cfg.seed)

        self.result_path = os.path.join(self.cfg.output, self.model_name, self.cfg.log_suffix)
        os.makedirs(self.result_path, exist_ok=True)

        self.logger = ext.logger.setting('log.txt', self.result_path, self.cfg.test, self.cfg.resume is not None)
        ext.trainer.setting(self.cfg)

        self.train_loader = ext.dataset.get_dataset_loader(self.cfg, train=True, use_cuda=False)
        self.val_loader = ext.dataset.get_dataset_loader(self.cfg, train=False, use_cuda=False)

        self.model = get_model(self.cfg)
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

        self.criterion = nn.MSELoss() if self.cfg.arch == 'AE' else nn.CrossEntropyLoss()
        ext.trainer.set_seed(self.cfg)

        taiyi_config = {
            ext.BatchNorm2dScaling: [['InputSndNorm', 'linear(5,0)'], ['OutputGradSndNorm', 'linear(5,0)']],
            nn.BatchNorm2d: [['InputSndNorm', 'linear(5,0)'], ['OutputGradSndNorm', 'linear(5,0)']],
            ext.LayerNormScaling: [['InputSndNorm', 'linear(5,0)'], ['OutputGradSndNorm', 'linear(5,0)']],
            nn.LayerNorm: [['InputSndNorm', 'linear(5,0)'], ['OutputGradSndNorm', 'linear(5,0)']],
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
                    "depth": self.cfg.depth,
                    "width": self.cfg.width,
                    "normalization": ext.normalization.setting(self.cfg),
                    "activation": ext.activation.setting(self.cfg),
                    "dropout_prob": self.cfg.dropout,
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

            if self.cfg.resume and hasattr(self, 'wandb_id') and self.wandb_id:
                print("resume wandb from id " + str(self.wandb_id))
                wandb_kwargs.update(dict(id=self.wandb_id, resume="must"))

            wandb.init(**wandb_kwargs)
            self.run_dir = os.path.dirname(wandb.run.dir)

            if self.cfg.taiyi:
                self.vis_wandb = Visualization(self.monitor, wandb)
                self.logger('==> taiyi config: {}'.format(taiyi_config))
            if self.cfg.vis:
                self.vis = ext.visualization.setting(
                    self.cfg,
                    self.model_name,
                    {'train loss': 'loss', 'test loss': 'loss', 'train accuracy': 'accuracy', 'test accuracy': 'accuracy'}
                )

    def add_arguments(self):
        parser = argparse.ArgumentParser('MNIST Classification')
        add_model_arguments(parser, task='classification')
        parser.add_argument('--offline', '-offline', action='store_true', help='offline mode')

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
        stages = getattr(ext.optimizer, "get_stages", lambda _cfg: None)(args)
        stage_total_epochs = getattr(ext.optimizer, "infer_total_epochs", lambda _stages: None)(stages)
        if stage_total_epochs is not None:
            args.epochs = stage_total_epochs
        return args

    def _rebuild_optim_sched_and_sync_saver(self):
        self.optimizer = ext.optimizer.setting(self.model, self.cfg)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.cfg)
        if hasattr(self, 'saver') and self.saver is not None:
            if hasattr(self.saver, 'optimizer'):
                self.saver.optimizer = self.optimizer
            if hasattr(self.saver, 'scheduler'):
                self.saver.scheduler = self.scheduler

    def _epoch_stage_plan(self):
        get_stages = getattr(ext.optimizer, 'get_stages', None)
        stages = get_stages(self.cfg) if callable(get_stages) else None
        explicit_total_epochs = getattr(ext.optimizer, 'infer_total_epochs', lambda _stages: None)(stages)

        epoch0 = self.cfg.start_epoch + 1
        epochN = explicit_total_epochs if explicit_total_epochs is not None else self.cfg.epochs

        if not stages:
            return [(1, None, epoch0, epochN)]

        plan = []
        cur = epoch0
        si = 1
        for st in stages:
            if cur >= epochN:
                break
            if st is None:
                continue

            if 'end_epoch' in st:
                end = int(st['end_epoch'])
            else:
                dur = st.get('epochs', st.get('epoch', None))
                end = cur + int(dur) if dur is not None else epochN

            end = max(cur, min(end, epochN))
            plan.append((si, st, cur, end))
            cur = end
            si += 1

        if explicit_total_epochs is None and cur < epochN:
            plan.append((si, stages[-1], cur, epochN))

        return plan

    def _apply_stage_to_cfg(self, st):
        if not st:
            return

        if 'optimizer' in st:
            self.cfg.optimizer = st['optimizer']
        elif 'name' in st:
            self.cfg.optimizer = st['name']

        if 'lr' in st:
            self.cfg.lr = st['lr']
        if 'weight_decay' in st:
            self.cfg.weight_decay = st['weight_decay']
        if 'optimizer_config' in st:
            self.cfg.optimizer_config = st['optimizer_config']
        if 'lr_method' in st:
            self.cfg.lr_method = st['lr_method']
        if 'lr_step' in st:
            self.cfg.lr_step = st['lr_step']
        if 'lr_gamma' in st:
            self.cfg.lr_gamma = st['lr_gamma']

    def train(self):
        if self.cfg.test:
            self.validate()
            return

        if not hasattr(self, 'step'):
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
                if self.cfg.lr_method != 'auto':
                    self.scheduler.step()

                self.train_epoch(epoch)

                if self.cfg.visualize:
                    wandb.log({"learning_rate": self.scheduler.get_last_lr()[0], "steps": self.step, "stage": si, "epochs": epoch})

                accuracy, val_loss = self.validate(epoch)

                if self.cfg.visualize:
                    self.saver.save_checkpoint(epoch=epoch, best_acc=self.best_acc, wandb_id=wandb.run.id, step=self.step)
                else:
                    self.saver.save_checkpoint(epoch=epoch, best_acc=self.best_acc)

                if self.cfg.lr_method == 'auto':
                    self.scheduler.step(val_loss)

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

    def train_epoch(self, epoch):
        self.logger('\nEpoch: {}, lr: {:.2g}, weight decay: {:.2g} on model {}'.format(
            epoch,
            self.optimizer.param_groups[0]['lr'],
            self.optimizer.param_groups[0]['weight_decay'],
            self.model_name
        ))
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        progress_bar = ext.ProgressBar(len(self.train_loader))
        for i, (inputs, targets) in enumerate(self.train_loader, 1):
            inputs = inputs.to(self.device)
            targets = inputs if self.cfg.arch == 'AE' else targets.to(self.device)

            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            if self.cfg.visualize and self.cfg.taiyi:
                self.monitor.track(self.step)
                self.vis_wandb.show(self.step)

            self.step += 1

            train_loss += losses.item() * targets.size(0)
            if self.cfg.arch == 'AE':
                correct = -train_loss
            else:
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(targets.view_as(pred)).sum().item()
            total += targets.size(0)
            if i % 10 == 0 or i == len(self.train_loader):
                progress_bar.step('Loss: {:.5g} | Accuracy: {:.2f}%'.format(
                    train_loss / total,
                    100. * correct / total
                ), 10)

        train_loss /= total
        accuracy = 100. * correct / total

        if self.cfg.visualize:
            wandb.log({"train_acc": accuracy, "train_loss": train_loss, "epochs": epoch, "steps": self.step})
        if self.cfg.vis:
            self.vis.add_value('train loss', train_loss)
            self.vis.add_value('train accuracy', accuracy)

        self.logger('Train on epoch {}: average loss={:.5g}, accuracy={:.2f}% ({}/{}), time: {}'.format(
            epoch, train_loss, accuracy, correct, total, progress_bar.time_used()
        ))

    def validate(self, epoch=-1):
        test_loss = 0
        correct = 0
        total = 0
        progress_bar = ext.ProgressBar(len(self.val_loader))
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = inputs if self.cfg.arch == 'AE' else targets.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets).item() * targets.size(0)
                if self.cfg.arch == 'AE':
                    correct = -test_loss
                else:
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

        self.logger('Test on epoch {}: average loss={:.5g}, accuracy={:.2f}% ({}/{}), time: {}'.format(
            epoch, test_loss, accuracy, correct, total, progress_bar.time_used()
        ))

        if not self.cfg.test and accuracy > self.best_acc:
            self.best_acc = accuracy
            self.saver.save_model('best.pth')
            self.logger('==> best accuracy: {:.2f}%'.format(self.best_acc))

        if self.cfg.arch == 'AE':
            pic = to_img(outputs[:64].cpu().data)
            save_image(pic, os.path.join(self.result_path, 'result_{}.png').format(epoch))

        return accuracy, test_loss


if __name__ == '__main__':
    Cs = MNIST()
    torch.set_num_threads(1)
    Cs.train()
