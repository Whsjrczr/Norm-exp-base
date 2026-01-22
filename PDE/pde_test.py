#!/usr/bin/env python3
import os
os.environ['DDE_BACKEND'] = 'pytorch'
import time
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt

import deepxde as dde
import torch
import wandb

from Taiyi.taiyi.monitor import Monitor
from Taiyi.visualize import Visualization

import sys

sys.path.append('../')
import extension as ext
from extension.utils import str2list

sys.path.append('../MLP')
from model.selection_tool import get_model
from pde_dataset import PDEBuilder


class PDETrainer:
    def __init__(self):
        self.cfg = self.add_arguments()
        self.model_name = 'PDE_' + self.cfg.pde_type + '_' + self.cfg.arch + '_d' + str(self.cfg.depth) + '_w' + str(
            self.cfg.width) + '_' + ext.normalization.setting(self.cfg) + '_' + ext.activation.setting(self.cfg) + '_lr' + str(self.cfg.lr) + '_epochs' + str(self.cfg.epochs) + '_seed' + str(self.cfg.seed)
        self.result_path = os.path.join(self.cfg.output, self.model_name, self.cfg.log_suffix)
        os.makedirs(self.result_path, exist_ok=True)
        self.logger = ext.logger.setting('log.txt', self.result_path, self.cfg.test, self.cfg.resume is not None)
        ext.trainer.setting(self.cfg)
        if self.cfg.float64:
            dde.config.set_default_float("float64")
            torch.set_default_dtype(torch.float64)
        # Set cfg for model selection
        self.cfg.im_size = [1]
        self.cfg.dataset_classes = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpu = torch.cuda.device_count()
        self.logger('==> use {:d} GPUs'.format(self.num_gpu))
        if self.num_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        self.model = get_model(self.cfg).to(self.device)
        self.logger('==> model [{}]: {}'.format(self.model_name, self.model))

        self.optimizer = ext.optimizer.setting(self.model, self.cfg)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.cfg)

        self.saver = ext.checkpoint.Checkpoint(self.model, self.cfg, self.optimizer, self.scheduler, self.result_path, not self.cfg.test)
        self.saver.load(self.cfg.load)

        # Define PDE problem
        pde_builder = PDEBuilder(self.cfg, self.model, self.optimizer)
        self.data, self.net, self.model = pde_builder.build()
        if hasattr(self.net, "to"):
            self.net = self.net.to(self.device)

        self.logger('==> model [{}]: {}'.format(self.model_name, self.model))

        ext.trainer.set_seed(self.cfg)

        # Monitor and wandb setup (simplified)
        taiyi_config = {}  # No specific layers for PDE
        self.monitor = Monitor(self.net, taiyi_config)

        if self.cfg.visualize:
            if self.cfg.offline:
                os.environ['WANDB_MODE'] = 'offline'
            if self.cfg.resume and hasattr(self, 'wandb_id'):
                print("resume wandb from id "+str(self.wandb_id))
                wandb.init(
                    project=self.subject_name,
                    entity="whsjrc-buaa",
                    name=self.model_name,
                    id=self.wandb_id,
                    resume="must",
                    notes=str(self.cfg),
                    config={
                        "pde_type": self.cfg.pde_type,
                        "arch": self.cfg.arch,
                        "depth": self.cfg.depth,
                        "width": self.cfg.width,
                        "normalization": ext.normalization.setting(self.cfg),
                        "num_per_group": self.cfg.norm_cfg.get('num_per_group', None),
                        "activation": self.cfg.activation,
                        "learning_rate": self.cfg.lr,
                        "epochs": self.cfg.epochs,
                        "seed": self.cfg.seed,
                        "optimizer": self.cfg.optimizer,
                        "scheduler": self.cfg.scheduler,
                        "scheduler_cfg": self.cfg.scheduler_cfg,
                    }
                )
            else:
                wandb.init(
                    project="PDE Solving Updated2",
                    entity="whsjrc-buaa",
                    name=self.model_name,
                    notes=str(self.cfg),
                    config={
                        "pde_type": self.cfg.pde_type,
                        "arch": self.cfg.arch,
                        "depth": self.cfg.depth,
                        "width": self.cfg.width,
                        "normalization": ext.normalization.setting(self.cfg),
                        "num_per_group": self.cfg.norm_cfg.get('num_per_group', None),
                        "activation": self.cfg.activation,
                        "learning_rate": self.cfg.lr,
                        "epochs": self.cfg.epochs,
                        "seed": self.cfg.seed,
                        "optimizer": self.cfg.optimizer,
                        "scheduler_cfg": self.cfg.scheduler_cfg,
                    }
                )
            self.run_dir = os.path.dirname(wandb.run.dir)
            self.vis_wandb = Visualization(self.monitor, wandb)
        else:
            self.vis_wandb = None
        return

    def add_arguments(self):
        parser = argparse.ArgumentParser('PDE Solving')
        model_names = ['MLP', 'PreNormMLP', 'CenDropScalingMLP', 'CenDropScalingPreNormMLP', 'ResCenDropScalingMLP']
        parser.add_argument('-a', '--arch', metavar='ARCH', default=model_names[0], choices=model_names,
                            help='model architecture: ' + ' | '.join(model_names))
        parser.add_argument('-width', '--width', type=int, default=50)
        parser.add_argument('-depth', '--depth', type=int, default=3)
        parser.add_argument('-dropout', '--dropout', type=float, default=0)
        parser.add_argument('--pde_type', default='poisson', choices=['poisson', 'helmholtz', 'helmholtz2d', 'allen_cahn', 'wave', 'klein_gordon', 'convdiff', 'cavity'], help='PDE type')
        parser.add_argument('--offline', action='store_true', help='offline mode')
        parser.add_argument('--visualize', action='store_true', help='use wandb for visualization and logging')
        parser.add_argument('--no_save_best', action='store_true', help='do not save best model during training')
        parser.add_argument('--display_every', type=int, default=1000, help='display and log every N iterations')
        parser.add_argument('--metrics', type=str2list, default='l2 relative error', help='comma-separated list of metrics to evaluate')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--float64', action='store_true', help='train with float64 precision')
        parser.add_argument('--subject_name', type=str, default='default_subject', help='subject name for logging')
        ext.trainer.add_arguments(parser)
        parser.set_defaults(epochs=10000)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.normalization.add_arguments(parser)
        ext.activation.add_arguments(parser)
        ext.optimizer.add_arguments(parser)
        ext.scheduler.add_arguments(parser)
        args = parser.parse_args()
        if args.resume:
            args = parser.parse_args(namespace=ext.checkpoint.Checkpoint.load_config(args.resume))
        return args

    def train(self):
        if self.cfg.test:
            self.validate()
            return
        # Train model
        losshistory, train_state = self.model.train(iterations=self.cfg.epochs, batch_size=self.cfg.batch_size, display_every=self.cfg.display_every)

        self.logger('==> Training completed.')
        # Log to wandb
        if self.cfg.visualize:
            for i in range(len(losshistory.loss_train)):
                log_dict = {"iterations": i + 1}
                train_losses = losshistory.loss_train[i]
                test_losses = losshistory.loss_test[i]
                metrics = losshistory.metrics_test[i]
                for j in range(len(train_losses)):
                    log_dict[f"train_loss_{j}"] = float(train_losses[j])
                for j in range(len(test_losses)):
                    log_dict[f"test_loss_{j}"] = float(test_losses[j])
                for j in range(len(metrics)):
                    metric_name = self.cfg.metrics[j].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('.', '')
                    log_dict[f"metrics_{metric_name}"] = float(metrics[j])
                wandb.log(log_dict)
            val_error = self.validate()
            wandb.log({"val_error": val_error})
        # Save model
        if not self.cfg.no_save_best:
            self.model.save(os.path.join(self.result_path, 'model'))
        # Finish
        now_date = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.logger('==> end time: {}'.format(now_date))

        if self.cfg.visualize:
            self.vis_wandb.close()
            self.monitor.get_output()
            self.logger("==> Wandb successfully get output.")

            new_log_filename = r'{}_{}.txt'.format(self.model_name, now_date)
            self.logger('==> Network training completed. Copy log file to {}'.format(new_log_filename))
            if self.cfg.offline:
                print(f"syncing wandb...{self.run_dir}")
                os.system(f"wandb sync {self.run_dir}")
            new_log_path = os.path.join(self.result_path, new_log_filename)
            shutil.copy(self.logger.filename, new_log_path)
        return

    def validate(self):
        # Predict and compute error
        x = np.linspace(-1, 1, 100).reshape(-1, 1)
        y_pred = self.model.predict(x)
        if self.cfg.pde_type == 'poisson':
            y_true = (x**2 - 1) / 2
        elif self.cfg.pde_type == 'helmholtz':
            y_true = np.sin(np.pi * x[:, 0])
        elif self.cfg.pde_type == 'allen_cahn':
            epsilon = 0.01
            y_true = np.tanh(x[:, 0] / np.sqrt(2 * epsilon))
        error = np.mean((y_pred - y_true)**2)
        self.logger('==> Validation L2 error: {:.5g}'.format(error))
        if self.cfg.visualize:
            wandb.log({"val_error": error})
        self.best_loss = getattr(self, 'best_loss', float('inf'))
        if not self.cfg.test and not self.cfg.no_save_best and error < self.best_loss:
            self.best_loss = error
            self.saver.save_model('best.pth')
            self.logger('==> best loss: {:.5g}'.format(self.best_loss))
        return error

if __name__ == '__main__':
    trainer = PDETrainer()
    trainer.train()
