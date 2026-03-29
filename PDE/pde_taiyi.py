#!/usr/bin/env python3
import os
os.environ['DDE_BACKEND'] = 'pytorch'
import time
import shutil
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import deepxde as dde
import torch

import sys

sys.path.append('../')
import extension as ext
from extension.utils import str2list

sys.path.append('../MLP')
from model.selection_tool import add_model_arguments, get_model
from pde_dataset import PDEBuilder


PDE_INPUT_DIMS = {
    'poisson': 1,
    'poisson_new': 1,
    'helmholtz': 1,
    'helmholtz_new': 1,
    'helmholtz_learnable_2': 1,
    'allen_cahn': 1,
    'allen_cahn_new': 1,
    'helmholtz2d': 2,
    'helmholtz_2d': 2,
    'wave': 2,
    'klein_gordon': 2,
    'convdiff': 3,
    'cavity': 3,
}


class TaiyiMonitorCallback(dde.callbacks.Callback):
    def __init__(self, taiyi_tracker, track_every):
        super().__init__()
        self.taiyi = taiyi_tracker
        self.track_every = max(int(track_every), 1)
        self._fallback_step = 0
        self._last_tracked_step = None

    def _track_if_needed(self):
        if not self.taiyi.enabled:
            return
        train_state = getattr(self.model, "train_state", None)
        step = None
        for attr in ("step", "iteration", "epoch"):
            value = getattr(train_state, attr, None)
            if value is not None:
                step = int(value)
                break
        if step is None:
            self._fallback_step += 1
            step = self._fallback_step
        if step % self.track_every == 0 and step != self._last_tracked_step:
            self.taiyi.track(step)
            self._last_tracked_step = step

    def on_epoch_end(self):
        self._track_if_needed()

    def on_batch_end(self):
        self._track_if_needed()


class PDETaiyiTrainer:
    def __init__(self):
        self.cfg = self.add_arguments()
        self.model_name = 'PDE_TAIYI_' + self.cfg.pde_type + '_' + self.cfg.arch + '_d' + str(self.cfg.depth) + '_w' + str(
            self.cfg.width) + '_' + ext.normalization.setting(self.cfg) + '_' + ext.activation.setting(self.cfg) + '_lr' + str(self.cfg.lr) + '_epochs' + str(self.cfg.epochs) + '_seed' + str(self.cfg.seed)
        self.result_path = os.path.join(self.cfg.output, self.model_name, self.cfg.log_suffix)
        os.makedirs(self.result_path, exist_ok=True)
        self.logger = ext.logger.setting('log.txt', self.result_path, self.cfg.test, self.cfg.resume is not None)
        ext.trainer.setting(self.cfg)
        if self.cfg.float64:
            dde.config.set_default_float("float64")
            torch.set_default_dtype(torch.float64)
            dde.backend.torch.torch.set_default_dtype(torch.float64)
        else:
            dde.config.set_default_float("float32")
            torch.set_default_dtype(torch.float32)
            dde.backend.torch.torch.set_default_dtype(torch.float32)

        self.cfg.im_size = [PDE_INPUT_DIMS[self.cfg.pde_type]]
        self.cfg.dataset_classes = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.num_gpu = torch.cuda.device_count()
        self.logger('==> use {:d} GPUs'.format(self.num_gpu))

        self.model = get_model(self.cfg).to(self.device)
        if self.num_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.logger('==> model [{}]: {}'.format(self.model_name, self.model))

        self.optimizer = ext.optimizer.setting(self.model, self.cfg)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.cfg)

        self.saver = ext.checkpoint.Checkpoint(self.model, self.cfg, self.optimizer, self.scheduler, self.result_path, not self.cfg.test)
        self.saver.load(self.cfg.load)

        pde_builder = PDEBuilder(self.cfg, self.model, self.optimizer)
        self.data, self.net, self.model = pde_builder.build()
        if hasattr(self.net, "to"):
            self.net = self.net.to(self.device)

        self.logger('==> model [{}]: {}'.format(self.model_name, self.model))

        ext.trainer.set_seed(self.cfg)

        taiyi_config = {
            'Linear': [
                ['InputCovStableRank', f'linear({self.cfg.taiyi_track_every},0)'],
                ['InputCovCondition50', f'linear({self.cfg.taiyi_track_every},0)'],
                ['InputCovMaxEig', f'linear({self.cfg.taiyi_track_every},0)'],
                ['InputSndNorm', f'linear({self.cfg.taiyi_track_every},0)'],
                ['OutputGradSndNorm', f'linear({self.cfg.taiyi_track_every},0)'],
                ['WeightNorm', f'linear({self.cfg.taiyi_track_every},0)'],
                ['WeightGradNorm', f'linear({self.cfg.taiyi_track_every},0)'],
                ['LinearDeadNeuronNum', f'linear({self.cfg.taiyi_track_every},0)'],
            ],
        }
        taiyi_modules = list(taiyi_config.keys())
        taiyi_quantities = {
            module_name: [item[0] if isinstance(item, (list, tuple)) else item for item in module_cfg]
            for module_name, module_cfg in taiyi_config.items()
        }

        wandb_kwargs = dict(
            project=self.cfg.subject_name,
            entity="whsjrc-buaa",
            name=self.model_name,
            notes=str(self.cfg) + " ---- " + json.dumps(taiyi_config, ensure_ascii=False),
            config={
                "pde_type": self.cfg.pde_type,
                "arch": self.cfg.arch,
                "depth": self.cfg.depth,
                "width": self.cfg.width,
                "normalization": ext.normalization.setting(self.cfg),
                "num_per_group": self.cfg.norm_cfg.get('num_per_group', None),
                "activation": ext.activation.setting(self.cfg),
                "learning_rate": self.cfg.lr,
                "epochs": self.cfg.epochs,
                "seed": self.cfg.seed,
                "optimizer": self.cfg.optimizer,
                "scheduler": self.cfg.lr_method,
                "scheduler_cfg": "step" + str(self.cfg.lr_step) + "_gamma" + str(self.cfg.lr_gamma),
                "loss_weights": self.cfg.loss_weights,
                "taiyi_track_every": self.cfg.taiyi_track_every,
                "taiyi_modules": taiyi_modules,
                "taiyi_quantities": taiyi_quantities,
                "taiyi_config": taiyi_config,
            },
        )
        if self.cfg.resume and hasattr(self, 'wandb_id'):
            print("resume wandb from id " + str(self.wandb_id))
            wandb_kwargs.update(dict(id=self.wandb_id, resume="must"))

        self.visualizer = ext.tracking.setting(
            self.cfg,
            env_name=self.model_name,
            vis_names=self._build_vis_names(),
            wandb_kwargs=wandb_kwargs,
        )
        self.taiyi = ext.taiyi.setting(
            self.cfg,
            model=self.net,
            monitor_config=taiyi_config,
            wandb=self.visualizer.wandb,
        )
        if self.taiyi.enabled:
            self.logger("==> taiyi config: {}".format(taiyi_config))

    def add_arguments(self):
        raw_argv = sys.argv[1:]
        parser = argparse.ArgumentParser('PDE Solving with Taiyi Layer Dynamics Tracking')
        add_model_arguments(parser, task='pde')
        parser.add_argument('--pde_type', default='poisson', choices=['poisson', 'helmholtz', 'helmholtz2d', 'helmholtz_2d', 'allen_cahn', 'wave', 'klein_gordon', 'convdiff', 'cavity', 'helmholtz_new', 'helmholtz_learnable_2', 'poisson_new', 'allen_cahn_new'], help='PDE type')
        parser.add_argument('--loss-weights', type=str2list, default='1.0,1.0', help='comma-separated list of loss weights')
        parser.add_argument('--offline', action='store_true', help='offline mode')
        parser.add_argument('--no_save_best', action='store_true', help='do not save best model during training')
        parser.add_argument('--display_every', type=int, default=1000, help='display and log every N iterations')
        parser.add_argument('--metrics', type=str2list, default='l2 relative error', help='comma-separated list of metrics to evaluate')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--float64', action='store_true', help='train with float64 precision')
        parser.add_argument('--subject_name', type=str, default='default_subject', help='subject name for logging')
        parser.add_argument('--taiyi_track_every', type=int, default=1000, help='track Taiyi layer dynamics every N iterations')
        ext.trainer.add_arguments(parser)
        parser.set_defaults(epochs=10000)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.normalization.add_arguments(parser)
        ext.activation.add_arguments(parser)
        ext.optimizer.add_arguments(parser)
        ext.scheduler.add_arguments(parser)
        ext.tracking.add_arguments(parser)
        ext.taiyi.add_arguments(parser)
        args = parser.parse_args()
        if args.pde_type == 'helmholtz_2d':
            args.pde_type = 'helmholtz_2d'
        if args.resume:
            args = parser.parse_args(namespace=ext.checkpoint.Checkpoint.load_config(args.resume))

        if not self._has_cli_flag(raw_argv, '--optimizer', '-oo'):
            args.optimizer = 'adam'
        if not self._has_cli_flag(raw_argv, '--lr'):
            args.lr = 1e-3
        if args.pde_type in {'helmholtz2d', 'helmholtz_2d'}:
            if not self._has_cli_flag(raw_argv, '--loss-weights'):
                args.loss_weights = [1.0, 10.0]
            if not self._has_cli_flag(raw_argv, '--float64'):
                args.float64 = True

        stages = getattr(ext.optimizer, "get_stages", lambda _cfg: None)(args)
        stage_total_iterations = getattr(ext.optimizer, "infer_total_iterations", lambda _stages: None)(stages)
        if stage_total_iterations is not None:
            args.epochs = stage_total_iterations
        if not getattr(args, "taiyi", False):
            args.taiyi = True
        if not getattr(args, "wandb", False):
            args.wandb = True
            args.visualize = True
        return ext.tracking.normalize_config(args)

    def _has_cli_flag(self, argv, *flags):
        for flag in flags:
            if flag in argv:
                return True
            prefix = flag + '='
            if any(arg.startswith(prefix) for arg in argv):
                return True
        return False

    def train(self):
        if self.cfg.test:
            self.validate()
            return

        callbacks = [TaiyiMonitorCallback(self.taiyi, self.cfg.taiyi_track_every)]
        stages = getattr(ext.optimizer, "get_stages", lambda _cfg: None)(self.cfg)
        all_histories = []
        iter_offset = 0

        if stages:
            self.logger(f"==> Multi-stage training enabled: {len(stages)} stages")
            for stage_idx, stage_cfg in enumerate(stages, 1):
                losshistory, train_state = self._train_stage(stage_idx, stage_cfg, callbacks=callbacks)
                all_histories.append((stage_idx, stage_cfg, losshistory, train_state, iter_offset))
                if losshistory is not None and hasattr(losshistory, "loss_train"):
                    iter_offset += len(losshistory.loss_train)
        else:
            losshistory, train_state = self.model.train(
                iterations=self.cfg.epochs,
                batch_size=self.cfg.batch_size,
                display_every=self.cfg.display_every,
                callbacks=callbacks,
            )
            all_histories.append((1, None, losshistory, train_state, 0))

        self.logger('==> Training completed.')

        if self.visualizer.wandb_enabled:
            self._log_histories_to_wandb(all_histories)
        if self.visualizer.visdom_enabled:
            self._log_histories_to_vis(all_histories)

        self.validate()
        # self._save_solution_plot()

        if not self.cfg.no_save_best:
            self.model.save(os.path.join(self.result_path, 'model'))

        now_date = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.logger('==> end time: {}'.format(now_date))

        taiyi_info = self.taiyi.finish()
        finish_info = self.visualizer.finish(sync_offline=self.cfg.offline)
        if taiyi_info["taiyi_output"]:
            self.logger("==> Taiyi output exported.")
        if finish_info["synced"]:
            self.logger("==> WandB offline run synced.")

        if self.visualizer.wandb_enabled:
            new_log_filename = r'{}_{}.txt'.format(self.model_name, now_date)
            self.logger('==> Network training completed. Copy log file to {}'.format(new_log_filename))
            new_log_path = os.path.join(self.result_path, new_log_filename)
            shutil.copy(self.logger.filename, new_log_path)

    def validate(self):
        if self._supports_curve_validation():
            _x, y_pred, y_true = self._predict_validation_curve()
            error = np.mean((y_pred - y_true)**2)
            self.logger('==> Validation L2 error: {:.5g}'.format(error))
        elif self._supports_grid_validation():
            _xy, y_pred, y_true, _xx, _yy = self._predict_validation_grid()
            error = np.mean((y_pred - y_true)**2)
            self.logger('==> Validation grid MSE: {:.5g}'.format(error))
        else:
            self.logger('==> Validation skipped: analytical reference is unavailable for this PDE.')
            return None

        if not np.isfinite(error):
            self.logger('==> Validation produced non-finite error; skip val logging and best-checkpoint update.')
            return None

        self.visualizer.log({"val_error": error})
        self.visualizer.add_value("val error", error)
        self.best_loss = getattr(self, 'best_loss', float('inf'))
        if not self.cfg.test and not self.cfg.no_save_best and error < self.best_loss:
            self.best_loss = error
            self.saver.save_model('best.pth')
            self.logger('==> best loss: {:.5g}'.format(self.best_loss))
        return error

    def _train_stage(self, stage_idx, stage_cfg, callbacks=None):
        opt_name = (stage_cfg.get("optimizer") or stage_cfg.get("name") or self.cfg.optimizer)
        lr = stage_cfg.get("lr", getattr(self.cfg, "lr", None))
        batch_size = stage_cfg.get("batch_size", self.cfg.batch_size)
        iterations = stage_cfg.get("iterations", stage_cfg.get("iters", stage_cfg.get("epochs", None)))
        loss_weights = stage_cfg.get("loss_weights", self.cfg.loss_weights)
        metrics = stage_cfg.get("metrics", self.cfg.metrics)
        optimizer_config = stage_cfg.get("optimizer_config", stage_cfg.get("config", {})) or {}

        opt_lower = str(opt_name).lower()
        if opt_lower in {"lbfgs", "l-bfgs", "l_bfgs"}:
            dde_opt = "L-BFGS"
            ext.optimizer.configure_dde_lbfgs(optimizer_config, logger=self.logger)
            compile_kwargs = {"loss_weights": loss_weights, "metrics": metrics}
            self.model.compile(dde_opt, **compile_kwargs)
        else:
            dde_opt = opt_lower
            compile_kwargs = {"lr": lr, "loss_weights": loss_weights, "metrics": metrics}
            self.model.compile(dde_opt, **compile_kwargs)

        self.logger(
            f"==> Stage {stage_idx}: optimizer={dde_opt}, lr={lr}, "
            f"iters={iterations}, batch_size={batch_size}, opt_cfg={optimizer_config}"
        )

        if iterations is None and dde_opt == "L-BFGS":
            return self.model.train(display_every=self.cfg.display_every, callbacks=callbacks)
        return self.model.train(
            iterations=int(iterations) if iterations is not None else self.cfg.epochs,
            batch_size=batch_size,
            display_every=self.cfg.display_every,
            callbacks=callbacks,
        )

    def _log_histories_to_wandb(self, histories):
        for stage_idx, stage_cfg, losshistory, _train_state, offset in histories:
            if losshistory is None:
                continue
            metrics_names = stage_cfg.get("metrics", self.cfg.metrics) if stage_cfg else self.cfg.metrics
            for i in range(len(losshistory.loss_train)):
                log_dict = {"iterations": offset + i + 1, "stage": stage_idx}
                train_losses = losshistory.loss_train[i]
                test_losses = losshistory.loss_test[i]
                metrics = losshistory.metrics_test[i]
                for j in range(len(train_losses)):
                    value = float(train_losses[j])
                    if np.isfinite(value):
                        log_dict[f"train_loss_{j}"] = value
                for j in range(len(test_losses)):
                    value = float(test_losses[j])
                    if np.isfinite(value):
                        log_dict[f"test_loss_{j}"] = value
                for j in range(len(metrics)):
                    metric_name = str(metrics_names[j]).replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('.', '')
                    value = float(metrics[j])
                    if np.isfinite(value):
                        log_dict[f"metrics_{metric_name}"] = value
                if len(log_dict) > 2:
                    self.visualizer.log(log_dict)

    def _log_histories_to_vis(self, histories):
        for _stage_idx, stage_cfg, losshistory, _train_state, _offset in histories:
            if losshistory is None:
                continue
            metrics_names = stage_cfg.get("metrics", self.cfg.metrics) if stage_cfg else self.cfg.metrics
            for i in range(len(losshistory.loss_train)):
                train_losses = losshistory.loss_train[i]
                test_losses = losshistory.loss_test[i]
                metrics = losshistory.metrics_test[i]
                train_total = float(np.sum(train_losses))
                test_total = float(np.sum(test_losses))
                if np.isfinite(train_total):
                    self.visualizer.add_value("train total loss", train_total)
                if np.isfinite(test_total):
                    self.visualizer.add_value("test total loss", test_total)
                for j in range(len(train_losses)):
                    value = float(train_losses[j])
                    if np.isfinite(value):
                        self.visualizer.add_value(f"train loss {j}", value)
                for j in range(len(test_losses)):
                    value = float(test_losses[j])
                    if np.isfinite(value):
                        self.visualizer.add_value(f"test loss {j}", value)
                for j in range(len(metrics)):
                    value = float(metrics[j])
                    if np.isfinite(value):
                        self.visualizer.add_value(self._metric_vis_name(metrics_names[j]), value)

    def _build_vis_names(self):
        names = {
            "train total loss": "loss",
            "test total loss": "loss",
            "val error": "error",
        }
        for idx in range(16):
            names[f"train loss {idx}"] = "loss component"
            names[f"test loss {idx}"] = "loss component"
        for metric in self.cfg.metrics:
            names[self._metric_vis_name(metric)] = "metric"
        return names

    def _metric_vis_name(self, metric_name):
        return f"metric {str(metric_name)}"

    def _predict_validation_curve(self):
        x = np.linspace(-1, 1, 100).reshape(-1, 1)
        y_pred = self.model.predict(x)
        y_true = self._reference_solution(x)
        return x, y_pred, y_true

    def _predict_validation_grid(self):
        grid_size = 101
        axis = np.linspace(-1, 1, grid_size)
        xx, yy = np.meshgrid(axis, axis)
        xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        y_pred = self.model.predict(xy)
        y_true = self._reference_solution(xy)
        return xy, y_pred, y_true, xx, yy

    def _supports_curve_validation(self):
        return self.cfg.pde_type in {
            'poisson',
            'poisson_new',
            'helmholtz',
            'helmholtz_new',
            'helmholtz_learnable_2',
            'allen_cahn',
            'allen_cahn_new',
        }

    def _supports_grid_validation(self):
        return self.cfg.pde_type in {
            'helmholtz2d',
            'helmholtz_2d',
        }

    def _reference_solution(self, x):
        if self.cfg.pde_type in {'poisson', 'poisson_new'}:
            return (x**2 - 1) / 2
        if self.cfg.pde_type in {'helmholtz', 'helmholtz_new', 'helmholtz_learnable_2'}:
            return np.sin(np.pi * x[:, 0:1])
        if self.cfg.pde_type in {'helmholtz2d', 'helmholtz_2d'}:
            return np.sin(np.pi * x[:, 0:1]) * np.sin(4 * np.pi * x[:, 1:2])
        if self.cfg.pde_type in {'allen_cahn', 'allen_cahn_new'}:
            epsilon = 0.01
            return np.tanh(x[:, 0:1] / np.sqrt(2 * epsilon))
        return None

    def _save_solution_plot(self):
        if self._supports_curve_validation():
            x, y_pred, y_true = self._predict_validation_curve()
            if y_true is None:
                return

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x[:, 0], y_true[:, 0], label='reference', linewidth=2)
            ax.plot(x[:, 0], y_pred[:, 0], label='prediction', linewidth=2)
            ax.set_title(self.cfg.pde_type)
            ax.set_xlabel('x')
            ax.set_ylabel('u(x)')
            ax.grid(True, alpha=0.3)
            ax.legend()

            fig_path = os.path.join(self.result_path, 'solution_curve.png')
            fig.savefig(fig_path, bbox_inches='tight')
            plt.close(fig)
            self.logger(f'==> Saved solution curve to {fig_path}')
            return

        if self._supports_grid_validation():
            _xy, y_pred, y_true, xx, yy = self._predict_validation_grid()
            if y_true is None:
                return

            pred_grid = y_pred.reshape(xx.shape)
            true_grid = y_true.reshape(xx.shape)
            err_grid = np.abs(pred_grid - true_grid)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            images = [
                axes[0].imshow(true_grid, extent=[-1, 1, -1, 1], origin='lower', aspect='auto'),
                axes[1].imshow(pred_grid, extent=[-1, 1, -1, 1], origin='lower', aspect='auto'),
                axes[2].imshow(err_grid, extent=[-1, 1, -1, 1], origin='lower', aspect='auto'),
            ]
            titles = ['reference', 'prediction', 'absolute error']
            for ax, image, title in zip(axes, images, titles):
                ax.set_title(title)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

            fig.suptitle(self.cfg.pde_type)
            fig_path = os.path.join(self.result_path, 'solution_grid.png')
            fig.savefig(fig_path, bbox_inches='tight')
            plt.close(fig)
            self.logger(f'==> Saved solution grid to {fig_path}')
            return

        self.logger('==> Solution plot skipped: analytical reference is unavailable for this PDE.')


if __name__ == '__main__':
    trainer = PDETaiyiTrainer()
    trainer.train()
