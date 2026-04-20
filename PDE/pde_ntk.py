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

import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
MLP_DIR = os.path.join(PROJECT_ROOT, 'MLP')
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import extension as ext
from extension.utils import str2list

if MLP_DIR not in sys.path:
    sys.path.append(MLP_DIR)
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


class NTKMonitorCallback(dde.callbacks.Callback):
    def __init__(self, trainer, track_every):
        super().__init__()
        self.trainer = trainer
        self.track_every = max(int(track_every), 1)
        self._fallback_step = 0
        self._last_tracked_step = None

    def _resolve_step(self):
        train_state = getattr(self.model, "train_state", None)
        for attr in ("step", "iteration", "epoch"):
            value = getattr(train_state, attr, None)
            if value is not None:
                return int(value)
        self._fallback_step += 1
        return self._fallback_step

    def _track_if_needed(self):
        step = self._resolve_step()
        if step % self.track_every == 0 and step != self._last_tracked_step:
            self.trainer.run_ntk_analysis('train', step=step)
            self._last_tracked_step = step

    def on_epoch_end(self):
        self._track_if_needed()

    def on_batch_end(self):
        self._track_if_needed()


class PDENTKTrainer:
    def __init__(self):
        self.cfg = self.add_arguments()
        self.model_name = 'PDE_NTK_' + self.cfg.pde_type + '_' + self.cfg.arch + '_d' + str(self.cfg.depth) + '_w' + str(
            self.cfg.width) + '_' + ext.normalization.setting(self.cfg) + '_' + ext.activation.setting(self.cfg) + '_lr' + str(self.cfg.lr) + '_epochs' + str(self.cfg.epochs) + '_seed' + str(self.cfg.seed)
        if self.cfg.arch == "MultiChannelMLP":
            self.model_name += "_" + ext.multichannel.setting(self.cfg)
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
        self.cfg.dataset_classes = 1 if self.cfg.pde_type != 'cavity' else 3

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.num_gpu = torch.cuda.device_count()
        self.logger('==> use {:d} GPUs'.format(self.num_gpu))

        self.model = ext.model.get_model(self.cfg).to(self.device)
        self.freeze_summary = ext.multichannel.summarize_freeze_state(self.model)
        if self.num_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        ext.multichannel.log_runtime_summary(self.logger, self.cfg, self.freeze_summary)
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

        wandb_kwargs = dict(
            project=self.cfg.subject_name,
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
                "activation": ext.activation.setting(self.cfg),
                "learning_rate": self.cfg.lr,
                "epochs": self.cfg.epochs,
                "seed": self.cfg.seed,
                "optimizer": self.cfg.optimizer,
                "scheduler": self.cfg.lr_method,
                "scheduler_cfg": "step" + str(self.cfg.lr_step) + "_gamma" + str(self.cfg.lr_gamma),
                "loss_weights": self.cfg.loss_weights,
                "ntk_points": self.cfg.ntk_points,
                "ntk_boundary_points": self.cfg.ntk_boundary_points,
                "ntk_when": self.cfg.ntk_when,
                "ntk_track_every": self.cfg.ntk_track_every,
                **ext.multichannel.get_runtime_config(self.cfg),
                "freeze_trainable_params": self.freeze_summary["trainable_params"],
                "freeze_frozen_params": self.freeze_summary["frozen_params"],
                "freeze_trainable_tensors": self.freeze_summary["trainable_tensors"],
                "freeze_frozen_tensors": self.freeze_summary["frozen_tensors"],
                "freeze_has_frozen_params": self.freeze_summary["has_frozen_params"],
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
        self.visualizer.update_wandb_summary(
            {
                "freeze_summary_text": ext.multichannel.format_freeze_summary(self.freeze_summary),
                "freeze_frozen_names": self.freeze_summary["frozen_names"],
                "freeze_multichannel_modules": self.freeze_summary["multichannel_modules"],
            }
        )
        self.metrics = ext.measurement.setting(
            result_path=self.result_path,
            visualizer=self.visualizer,
            logger=self.logger,
        )
        taiyi_config = {
            'ResidualBlock': [['ResidualInputAngleMean', 'linear(5,0)'], ['ResidualStreamOutputAngleMean', 'linear(5,0)']],
            'ResBlockDropout': [['ResidualInputAngleMean', 'linear(5,0)'], ['ResidualStreamOutputAngleMean', 'linear(5,0)']],
            'BasicBlock': [['ResidualInputAngleMean', 'linear(5,0)'], ['ResidualStreamOutputAngleMean', 'linear(5,0)']],
            'Bottleneck': [['ResidualInputAngleMean', 'linear(5,0)'], ['ResidualStreamOutputAngleMean', 'linear(5,0)']],
        }
        self.taiyi = ext.taiyi.setting(
            self.cfg,
            model=self.net,
            monitor_config=taiyi_config,
            wandb=self.visualizer.wandb,
            output_dir=self.result_path,
        )
        self.ntk = ext.ntk.EmpiricalNTK(
            result_path=self.result_path,
            metrics=self.metrics,
            save_eigvals=self.cfg.ntk_save_eigvals,
        )

    def add_arguments(self):
        raw_argv = sys.argv[1:]
        parser = argparse.ArgumentParser('PDE Solving with NTK Analysis')
        ext.model.add_model_arguments(parser, task='pde', default_family='mlp')
        parser.add_argument('--pde_type', default='poisson', choices=['poisson', 'helmholtz', 'helmholtz2d', 'helmholtz_2d', 'allen_cahn', 'wave', 'klein_gordon', 'convdiff', 'cavity', 'helmholtz_new', 'helmholtz_learnable_2', 'poisson_new', 'allen_cahn_new'], help='PDE type')
        parser.add_argument('--loss-weights', type=str2list, default='1.0,1.0', help='comma-separated list of loss weights')
        parser.add_argument('--offline', action='store_true', help='offline mode')
        parser.add_argument('--no_save_best', action='store_true', help='do not save best model during training')
        parser.add_argument('--display_every', type=int, default=1000, help='display and log every N iterations')
        parser.add_argument('--metrics', type=str2list, default='l2 relative error', help='comma-separated list of metrics to evaluate')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--float64', action='store_true', help='train with float64 precision')
        parser.add_argument('--subject_name', type=str, default='default_subject', help='subject name for logging')
        parser.add_argument('--ntk_points', type=int, default=64, help='number of interior points used for NTK')
        parser.add_argument('--ntk_boundary_points', type=int, default=32, help='number of boundary points used for NTK')
        parser.add_argument('--ntk_when', type=str2list, default='init,final', help='when to run NTK analysis: init,final')
        parser.add_argument('--ntk_track_every', type=int, default=1000, help='track NTK every N iterations during training; <=0 disables periodic tracking')
        parser.add_argument('--ntk_save_eigvals', action='store_true', help='save NTK eigenvalues to .npy files')
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
        if args.resume:
            args = parser.parse_args(namespace=ext.checkpoint.Checkpoint.load_config(args.resume))

        if args.pde_type in {'helmholtz2d', 'helmholtz_2d'}:
            if not self._has_cli_flag(raw_argv, '--loss-weights'):
                args.loss_weights = [1.0, 10.0]
            if not self._has_cli_flag(raw_argv, '--float64'):
                args.float64 = True

        stages = getattr(ext.optimizer, "get_stages", lambda _cfg: None)(args)
        stage_total_iterations = getattr(ext.optimizer, "infer_total_iterations", lambda _stages: None)(stages)
        if stage_total_iterations is not None:
            args.epochs = stage_total_iterations
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
        when = {str(item).lower() for item in self.cfg.ntk_when}
        if 'init' in when:
            self.run_ntk_analysis('init', step=0)

        if self.cfg.test:
            self.validate()
        else:
            callbacks = []
            if self.cfg.ntk_track_every > 0:
                callbacks.append(NTKMonitorCallback(self, self.cfg.ntk_track_every))

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
            self._save_solution_plot()

            if not self.cfg.no_save_best:
                self.model.save(os.path.join(self.result_path, 'model'))
            self.saver.save_checkpoint(
                epoch=self.cfg.epochs - 1,
                best_loss=getattr(self, "best_loss", None),
                step=self._current_train_step(default=self.cfg.epochs),
                wandb_id=self.visualizer.run_id,
            )

        if 'final' in when:
            self.run_ntk_analysis('final', step=self._current_train_step(default=self.cfg.epochs))

        now_date = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.logger('==> end time: {}'.format(now_date))

        taiyi_info = self.taiyi.finish()
        finish_info = self.visualizer.finish(sync_offline=self.cfg.offline)
        if taiyi_info["taiyi_output"]:
            self.logger("==> Taiyi monitor collected output.")
        if finish_info["synced"]:
            self.logger("==> WandB offline run synced.")

        if self.visualizer.wandb_enabled:
            new_log_filename = r'{}_{}.txt'.format(self.model_name, now_date)
            self.logger('==> Network training completed. Copy log file to {}'.format(new_log_filename))
            new_log_path = os.path.join(self.result_path, new_log_filename)
            shutil.copy(self.logger.filename, new_log_path)

    def run_ntk_analysis(self, phase, step=None):
        point_sets = self._collect_ntk_point_sets()
        if not point_sets:
            self.logger(f'==> NTK [{phase}] skipped: no available sampling points.')
            return

        if step is None:
            step = self._current_train_step(default=-1)
        self.logger(f'==> NTK [{phase}] analyzing {len(point_sets)} point sets.')
        for set_name, points in point_sets.items():
            x = torch.as_tensor(points, device=self.device, dtype=next(self.net.parameters()).dtype)
            jacobian = self.ntk.compute_empirical_jacobian(self.net, x)
            stats = self.ntk.compute_ntk_spectrum(jacobian)
            record = {
                "phase": phase,
                "step": int(step),
                "point_set": set_name,
                "n_points": int(x.shape[0]),
                "input_dim": int(x.shape[1]),
                "output_dim": int(self.ntk.infer_output_dim(self.net, x)),
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
                f"mean_self_kernel={record['mean_self_kernel']:.3e}"
            )
        self.ntk.save_records()

    def _current_train_step(self, default=-1):
        train_state = getattr(self.model, "train_state", None)
        for attr in ("step", "iteration", "epoch"):
            value = getattr(train_state, attr, None)
            if value is not None:
                return int(value)
        return int(default)

    def _collect_ntk_point_sets(self):
        domain_points = self._sample_domain_points(self.cfg.ntk_points)
        boundary_points = self._sample_boundary_points(self.cfg.ntk_boundary_points)
        point_sets = {}
        if domain_points is not None and len(domain_points) > 0:
            point_sets["domain"] = domain_points
        if boundary_points is not None and len(boundary_points) > 0:
            point_sets["boundary"] = boundary_points
        if domain_points is not None and boundary_points is not None and len(domain_points) > 0 and len(boundary_points) > 0:
            point_sets["combined"] = np.concatenate([domain_points, boundary_points], axis=0)
        return point_sets

    def _sample_domain_points(self, num_points):
        if num_points <= 0:
            return None
        geom = getattr(self.data, "geom", None)
        if geom is not None:
            for method_name in ("uniform_points", "random_points"):
                method = getattr(geom, method_name, None)
                if method is None:
                    continue
                try:
                    points = method(num_points, boundary=False)
                except TypeError:
                    try:
                        points = method(num_points)
                    except TypeError:
                        continue
                points = np.asarray(points)
                if points.size > 0:
                    return points[:num_points]

        points = getattr(self.data, "train_x_all", None)
        if points is None:
            points = getattr(self.data, "train_x", None)
        if points is None:
            return None
        points = np.asarray(points)
        return points[:min(num_points, len(points))]

    def _sample_boundary_points(self, num_points):
        if num_points <= 0:
            return None
        geom = getattr(self.data, "geom", None)
        if geom is not None:
            for method_name in ("uniform_boundary_points", "random_boundary_points"):
                method = getattr(geom, method_name, None)
                if method is None:
                    continue
                try:
                    points = method(num_points)
                except TypeError:
                    try:
                        points = method(num_points, random="pseudo")
                    except TypeError:
                        continue
                points = np.asarray(points)
                if points.size > 0:
                    return points[:num_points]

        points = getattr(self.data, "train_x_bc", None)
        if points is None:
            return None
        points = np.asarray(points)
        return points[:min(num_points, len(points))]

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

        self.metrics.log_validation("val error", error, wandb_key="val_error", step=self._current_train_step(default=-1))
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
                    self.metrics.log_scalars(log_dict, step=offset + i + 1, step_key="iterations")

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
                vis_scalars = {}
                if np.isfinite(train_total):
                    vis_scalars["train total loss"] = train_total
                if np.isfinite(test_total):
                    vis_scalars["test total loss"] = test_total
                for j in range(len(train_losses)):
                    value = float(train_losses[j])
                    if np.isfinite(value):
                        vis_scalars[f"train loss {j}"] = value
                for j in range(len(test_losses)):
                    value = float(test_losses[j])
                    if np.isfinite(value):
                        vis_scalars[f"test loss {j}"] = value
                for j in range(len(metrics)):
                    value = float(metrics[j])
                    if np.isfinite(value):
                        vis_scalars[self._metric_vis_name(metrics_names[j])] = value
                if vis_scalars:
                    self.metrics.log_scalars({}, vis_scalars=vis_scalars)

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
        for phase in ("init", "train", "final"):
            for point_set in ("domain", "boundary", "combined"):
                for metric in ("cond", "eff_rank_90", "numerical_rank", "mean_self_kernel", "trace", "stable_rank"):
                    names[f"ntk {phase} {point_set} {metric}"] = "ntk"
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
    trainer = PDENTKTrainer()
    trainer.train()
