#!/usr/bin/env python3
import argparse
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import extension as ext
from extension.model.nanogpt import format_nanogpt_activation_setting, format_nanogpt_norm_setting
from nanoGPT.tinyshakespeare import TinyShakespeareBatches, prepare_tinyshakespeare


class NanoGPTTrainer:
    def __init__(self):
        self.cfg = self.add_arguments()
        ext.normalization.setting(self.cfg)
        ext.activation.setting(self.cfg)
        norm_flag = format_nanogpt_norm_setting(self.cfg)
        activation_flag = format_nanogpt_activation_setting(self.cfg)
        self.model_name = (
            f"nanoGPT_tinyshakespeare_L{self.cfg.n_layer}_H{self.cfg.n_head}_D{self.cfg.n_embd}"
            f"_ctx{self.cfg.block_size}_{norm_flag}_{activation_flag}"
            f"_lr{self.cfg.lr}_bs{self.cfg.batch_size[0]}_wd{self.cfg.weight_decay}_seed{self.cfg.seed}"
        )
        self.result_path = os.path.join(self.cfg.output, self.model_name, self.cfg.log_suffix)
        os.makedirs(self.result_path, exist_ok=True)
        self.logger = ext.logger.setting("log.txt", self.result_path, self.cfg.test, bool(self.cfg.resume))
        ext.trainer.setting(self.cfg)

        self.device = self._resolve_device()
        self.logger(f"==> device: {self.device}; cuda devices: {torch.cuda.device_count()}")
        if self.cfg.auto_prepare:
            self._prepare_data_if_needed()
        self.dataset = TinyShakespeareBatches(self.cfg.data_dir, self.cfg.block_size, self.device)
        dataset_vocab_size = int(self.dataset.meta["vocab_size"])
        if self.cfg.vocab_size is not None and self.cfg.vocab_size != dataset_vocab_size:
            raise ValueError(
                f"Configured vocab_size={self.cfg.vocab_size} does not match dataset vocab={dataset_vocab_size}."
            )
        self.cfg.vocab_size = dataset_vocab_size
        self.logger(
            "==> dataset tinyshakespeare: vocab={}, train_tokens={}, val_tokens={}".format(
                self.cfg.vocab_size,
                self.dataset.meta["train_tokens"],
                self.dataset.meta["val_tokens"],
            )
        )

        self.model = ext.model.get_model(self.cfg).to(self.device)
        if self.cfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.logger(f"==> model [{self.model_name}]: parameters={self._base_model().get_num_params():,}")

        self.optimizer = ext.optimizer.setting(self.model, self.cfg)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.cfg)
        self.saver = ext.checkpoint.Checkpoint(
            self._base_model(), self.cfg, self.optimizer, self.scheduler, self.result_path, not self.cfg.test
        )
        self.saver.load(self.cfg.load)

        self.step = 0
        self.best_val_loss = float("inf")
        if self.cfg.resume:
            saved = self.saver.resume(self.cfg.resume)
            self.cfg.start_epoch = int(saved.get("epoch", self.cfg.start_epoch))
            self.step = int(saved.get("step", 0))
            self.best_val_loss = float(saved.get("best_val_loss", self.best_val_loss))
            self.cfg.seed = saved.get("seed", self.cfg.seed)
            self.wandb_id = saved.get("wandb_id")

        scaler_enabled = self.device.type == "cuda" and self.cfg.dtype == "float16"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.visualizer = ext.tracking.setting(
            self.cfg,
            env_name=self.model_name,
            vis_names={"train loss": "loss", "val loss": "loss", "val perplexity": "perplexity"},
            wandb_kwargs=self._wandb_kwargs(),
        )
        self.metrics = ext.measurement.setting(
            result_path=self.result_path, visualizer=self.visualizer, logger=self.logger
        )
        taiyi_config = self._nanogpt_diagnostics_config() if self.cfg.diagnostics else {}
        self.taiyi = ext.taiyi.setting(
            self.cfg,
            model=self.model,
            monitor_config=taiyi_config,
            wandb=self.visualizer.wandb,
        )
        if self.taiyi.enabled:
            self.logger(f"==> taiyi config: {taiyi_config}")
            if self.cfg.diagnostics:
                self.logger("==> nanoGPT Taiyi diagnostics enabled; metrics are logged through wandb")

    def add_arguments(self):
        parser = argparse.ArgumentParser("nanoGPT TinyShakespeare Language Modeling")
        ext.model.add_model_arguments(parser, task="classification", default_family="nanogpt")
        group = parser.add_argument_group("Language Modeling Data Options")
        group.add_argument("--data-dir", default="./dataset/tinyshakespeare")
        group.add_argument("--auto-prepare", action=argparse.BooleanOptionalAction, default=True)
        group.add_argument("--batch-size", type=ext.utils.str2list, default="64,64")
        group.add_argument("--iters-per-epoch", type=int, default=100)
        group.add_argument("--eval-iters", type=int, default=50)
        group.add_argument("--gradient-accumulation-steps", type=int, default=1)
        group.add_argument("--grad-clip", type=float, default=1.0)
        group.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
        group.add_argument("--compile", action="store_true")
        group.add_argument("--sample-prompt", default="\n")
        group.add_argument("--sample-tokens", type=int, default=300)
        group.add_argument("--temperature", type=float, default=0.8)
        group.add_argument("--top-k", type=int, default=200)
        group.add_argument("--sample-every", type=int, default=5)
        parser.add_argument("--offline", "-offline", action="store_true", help="offline mode")
        parser.add_argument("--diagnostics", action="store_true", help="enable Taiyi norm/residual diagnostics through wandb")
        ext.trainer.add_arguments(parser)
        parser.set_defaults(epochs=50, seed=0)
        ext.scheduler.add_arguments(parser)
        parser.set_defaults(lr_method="cos", lr=6e-4, lr_gamma=6e-5)
        ext.optimizer.add_arguments(parser)
        parser.set_defaults(optimizer="adamw", weight_decay=0.1)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.normalization.add_arguments(parser)
        parser.set_defaults(norm="LN", dropout=0.2)
        ext.activation.add_arguments(parser)
        parser.set_defaults(activation="gelu")
        ext.tracking.add_arguments(parser)
        ext.taiyi.add_arguments(parser)

        args = parser.parse_args()
        if args.resume:
            args = parser.parse_args(namespace=ext.checkpoint.Checkpoint.load_config(args.resume))
        if len(args.batch_size) == 1:
            args.batch_size = [args.batch_size[0], args.batch_size[0]]
        if args.lr_method == "cos" and args.lr_step == 30:
            args.lr_step = args.epochs
        if args.diagnostics:
            args.taiyi = True
            args.wandb = True
            args.visualize = True
        if args.gradient_accumulation_steps < 1:
            parser.error("--gradient-accumulation-steps must be >= 1")
        return ext.tracking.normalize_config(args)

    def _resolve_device(self):
        if torch.cuda.is_available():
            if self.cfg.gpu is not None:
                return torch.device(f"cuda:{self.cfg.gpu}")
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _prepare_data_if_needed(self):
        required = ["train.bin", "val.bin", "meta.json"]
        if all(os.path.exists(os.path.join(self.cfg.data_dir, name)) for name in required):
            return
        self.logger(f"==> preparing TinyShakespeare at {self.cfg.data_dir}")
        prepare_tinyshakespeare(self.cfg.data_dir)

    def _base_model(self):
        return self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

    def _autocast(self):
        if self.device.type != "cuda" or self.cfg.dtype == "float32":
            return nullcontext()
        dtype = torch.bfloat16 if self.cfg.dtype == "bfloat16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _nanogpt_diagnostics_config(self):
        block_indices = sorted({0, max(0, self.cfg.n_layer // 2), max(0, self.cfg.n_layer - 1)})
        norm_quantities = [
            ["NormMechanismStats", "linear(1,0)"],
            ["OutputGradSndNorm", "linear(5,0)"],
        ]
        block_quantities = [
            ["ResidualMechanismStats", "linear(1,0)"],
        ]
        config = {}
        for block_idx in block_indices:
            block_prefix = f"transformer.h.{block_idx}"
            config[block_prefix] = block_quantities
            config[f"{block_prefix}.ln_1"] = norm_quantities
            config[f"{block_prefix}.ln_2"] = norm_quantities
        config["transformer.ln_f"] = norm_quantities
        config["lm_head"] = [["ViTLogitsStats", "linear(1,0)"]]
        return config

    def _wandb_kwargs(self):
        kwargs = dict(
            project=self.cfg.wandb_project,
            name=self.model_name,
            notes=str(self.cfg),
            config={
                "model": self.cfg.arch,
                "n_layer": self.cfg.n_layer,
                "n_head": self.cfg.n_head,
                "n_embd": self.cfg.n_embd,
                "block_size": self.cfg.block_size,
                "normalization": ext.normalization.setting(self.cfg),
                "activation": ext.activation.setting(self.cfg),
                "nanogpt_norm": format_nanogpt_norm_setting(self.cfg),
                "nanogpt_activation": format_nanogpt_activation_setting(self.cfg),
                "attn_norm": self.cfg.attn_norm,
                "mlp_norm": self.cfg.mlp_norm,
                "final_norm": self.cfg.final_norm,
                "mlp_activation": self.cfg.mlp_activation,
                "optimizer": self.cfg.optimizer,
                "learning_rate": self.cfg.lr,
                "batch_size": self.cfg.batch_size[0],
                "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
                "weight_decay": self.cfg.weight_decay,
                "epochs": self.cfg.epochs,
                "iters_per_epoch": self.cfg.iters_per_epoch,
                "seed": self.cfg.seed,
            },
        )
        if self.cfg.resume and getattr(self, "wandb_id", None):
            kwargs.update(dict(id=self.wandb_id, resume="must"))
        return kwargs

    def train(self):
        if self.cfg.test:
            val_loss = self.validate()
            self.sample("sample_test.txt")
            return val_loss

        for epoch in range(self.cfg.start_epoch + 1, self.cfg.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            perplexity = math.exp(min(val_loss, 20.0))
            is_best = val_loss <= self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.metrics.log_scalars(
                {"learning_rate": self.optimizer.param_groups[0]["lr"]},
                step=self.step,
                epoch=epoch,
            )
            checkpoint_kwargs = dict(
                epoch=epoch,
                best_val_loss=self.best_val_loss,
                step=self.step,
                seed=self.cfg.seed,
            )
            if self.visualizer.wandb_enabled:
                checkpoint_kwargs["wandb_id"] = self.visualizer.run_id
            self.saver.save_checkpoint(**checkpoint_kwargs)
            if is_best:
                self.saver.save_checkpoint(name="best.pth", **checkpoint_kwargs)
                self.logger(f"==> best validation loss: {val_loss:.5g}; perplexity={perplexity:.4f}")
            if self.cfg.sample_every > 0 and (epoch + 1) % self.cfg.sample_every == 0:
                self.sample(f"sample_epoch_{epoch:04d}.txt")
            if self.cfg.lr_method == "auto":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

        self.sample("sample_final.txt")
        now_date = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
        self.logger(f"==> end time: {now_date}")
        taiyi_info = self.taiyi.finish()
        self.visualizer.finish(sync_offline=self.cfg.offline)
        if taiyi_info["taiyi_output"]:
            self.logger(f"==> Taiyi monitor uploaded artifact: {taiyi_info.get('taiyi_artifact')}")
        shutil.copy(self.logger.filename, os.path.join(self.result_path, f"{self.model_name}_{now_date}.txt"))

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
        running_loss = 0.0
        progress_bar = ext.ProgressBar(self.cfg.iters_per_epoch)
        for iteration in range(1, self.cfg.iters_per_epoch + 1):
            self.optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(self.cfg.gradient_accumulation_steps):
                inputs, targets = self.dataset.get_batch("train", self.cfg.batch_size[0])
                with self._autocast():
                    _logits, loss = self.model(inputs, targets)
                    scaled_loss = loss / self.cfg.gradient_accumulation_steps
                self.scaler.scale(scaled_loss).backward()
                loss_value += float(scaled_loss.detach())
            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.taiyi.track(self.step)
            self.step += 1
            running_loss += loss_value
            if iteration % 10 == 0 or iteration == self.cfg.iters_per_epoch:
                progress_bar.step(f"Loss: {running_loss / iteration:.5g}", 10)
        average_loss = running_loss / self.cfg.iters_per_epoch
        self.metrics.log_scalars(
            {"train_loss": average_loss},
            step=self.step,
            epoch=epoch,
            vis_scalars={"train loss": average_loss},
        )
        self.logger(f"Train on epoch {epoch}: average loss={average_loss:.5g}, time: {progress_bar.time_used()}")
        return average_loss

    @torch.no_grad()
    def validate(self, epoch=-1):
        self.model.eval()
        total_loss = 0.0
        for _ in range(self.cfg.eval_iters):
            inputs, targets = self.dataset.get_batch("val", self.cfg.batch_size[1])
            with self._autocast():
                _logits, loss = self.model(inputs, targets)
            total_loss += float(loss)
        val_loss = total_loss / self.cfg.eval_iters
        perplexity = math.exp(min(val_loss, 20.0))
        self.metrics.log_scalars(
            {"val_loss": val_loss, "val_perplexity": perplexity},
            step=self.step,
            epoch=epoch,
            vis_scalars={"val loss": val_loss, "val perplexity": perplexity},
        )
        self.logger(f"Validate on epoch {epoch}: loss={val_loss:.5g}, perplexity={perplexity:.4f}")
        return val_loss

    @torch.no_grad()
    def sample(self, filename):
        model = self._base_model()
        was_training = model.training
        model.eval()
        prompt = torch.tensor([self.dataset.encode(self.cfg.sample_prompt)], dtype=torch.long, device=self.device)
        tokens = model.generate(
            prompt,
            max_new_tokens=self.cfg.sample_tokens,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
        )[0].tolist()
        text = self.dataset.decode(tokens)
        output_path = os.path.join(self.result_path, filename)
        with open(output_path, "w", encoding="utf-8") as output:
            output.write(text)
        self.logger(f"==> generated sample: {output_path}")
        if was_training:
            model.train()
        return text


if __name__ == "__main__":
    trainer = NanoGPTTrainer()
    trainer.train()
