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
from BERT.translation_data import TranslationBatches, prepare_translation_data


class BERTTranslationTrainer:
    def __init__(self):
        self.cfg = self.add_arguments()
        ext.normalization.setting(self.cfg)
        ext.activation.setting(self.cfg)
        self.model_name = (
            f"BERT_translation_L{self.cfg.bert_layers}_H{self.cfg.bert_heads}_D{self.cfg.bert_embd}"
            f"_src{self.cfg.max_src_len}_tgt{self.cfg.max_tgt_len}"
            f"_{ext.normalization.setting(self.cfg)}_{ext.activation.setting(self.cfg)}"
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
        self.dataset = TranslationBatches(self.cfg.data_dir, self.cfg.max_src_len, self.cfg.max_tgt_len, self.device)
        self.cfg.vocab_size = int(self.dataset.meta["vocab_size"])
        self.cfg.pad_token_id = self.dataset.vocab.pad_id
        self.cfg.bos_token_id = self.dataset.vocab.bos_id
        self.cfg.eos_token_id = self.dataset.vocab.eos_id
        self.cfg.unk_token_id = self.dataset.vocab.unk_id
        self.logger(
            "==> dataset translation: vocab={}, train_pairs={}, val_pairs={}".format(
                self.cfg.vocab_size,
                self.dataset.meta["train_pairs"],
                self.dataset.meta["val_pairs"],
            )
        )

        self.model = ext.model.get_model(self.cfg).to(self.device)
        self.logger(f"==> model [{self.model_name}]: parameters={self.model.get_num_params():,}")
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
            vis_names={"train loss": "loss", "val loss": "loss", "val token accuracy": "accuracy"},
            wandb_kwargs=self._wandb_kwargs(),
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

    def add_arguments(self):
        parser = argparse.ArgumentParser("BERT-style Text Translation")
        ext.model.add_model_arguments(parser, task="translation", default_family="bert")
        group = parser.add_argument_group("Translation Data Options")
        group.add_argument("--data-dir", default="./dataset/text_translation")
        group.add_argument("--input-file", default=None, help="TSV file with source<TAB>target per line")
        group.add_argument("--auto-prepare", action=argparse.BooleanOptionalAction, default=True)
        group.add_argument("--train-ratio", type=float, default=0.9)
        group.add_argument("--min-freq", type=int, default=1)
        group.add_argument("--overwrite-data", action="store_true")
        group.add_argument("--batch-size", type=ext.utils.str2list, default="64,64")
        group.add_argument("--iters-per-epoch", type=int, default=100)
        group.add_argument("--eval-iters", type=int, default=50)
        group.add_argument("--gradient-accumulation-steps", type=int, default=1)
        group.add_argument("--grad-clip", type=float, default=1.0)
        group.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
        group.add_argument("--sample-src", default="hello")
        group.add_argument("--sample-every", type=int, default=5)
        parser.add_argument("--offline", "-offline", action="store_true", help="offline mode")

        ext.trainer.add_arguments(parser)
        parser.set_defaults(epochs=50, seed=0)
        ext.scheduler.add_arguments(parser)
        parser.set_defaults(lr_method="cos", lr=3e-4, lr_gamma=1e-5)
        ext.optimizer.add_arguments(parser)
        parser.set_defaults(optimizer="adamw", weight_decay=0.01)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.normalization.add_arguments(parser)
        parser.set_defaults(norm="LN", dropout=0.1)
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
        required = ["train.jsonl", "val.jsonl", "meta.json"]
        if (
            not self.cfg.overwrite_data
            and all(os.path.exists(os.path.join(self.cfg.data_dir, name)) for name in required)
        ):
            return
        self.logger(f"==> preparing translation data at {self.cfg.data_dir}")
        prepare_translation_data(
            self.cfg.data_dir,
            input_file=self.cfg.input_file,
            train_ratio=self.cfg.train_ratio,
            min_freq=self.cfg.min_freq,
            overwrite=self.cfg.overwrite_data,
        )

    def _autocast(self):
        if self.device.type != "cuda" or self.cfg.dtype == "float32":
            return nullcontext()
        dtype = torch.bfloat16 if self.cfg.dtype == "bfloat16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _wandb_kwargs(self):
        kwargs = dict(
            project=self.cfg.wandb_project,
            name=self.model_name,
            notes=str(self.cfg),
            config={
                "model": self.cfg.arch,
                "bert_layers": self.cfg.bert_layers,
                "bert_heads": self.cfg.bert_heads,
                "bert_embd": self.cfg.bert_embd,
                "max_src_len": self.cfg.max_src_len,
                "max_tgt_len": self.cfg.max_tgt_len,
                "normalization": ext.normalization.setting(self.cfg),
                "activation": ext.activation.setting(self.cfg),
                "optimizer": self.cfg.optimizer,
                "learning_rate": self.cfg.lr,
                "batch_size": self.cfg.batch_size[0],
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
            val_loss, token_acc = self.validate()
            self.sample("sample_test.txt")
            return val_loss, token_acc

        for epoch in range(self.cfg.start_epoch + 1, self.cfg.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, token_acc = self.validate(epoch)
            is_best = val_loss <= self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.metrics.log_scalars(
                {"learning_rate": self.optimizer.param_groups[0]["lr"]},
                step=self.step,
                epoch=epoch,
            )
            checkpoint_kwargs = dict(epoch=epoch, best_val_loss=self.best_val_loss, step=self.step, seed=self.cfg.seed)
            if self.visualizer.wandb_enabled:
                checkpoint_kwargs["wandb_id"] = self.visualizer.run_id
            self.saver.save_checkpoint(**checkpoint_kwargs)
            if is_best:
                self.saver.save_checkpoint(name="best.pth", **checkpoint_kwargs)
                self.logger(f"==> best validation loss: {val_loss:.5g}; token_acc={token_acc:.2f}%")
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
            self.logger("==> Taiyi monitor collected output.")
        shutil.copy(self.logger.filename, os.path.join(self.result_path, f"{self.model_name}_{now_date}.txt"))
        return train_loss

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
                src, tgt_input, tgt_labels = self.dataset.get_batch("train", self.cfg.batch_size[0])
                with self._autocast():
                    _logits, loss = self.model(src, tgt_input, tgt_labels)
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
        total_correct = 0
        total_tokens = 0
        for _ in range(self.cfg.eval_iters):
            src, tgt_input, tgt_labels = self.dataset.get_batch("val", self.cfg.batch_size[1])
            with self._autocast():
                logits, loss = self.model(src, tgt_input, tgt_labels)
            total_loss += float(loss)
            mask = tgt_labels.ne(self.dataset.vocab.pad_id)
            total_correct += int(logits.argmax(dim=-1).eq(tgt_labels).masked_select(mask).sum().item())
            total_tokens += int(mask.sum().item())
        val_loss = total_loss / self.cfg.eval_iters
        token_acc = 100.0 * total_correct / max(total_tokens, 1)
        perplexity = math.exp(min(val_loss, 20.0))
        self.metrics.log_scalars(
            {"val_loss": val_loss, "val_token_acc": token_acc, "val_perplexity": perplexity},
            step=self.step,
            epoch=epoch,
            vis_scalars={"val loss": val_loss, "val token accuracy": token_acc},
        )
        self.logger(
            f"Validate on epoch {epoch}: loss={val_loss:.5g}, perplexity={perplexity:.4f}, token_acc={token_acc:.2f}%"
        )
        return val_loss, token_acc

    @torch.no_grad()
    def sample(self, filename):
        was_training = self.model.training
        self.model.eval()
        src = self.dataset.encode_source(self.cfg.sample_src)
        tokens = self.model.generate(src, max_new_tokens=self.cfg.max_tgt_len)[0].tolist()
        translation = self.dataset.decode_target(tokens)
        output_path = os.path.join(self.result_path, filename)
        with open(output_path, "w", encoding="utf-8") as output:
            output.write(f"source: {self.cfg.sample_src}\ntranslation: {translation}\n")
        self.logger(f"==> generated translation sample: {output_path}")
        if was_training:
            self.model.train()
        return translation


if __name__ == "__main__":
    trainer = BERTTranslationTrainer()
    trainer.train()
