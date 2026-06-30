#!/usr/bin/env python3
"""Probe dimension-wise statistics for ViT SeqBN/SBN/CFBN variants."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import extension as ext
from extension.model.vit import build_vit_norm_layers, build_vit_norm_layer
from extension.model.vit.vision_transformer import VisionTransformer
from nanoGPT.norm_dimension_probe import _as_float, _resolve_device, _split_csv, tensor_dimension_stats, write_csv, write_jsonl


DEFAULT_NORMS = "LN,SBN,CSBN,SeqBN,CSeqBN,DSeqBN,CDSeqBN,CFBN,CCFBN"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SyntheticImageBatches:
    image_size: int
    in_chans: int
    num_classes: int
    device: torch.device

    def get_batch(self, split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        del split
        images = torch.randn(batch_size, self.in_chans, self.image_size, self.image_size, device=self.device)
        labels = torch.randint(self.num_classes, (batch_size,), device=self.device)
        return images, labels


def _take_loader_batch(loader, device: torch.device):
    images, labels = next(iter(loader))
    images = images.to(device, non_blocking=device.type == "cuda")
    labels = labels.to(device, non_blocking=device.type == "cuda")
    return images, labels


class LoaderImageBatches:
    def __init__(self, train_loader, val_loader, device: torch.device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def get_batch(self, split: str, _batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        loader = self.train_loader if split == "train" else self.val_loader
        return _take_loader_batch(loader, self.device)


def build_batch_provider(args, device: torch.device):
    if args.data_mode == "synthetic":
        return SyntheticImageBatches(args.image_size, args.in_chans, args.num_classes, device)

    cfg = SimpleNamespace(
        dataset=args.dataset,
        dataset_cfg={"loader": "vit", "image_size": args.image_size, "val_resize_size": args.val_resize_size},
        dataset_root=args.dataset_root,
        batch_size=[args.batch_size, args.batch_size],
        val_batch_size=args.batch_size,
        workers=args.workers,
        im_size=[args.image_size],
        in_chans=args.in_chans,
        dataset_classes=args.num_classes,
        disable_train_shuffle=False,
    )
    ext.dataset.setting(cfg)
    train_loader = ext.dataset.get_dataset_loader(cfg, train=True, use_cuda=device.type == "cuda")
    val_loader = ext.dataset.get_dataset_loader(cfg, train=False, use_cuda=device.type == "cuda")
    args.num_classes = int(getattr(cfg, "dataset_classes", args.num_classes))
    return LoaderImageBatches(train_loader, val_loader, device)


def build_cfg(args, norm: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_family="vit",
        arch="custom_vit_probe",
        norm=norm,
        norm_cfg=dict(args.norm_cfg or {}),
        norm_no_affine=args.norm_no_affine,
        norm_site=args.norm_site,
        norm_sites=args.norm_sites,
        im_size=[args.image_size],
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        dataset_classes=args.num_classes,
        num_classes=args.num_classes,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        activation=args.activation,
        activation_cfg={},
        mean_shift_alpha=0.0,
        mean_shift_target="pre_norm",
        centering_rescue="none",
        init_preset="default",
        init_gain=1.0,
        init_std=0.02,
        init_bias=0.0,
    )


def build_model(args, norm: str, device: torch.device) -> torch.nn.Module:
    cfg = build_cfg(args, norm)
    ext.normalization.setting(cfg)
    ext.activation.setting(cfg)
    model = VisionTransformer(
        img_size=[args.image_size],
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=True,
        drop_rate=args.dropout,
        drop_path_rate=args.drop_path_rate,
        norm_layer=build_vit_norm_layer(cfg),
        act_layer=ext.activation.Activation,
        **build_vit_norm_layers(cfg),
    )
    return model.to(device)


def _is_vit_norm_probe_site(name: str) -> bool:
    return name == "norm" or name.endswith(".norm1") or name.endswith(".norm2")


@contextmanager
def capture_vit_norm_activations(model: torch.nn.Module):
    records: list[dict] = []
    handles = []

    def hook(name: str):
        def _hook(_module, inputs, output):
            if not inputs:
                return
            inp = inputs[0]
            if isinstance(inp, torch.Tensor) and isinstance(output, torch.Tensor):
                records.append({"module": name, "input": inp.detach().cpu(), "output": output.detach().cpu()})

        return _hook

    for name, module in model.named_modules():
        if _is_vit_norm_probe_site(name):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


def _cls_patch_stats(tensor: torch.Tensor) -> dict[str, float]:
    x = tensor.detach().float()
    if x.dim() != 3 or x.shape[1] < 2:
        return {}
    cls = x[:, 0, :]
    patch = x[:, 1:, :]
    cls_rms = torch.sqrt(cls.square().mean())
    patch_rms = torch.sqrt(patch.square().mean())
    return {
        "cls_rms": _as_float(cls_rms),
        "patch_rms": _as_float(patch_rms),
        "cls_patch_rms_ratio": _as_float(cls_rms / patch_rms.clamp_min(torch.finfo(patch_rms.dtype).tiny)),
        "cls_mean_abs": _as_float(cls.mean().abs()),
        "patch_mean_abs": _as_float(patch.mean().abs()),
    }


@torch.no_grad()
def collect_stats(model: torch.nn.Module, probe_images: torch.Tensor, norm: str, step: int, split: str) -> list[dict]:
    was_training = model.training
    model.eval()
    with capture_vit_norm_activations(model) as records:
        model(probe_images)
    if was_training:
        model.train()

    rows = []
    for record in records:
        for tensor_name in ("input", "output"):
            row = {
                "norm": norm,
                "step": step,
                "split": split,
                "module": record["module"],
                "tensor": tensor_name,
            }
            row.update(tensor_dimension_stats(record[tensor_name]))
            row.update({f"vit_{key}": value for key, value in _cls_patch_stats(record[tensor_name]).items()})
            rows.append(row)
    return rows


@torch.no_grad()
def collect_patch_sensitivity_rows(model: torch.nn.Module, probe_images: torch.Tensor, norm: str) -> list[dict]:
    was_training = model.training
    model.eval()
    changed_images = probe_images.clone()
    image_mid = changed_images.shape[-1] // 2
    changed_images[:, :, :, image_mid:] = -changed_images[:, :, :, image_mid:]

    with capture_vit_norm_activations(model) as base_records:
        model(probe_images)
    with capture_vit_norm_activations(model) as changed_records:
        model(changed_images)
    if was_training:
        model.train()

    rows = []
    by_name = {record["module"]: record for record in changed_records}
    for base in base_records:
        changed = by_name.get(base["module"])
        if changed is None:
            continue
        for tensor_name in ("input", "output"):
            x = base[tensor_name].float()
            y = changed[tensor_name].float()
            cls_delta = (x[:, :1, :] - y[:, :1, :]).abs()
            if x.shape[1] > 2:
                split = 1 + (x.shape[1] - 1) // 2
                early_patch_delta = (x[:, 1:split, :] - y[:, 1:split, :]).abs()
                late_patch_delta = (x[:, split:, :] - y[:, split:, :]).abs()
            else:
                early_patch_delta = cls_delta.new_zeros(1)
                late_patch_delta = (x[:, 1:, :] - y[:, 1:, :]).abs()
            rows.append(
                {
                    "norm": norm,
                    "module": base["module"],
                    "tensor": tensor_name,
                    "cls_abs_mean": _as_float(cls_delta.mean()),
                    "cls_abs_max": _as_float(cls_delta.max()),
                    "early_patch_abs_mean": _as_float(early_patch_delta.mean()),
                    "late_patch_abs_mean": _as_float(late_patch_delta.mean()),
                }
            )
    return rows


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, batches, batch_size: int, eval_iters: int) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for _ in range(eval_iters):
        images, labels = batches.get_batch("val", batch_size)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += float(loss.detach())
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += int(labels.numel())
    if was_training:
        model.train()
    return total_loss / max(1, eval_iters), 100.0 * total_correct / max(1, total_count)


def train_steps(model: torch.nn.Module, batches, args, norm: str, probe_images: torch.Tensor) -> tuple[list[dict], list[dict]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stats_rows = collect_stats(model, probe_images, norm=norm, step=0, split="probe")
    train_rows = []
    model.train()

    for step in range(1, args.train_steps + 1):
        images, labels = batches.get_batch("train", args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        acc1 = 100.0 * float((logits.argmax(dim=1) == labels).float().mean().detach())

        train_rows.append(
            {
                "norm": norm,
                "step": step,
                "train_loss": float(loss.detach()),
                "train_acc1": acc1,
                "grad_norm": _as_float(grad_norm),
            }
        )
        if args.stat_every > 0 and (step % args.stat_every == 0 or step == args.train_steps):
            stats_rows.extend(collect_stats(model, probe_images, norm=norm, step=step, split="probe"))
    return stats_rows, train_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ViT dimension-wise SeqBN/SBN/CFBN probe")
    parser.add_argument("--norms", default=DEFAULT_NORMS)
    parser.add_argument("--output-dir", default="results/vit-norm-dimension-probe")
    parser.add_argument("--data-mode", choices=("synthetic", "repo_dataset"), default="synthetic")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--val-resize-size", type=int, default=36)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--in-chans", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--norm-cfg", type=ext.utils.str2dict, default={})
    parser.add_argument("--norm-no-affine", action="store_true")
    parser.add_argument("--norm-site", default=None, choices=("all", "norm1", "norm2", "final", "none"))
    parser.add_argument("--norm-sites", default=None)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--stat-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = _resolve_device(args.device)
    norms = _split_csv(args.norms)
    if not norms:
        raise ValueError("--norms must contain at least one normalization variant.")

    output_dir = Path(args.output_dir)
    batches = build_batch_provider(args, device)
    probe_images, _probe_labels = batches.get_batch("val", args.probe_batch_size)

    all_stats_rows = []
    all_train_rows = []
    all_effect_rows = []
    all_sensitivity_rows = []
    for norm in norms:
        _set_seed(args.seed)
        model = build_model(args, norm, device)
        initial_val_loss, initial_val_acc1 = evaluate_loss(model, batches, args.batch_size, args.eval_iters)
        stats_rows, train_rows = train_steps(model, batches, args, norm, probe_images)
        final_val_loss, final_val_acc1 = evaluate_loss(model, batches, args.batch_size, args.eval_iters)
        sensitivity_rows = collect_patch_sensitivity_rows(model, probe_images, norm)

        all_stats_rows.extend(stats_rows)
        all_train_rows.extend(train_rows)
        all_sensitivity_rows.extend(sensitivity_rows)
        all_effect_rows.append(
            {
                "norm": norm,
                "seed": args.seed,
                "data_mode": args.data_mode,
                "train_steps": args.train_steps,
                "initial_val_loss": initial_val_loss,
                "initial_val_acc1": initial_val_acc1,
                "final_train_loss": train_rows[-1]["train_loss"] if train_rows else float("nan"),
                "final_train_acc1": train_rows[-1]["train_acc1"] if train_rows else float("nan"),
                "final_val_loss": final_val_loss,
                "final_val_acc1": final_val_acc1,
                "final_val_perplexity_proxy": math.exp(min(final_val_loss, 20.0)),
            }
        )
        print(
            f"{norm}: initial_val_loss={initial_val_loss:.4f}, "
            f"final_val_loss={final_val_loss:.4f}, final_val_acc1={final_val_acc1:.2f}, "
            f"train_steps={args.train_steps}"
        )

    write_csv(output_dir / "dimension_stats.csv", all_stats_rows)
    write_csv(output_dir / "train_trace.csv", all_train_rows)
    write_csv(output_dir / "effect_summary.csv", all_effect_rows)
    write_csv(output_dir / "patch_sensitivity_probe.csv", all_sensitivity_rows)
    write_jsonl(output_dir / "dimension_stats.jsonl", all_stats_rows)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote ViT probe outputs to {output_dir}")


if __name__ == "__main__":
    main()
