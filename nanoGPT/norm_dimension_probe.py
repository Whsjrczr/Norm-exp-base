#!/usr/bin/env python3
"""Probe dimension-wise statistics for sequence normalization variants.

The script is intentionally lightweight: it reuses the repository's nanoGPT
model factory, runs each normalization variant on the same probe batches, and
writes CSV/JSONL files that can be aggregated or plotted later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import extension as ext
from nanoGPT.tinyshakespeare import TinyShakespeareBatches, prepare_tinyshakespeare


DEFAULT_NORMS = "LN,SBN,CSBN,SeqBN,CSeqBN,DSeqBN,CDSeqBN,CFBN,CCFBN"
NONCAUSAL_NORMS = {"BN", "BNc", "BNs", "SBN", "SBNc", "SBNs", "SeqBN", "SeqBNc", "SeqBNs", "DSeqBN", "DSeqBNc", "DSeqBNs", "CFBN", "CFBNc", "CFBNs"}


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().float()
        if value.numel() == 1:
            return float(value.item())
        return float(value.mean().item())
    return float(value)


def _safe_std(x: torch.Tensor, dim=None) -> torch.Tensor:
    return x.std(dim=dim, unbiased=False)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    denominator = denominator.abs().clamp_min(torch.finfo(denominator.dtype).tiny)
    return numerator / denominator


def _axis_summary(prefix: str, values: torch.Tensor) -> dict[str, float]:
    x = values.detach().float()
    return {
        f"{prefix}_mean": _as_float(x.mean()),
        f"{prefix}_abs_mean": _as_float(x.abs().mean()),
        f"{prefix}_abs_max": _as_float(x.abs().max()),
        f"{prefix}_std": _as_float(_safe_std(x)),
        f"{prefix}_cv": _as_float(_safe_ratio(_safe_std(x), x.mean())),
    }


def tensor_dimension_stats(tensor: torch.Tensor) -> dict[str, float]:
    """Summarize a B,T,C activation along the axes relevant to SBN/SeqBN/CFBN."""
    x = tensor.detach().float()
    stats = {
        "global_mean": _as_float(x.mean()),
        "global_mean_abs": _as_float(x.mean().abs()),
        "global_std": _as_float(_safe_std(x)),
        "global_var": _as_float(x.var(unbiased=False)),
        "global_rms": _as_float(torch.sqrt(torch.mean(x.square()))),
        "global_abs_max": _as_float(x.abs().max()),
        "nan_count": _as_float(torch.isnan(x).sum()),
        "inf_count": _as_float(torch.isinf(x).sum()),
    }
    if x.dim() != 3:
        return stats

    # Feature axis: LayerNorm/RMS-style behavior per sample/token.
    stats.update(_axis_summary("feature_mean", x.mean(dim=2)))
    stats.update(_axis_summary("feature_var", x.var(dim=2, unbiased=False)))

    # Sequence axis: SBN/CSBN-style behavior per sample/channel.
    stats.update(_axis_summary("sequence_mean", x.mean(dim=1)))
    stats.update(_axis_summary("sequence_var", x.var(dim=1, unbiased=False)))

    # Batch+feature per token: fixed-position SeqBN behavior.
    stats.update(_axis_summary("batch_feature_mean", x.mean(dim=(0, 2))))
    stats.update(_axis_summary("batch_feature_var", x.var(dim=(0, 2), unbiased=False)))

    # Batch+sequence per channel: CFBN behavior.
    stats.update(_axis_summary("batch_sequence_mean", x.mean(dim=(0, 1))))
    stats.update(_axis_summary("batch_sequence_var", x.var(dim=(0, 1), unbiased=False)))

    # Batch-only dispersion at each token/channel, useful for CFBN vs SBN.
    stats.update(_axis_summary("batch_mean", x.mean(dim=0)))
    stats.update(_axis_summary("batch_var", x.var(dim=0, unbiased=False)))
    return stats


def _is_norm_probe_site(name: str, module: torch.nn.Module) -> bool:
    if name.endswith(("ln_1", "ln_2")) or name == "transformer.ln_f":
        return True
    cls_name = module.__class__.__name__.lower()
    return cls_name.endswith("norm") and name


@contextmanager
def capture_norm_activations(model: torch.nn.Module):
    records: list[dict] = []
    handles = []

    def hook(name: str):
        def _hook(_module, inputs, output):
            if not inputs:
                return
            inp = inputs[0]
            if isinstance(inp, torch.Tensor) and isinstance(output, torch.Tensor):
                records.append(
                    {
                        "module": name,
                        "input": inp.detach().cpu(),
                        "output": output.detach().cpu(),
                    }
                )

        return _hook

    for name, module in model.named_modules():
        if _is_norm_probe_site(name, module):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


@dataclass
class BatchProvider:
    vocab_size: int
    block_size: int
    device: torch.device
    dataset: TinyShakespeareBatches | None = None

    def get_batch(self, split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dataset is not None:
            return self.dataset.get_batch(split, batch_size)
        x = torch.randint(self.vocab_size, (batch_size, self.block_size), device=self.device)
        y = torch.randint(self.vocab_size, (batch_size, self.block_size), device=self.device)
        return x, y


def build_batch_provider(args, device: torch.device) -> BatchProvider:
    if args.data_mode == "synthetic":
        return BatchProvider(vocab_size=args.vocab_size, block_size=args.block_size, device=device)

    data_dir = Path(args.data_dir)
    required = [data_dir / "train.bin", data_dir / "val.bin", data_dir / "meta.json"]
    if args.auto_prepare and not all(path.exists() for path in required):
        prepare_tinyshakespeare(data_dir)
    dataset = TinyShakespeareBatches(data_dir, args.block_size, device)
    return BatchProvider(
        vocab_size=int(dataset.meta["vocab_size"]),
        block_size=args.block_size,
        device=device,
        dataset=dataset,
    )


def build_cfg(args, norm: str, vocab_size: int) -> SimpleNamespace:
    names = _split_csv(norm.replace("+", ","))
    allow_noncausal = args.allow_noncausal_norm or any(name in NONCAUSAL_NORMS for name in names)
    return SimpleNamespace(
        model_family="nanogpt",
        arch="nanoGPT",
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        norm=norm,
        norm_cfg=dict(args.norm_cfg or {}),
        norm_no_affine=args.norm_no_affine,
        attn_norm=args.attn_norm,
        attn_norm_cfg=None,
        mlp_norm=args.mlp_norm,
        mlp_norm_cfg=None,
        final_norm=args.final_norm,
        final_norm_cfg=None,
        activation=args.activation,
        activation_cfg={},
        mlp_activation=None,
        mlp_activation_cfg=None,
        allow_noncausal_norm=allow_noncausal,
        norm_site=args.norm_site,
        norm_sites=args.norm_sites,
        mean_shift_alpha=0.0,
        mean_shift_target="pre_norm",
        centering_rescue="none",
        init_preset="default",
        init_gain=1.0,
        init_std=0.02,
        init_bias=0.0,
    )


def build_model(args, norm: str, vocab_size: int, device: torch.device) -> torch.nn.Module:
    cfg = build_cfg(args, norm, vocab_size)
    ext.normalization.setting(cfg)
    ext.activation.setting(cfg)
    model = ext.model.get_model(cfg).to(device)
    return model


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, batches: BatchProvider, batch_size: int, eval_iters: int) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(eval_iters):
        inputs, targets = batches.get_batch("val", batch_size)
        _logits, loss = model(inputs, targets)
        total += float(loss.detach())
    if was_training:
        model.train()
    return total / max(1, eval_iters)


@torch.no_grad()
def collect_stats(
    model: torch.nn.Module,
    probe_inputs: torch.Tensor,
    norm: str,
    step: int,
    split: str,
) -> list[dict[str, float | str | int]]:
    was_training = model.training
    model.eval()
    with capture_norm_activations(model) as records:
        model(probe_inputs)
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
            rows.append(row)
    return rows


@torch.no_grad()
def collect_causality_rows(
    model: torch.nn.Module,
    probe_inputs: torch.Tensor,
    norm: str,
    changed_pos: int,
) -> list[dict[str, float | str | int]]:
    if changed_pos <= 0 or changed_pos >= probe_inputs.shape[1]:
        raise ValueError("--causal-change-pos must be inside [1, block_size - 1].")

    was_training = model.training
    model.eval()
    base_inputs = probe_inputs.clone()
    changed_inputs = probe_inputs.clone()
    changed_inputs[:, changed_pos:] = (changed_inputs[:, changed_pos:] + 1) % int(model.lm_head.out_features)

    with capture_norm_activations(model) as base_records:
        model(base_inputs)
    with capture_norm_activations(model) as changed_records:
        model(changed_inputs)
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
            prefix_delta = (x[:, :changed_pos, :] - y[:, :changed_pos, :]).abs()
            suffix_delta = (x[:, changed_pos:, :] - y[:, changed_pos:, :]).abs()
            rows.append(
                {
                    "norm": norm,
                    "module": base["module"],
                    "tensor": tensor_name,
                    "changed_pos": changed_pos,
                    "prefix_abs_max": _as_float(prefix_delta.max()),
                    "prefix_abs_mean": _as_float(prefix_delta.mean()),
                    "suffix_abs_mean": _as_float(suffix_delta.mean()),
                }
            )
    return rows


def train_steps(
    model: torch.nn.Module,
    batches: BatchProvider,
    args,
    norm: str,
    probe_inputs: torch.Tensor,
) -> tuple[list[dict], list[dict]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stats_rows = collect_stats(model, probe_inputs, norm=norm, step=0, split="probe")
    train_rows = []
    model.train()

    for step in range(1, args.train_steps + 1):
        inputs, targets = batches.get_batch("train", args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = model(inputs, targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        train_rows.append(
            {
                "norm": norm,
                "step": step,
                "train_loss": float(loss.detach()),
                "grad_norm": _as_float(grad_norm),
            }
        )
        if args.stat_every > 0 and (step % args.stat_every == 0 or step == args.train_steps):
            stats_rows.extend(collect_stats(model, probe_inputs, norm=norm, step=step, split="probe"))
    return stats_rows, train_rows


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Dimension-wise SeqBN/SBN/CFBN probe")
    parser.add_argument("--norms", default=DEFAULT_NORMS, help="comma-separated norm variants")
    parser.add_argument("--output-dir", default="results/norm-dimension-probe")
    parser.add_argument("--data-mode", choices=("synthetic", "tinyshakespeare"), default="synthetic")
    parser.add_argument("--data-dir", default="dataset/tinyshakespeare")
    parser.add_argument("--auto-prepare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=65)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--norm-cfg", type=ext.utils.str2dict, default={})
    parser.add_argument("--norm-no-affine", action="store_true")
    parser.add_argument("--allow-noncausal-norm", action="store_true")
    parser.add_argument("--norm-site", default=None, choices=("all", "attn", "mlp", "final", "none"))
    parser.add_argument("--norm-sites", default=None)
    parser.add_argument("--attn-norm", default=None)
    parser.add_argument("--mlp-norm", default=None)
    parser.add_argument("--final-norm", default=None)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--stat-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--causal-change-pos", type=int, default=None)
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
    probe_inputs, _probe_targets = batches.get_batch("val", args.probe_batch_size)
    changed_pos = args.causal_change_pos if args.causal_change_pos is not None else max(1, args.block_size // 2)

    all_stats_rows = []
    all_train_rows = []
    all_effect_rows = []
    all_causality_rows = []

    for norm in norms:
        _set_seed(args.seed)
        model = build_model(args, norm, batches.vocab_size, device)
        initial_val_loss = evaluate_loss(model, batches, args.batch_size, args.eval_iters)
        stats_rows, train_rows = train_steps(model, batches, args, norm, probe_inputs)
        final_val_loss = evaluate_loss(model, batches, args.batch_size, args.eval_iters)
        causality_rows = collect_causality_rows(model, probe_inputs, norm, changed_pos=changed_pos)

        all_stats_rows.extend(stats_rows)
        all_train_rows.extend(train_rows)
        all_causality_rows.extend(causality_rows)
        final_train_loss = train_rows[-1]["train_loss"] if train_rows else float("nan")
        all_effect_rows.append(
            {
                "norm": norm,
                "seed": args.seed,
                "data_mode": args.data_mode,
                "train_steps": args.train_steps,
                "initial_val_loss": initial_val_loss,
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
                "final_val_perplexity": math.exp(min(final_val_loss, 20.0)),
            }
        )
        print(
            f"{norm}: initial_val_loss={initial_val_loss:.4f}, "
            f"final_val_loss={final_val_loss:.4f}, train_steps={args.train_steps}"
        )

    write_csv(output_dir / "dimension_stats.csv", all_stats_rows)
    write_csv(output_dir / "train_trace.csv", all_train_rows)
    write_csv(output_dir / "effect_summary.csv", all_effect_rows)
    write_csv(output_dir / "causality_probe.csv", all_causality_rows)
    write_jsonl(output_dir / "dimension_stats.jsonl", all_stats_rows)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote probe outputs to {output_dir}")


if __name__ == "__main__":
    main()
