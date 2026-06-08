#!/usr/bin/env python3
import argparse
import csv
import os
import re


RUN_RE = re.compile(
    r"BERT_translation_L(?P<layers>\d+)_H(?P<heads>\d+)_D(?P<embd>\d+)"
    r"_src(?P<src_len>\d+)_tgt(?P<tgt_len>\d+)"
    r"_(?P<norm>.+?)_(?P<activation>[^_]+)"
    r"_lr(?P<lr>[^_]+)_bs(?P<batch>[^_]+)_wd(?P<weight_decay>[^_]+)_seed(?P<seed>\d+)"
)
VAL_RE = re.compile(
    r"Validate on epoch (?P<epoch>-?\d+): loss=(?P<loss>[0-9.eE+-]+), "
    r"perplexity=(?P<perplexity>[0-9.eE+-]+), token_acc=(?P<token_acc>[0-9.eE+-]+)%"
)


def parse_log(log_path):
    best = None
    last = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = VAL_RE.search(line)
            if not match:
                continue
            record = {
                "epoch": int(match.group("epoch")),
                "val_loss": float(match.group("loss")),
                "val_perplexity": float(match.group("perplexity")),
                "val_token_acc": float(match.group("token_acc")),
            }
            last = record
            if best is None or record["val_loss"] < best["val_loss"]:
                best = record
    return best, last


def iter_runs(results_root):
    for dirpath, _dirnames, filenames in os.walk(results_root):
        if "log.txt" not in filenames:
            continue
        run_name = os.path.basename(dirpath)
        match = RUN_RE.match(run_name)
        if not match:
            continue
        best, last = parse_log(os.path.join(dirpath, "log.txt"))
        if best is None:
            continue
        row = match.groupdict()
        row.update(
            {
                "run_dir": dirpath,
                "best_epoch": best["epoch"],
                "best_val_loss": best["val_loss"],
                "best_val_perplexity": best["val_perplexity"],
                "best_val_token_acc": best["val_token_acc"],
                "last_epoch": last["epoch"],
                "last_val_loss": last["val_loss"],
                "last_val_perplexity": last["val_perplexity"],
                "last_val_token_acc": last["val_token_acc"],
            }
        )
        yield row


def main():
    parser = argparse.ArgumentParser("Summarize BERT translation norm/LR results")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = sorted(
        iter_runs(args.results_root),
        key=lambda row: (row["norm"], float(row["lr"]), int(row["seed"])),
    )
    fieldnames = [
        "norm",
        "lr",
        "seed",
        "best_epoch",
        "best_val_loss",
        "best_val_perplexity",
        "best_val_token_acc",
        "last_epoch",
        "last_val_loss",
        "last_val_perplexity",
        "last_val_token_acc",
        "layers",
        "heads",
        "embd",
        "src_len",
        "tgt_len",
        "activation",
        "batch",
        "weight_decay",
        "run_dir",
    ]

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.output}")
        return

    writer = csv.DictWriter(os.sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
