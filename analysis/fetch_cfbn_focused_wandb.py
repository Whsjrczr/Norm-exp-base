import csv
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import wandb


PROJECTS = {
    "nanoGPT": "whsjrc-buaa/nanoGPT-CFBN-focused",
    "ViT": "whsjrc-buaa/ViT-CFBN-focused",
}

OUT_DIR = Path(__file__).resolve().parent / "cfbn_focused_wandb"

NANOGPT_RE = re.compile(
    r"nanoGPT_tinyshakespeare_L(?P<n_layer>\d+)_H(?P<n_head>\d+)_D(?P<n_embd>\d+)"
    r"_ctx(?P<block_size>\d+)_(?P<norm_tag>.+)_(?P<activation>[^_]+)"
    r"_lr(?P<lr>[^_]+)_bs(?P<batch_size>[^_]+)_wd(?P<weight_decay>[^_]+)"
    r"_seed(?P<seed>\d+)"
)

VIT_RE = re.compile(
    r"ViT_(?P<arch>[^_]+_[^_]+)_(?P<dataset>[^_]+)_img(?P<image_size>\d+)"
    r"_patch(?P<patch_size>\d+)_(?P<norm>[^_]+)_(?P<activation>[^_]+)"
    r"_lr(?P<lr>[^_]+)_bs(?P<batch_size>\d+)_dropout(?P<dropout>[^_]+)"
    r"_droppath(?P<drop_path>[^_]+)_wd(?P<weight_decay>[^_]+)_seed(?P<seed>\d+)"
)

NANOGPT_KEYS = ["train_loss", "val_loss", "val_perplexity", "epochs", "steps", "learning_rate"]
VIT_KEYS = [
    "train_acc",
    "test_acc",
    "train_acc5",
    "test_acc5",
    "train_loss",
    "test_loss",
    "epochs",
    "steps",
    "learning_rate",
]


def as_float(value):
    try:
        if value is None:
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value, digits=4):
    value = as_float(value)
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def pct_delta(value, base):
    value = as_float(value)
    base = as_float(base)
    if math.isnan(value) or math.isnan(base) or base == 0:
        return math.nan
    return 100.0 * (value - base) / base


def parse_nanogpt(name):
    match = NANOGPT_RE.match(name)
    parsed = match.groupdict() if match else {}
    tag = parsed.get("norm_tag", "")
    parts = dict(re.findall(r"(attn|mlp|final)([A-Za-z0-9]+)", tag))
    if tag in {"CCFBN", "CCFBNc", "CCFBNs"}:
        parsed["slot"] = "all"
        parsed["norm"] = tag
    elif tag.startswith("all"):
        parsed["slot"] = "all"
        parsed["norm"] = tag[3:]
    else:
        active = [(slot, norm) for slot, norm in parts.items() if norm != "LN"]
        parsed["slot"] = active[0][0] if active else ""
        parsed["norm"] = active[0][1] if active else ""
    return parsed


def parse_vit(name):
    match = VIT_RE.match(name)
    parsed = match.groupdict() if match else {}
    return parsed


def numericize(row):
    for key in list(row):
        if key in {
            "n_layer",
            "n_head",
            "n_embd",
            "block_size",
            "seed",
            "image_size",
            "patch_size",
            "batch_size",
        }:
            try:
                row[key] = int(row[key])
            except (TypeError, ValueError):
                pass
        elif key in {"lr", "weight_decay", "dropout", "drop_path"}:
            row[key] = as_float(row[key])
    return row


def extrema(points, key, mode):
    vals = [(as_float(p.get(key)), p) for p in points]
    vals = [(v, p) for v, p in vals if not math.isnan(v)]
    if not vals:
        return math.nan, math.nan
    value, point = (min(vals, key=lambda x: x[0]) if mode == "min" else max(vals, key=lambda x: x[0]))
    return value, as_float(point.get("epochs"))


def scan_points(run, keys):
    rows = []
    # Train and validation metrics are logged as separate W&B points at the
    # same epoch, so key-filtered scan_history can return no rows.
    for point in run.scan_history(page_size=500):
        rows.append(point)
    return rows


def load_runs():
    api = wandb.Api()
    raw_rows = []
    history_rows = []

    for model, project in PROJECTS.items():
        keys = NANOGPT_KEYS if model == "nanoGPT" else VIT_KEYS
        for run in api.runs(project, per_page=100):
            summary = dict(run.summary)
            parsed = parse_nanogpt(run.name) if model == "nanoGPT" else parse_vit(run.name)
            row = {
                "model": model,
                "project": project,
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": getattr(run, "created_at", ""),
                "url": run.url,
                "runtime_sec": as_float(summary.get("_runtime")),
            }
            row.update(numericize(parsed))
            for key in keys:
                row[f"final_{key}"] = as_float(summary.get(key))

            points = scan_points(run, keys)
            for point in points:
                history_rows.append(
                    {
                        "model": model,
                        "run_id": run.id,
                        "name": run.name,
                        "epoch": point.get("epochs"),
                        "step": point.get("steps", point.get("_step")),
                        **{key: point.get(key) for key in keys},
                    }
                )

            if model == "nanoGPT":
                row["best_val_loss"], row["best_val_loss_epoch"] = extrema(points, "val_loss", "min")
                row["best_val_perplexity"], row["best_val_perplexity_epoch"] = extrema(
                    points, "val_perplexity", "min"
                )
                row["min_train_loss"], row["min_train_loss_epoch"] = extrema(points, "train_loss", "min")
            else:
                row["best_test_acc"], row["best_test_acc_epoch"] = extrema(points, "test_acc", "max")
                row["best_train_acc"], row["best_train_acc_epoch"] = extrema(points, "train_acc", "max")
                row["min_test_loss"], row["min_test_loss_epoch"] = extrema(points, "test_loss", "min")
                row["min_train_loss"], row["min_train_loss_epoch"] = extrema(points, "train_loss", "min")
                row["final_gap"] = row.get("final_train_acc", math.nan) - row.get("final_test_acc", math.nan)
            raw_rows.append(row)

    return raw_rows, history_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "model",
        "run_id",
        "state",
        "created_at",
        "slot",
        "norm",
        "patch_size",
        "final_val_loss",
        "final_val_perplexity",
        "best_val_loss",
        "best_val_perplexity",
        "final_test_acc",
        "best_test_acc",
        "final_train_acc",
        "final_gap",
        "final_test_loss",
        "runtime_sec",
        "name",
        "url",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(rows):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    nano = sorted([r for r in rows if r["model"] == "nanoGPT"], key=lambda r: (r.get("slot", ""), r.get("norm", "")))
    vit = sorted([r for r in rows if r["model"] == "ViT"], key=lambda r: (r.get("patch_size", 0), r.get("norm", "")))

    best_nano = min(nano, key=lambda r: as_float(r.get("best_val_loss")))
    best_vit = max(vit, key=lambda r: as_float(r.get("best_test_acc")))

    by_norm_nano = {}
    for r in nano:
        by_norm_nano.setdefault(r.get("norm", ""), []).append(r)
    by_patch_vit = {}
    for r in vit:
        by_patch_vit.setdefault(r.get("patch_size"), []).append(r)

    lines = [
        "# CFBN Focused W&B Summary",
        "",
        f"- generated: {now}",
        f"- nanoGPT project: `{PROJECTS['nanoGPT']}` ({len(nano)} runs)",
        f"- ViT project: `{PROJECTS['ViT']}` ({len(vit)} runs)",
        "- all runs are seed 0 and finished unless noted in the tables.",
        "",
        "## nanoGPT TinyShakespeare",
        "",
        "| slot | norm | final val loss | best val loss | best ppl | final train loss | epoch | runtime(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in nano:
        lines.append(
            "| {slot} | {norm} | {fvl} | {bvl} | {bppl} | {ftl} | {ep} | {rt} |".format(
                slot=r.get("slot", ""),
                norm=r.get("norm", ""),
                fvl=fmt(r.get("final_val_loss")),
                bvl=fmt(r.get("best_val_loss")),
                bppl=fmt(r.get("best_val_perplexity")),
                ftl=fmt(r.get("final_train_loss")),
                ep=fmt(r.get("best_val_loss_epoch"), 0),
                rt=fmt(r.get("runtime_sec"), 0),
            )
        )
    lines += [
        "",
        f"Best nanoGPT: `{best_nano.get('slot')}/{best_nano.get('norm')}` with best val loss {fmt(best_nano.get('best_val_loss'))} and perplexity {fmt(best_nano.get('best_val_perplexity'))}.",
        "",
        "### nanoGPT by norm",
        "",
        "| norm | mean best val loss | mean best ppl | best slot | best val loss |",
        "|---|---:|---:|---|---:|",
    ]
    for norm, group in sorted(by_norm_nano.items()):
        mean_loss = sum(as_float(r.get("best_val_loss")) for r in group) / len(group)
        mean_ppl = sum(as_float(r.get("best_val_perplexity")) for r in group) / len(group)
        best = min(group, key=lambda r: as_float(r.get("best_val_loss")))
        lines.append(f"| {norm} | {mean_loss:.4f} | {mean_ppl:.4f} | {best.get('slot')} | {fmt(best.get('best_val_loss'))} |")

    lines += [
        "",
        "## ViT CIFAR-10",
        "",
        "| patch | norm | final test acc | best test acc | final train acc | gap | final test loss | epoch | runtime(s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in vit:
        lines.append(
            "| {patch} | {norm} | {fta} | {bta} | {ftra} | {gap} | {ftl} | {ep} | {rt} |".format(
                patch=r.get("patch_size", ""),
                norm=r.get("norm", ""),
                fta=fmt(r.get("final_test_acc")),
                bta=fmt(r.get("best_test_acc")),
                ftra=fmt(r.get("final_train_acc")),
                gap=fmt(r.get("final_gap")),
                ftl=fmt(r.get("final_test_loss")),
                ep=fmt(r.get("best_test_acc_epoch"), 0),
                rt=fmt(r.get("runtime_sec"), 0),
            )
        )
    lines += [
        "",
        f"Best ViT: `patch{best_vit.get('patch_size')}/{best_vit.get('norm')}` with best test acc {fmt(best_vit.get('best_test_acc'))}.",
        "",
        "### ViT by patch",
        "",
        "| patch | best norm | best test acc | CFBNc-CFBN delta | CFBNs-CFBN delta |",
        "|---:|---|---:|---:|---:|",
    ]
    for patch, group in sorted(by_patch_vit.items()):
        best = max(group, key=lambda r: as_float(r.get("best_test_acc")))
        base = next((r for r in group if r.get("norm") == "CFBN"), None)
        c = next((r for r in group if r.get("norm") == "CFBNc"), None)
        s = next((r for r in group if r.get("norm") == "CFBNs"), None)
        base_acc = as_float(base.get("best_test_acc")) if base else math.nan
        delta_c = as_float(c.get("best_test_acc")) - base_acc if c else math.nan
        delta_s = as_float(s.get("best_test_acc")) - base_acc if s else math.nan
        lines.append(f"| {patch} | {best.get('norm')} | {fmt(best.get('best_test_acc'))} | {fmt(delta_c)} | {fmt(delta_s)} |")

    return "\n".join(lines) + "\n"


def main():
    rows, history = load_runs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "runs.csv", rows)
    write_csv(OUT_DIR / "history.csv", history)
    md = build_markdown(rows)
    (OUT_DIR / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"saved {OUT_DIR / 'runs.csv'}")
    print(f"saved {OUT_DIR / 'history.csv'}")
    print(f"saved {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
