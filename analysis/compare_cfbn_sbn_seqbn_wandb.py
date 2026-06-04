import csv
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import wandb


OUT_DIR = Path(__file__).resolve().parent / "cfbn_sbn_seqbn_compare"

PROJECTS = {
    "cfbn_nano": "whsjrc-buaa/nanoGPT-CFBN-focused",
    "seqbn_nano": "whsjrc-buaa/nanoGPT-SeqBN",
    "cfbn_vit": "whsjrc-buaa/ViT-CFBN-focused",
    "vit_patch8": "whsjrc-buaa/ViT-SeqBN-patch8",
    "vit_seed": "whsjrc-buaa/ViT-SeqBN-seed-validation",
    "vit_diag": "whsjrc-buaa/ViT-SeqBN-failure-diagnostics",
}

NANO_RE = re.compile(
    r"nanoGPT_tinyshakespeare_L(?P<n_layer>\d+)_H(?P<n_head>\d+)_D(?P<n_embd>\d+)"
    r"_ctx(?P<block_size>\d+)_(?P<norm_tag>.+)_(?P<activation>[^_]+)"
    r"_lr(?P<lr>[^_]+)_bs(?P<batch_size>[^_]+)(?:_drop(?P<dropout>[^_]+))?"
    r"_wd(?P<weight_decay>[^_]+)_s(?:eed)?(?P<seed>\d+)"
)

VIT_RE = re.compile(
    r"ViT_(?P<arch>[^_]+_[^_]+)_(?P<dataset>[^_]+)_img(?P<image_size>\d+)"
    r"_patch(?P<patch_size>\d+)_(?P<norm>[^_]+)_(?P<activation>[^_]+)"
    r"_lr(?P<lr>[^_]+)_bs(?P<batch_size>\d+)_drop(?:out)?(?P<dropout>[^_]+)"
    r"_d(?:rop)?path(?P<drop_path>[^_]+)_wd(?P<weight_decay>[^_]+)_s(?:eed)?(?P<seed>\d+)"
)


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


def parse_num(row, keys):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            row[key] = as_float(row[key])


def parse_nano(name):
    match = NANO_RE.match(name)
    row = match.groupdict() if match else {}
    tag = row.get("norm_tag", "")
    parts = dict(re.findall(r"(attn|mlp|final)([A-Za-z0-9+]+)", tag))
    if tag.startswith("control"):
        row["slot"] = "baseline"
        row["norm"] = tag[len("control") :]
    elif tag.startswith("all"):
        row["slot"] = "all"
        row["norm"] = tag[len("all") :]
    elif re.fullmatch(r"[A-Za-z0-9+]+", tag):
        row["slot"] = "all"
        row["norm"] = tag
    else:
        active = [(slot, norm) for slot, norm in parts.items() if norm != "LN"]
        if active:
            row["slot"] = active[0][0]
            row["norm"] = active[0][1]
        elif parts and all(norm == "LN" for norm in parts.values()):
            row["slot"] = "all"
            row["norm"] = "LN"
        else:
            row["slot"] = ""
            row["norm"] = ""
    parse_num(row, ["lr", "weight_decay", "dropout", "seed"])
    return row


def parse_vit(name):
    match = VIT_RE.match(name)
    row = match.groupdict() if match else {}
    parse_num(row, ["image_size", "patch_size", "batch_size", "lr", "dropout", "drop_path", "weight_decay", "seed"])
    return row


def load_runs():
    api = wandb.Api(timeout=60)
    rows = []
    for label, project in PROJECTS.items():
        model = "nanoGPT" if "nano" in label else "ViT"
        for run in api.runs(project, per_page=300):
            try:
                summary = dict(run.summary)
            except Exception as exc:
                rows.append(
                    {
                        "source": label,
                        "project": project,
                        "model": model,
                        "run_id": run.id,
                        "name": run.name,
                        "state": "summary_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            row = {
                "source": label,
                "project": project,
                "model": model,
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": getattr(run, "created_at", ""),
                "url": run.url,
                "runtime_sec": as_float(summary.get("_runtime")),
            }
            row.update(parse_nano(run.name) if model == "nanoGPT" else parse_vit(run.name))
            if model == "nanoGPT":
                row["final_train_loss"] = as_float(summary.get("train_loss"))
                row["final_val_loss"] = as_float(summary.get("val_loss"))
                row["final_val_perplexity"] = as_float(summary.get("val_perplexity"))
                row["best_val_loss"] = row["final_val_loss"]
                row["best_val_perplexity"] = row["final_val_perplexity"]
            else:
                row["final_train_acc"] = as_float(summary.get("train_acc"))
                row["final_test_acc"] = as_float(summary.get("test_acc"))
                row["final_train_loss"] = as_float(summary.get("train_loss"))
                row["final_test_loss"] = as_float(summary.get("test_loss"))
                row["best_test_acc"] = row["final_test_acc"]
            rows.append(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "model",
        "source",
        "norm",
        "slot",
        "patch_size",
        "lr",
        "seed",
        "final_val_loss",
        "best_val_loss",
        "best_val_perplexity",
        "final_test_acc",
        "best_test_acc",
        "final_train_acc",
        "final_test_loss",
        "name",
        "url",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_mean(rows, key):
    vals = [as_float(r.get(key)) for r in rows]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return math.nan, 0
    return sum(vals) / len(vals), len(vals)


def build_markdown(rows):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# CFBN vs SBN/SeqBN W&B Comparison",
        "",
        f"- generated: {now}",
        f"- projects: {', '.join(f'`{p}`' for p in PROJECTS.values())}",
        "",
        "## nanoGPT, lr=6e-4, seed=0",
        "",
        "| family | norm | slot | best val loss | best ppl | final val loss |",
        "|---|---|---|---:|---:|---:|",
    ]
    nano = [
        r
        for r in rows
        if r["model"] == "nanoGPT"
        and r.get("state") == "finished"
        and abs(as_float(r.get("lr")) - 0.0006) < 1e-12
        and as_float(r.get("seed")) == 0
        and not math.isnan(as_float(r.get("best_val_loss")))
    ]
    family_order = {"cfbn_nano": 0, "seqbn_nano": 1}
    nano = sorted(nano, key=lambda r: (family_order.get(r["source"], 9), r.get("norm", ""), r.get("slot", "")))
    for r in nano:
        family = "CFBN" if r["source"] == "cfbn_nano" else "SBN/SeqBN"
        lines.append(
            f"| {family} | {r.get('norm','')} | {r.get('slot','')} | "
            f"{fmt(r.get('best_val_loss'))} | {fmt(r.get('best_val_perplexity'))} | {fmt(r.get('final_val_loss'))} |"
        )

    lines += [
        "",
        "### nanoGPT slot/family summary",
        "",
        "| family | norm | mean best val loss | n | best slot | best val loss |",
        "|---|---|---:|---:|---|---:|",
    ]
    for source in ["cfbn_nano", "seqbn_nano"]:
        source_rows = [r for r in nano if r["source"] == source]
        for norm in sorted({r.get("norm", "") for r in source_rows}):
            group = [r for r in source_rows if r.get("norm") == norm]
            mean_loss, n = group_mean(group, "best_val_loss")
            best = min(group, key=lambda r: as_float(r.get("best_val_loss")))
            family = "CFBN" if source == "cfbn_nano" else "SBN/SeqBN"
            lines.append(f"| {family} | {norm} | {fmt(mean_loss)} | {n} | {best.get('slot','')} | {fmt(best.get('best_val_loss'))} |")

    lines += [
        "",
        "## ViT comparable slices",
        "",
        "| slice | norm | source | mean best test acc | n | final test acc mean |",
        "|---|---|---|---:|---:|---:|",
    ]
    vit = [r for r in rows if r["model"] == "ViT" and r.get("state") == "finished" and not math.isnan(as_float(r.get("best_test_acc")))]
    slices = [
        ("patch4 lr1e-4", lambda r: as_float(r.get("patch_size")) == 4 and abs(as_float(r.get("lr")) - 0.0001) < 1e-12),
        ("patch8 lr1e-4", lambda r: as_float(r.get("patch_size")) == 8 and abs(as_float(r.get("lr")) - 0.0001) < 1e-12),
        ("patch16 lr1e-3", lambda r: as_float(r.get("patch_size")) == 16 and abs(as_float(r.get("lr")) - 0.001) < 1e-12),
        ("patch16 lr1e-4", lambda r: as_float(r.get("patch_size")) == 16 and abs(as_float(r.get("lr")) - 0.0001) < 1e-12),
    ]
    for slice_name, pred in slices:
        slice_rows = [r for r in vit if pred(r) and r.get("norm") in {"CFBN", "CFBNc", "CFBNs", "SBN", "SBNs", "SeqBN", "SeqBNs", "DSeqBN", "DSeqBNs", "LN"}]
        for norm in sorted({r.get("norm", "") for r in slice_rows}):
            group = [r for r in slice_rows if r.get("norm") == norm]
            mean_best, n = group_mean(group, "best_test_acc")
            mean_final, _ = group_mean(group, "final_test_acc")
            sources = ",".join(sorted({r["source"] for r in group}))
            lines.append(f"| {slice_name} | {norm} | {sources} | {fmt(mean_best, 2)} | {n} | {fmt(mean_final, 2)} |")

    return "\n".join(lines) + "\n"


def main():
    rows = load_runs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "runs.csv", rows)
    md = build_markdown(rows)
    (OUT_DIR / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"saved {OUT_DIR / 'runs.csv'}")
    print(f"saved {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
