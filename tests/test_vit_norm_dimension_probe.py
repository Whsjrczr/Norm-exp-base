import csv
import subprocess
import sys

import pytest

pytest.importorskip("torch")


def test_vit_norm_dimension_probe_cli_writes_outputs(tmp_path):
    output_dir = tmp_path / "vit_probe"
    cmd = [
        sys.executable,
        "ViT/vit_norm_dimension_probe.py",
        "--data-mode",
        "synthetic",
        "--output-dir",
        str(output_dir),
        "--norms",
        "LN,SBN,CFBN",
        "--image-size",
        "16",
        "--patch-size",
        "8",
        "--num-classes",
        "4",
        "--embed-dim",
        "16",
        "--depth",
        "1",
        "--num-heads",
        "1",
        "--batch-size",
        "2",
        "--probe-batch-size",
        "2",
        "--train-steps",
        "1",
        "--eval-iters",
        "1",
        "--stat-every",
        "1",
        "--device",
        "cpu",
    ]

    subprocess.run(cmd, check=True)

    effect_path = output_dir / "effect_summary.csv"
    stats_path = output_dir / "dimension_stats.csv"
    sensitivity_path = output_dir / "patch_sensitivity_probe.csv"
    assert effect_path.exists()
    assert stats_path.exists()
    assert sensitivity_path.exists()

    with effect_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["norm"] for row in rows} == {"LN", "SBN", "CFBN"}
    assert all(float(row["final_val_loss"]) > 0 for row in rows)

    with stats_path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "feature_var_abs_mean" in header
    assert "batch_sequence_mean_abs_mean" in header
    assert "vit_cls_patch_rms_ratio" in header
