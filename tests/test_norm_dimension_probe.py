import csv
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from nanoGPT.norm_dimension_probe import tensor_dimension_stats


def test_tensor_dimension_stats_reports_expected_axes():
    x = torch.randn(3, 5, 7)

    stats = tensor_dimension_stats(x)

    assert "feature_mean_abs_mean" in stats
    assert "sequence_var_abs_mean" in stats
    assert "batch_feature_mean_abs_mean" in stats
    assert "batch_sequence_var_abs_mean" in stats
    assert stats["nan_count"] == 0.0


def test_norm_dimension_probe_cli_writes_outputs(tmp_path):
    output_dir = tmp_path / "probe"
    cmd = [
        sys.executable,
        "nanoGPT/norm_dimension_probe.py",
        "--data-mode",
        "synthetic",
        "--output-dir",
        str(output_dir),
        "--norms",
        "LN,CSBN",
        "--n-layer",
        "1",
        "--n-head",
        "1",
        "--n-embd",
        "16",
        "--block-size",
        "8",
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
    causality_path = output_dir / "causality_probe.csv"
    assert effect_path.exists()
    assert stats_path.exists()
    assert causality_path.exists()

    with effect_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["norm"] for row in rows} == {"LN", "CSBN"}
    assert all(float(row["final_val_loss"]) > 0 for row in rows)

    with stats_path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "feature_var_abs_mean" in header
    assert "batch_sequence_mean_abs_mean" in header
