import math
import os
import sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(__file__))
TAIYI_ROOT = os.path.join(ROOT, "Taiyi")
if TAIYI_ROOT not in sys.path:
    sys.path.insert(0, TAIYI_ROOT)

from Taiyi.taiyi.monitor import Monitor


def test_input_value_histogram_tracks_value_and_abs_value_distributions():
    layer = nn.Identity()
    monitor = Monitor(layer, {"": ["InputValueHistogram"]})

    x = torch.tensor([[1.0, -1.0, 1.0, 3.0, 3.0, 1.0]])
    _ = layer(x)
    monitor.track(0)

    output = monitor.get_output()[""]["InputValueHistogram"][0]
    assert output["num_samples"] == 1
    assert output["num_dims_per_sample"] == 6
    assert int(output["value_hist"].sum()) == 6
    assert int(output["abs_value_hist"].sum()) == 6
    assert math.isclose(float(output["value_mean"]), float(x.mean()), rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(output["abs_value_mean"]), float(x.abs().mean()), rel_tol=0.0, abs_tol=1e-6)


def test_input_norm_contribution_identifies_single_dimension_dominance():
    layer = nn.Identity()
    monitor = Monitor(layer, {"": ["InputNormContribution"]})

    x = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    _ = layer(x)
    monitor.track(0)

    output = monitor.get_output()[""]["InputNormContribution"][0]
    rms = output["rms"]
    ln = output["ln"]

    assert rms["valid_samples"] == 1
    assert math.isclose(float(rms["top1_share_mean"]), 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(rms["effective_dims_mean"]), 1.0, rel_tol=0.0, abs_tol=1e-6)

    assert ln["valid_samples"] == 1
    assert float(ln["top1_share_mean"]) > 0.7
    assert float(ln["effective_dims_mean"]) < 2.0


def test_input_norm_contribution_identifies_uniform_multi_dimension_control():
    layer = nn.Identity()
    monitor = Monitor(layer, {"": ["InputNormContribution"]})

    x = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    _ = layer(x)
    monitor.track(0)

    output = monitor.get_output()[""]["InputNormContribution"][0]
    rms = output["rms"]
    ln = output["ln"]

    assert math.isclose(float(rms["top1_share_mean"]), 0.25, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(rms["effective_dims_mean"]), 4.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(ln["top1_share_mean"]), 0.25, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(ln["effective_dims_mean"]), 4.0, rel_tol=0.0, abs_tol=1e-6)
