import pytest

torch = pytest.importorskip("torch")

import extension as ext
from Taiyi.taiyi.monitor import Monitor
from extension.model.vit.vision_transformer import vit_tiny


def test_vit_diagnostics_are_taiyi_quantities():
    model = vit_tiny(img_size=32, patch_size=16, num_classes=3, drop_path_rate=0.0, act_layer=ext.activation.Activation)
    monitor = Monitor(
        model,
        {
            "blocks.0": [["ViTResidualStats", "linear(1,0)"]],
            "blocks.0.norm1": [["ViTNormStats", "linear(1,0)"]],
            "norm": [["ViTNormStats", "linear(1,0)"]],
            "fc": [["ViTLogitsStats", "linear(1,0)"]],
        },
    )

    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 3)
    monitor.track(0)
    output = monitor.get_output()

    norm_stats = output["blocks.0.norm1"]["ViTNormStats"][0]
    assert "input_mean" in norm_stats
    assert "output_across_sequence_std_abs_max" in norm_stats
    assert "input_cls_patch_rms_ratio" in norm_stats

    residual_stats = output["blocks.0"]["ViTResidualStats"][0]
    assert "attn_branch_to_stream_ratio" in residual_stats
    assert "mlp_branch_to_stream_ratio" in residual_stats

    logits_stats = output["fc"]["ViTLogitsStats"][0]
    assert "softmax_entropy_mean" in logits_stats
    assert "softmax_max_prob_mean" in logits_stats
