import torch
from types import SimpleNamespace

import extension.normalization as normalization
from extension.my_modules.norm.bn1d_modules import (
    BatchNorm1dCentering,
    BatchNorm1dScaling,
    BatchNorm1dScalingRMS,
)
from extension.my_modules.norm.seq_bn import (
    DynamicSequenceBatchNorm1d,
    DynamicSequenceBatchNorm1dCentering,
    DynamicSequenceBatchNorm1dScaling,
    SequenceBatchNorm1d,
    SequenceBatchNorm1dCentering,
    SequenceBatchNorm1dScaling,
)
from extension.my_modules.norm.bn2d_modules import (
    BatchNorm2dCentering,
    BatchNorm2dScaling,
    BatchNorm2dScalingRMS,
)
from extension.my_modules.norm.gn_modules import GroupNormScaling
from extension.my_modules.activation.pgn_modules import (
    PointwiseGroupNorm,
    PointwiseGroupNormCentering,
    PointwiseGroupNormScaling,
    PointwiseGroupNormScalingRMS,
)
from extension.my_modules.activation.mlp_activation import MLPActivation
from extension.my_modules.norm.pln import ParallelLN
from extension.my_modules.activation.pq_activation import PQActivation
from extension.my_modules.norm.pq_norm import PQNorm
from extension.my_modules.activation.sinarctan import SinArctan
import extension.activation as activation
from extension.normalization import _ParallelLayerNorm, _ParallelLayerScaling
from extension.model import build_vit_norm_layer


def test_bn1d_variants_support_channel_first_and_channel_last():
    inputs = [torch.randn(2, 384, 5), torch.randn(2, 5, 384)]
    modules = [BatchNorm1dCentering(384), BatchNorm1dScaling(384), BatchNorm1dScalingRMS(384)]

    for x in inputs:
        for module in modules:
            y = module(x)
            assert y.shape == x.shape


def test_sequence_bn1d_variants_support_token_first_and_token_last():
    inputs = [torch.randn(2, 5, 384), torch.randn(2, 384, 5)]
    modules = [SequenceBatchNorm1dCentering(5), SequenceBatchNorm1dScaling(5)]

    for x in inputs:
        for module in modules:
            y = module(x)
            assert y.shape == x.shape


def test_sequence_bn1d_centering_and_scaling_normalize_sequence_axis():
    x = torch.randn(4, 6, 8)

    centered = SequenceBatchNorm1dCentering(6, affine=False, track_running_stats=False)(x)
    center_mean = centered.mean(dim=(0, 2))
    assert torch.allclose(center_mean, torch.zeros_like(center_mean), atol=1e-6, rtol=1e-6)

    scaled = SequenceBatchNorm1dScaling(6, affine=False, track_running_stats=False)(x)
    scale_var = scaled.var(dim=(0, 2), unbiased=False)
    assert torch.allclose(scale_var, torch.ones_like(scale_var), atol=1e-4, rtol=1e-4)


def test_sequence_bn1d_supports_higher_dimensional_inputs():
    x_last = torch.randn(2, 5, 3, 7)
    y_last = SequenceBatchNorm1dCentering(5, affine=False, track_running_stats=False, layout="last")(x_last)
    mean_last = y_last.mean(dim=(0, 2, 3))
    assert torch.allclose(mean_last, torch.zeros_like(mean_last), atol=1e-6, rtol=1e-6)

    x_first = torch.randn(2, 3, 7, 5)
    y_first = SequenceBatchNorm1dScaling(5, affine=False, track_running_stats=False, layout="first")(x_first)
    var_first = y_first.var(dim=(0, 1, 2), unbiased=False)
    assert torch.allclose(var_first, torch.ones_like(var_first), atol=1e-4, rtol=1e-4)


def test_sequence_bn_combined_module_matches_center_then_scale():
    x = torch.randn(3, 6, 8)
    module = SequenceBatchNorm1d(6, affine=False, track_running_stats=False)
    manual = SequenceBatchNorm1dScaling(6, affine=False, track_running_stats=False)(
        SequenceBatchNorm1dCentering(6, affine=False, track_running_stats=False)(x)
    )
    y = module(x)
    assert torch.allclose(y, manual, atol=1e-6, rtol=1e-6)


def test_dynamic_sequence_bn_reuses_same_module_for_different_lengths():
    module = DynamicSequenceBatchNorm1d(layout="last")
    x1 = torch.randn(2, 5, 8)
    x2 = torch.randn(2, 9, 8)
    y1 = module(x1)
    y2 = module(x2)
    assert y1.shape == x1.shape
    assert y2.shape == x2.shape


def test_dynamic_sequence_bn_centering_and_scaling_normalize_sequence_axis():
    x = torch.randn(4, 6, 8)
    centered = DynamicSequenceBatchNorm1dCentering(layout="last")(x)
    center_mean = centered.mean(dim=(0, 2))
    assert torch.allclose(center_mean, torch.zeros_like(center_mean), atol=1e-6, rtol=1e-6)

    scaled = DynamicSequenceBatchNorm1dScaling(layout="last")(x)
    scale_var = scaled.var(dim=(0, 2), unbiased=False)
    assert torch.allclose(scale_var, torch.ones_like(scale_var), atol=1e-4, rtol=1e-4)


def test_bn2d_variants_keep_running_stats_flat_after_training():
    modules = [BatchNorm2dCentering(4), BatchNorm2dScaling(4), BatchNorm2dScalingRMS(4)]

    for module in modules:
        _ = module(torch.randn(2, 4, 3, 3))
        module.eval()
        y = module(torch.randn(2, 4, 3, 3))
        assert y.shape == (2, 4, 3, 3)
        if hasattr(module, "running_mean") and module.running_mean is not None:
            assert module.running_mean.shape == (4,)
        if hasattr(module, "running_var") and module.running_var is not None:
            assert module.running_var.shape == (4,)
        if hasattr(module, "running_norm") and module.running_norm is not None:
            assert module.running_norm.shape == (4,)


def test_group_norm_scaling_supports_affine_bias():
    module = GroupNormScaling(2, 8, affine=True, bias=True)
    y = module(torch.randn(2, 8, 4))
    assert y.shape == (2, 8, 4)


def test_pointwise_group_norm_variants_support_nchw_affine():
    x = torch.randn(2, 6, 4, 2)
    modules = [
        PointwiseGroupNorm(2, 6, affine=True),
        PointwiseGroupNormCentering(2, 6, affine=True),
        PointwiseGroupNormScaling(2, 6, affine=True),
        PointwiseGroupNormScalingRMS(2, 6, affine=True),
    ]

    for module in modules:
        y = module(x)
        assert y.shape == x.shape


def test_parallel_ln_supports_common_layouts():
    cases = [
        torch.randn(2, 5, 8),
        torch.randn(2, 8, 5),
        torch.randn(2, 8, 4, 4),
        torch.randn(2, 4, 4, 8),
    ]

    module = ParallelLN(8, num_per_group=4)
    for x in cases:
        y = module(x)
        assert y.shape == x.shape


def test_parallel_ln_factories_work_for_vit_layout():
    x = torch.randn(2, 5, 8)
    for factory in (_ParallelLayerNorm, _ParallelLayerScaling):
        y = factory(8, num_per_group=4)(x)
        assert y.shape == x.shape


def test_pq_norm_supports_common_layouts():
    cases = [
        torch.randn(2, 5, 8),
        torch.randn(2, 8, 5),
        torch.randn(2, 8, 4, 4),
        torch.randn(2, 4, 4, 8),
    ]

    module = PQNorm(8, p=4, q=2)
    for x in cases:
        y = module(x)
        assert y.shape == x.shape


def test_pq_norm_matches_definition_5_1_constraint():
    x = torch.randn(4, 6)
    y = PQNorm(6, p=4, q=2, centering=False, affine=False)(x)
    q_moment = torch.mean(torch.abs(y).pow(2), dim=-1)
    assert torch.allclose(q_moment, torch.ones_like(q_moment), atol=1e-4, rtol=1e-4)


def test_grouped_pq_norm_matches_parallel_ln_scaling_when_p_equals_q_equals_2():
    x = torch.randn(3, 8, 5)
    pqn = PQNorm(8, num_per_group=4, p=2, q=2, centering=False, affine=False)
    pln = ParallelLN(8, num_per_group=4, centering=False, affine=False)
    y_pqn = pqn(x)
    y_pln = pln(x)
    assert torch.allclose(y_pqn, y_pln, atol=1e-6, rtol=1e-6)


def test_centered_grouped_pq_norm_matches_parallel_ln_when_p_equals_q_equals_2():
    x = torch.randn(3, 8, 5)
    pqn = PQNorm(8, num_per_group=4, p=2, q=2, centering=True, affine=False)
    pln = ParallelLN(8, num_per_group=4, centering=True, affine=False)
    y_pqn = pqn(x)
    y_pln = pln(x)
    assert torch.allclose(y_pqn, y_pln, atol=1e-6, rtol=1e-6)


def test_pq_activation_matches_sinarctan_when_p_equals_q_equals_2():
    x = torch.randn(4, 7)
    y_pq = PQActivation(p=2, q=2)(x)
    y_sat = SinArctan(num_features=7)(x)
    assert torch.allclose(y_pq, y_sat, atol=1e-6, rtol=1e-6)


def test_mlp_activation_supports_common_shapes_and_has_trainable_parameters():
    module = MLPActivation(hidden_dim=8, act="tanh")
    x2 = torch.randn(4, 7, requires_grad=True)
    x3 = torch.randn(2, 5, 8, requires_grad=True)

    y2 = module(x2)
    y3 = module(x3)

    assert y2.shape == x2.shape
    assert y3.shape == x3.shape

    (y2.sum() + y3.sum()).backward()
    assert module.fc1.weight.grad is not None
    assert module.fc2.weight.grad is not None


def test_activation_factory_builds_mlpact_and_formats_flag():
    activation._config.activation = "mlpact"
    activation._config.activation_cfg = {"n": 11, "act": "gelu"}

    module = activation.Activation(32)
    assert isinstance(module, MLPActivation)
    assert module.hidden_dim == 11
    assert module.act_name == "gelu"
    assert activation.getActivationConfigFlag() == "mlpact_H11_Agelu"


def test_norm_factory_supports_mlp_2d_for_bn_and_gn():
    x = torch.randn(2, 8)
    cases = [
        ("BN", {"dim": 2}),
        ("BNc", {"dim": 2}),
        ("BNs", {"dim": 2}),
        ("GN", {"dim": 2, "num_groups": 4}),
        ("GNc", {"dim": 2, "num_groups": 4}),
        ("GNs", {"dim": 2, "num_groups": 4}),
        ("PQN", {"dim": 2, "p": 4, "q": 2}),
        ("PQN", {"dim": 2, "num_per_group": 4, "p": 2, "q": 2}),
        ("PQN", {"dim": 2, "num_per_group": 4, "p": 2, "q": 2, "centering": True}),
    ]

    for norm_name, kwargs in cases:
        normalization._config.norm = norm_name
        normalization._config.norm_cfg = {}
        y = normalization.Norm(8, **kwargs)(x)
        assert y.shape == x.shape


def test_instance_norm_rejects_dim_2_inputs():
    normalization._config.norm = "IN"
    normalization._config.norm_cfg = {}
    try:
        normalization.Norm(8, dim=2)
    except ValueError as exc:
        assert "dim=2" in str(exc)
    else:
        raise AssertionError("Expected ValueError for dim=2 InstanceNorm")


def test_norm_factory_supports_vit_token_last_for_bn_in_gn():
    x = torch.randn(2, 5, 8)
    cases = [
        ("BN", {"dim": 3, "layout": "last"}),
        ("BNc", {"dim": 3, "layout": "last"}),
        ("BNs", {"dim": 3, "layout": "last"}),
        ("SeqBN", {"dim": 3, "layout": "last"}),
        ("SeqBNc", {"dim": 3, "layout": "last"}),
        ("SeqBNs", {"dim": 3, "layout": "last"}),
        ("DSeqBN", {"dim": 3, "layout": "last"}),
        ("DSeqBNc", {"dim": 3, "layout": "last"}),
        ("DSeqBNs", {"dim": 3, "layout": "last"}),
        ("IN", {"dim": 3, "layout": "last"}),
        ("GN", {"dim": 3, "layout": "last", "num_groups": 4}),
        ("GNc", {"dim": 3, "layout": "last", "num_groups": 4}),
        ("GNs", {"dim": 3, "layout": "last", "num_groups": 4}),
        ("PQN", {"dim": 3, "layout": "last", "p": 4, "q": 2}),
        ("PQN", {"dim": 3, "layout": "last", "num_per_group": 4, "p": 2, "q": 2}),
        ("PQN", {"dim": 3, "layout": "last", "num_per_group": 4, "p": 2, "q": 2, "centering": True}),
    ]

    for norm_name, kwargs in cases:
        normalization._config.norm = norm_name
        normalization._config.norm_cfg = {}
        if norm_name.startswith("SeqBN"):
            y = normalization.Norm(x.shape[1], **kwargs)(x)
        elif norm_name.startswith("DSeqBN"):
            y = normalization.Norm(**kwargs)(x)
        else:
            y = normalization.Norm(x.shape[2], **kwargs)(x)
        assert y.shape == x.shape


def test_norm_factory_supports_sequence_bn_for_dim_4():
    cases = [
        (torch.randn(2, 5, 7, 3), "last", 5),
        (torch.randn(2, 3, 7, 5), "first", 5),
    ]
    for norm_name in ("SeqBN", "SeqBNc", "SeqBNs"):
        normalization._config.norm = norm_name
        normalization._config.norm_cfg = {}
        for x, layout, num_features in cases:
            y = normalization.Norm(num_features, dim=4, layout=layout)(x)
            assert y.shape == x.shape


def test_norm_factory_supports_dynamic_sequence_bn_for_dim_4():
    cases = [
        (torch.randn(2, 5, 7, 3), "last"),
        (torch.randn(2, 3, 7, 5), "first"),
    ]
    for norm_name in ("DSeqBN", "DSeqBNc", "DSeqBNs"):
        normalization._config.norm = norm_name
        normalization._config.norm_cfg = {}
        for x, layout in cases:
            y = normalization.Norm(dim=4, layout=layout)(x)
            assert y.shape == x.shape


def test_vit_build_vit_norm_layer_binds_sequence_length_for_seqbn():
    cfg = SimpleNamespace(norm="SeqBN", im_size=(32, 32), patch_size=4)
    normalization._config.norm = "SeqBN"
    normalization._config.norm_cfg = {}
    module = build_vit_norm_layer(cfg)(384)
    assert isinstance(module, SequenceBatchNorm1d)
    assert module[0].num_features == 65


def test_vit_build_vit_norm_layer_uses_dynamic_sequence_norm_without_fixed_length():
    cfg = SimpleNamespace(norm="DSeqBN", im_size=(32, 32), patch_size=4)
    normalization._config.norm = "DSeqBN"
    normalization._config.norm_cfg = {}
    module = build_vit_norm_layer(cfg)(384)
    assert isinstance(module, DynamicSequenceBatchNorm1d)

