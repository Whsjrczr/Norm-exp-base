import torch

import extension.normalization as normalization
from extension.my_modules.norm.bn1d_modules import (
    BatchNorm1dCentering,
    BatchNorm1dScaling,
    BatchNorm1dScalingRMS,
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
from extension.my_modules.norm.pln import ParallelLN
from extension.my_modules.activation.pq_activation import PQActivation
from extension.my_modules.norm.pq_norm import PQNorm
from extension.my_modules.activation.sinarctan import SinArctan
from extension.normalization import _ParallelLayerNorm, _ParallelLayerScaling


def test_bn1d_variants_support_channel_first_and_channel_last():
    inputs = [torch.randn(2, 384, 5), torch.randn(2, 5, 384)]
    modules = [BatchNorm1dCentering(384), BatchNorm1dScaling(384), BatchNorm1dScalingRMS(384)]

    for x in inputs:
        for module in modules:
            y = module(x)
            assert y.shape == x.shape


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
    y = PQNorm(6, p=4, q=2, affine=False)(x)
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
        y = normalization.Norm(8, **kwargs)(x)
        assert y.shape == x.shape
