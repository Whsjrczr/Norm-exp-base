import argparse
from functools import partial
import torch.nn as nn

from .my_modules.norm.ln_modules import *
from .my_modules.norm.bn1d_modules import *
from .my_modules.norm.bn2d_modules import *
from .my_modules.norm.gn_modules import *
from .my_modules.norm.pln import ParallelLN
from .my_modules.norm.pq_norm import PQNorm

from .utils import str2dict


class _LayoutAdapter(nn.Module):
    def __init__(self, module, dim, layout):
        super().__init__()
        self.module = module
        self.dim = dim
        self.layout = layout

    def forward(self, x):
        if self.dim == 2:
            return self.module(x.unsqueeze(-1)).squeeze(-1)

        if self.dim == 3 and self.layout == "last":
            return self.module(x.transpose(1, 2)).transpose(1, 2)

        if self.dim == 4 and self.layout == "last":
            return self.module(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return self.module(x)


def _normalize_layout(dim, layout):
    if dim not in (2, 3, 4):
        raise ValueError(f"Unsupported norm dim: {dim}. Expected one of 2, 3, 4.")

    if layout is None:
        if dim == 2:
            return "last"
        return "first"

    if layout not in ("first", "last"):
        raise ValueError(f"Unsupported norm layout: {layout}. Expected 'first' or 'last'.")
    return layout


def _wrap_layout(module, dim, layout):
    if dim == 2:
        return _LayoutAdapter(module, dim=2, layout="last")
    if layout == "last":
        return _LayoutAdapter(module, dim=dim, layout=layout)
    return module


def make_norm_factory(**bound_kwargs):
    return partial(Norm, **bound_kwargs)



# GN
def _GroupNorm(num_features, num_groups=32, eps=1e-5, affine=True, dim=4, layout=None, *args, **kwargs):
    layout = _normalize_layout(dim, layout)
    module = nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine)
    return _wrap_layout(module, dim=dim, layout=layout)

def _GroupNormCentering(num_features, num_groups=32, eps=1e-5, affine=True, dim=4, layout=None, *args, **kwargs):
    layout = _normalize_layout(dim, layout)
    module = GroupNormCentering(num_groups, num_features, affine=affine)
    return _wrap_layout(module, dim=dim, layout=layout)

def _GroupNormScaling(num_features, num_groups=32, eps=1e-5, affine=True, dim=4, layout=None, *args, **kwargs):
    layout = _normalize_layout(dim, layout)
    module = GroupNormScaling(num_groups, num_features, eps=eps, affine=affine)
    return _wrap_layout(module, dim=dim, layout=layout)

# LN
def _LayerNorm(normalized_shape, eps=1e-5, affine=True, *args, **kwargs):
    return nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)

def _LayerNormCentering(normalized_shape, eps=1e-5, affine=True, *args, **kwargs):
    return LayerNormCentering(normalized_shape, elementwise_affine=affine)

def _LayerNormScaling(normalized_shape, eps=1e-5, affine=True, *args, **kwargs):
    return LayerNormScaling(normalized_shape, eps=eps, elementwise_affine=affine)

def _RMSNorm(normalized_shape, eps=1e-5, affine=True, *args, **kwargs):
    return LayerNormScalingRMS(normalized_shape, eps=eps, elementwise_affine=affine)


def _CenteringDropoutScaling(normalized_shape,dropout_prob=0.0, eps=1e-05, affine=True, *args, **kwargs):
    return nn.Sequential(
        LayerNormCentering(normalized_shape, elementwise_affine=False),
        nn.Dropout(p=dropout_prob),
        LayerNormScaling(normalized_shape, elementwise_affine=affine,bias=affine, eps=eps)
    )

def _bCenteringDropoutScaling(num_features,dropout_prob=0.0, eps=1e-05, affine=True, *args, **kwargs):
    return nn.Sequential(
        BatchNorm1dCentering(num_features, affine=False),
        nn.Dropout(p=dropout_prob),
        LayerNormScaling(num_features, elementwise_affine=affine,bias=affine, eps=eps)
    )

def _bCenlCenDropScaling(num_features,dropout_prob=0.0, eps=1e-05, affine=True, *args, **kwargs):
    return nn.Sequential(
        BatchNorm1dCentering(num_features, affine=False),
        LayerNormCentering(num_features, elementwise_affine=False),
        nn.Dropout(p=dropout_prob),
        LayerNormScaling(num_features, elementwise_affine=affine,bias=affine, eps=eps)
    )

def _bCLayerNorm(num_features, eps=1e-5, affine=True, *args, **kwargs):
    return nn.Sequential(
        BatchNorm1dCentering(num_features, affine=False),
        nn.LayerNorm(num_features, eps=eps, elementwise_affine=affine)
    )

def _bCRMSNorm(num_features, eps=1e-5, affine=True, *args, **kwargs):
    return nn.Sequential(
        BatchNorm1dCentering(num_features, affine=False),
        LayerNormScalingRMS(num_features, eps=eps, elementwise_affine=affine)
    )


# BN
def _BatchNorm(num_features, dim=4, layout=None, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, *args, **kwargs):
    layout = _normalize_layout(dim, layout)
    module_cls = nn.BatchNorm2d if dim == 4 else nn.BatchNorm1d
    module = module_cls(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    return _wrap_layout(module, dim=dim, layout=layout)

def _BatchNormCentering(num_features, dim=4, layout=None, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, *args, **kwargs):
    layout = _normalize_layout(dim, layout)
    module_cls = BatchNorm2dCentering if dim == 4 else BatchNorm1dCentering
    module = module_cls(num_features, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    return _wrap_layout(module, dim=dim, layout=layout)

def _BatchNormScaling(num_features, dim=4, layout=None, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, *args, **kwargs):
    layout = _normalize_layout(dim, layout)
    module_cls = BatchNorm2dScaling if dim == 4 else BatchNorm1dScaling
    module = module_cls(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    return _wrap_layout(module, dim=dim, layout=layout)

# IN
def _InstanceNorm(num_features, dim=4, layout=None, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, *args,
                  **kwargs):
    layout = _normalize_layout(dim, layout)
    if dim == 2:
        raise ValueError("InstanceNorm does not support dim=2 inputs without a spatial axis. Use LN/RMS/BN/GN instead.")
    module_cls = nn.InstanceNorm2d if dim == 4 else nn.InstanceNorm1d
    module = module_cls(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    return _wrap_layout(module, dim=dim, layout=layout)


def _Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, *args, **kwargs):
    """return first input"""
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

'''def _IdentityModule(x, *args, **kwargs):
    """return first input"""
    return IdentityModule()'''

def _IdentityModule(x, *args, **kwargs):
    return nn.Identity(x, *args, **kwargs)


def _Identity_fn(x, *args, **kwargs):
    """return first input"""
    return x

def _ParallelLayerNorm(num_features, num_per_group=8, eps=1e-5, centering=True, dim=4, layout=None, *args, **kwargs):
    layout = _normalize_layout(dim, layout)
    return ParallelLN(num_features, num_per_group=num_per_group, eps=eps, centering=centering, dim=dim, layout=layout, *args, **kwargs)

def _ParallelLayerScaling(num_features, num_per_group=8, eps=1e-5, centering=False, dim=4, layout=None, *args, **kwargs):
    layout = _normalize_layout(dim, layout)
    return ParallelLN(num_features, num_per_group=num_per_group, eps=eps, centering=centering, dim=dim, layout=layout, *args, **kwargs)

def _PQNorm(num_features, num_per_group=None, p=2, q=2, eps=1e-5, centering=True, affine=True, dim=4, layout=None, *args, **kwargs):
    layout = _normalize_layout(dim, layout)
    module = PQNorm(
        num_features,
        num_per_group=num_per_group,
        p=p,
        q=q,
        eps=eps,
        centering=centering,
        affine=affine,
        dim=dim,
        *args,
        **kwargs,
    )
    return _wrap_layout(module, dim=dim, layout=layout)



class _config:
    norm = 'BN'
    norm_cfg = {}
    norm_methods = {'BN': _BatchNorm,
                    'GN': _GroupNorm,
                    'LN': _LayerNorm,
                    'IN': _InstanceNorm,
                    'LNc': _LayerNormCentering,
                    'LNs': _LayerNormScaling,
                    'RMS': _RMSNorm,
                    'CDS': _CenteringDropoutScaling,
                    'BNc': _BatchNormCentering,
                    'BNs': _BatchNormScaling,
                    'bCDS':_bCenteringDropoutScaling,
                    'bClCDS':_bCenlCenDropScaling,
                    'bCLN': _bCLayerNorm,
                    'bCRMS':_bCRMSNorm,
                    'GNc': _GroupNormCentering,
                    'GNs': _GroupNormScaling,
                    "PLN": _ParallelLayerNorm, #partial(ParallelLN, centering=True, norm_method="default"),
                    "PLS": _ParallelLayerScaling,#partial(ParallelLN, centering=False, norm_method="default"),
                    "PQN": _PQNorm,
                    'No': _IdentityModule,
                    'no': _IdentityModule,}  # 'No': _LayerNorm, 


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Normalization Options')
    group.add_argument('--norm', default='No', help='Use which normalization layers? {' + ', '.join(
        _config.norm_methods.keys()) + '}' + ' (defalut: {})'.format(_config.norm))
    group.add_argument('--norm-cfg', type=str2dict, default={}, metavar='DICT', help='layers config.')
    return group

def getNormConfigFlag():
    flag = ''
    flag += _config.norm
    if str.find(_config.norm, 'GW')>-1 or str.find(_config.norm, 'GN') > -1:
        if _config.norm_cfg.get('num_groups') != None:
            flag += '_NG' + str(_config.norm_cfg.get('num_groups'))
    if str.find(_config.norm,'ItN') > -1:
        if _config.norm_cfg.get('T') != None:
            flag += '_T' + str(_config.norm_cfg.get('T'))
        if _config.norm_cfg.get('num_channels') != None:
            flag += '_NC' + str(_config.norm_cfg.get('num_channels'))

    if str.find(_config.norm,'DBN') > -1:
        flag += '_NC' + str(_config.norm_cfg.get('num_channels'))
    if _config.norm_cfg.get('affine') == False:
        flag += '_NoA'
    if _config.norm_cfg.get('momentum') != None:
        flag += '_MM' + str(_config.norm_cfg.get('momentum'))
    #print(_config.normConv_cfg)
    if str.find(_config.norm, "PL") > -1:
        if _config.norm_cfg.get("num_per_group") != None:
            flag += str(_config.norm_cfg.get("num_per_group"))
        else:
            flag += "8"
        if _config.norm_cfg.get("affine") == False:
            flag += "_NoAF"
        if _config.norm_cfg.get("norm_p") != None:
            flag += "P" + str(_config.norm_cfg.get("norm_p"))

    if _config.norm == "PQN":
        if _config.norm_cfg.get("num_per_group") is not None:
            flag += "_NPG" + str(_config.norm_cfg.get("num_per_group"))
        if _config.norm_cfg.get("p") is not None:
            flag += "_P" + str(_config.norm_cfg.get("p"))
        if _config.norm_cfg.get("q") is not None:
            flag += "_Q" + str(_config.norm_cfg.get("q"))
        if _config.norm_cfg.get("centering") == True:
            flag += "_Ctr"


    return flag

def setting(cfg: argparse.Namespace):
    # print(_config.__dict__)
    for key, value in vars(cfg).items():
        #print(key)
        #print(value)
        if key in _config.__dict__:
            setattr(_config, key, value)
    # print(_config.__dict__)
    flagName = getNormConfigFlag()
    # print(flagName)
    return flagName


def Norm(*args, **kwargs):
    kwargs.update(_config.norm_cfg)
    if _config.norm == 'None':
        return None
    return _config.norm_methods[_config.norm](*args, **kwargs)
