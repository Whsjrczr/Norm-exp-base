import argparse
import torch
import torch.nn as nn
from .utils import str2dict
from .my_modules.pgn_modules import PointwiseGroupNorm

def _ReLU(num_features, inplace=False, *args, **kwargs):
    return nn.ReLU(inplace=inplace)

def _sigmoid(num_features,*args, **kwargs):
    return nn.Sigmoid()

def _tanh(num_features, *args, **kwargs):
    return nn.Tanh()

def _GroupNorm(num_feature, num_groups=32, eps=1e-5, affine=True, *args, **kwargs):
    return nn.GroupNorm(num_groups, num_feature, eps=eps, affine=affine)

def _IdentityModule(x, *args, **kwargs):
    return nn.Identity(x, *args, **kwargs)

def _PointwiseGroupNorm(num_feature, num_groups=32, eps=1e-5, affine=True, *args, **kwargs):
    return PointwiseGroupNorm(num_groups, num_feature, eps=eps, affine=affine)

class _config:
    activation = 'relu'
    activation_cfg = {}
    _methods = {'relu': _ReLU, 'sigmoid': _sigmoid, 'tanh': _tanh,'gn': _GroupNorm,'pgn': _PointwiseGroupNorm,'no': torch.nn.Identity}

def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Activation Option:')
    group.add_argument('--activation', default='relu', choices=_config._methods.keys(),
                       help='the activation method to train network {' + ', '.join(_config._methods.keys()) + '}')
    group.add_argument('--activation-cfg', type=str2dict, default={}, metavar='DICT', help='norm config.')
    return group


'''
def add_grouped_weight_decay(model, weight_decay=1e-4):
    decay = []
    no_decay = []
    print(model.named_parameters())
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.find('WNScale') != -1:
            print('-----------WNScale no weight decay----------------')
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.}]
'''
def getActivationConfigFlag():
    flag = ''
    flag += _config.activation
    if str.find(_config.activation, 'gn')>-1:
        if _config.activation_cfg.get('num_groups') != None:
            flag += '_NG' + str(_config.activation_cfg.get('num_groups'))
    if str.find(_config.activation, 'relu')>-1:
        if _config.activation_cfg.get('inplace')==True:
            flag += '_InP'
    return flag


def setting(cfg: argparse.Namespace, **kwargs):
    for key, value in vars(cfg).items():
        if key in _config.__dict__:
            setattr(_config, key, value)
    flagname = getActivationConfigFlag()
    return flagname

def Activation(*args, **kwargs):
    kwargs.update(_config.activation_cfg)
    return _config._methods[_config.activation](*args, **kwargs)
