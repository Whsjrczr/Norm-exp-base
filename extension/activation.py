import argparse
import torch
from .utils import str2dict

class _config:
    activation = 'relu'
    activation_cfg = {}
    _methods = {'relu': torch.nn.ReLU(False), 'relu_inplace': torch.nn.ReLU(True), 'sigmoid': torch.nn.Sigmoid, 'tanh': torch.nn.Tanh,'no': torch.nn.Identity}

# _methods = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'adamax': torch.optim.Adamax,
#             'RMSprop': torch.optim.RMSprop}


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Activation Option:')
    group.add_argument('--activation', default='relu', choices=_config._methods.keys(),
                       help='the activation method to train network {' + ', '.join(_config._methods.keys()) + '}')
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



def setting(cfg: argparse.Namespace, **kwargs):
    for key, value in vars(cfg).items():
        if key in _config.__dict__:
            setattr(_config, key, value)
    return _config.activation

def Activation():
    return _config._methods[_config.activation]

