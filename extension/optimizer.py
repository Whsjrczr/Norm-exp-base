import argparse
import ast
import re
import torch
from .utils import str2dict, str2num
from .logger import get_logger

_methods = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'adamax': torch.optim.Adamax,
    'RMSprop': torch.optim.RMSprop,
    'lbfgs': torch.optim.LBFGS,
}

_STAGE_KEYS = {
    'optimizer', 'name', 'lr', 'weight_decay',
    'epochs', 'epoch', 'end_epoch', 'iterations', 'iters',
    'batch_size', 'loss_weights', 'metrics',
    'optimizer_config', 'config',
    'lr_method', 'lr_step', 'lr_gamma',
}


def _parse_stage_dict(spec: str) -> dict:
    tokens = [token.strip() for token in spec.split(',') if token.strip()]
    stage = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if '=' not in token:
            raise ValueError(f"Invalid stage token: {token}")

        key, value = token.split('=', 1)
        key = key.strip()
        value = value.strip()

        if key in {'optimizer_config', 'config'}:
            nested_tokens = [value] if value else []
            idx += 1
            while idx < len(tokens):
                next_token = tokens[idx]
                if '=' not in next_token:
                    nested_tokens.append(next_token)
                    idx += 1
                    continue
                next_key, _ = next_token.split('=', 1)
                if next_key.strip() in _STAGE_KEYS:
                    break
                nested_tokens.append(next_token)
                idx += 1
            stage[key] = str2dict(','.join(nested_tokens))
            continue

        stage[key] = str2num(value)
        idx += 1

    return stage



def _str2stages(x):
    """Parse optimizer stages from CLI (simple version).

    Supported:
      - Python literal / JSON list-of-dicts (recommended for complex configs)
      - Simple string with ';' separated stages:
          "adam:lr=3e-4,epochs=50; sgd:lr=1e-2,epochs=150,momentum=0.9"
        You may also omit ':' and write:
          "optimizer=adam,lr=3e-4,epochs=50; optimizer=sgd,lr=1e-2,epochs=150"
        Or treat the first token as optimizer:
          "adam,lr=3e-4,epochs=50; sgd,lr=1e-2,epochs=150,momentum=0.9"
      - Read from file: "@path/to/stages.txt"

    Notes:
      - key/value parsing uses utils.str2dict, same as --optimizer-config.
      - optimizer_config can itself be a str2dict string, e.g.:
          "lbfgs:optimizer_config=maxiter=5000,maxcor=50"
    """
    if x is None:
        return []
    if not isinstance(x, str):
        return list(x)

    s = x.strip()
    if s == '':
        return []

    # Load from file
    if s.startswith('@'):
        with open(s[1:].strip(), 'r', encoding='utf-8') as f:
            s = f.read().strip()
        if s == '':
            return []

    # Try Python literal / JSON-like
    try:
        obj = ast.literal_eval(s)
    except Exception:
        obj = None

    if obj is not None:
        if isinstance(obj, dict) and 'stages' in obj:
            obj = obj['stages']
        if isinstance(obj, dict):
            obj = [obj]
        if isinstance(obj, list):
            stages = []
            for st in obj:
                if st is None:
                    continue
                stage = dict(st)
                if 'optimizer_config' in stage and isinstance(stage['optimizer_config'], str):
                    stage['optimizer_config'] = str2dict(stage['optimizer_config'])
                stages.append(stage)
            return stages

    # Simple ';' separated stages
    parts = [p.strip() for p in s.split(';') if p.strip()]
    stages = []
    for part in parts:
        stage = {}

        # Form: "adam:lr=...,epochs=..."
        if ':' in part:
            opt, rest = part.split(':', 1)
            opt = opt.strip()
            rest = rest.strip()
            if rest:
                stage.update(_parse_stage_dict(rest))
            stage.setdefault('optimizer', opt)

        else:
            # Form: "adam,lr=...,epochs=..." (first token is optimizer)
            toks = [t.strip() for t in part.split(',') if t.strip()]
            if toks and ('=' not in toks[0]):
                stage.setdefault('optimizer', toks[0])
                rest = ','.join(toks[1:])
                if rest:
                    stage.update(_parse_stage_dict(rest))
            else:
                # Form: "optimizer=adam,lr=...,epochs=..."
                stage.update(_parse_stage_dict(part))

        if 'optimizer_config' in stage and isinstance(stage['optimizer_config'], str):
            stage['optimizer_config'] = str2dict(stage['optimizer_config'])

        if stage:
            stages.append(stage)

    return stages


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Optimizer Option:')
    group.add_argument(
        '-oo', '--optimizer', default='sgd', choices=_methods.keys(),
        help='the optimizer method to train network {' + ', '.join(_methods.keys()) + '}'
    )
    group.add_argument(
        '-oc', '--optimizer-config', default={}, type=str2dict, metavar='DICT',
        help='The configure for optimizer (single-stage).'
    )
    group.add_argument(
        '-os', '--optimizer-stages', default=None, type=_str2stages, metavar='STAGES',
        help=('Multi-stage optimizer schedule as a python literal list of dicts. '
              'Each stage dict can contain: optimizer, lr, weight_decay, optimizer_config, '
              'and a duration key (epochs for epoch-based training; iterations for DeepXDE). '
              'Example: "[{\'optimizer\':\'adam\',\'lr\':1e-3,\'epochs\':50},'
              '{\'optimizer\':\'sgd\',\'lr\':1e-2,\'momentum\':0.9,\'epochs\':150}]"')
    )
    group.add_argument(
        '-wd', '--weight-decay', default=0, type=float, metavar='FLOAT',
        help='weight decay (default: 0).'
    )
    return



def get_stages(cfg: argparse.Namespace):
    """Return list[dict] stage specs, or None if single-stage."""
    stages = getattr(cfg, 'optimizer_stages', None)
    if stages:
        return stages
    # Backward-compatible: allow putting stages inside optimizer_config
    oc = getattr(cfg, 'optimizer_config', None)
    if isinstance(oc, dict) and 'stages' in oc:
        st = oc.get('stages')
        if isinstance(st, list) and all(isinstance(d, dict) for d in st):
            return st
    return None


def infer_total_epochs(stages):
    """Infer total epoch budget from stage specs.

    Returns None when any stage lacks an explicit epoch boundary.
    """
    if not stages:
        return None

    total = 0
    for stage in stages:
        if not isinstance(stage, dict):
            return None
        if 'end_epoch' in stage:
            total = int(stage['end_epoch'])
        elif 'epochs' in stage:
            total += int(stage['epochs'])
        elif 'epoch' in stage:
            total += int(stage['epoch'])
        else:
            return None
    return total


def infer_total_iterations(stages):
    """Infer total iteration budget from stage specs for DeepXDE/PDE training.

    Returns None when any stage lacks an explicit iteration count.
    """
    if not stages:
        return None

    total = 0
    for stage in stages:
        if not isinstance(stage, dict):
            return None
        if 'iterations' in stage:
            total += int(stage['iterations'])
        elif 'iters' in stage:
            total += int(stage['iters'])
        elif 'epochs' in stage:
            total += int(stage['epochs'])
        else:
            return None
    return total


def build_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    lr: float | None,
    weight_decay: float,
    optimizer_config: dict | None = None,
    **kwargs,
):
    """Build a torch optimizer with consistent defaults."""
    name = optimizer_name.lower()
    if name not in _methods:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    cfg_dict = {}
    if optimizer_config:
        cfg_dict.update(optimizer_config)

    # Common defaults
    if name == 'sgd':
        cfg_dict.setdefault('momentum', 0.9)

    if lr is not None:
        cfg_dict['lr'] = lr
    cfg_dict['weight_decay'] = weight_decay
    cfg_dict.update(kwargs)

    params = model.parameters()
    opt = _methods[name](params, **cfg_dict)
    logger = get_logger()
    logger('==> Optimizer {}'.format(opt))
    return opt


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

def setting(model: torch.nn.Module, cfg: argparse.Namespace, **kwargs):
    """Single-stage optimizer creation (kept for backward compatibility).

    If you pass --optimizer-stages, callers should handle stage switching themselves.
    Here we simply create the optimizer described by cfg.optimizer/cfg.lr/cfg.optimizer_config.
    """
    lr = getattr(cfg, 'lr', None)
    optimizer_config = getattr(cfg, 'optimizer_config', {}) or {}
    weight_decay = getattr(cfg, 'weight_decay', 0.0)

    # Preserve previous behavior: allow kwargs override + cfg.optimizer_config merge
    merged_cfg = {}
    merged_cfg.update(optimizer_config)
    merged_cfg.update(kwargs)

    return build_optimizer(
        model=model,
        optimizer_name=getattr(cfg, 'optimizer', 'sgd'),
        lr=lr,
        weight_decay=weight_decay,
        optimizer_config=merged_cfg,
    )
