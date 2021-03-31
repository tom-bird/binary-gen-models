import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import gammaln
import os
import shutil
import math


def transfer_fun(transfer_type):
    if transfer_type is None:
        f = lambda s: s.data
    elif transfer_type == 'clip':
        f = lambda s: s.data.clamp(-1., 1.)
    elif transfer_type == 'v_init':
        # this is to initialise only the v params in wn modules
        # make the init values be ~ N(0,0.05)
        f = lambda s: (s.data - s.mean()) / (20 * s.data.std())
    else:
        raise NotImplementedError
    return f


def transfer_params(source_model, target_model, selection=None, transfer_type=None, **kwargs):
    source_sd = source_model.state_dict()
    target_sd = target_model.state_dict()
    t_fun = transfer_fun(transfer_type)

    for name, p in source_sd.items():
        if selection is None or name in selection:
            target_sd[name] = t_fun(p)
    target_model.load_state_dict(target_sd)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def pack_bits(t):
    """t is a byte tensor with binary values (0, 1).
    Pack bits in the first dimension"""
    assert t.shape[0] % 8 == 0
    n = t.shape[0] // 8
    packed = torch.zeros(n, *t.shape[1:]).byte().to(t.device)
    for i in range(8):
        packed += t[i*n:(i+1)*n] << i
    return packed


def unpack_bits(t):
    """t is a tensor of packed bits. Unpack in the first dim"""
    n = t.shape[0]
    unpacked = torch.zeros(8*n, *t.shape[1:]).to(t.device)
    for i in range(8):
        unpacked[i*n:(i+1)*n] = (t >> i) & 1
    return unpacked


def sumflat(x: torch.Tensor):
    return x.view(x.shape[0], -1).sum(1)


def standard_normal_logp(x):
    return -0.5 * x ** 2 - 0.5 * math.log(math.tau)


def inverse_sigmoid(x: torch.Tensor):
    return torch.log(x) - torch.log1p(-x)
