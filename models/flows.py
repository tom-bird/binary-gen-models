import math
import time
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from models.logistic import mixlogistic_logcdf, mixlogistic_logpdf, mixlogistic_invcdf, force_accurate_mixlogistic_invcdf
from models.modules import init_mode, is_init_enabled, LearnedNorm
from utils import sumflat, inverse_sigmoid, standard_normal_logp


class BaseFlow(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_, *, aux, inverse: bool) -> Tuple[Any, Any]:
        """
        Apply the flow, either the forward direction (x -> z) or the inverse direction (z -> x)
        :param input_: input to the flow. Should be Tensor or tuple of Tensors, each has shape (batch_size, ...)
        :param aux: auxiliary data, useful for conditional flows
        :param inverse: True to run the inverse direction; False for forward
        return: tuple (output, logd). output is the output of the flow (batch_size, ...), logd is a vector of shape
            (batch_size,) containing the log determinant of the Jacobian of the flow for each element of the batch
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class ElementwiseFlow(BaseFlow):
    def _forward_elementwise(self, input_, *, aux, inverse: bool):
        raise NotImplementedError

    def forward(self, input_, *, aux, inverse: bool):
        output, logd = self._forward_elementwise(input_, aux=aux, inverse=inverse)
        assert output.shape == logd.shape and output.shape[0] == input_.shape[0]
        return output, sumflat(logd)


class Inverse(BaseFlow):
    def __init__(self, base):
        super().__init__()
        assert isinstance(base, BaseFlow)
        self.base = base

    def forward(self, input_, *, aux, inverse: bool):
        return self.base.forward(input_, aux=aux, inverse=not inverse)

    def __str__(self):
        return f'Inverse({str(self.base)})'


class Compose(BaseFlow):
    def __init__(self, flows, dbgprint=False):
        super().__init__()
        self.flows = ModuleList(flows)
        self.dbgprint = dbgprint

    def forward(self, input_, *, aux, inverse):
        bs = input_.shape[0]
        x = input_
        total_logd = None

        for flow in (reversed(self.flows) if inverse else self.flows):
            tstart = time.time()

            x, logd = flow(x, aux=aux, inverse=inverse)
            if isinstance(x, torch.Tensor):
                assert x.shape[0] == bs
                # print(f'{flow.__class__.__name__} mean {x.mean().item()} std {x.std(unbiased=False).item()}')
            elif isinstance(x, tuple):
                assert all(entry.shape[0] == bs for entry in x)
                # for k, entry in enumerate(x):
                #     print(f'{flow.__class__.__name__} [{k}] mean {entry.mean().item()} std {entry.std(unbiased=False).item()}')
            else:
                assert False

            if logd is not None:
                assert logd.shape == (bs,)
                if total_logd is None:
                    total_logd = torch.zeros_like(logd)
                total_logd += logd

            if self.dbgprint:
                _dbgprint(flow, x, time=time.time() - tstart)

        return x, total_logd


def test_compose():
    _run_flow_test(lambda: Compose([ImgProc(), Sigmoid()]), x_bounds=(0., 256.))


class ImgProc(ElementwiseFlow):
    def _forward_elementwise(self, input_, *, aux, inverse: bool):
        if not inverse:
            x = input_
            if not ((x >= 0).all() and (x <= 256).all()):
                print('WARNING: ImgProc input out of [0, 256] bounds!')
            x = x * (.9 / 256) + .05  # [0, 256] -> [.05, .95]
            x = inverse_sigmoid(x)
            logd = math.log(.9 / 256) + F.softplus(x) + F.softplus(-x)
            return x, logd
        else:
            y = input_
            y = torch.sigmoid(y)
            logd = torch.log(y) + torch.log1p(-y)
            y = (y - .05) / (.9 / 256)  # [.05, .95] -> [0, 256]
            logd -= math.log(.9 / 256)
            return y, logd


def test_imgproc():
    _run_flow_test(ImgProc, x_bounds=(0., 256.))


class Sigmoid(ElementwiseFlow):
    def _forward_elementwise(self, input_, *, aux, inverse: bool):
        if not inverse:
            x = input_
            y = torch.sigmoid(x)
            logd = -F.softplus(x) - F.softplus(-x)
            return y, logd
        else:
            y = input_
            if not ((y >= 0).all() and (y <= 1).all()):
                print('WARNING: Inverse(Sigmoid) input out of [0, 1] bounds!')
            x = inverse_sigmoid(y)
            logd = -torch.log(y) - torch.log1p(-y)
            return x, logd


def test_sigmoid():
    _run_flow_test(Sigmoid)


############## Parameterized flows ##############

def _dbgprint(flow, input_, time=''):
    if is_init_enabled():
        print('{:<40} mean={:7.4f} std={:7.4f} min={:7.4f} max={:7.4f} shape={} t={}'.format(
            str(flow),
            input_.mean(), input_.std(), input_.min(), input_.max(),
            tuple(input_.shape), time
        ))


class Normalize(ElementwiseFlow):
    def __init__(self, shape):
        super().__init__()
        self.normalize = LearnedNorm(shape)

    def _forward_elementwise(self, input_, *, aux, inverse: bool):
        out = self.normalize(input_, inverse=inverse)
        logd = self.normalize.get_gain().log()
        if inverse:
            logd *= -1
        # repeat logd along the batch axis
        assert logd.shape == input_.shape[1:]
        logd = logd[None].repeat(input_.shape[0], *([1] * len(input_.shape[1:])))  # repeat along batch axis
        assert logd.shape == input_.shape
        return out, logd


def test_normalize():
    x_shape = (3, 8, 8)
    _run_flow_test(lambda: Normalize(shape=x_shape), x_shape=x_shape)


class Pointwise(BaseFlow):
    def __init__(self, channels, noisy_identity_init=0.001):
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.eye(channels, channels) + noisy_identity_init * torch.randn(channels, channels)
        )

    @staticmethod
    def _flatten(x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) if isinstance(x, torch.Tensor) else x.transpose(0, 2, 3, 1)
        x = x.reshape(B * H * W, C)
        return x

    @staticmethod
    def _unflatten(y, orig_shape):
        B, C, H, W = orig_shape
        y = y.reshape(B, H, W, C)
        y = y.permute(0, 3, 1, 2) if isinstance(y, torch.Tensor) else y.transpose(0, 3, 1, 2)
        return y

    def forward(self, input_, *, aux, inverse: bool):
        x_bchw = input_
        B, C, H, W = x_bchw.shape
        assert self.w.shape == (C, C)
        xflat = self._flatten(x_bchw)
        yflat = xflat.matmul((torch.inverse(self.w) if inverse else self.w).t())
        y_bchw = self._unflatten(yflat, x_bchw.shape)
        logd = ((-1 if inverse else 1) * H * W * torch.slogdet(self.w.double())[1].type(self.w.dtype)
                * torch.ones(B, dtype=input_.dtype, device=input_.device))
        assert y_bchw.shape == x_bchw.shape and logd.shape == (B,)
        return y_bchw, logd


def test_pointwise():
    x_shape = (3, 8, 8)
    # TODO try better init
    _run_flow_test(lambda: Pointwise(channels=x_shape[0], noisy_identity_init=1.0), x_shape=x_shape, check_logd=False)


############## Data-parameterized flows ##############

class ElementwiseAffine(ElementwiseFlow):
    def __init__(self, *, logscales, translations):
        super().__init__()
        self.logscales = logscales
        self.translations = translations

    def _forward_elementwise(self, input_, *, aux, inverse: bool):
        assert input_.shape == self.logscales.shape == self.translations.shape
        if not inverse:
            return (input_ * torch.exp(self.logscales) + self.translations), self.logscales
        else:
            return ((input_ - self.translations) * torch.exp(-self.logscales)), -self.logscales


class MixLogisticCDF(ElementwiseFlow):
    """
    Elementwise transformation by the CDF of a mixture of logistics
    """

    def __init__(self, *, logits, means, logscales, mix_dim):
        super().__init__()
        self.logits = logits
        self.means = means
        self.logscales = logscales
        self.mix_dim = mix_dim

    def _forward_elementwise(self, input_, *, aux, inverse: bool):
        logistic_kwargs = dict(logits=self.logits, means=self.means, logscales=self.logscales, mix_dim=self.mix_dim)
        if not inverse:
            out = torch.exp(mixlogistic_logcdf(input_, **logistic_kwargs))
            logd = mixlogistic_logpdf(input_, **logistic_kwargs)
        else:
            if not ((input_ >= 0).all() and (input_ <= 1).all()):
                print('WARNING: Inverse(MixLogisticCDF) input out of [0, 1] bounds!')
            out = mixlogistic_invcdf(input_, **logistic_kwargs)
            logd = -mixlogistic_logpdf(out, **logistic_kwargs)
        return out, logd


def test_mix_logistic_cdf():
    mixlogistic_data_shape = (5, 7, 3, 2, 2)  # (bs, mix_components, channels, height, width)
    input_shape = (mixlogistic_data_shape[0], *mixlogistic_data_shape[2:])
    mix_dim = 1
    logits = 0.1 * torch.randn(*mixlogistic_data_shape, dtype=torch.float64)
    means = 0.1 * torch.randn(*mixlogistic_data_shape, dtype=torch.float64)
    logscales = 0.1 * torch.randn(*mixlogistic_data_shape, dtype=torch.float64)
    ctor = lambda: MixLogisticCDF(logits=logits, means=means, logscales=logscales,
                                  mix_dim=mix_dim)  # , inv_bisection_tol=1e-12),
    with force_accurate_mixlogistic_invcdf():
        _run_flow_test(ctor, bs=input_shape[0], x_shape=input_shape[1:])


############## Unit tests ##############


@torch.no_grad()
def _finitediff(m, x, *, inverse, eps, aux):
    """log partial derivatives on the diagonal of the jacobian"""
    assert len(x.shape) == 4
    finitediff_logd = torch.zeros_like(x)
    for b in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                for k in range(x.shape[3]):
                    orig = x[b, i, j, k].item()
                    x[b, i, j, k] = orig + eps
                    z2 = m(x, inverse=inverse, aux=aux)[0][b, i, j, k].item()
                    x[b, i, j, k] = orig - eps
                    z1 = m(x, inverse=inverse, aux=aux)[0][b, i, j, k].item()
                    x[b, i, j, k] = orig
                    finitediff_logd[b, i, j, k] = math.log(z2 - z1) - math.log(2 * eps)
    finitediff_logd = finitediff_logd.view(x.shape[0], -1).sum(1)
    return finitediff_logd


def _make_test_data(bs, x_shape, aux_shape, x_bounds):
    # Make some random input
    if x_bounds is not None:
        assert len(x_bounds) == 2
        x = torch.rand(bs, *x_shape, dtype=torch.float64) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    else:
        x = torch.randn(bs, *x_shape, dtype=torch.float64)

    if aux_shape is None:
        aux_shape = x_shape
    aux = torch.randn(bs, *aux_shape, dtype=torch.float64)

    return x, aux


@torch.no_grad()
def _run_flow_test(m_ctor, *, x_bounds=None, x_shape=(3, 8, 8), aux_shape=None, bs=5, check_logd=True,
                   finitediff_eps=1e-7):
    """
    - Check that a flow and its inverse compose to yield the identity
    - Check that the log det Jacobian is calculated correctly via finite differences
    """
    m = m_ctor().type(torch.float64)
    x, aux = _make_test_data(bs, x_shape, aux_shape, x_bounds)

    # Initialize
    with init_mode():
        m(x, inverse=False, aux=aux)

    # Check inverse
    y, logd = m(x, inverse=False, aux=aux)
    x2, invlogd = m(y, inverse=True, aux=aux)
    assert torch.allclose(x, x2)
    assert x.shape == x2.shape == y.shape
    assert (logd is None) == (invlogd is None)
    if logd is not None:
        assert torch.allclose(logd, -invlogd)
        assert logd.shape == invlogd.shape == (bs,)

    # Check logd by finite differences (only works for triangular jacobians)
    if check_logd:
        finitediff_logd = _finitediff(m, x, eps=finitediff_eps, inverse=False, aux=aux)
        if logd is None:
            assert finitediff_logd.abs().max() < 1e-6
        else:
            assert torch.allclose(logd, finitediff_logd)

        finitediff_invlogd = _finitediff(m, y, eps=finitediff_eps, inverse=True, aux=aux)
        if invlogd is None:
            assert finitediff_invlogd.abs().max() < 1e-6
        else:
            assert (invlogd - finitediff_invlogd).abs().max() < 1e-5
