from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

_SMALL = 1e-10

_INIT_ENABLED = False


def is_init_enabled():
    return _INIT_ENABLED


@contextmanager
def init_mode():
    global _INIT_ENABLED
    assert not _INIT_ENABLED
    _INIT_ENABLED = True
    yield
    _INIT_ENABLED = False


def softplus(x):
    return -F.logsigmoid(-x)

# PyTorch module that is used to only pass through values
class Pass(nn.Module):
    def __init__(self):
        super(Pass, self).__init__()

    def forward(self, x, **kwargs):
        return x

    def inverse(self, x, **kwargs):
        return x

### Binary layers
class Binarise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        """Pass gradients through"""
        return grad_output
binarise = Binarise.apply


class Sign(nn.Module):
    def forward(self, input):
        return binarise(torch.clamp(input, -1, 1))


class BinarisedLayer:
    def clamp_weights(self):
        params = self.state_dict()
        for p in params:
            params[p] = torch.clamp(params[p], -1, 1)
        self.load_state_dict(params)


class BinarisedWnLayer:
    def clamp_weights(self):
        params = self.state_dict()
        for p in params:
            if (not p == 'gain') and (not p == 'b'):
                params[p] = torch.clamp(params[p], -1, 1)
        self.load_state_dict(params)


# PyTorch module that applies Data Dependent Initialization + Weight Normalization
class WnModule(nn.Module):
    """
    Module with data-dependent initialization
    """

    def __init__(self):
        super().__init__()
        self.q = None

    def _init(self, *args, **kwargs):
        """
        Data-dependent initialization. Will be called on the first forward()
        """
        raise NotImplementedError

    def _forward(self, *args, **kwargs):
        """
        The standard forward pass
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Calls _init (with no_grad) if not initialized.
        If initialized already, calls _forward.
        """
        if _INIT_ENABLED:
            with torch.no_grad():  # no gradients for the init pass
                return self._init(*args, **kwargs)
        return self._forward(*args, **kwargs)


class WnLinear(WnModule):
    def __init__(self, in_features, out_features, init_scale=1.0, init_stdv=0.05,
                 loggain=True, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.init_scale = init_scale
        self.loggain = loggain
        self.v = nn.Parameter(torch.Tensor(out_features, in_features))
        self.gain = nn.Parameter(torch.Tensor(out_features))
        self.b = nn.Parameter(torch.Tensor(out_features), requires_grad=True if self.bias else False)

        nn.init.normal_(self.v, 0., init_stdv)
        if self.loggain:
            nn.init.zeros_(self.gain)
        else:
            nn.init.ones_(self.gain)
        nn.init.zeros_(self.b)

    def _init(self, x):
        # calculate unnormalized activations
        y = self._forward(x)

        # set g and b so that activations are normalized
        m = y.mean(dim=0)
        s = self.init_scale / (y.std(dim=0) + _SMALL)
        assert m.shape == s.shape == self.gain.shape == self.b.shape

        if self.loggain:
            loggain = torch.clamp(torch.log(s), min=-10., max=None)
            self.gain.data.copy_(loggain)
        else:
            self.gain.data.copy_(s)

        if self.bias:
            self.b.data.sub_(m * s)

        # forward pass again, now normalized
        return self._forward(x)

    def _forward(self, x):
        if self.loggain:
            g = softplus(self.gain)
        else:
            g = self.gain
        vnorm = self.v.norm(p=2, dim=1)
        assert vnorm.shape == self.gain.shape == self.b.shape
        w = self.v * (g / (vnorm + _SMALL)).view(self.out_features, 1)
        return F.linear(x, w, self.b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, init_scale={}, loggain={}'.format(
            self.in_features, self.out_features, self.init_scale, self.loggain
        )


class BinarisedWnLinear(WnLinear, BinarisedWnLayer):
    def _forward(self, x):
        if self.loggain:
            g = softplus(self.gain)
        else:
            g = self.gain
        vnorm = np.sqrt(self.in_features)
        # Note we can scale after the op for fast binary implementation
        w = binarise(self.v) * (g / vnorm).view(self.out_features, 1)
        return F.linear(x, w, self.b)


# Data-Dependent Initialization + Weight Normalization extension of a "Conv2D" module of PyTorch
class WnConv2d(WnModule):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0,
                 init_scale=0.1, init_stdv=0.05, loggain=True, bias=True):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.bias = bias
        self.init_scale = init_scale
        self.init_stdv = init_stdv
        self.loggain = loggain

        self.v = nn.Parameter(torch.Tensor(out_dim, in_dim, *self.kernel_size))
        self.gain = nn.Parameter(torch.Tensor(out_dim))
        self.b = nn.Parameter(torch.Tensor(out_dim), requires_grad=True if self.bias else False)

        nn.init.normal_(self.v, 0., init_stdv)
        if self.loggain:
            nn.init.zeros_(self.gain)
        else:
            nn.init.ones_(self.gain)
        nn.init.zeros_(self.b)

    def _init(self, x):
        # calculate unnormalized activations
        y_bchw = self._forward(x)
        assert len(y_bchw.shape) == 4 and y_bchw.shape[:2] == (x.shape[0], self.out_dim)

        # set g and b so that activations are normalized
        y_c = y_bchw.transpose(0, 1).reshape(self.out_dim, -1)
        m = y_c.mean(dim=1)
        s = self.init_scale / (y_c.std(dim=1) + _SMALL)
        assert m.shape == s.shape == self.gain.shape == self.b.shape

        if self.loggain:
            loggain = torch.clamp(torch.log(s), min=-10., max=None)
            self.gain.data.copy_(loggain)
        else:
            self.gain.data.copy_(s)

        if self.bias:
            self.b.data.sub_(m * s)

        # forward pass again, now normalized
        return self._forward(x)

    def _forward(self, x):
        if self.loggain:
            g = softplus(self.gain)
        else:
            g = self.gain
        vnorm = self.v.view(self.out_dim, -1).norm(p=2, dim=1)
        assert vnorm.shape == self.gain.shape == self.b.shape
        w = self.v * (g / (vnorm + _SMALL)).view(self.out_dim, 1, 1, 1)
        return F.conv2d(x, w, self.b, stride=self.stride, padding=self.padding)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}, kernel_size={}, stride={}, padding={}, init_scale={}, loggain={}'\
            .format(self.in_dim, self.out_dim, self.kernel_size,
                    self.stride, self.padding, self.init_scale, self.loggain)


class WnConvTranpose2d(WnConv2d):
    def __init__(self, *args, **kwargs):
        if 'output_padding' in kwargs:
            self.output_padding = _pair(kwargs['output_padding'])
            del kwargs['output_padding']
        super().__init__(*args, **kwargs)

    def _forward(self, x):
        if self.loggain:
            g = softplus(self.gain)
        else:
            g = self.gain
        vnorm = self.v.view(self.out_dim, -1).norm(p=2, dim=1)
        assert vnorm.shape == self.gain.shape == self.b.shape
        w = self.v * (g / (vnorm + _SMALL)).view(self.out_dim, 1, 1, 1)
        return F.conv_transpose2d(x, w.transpose(0, 1), self.b, self.stride, self.padding, self.output_padding)


class BinarisedWnConv2d(WnConv2d, BinarisedWnLayer):
    def _forward(self, x):
        if self.loggain:
            g = softplus(self.gain)
        else:
            g = self.gain
        vnorm = np.sqrt(self.in_dim * np.prod(self.kernel_size))  # all dims but out dim
        w = binarise(self.v) * (g / vnorm).view(self.out_dim, 1, 1, 1)
        # Note we can scale after the op for fast binary implementation
        return F.conv2d(x, w, self.b, stride=self.stride, padding=self.padding)


class BinarisedWnConvTranpose2d(BinarisedWnConv2d):
    def __init__(self, *args, **kwargs):
        if 'output_padding' in kwargs:
            self.output_padding = _pair(kwargs['output_padding'])
            del kwargs['output_padding']
        super().__init__(*args, **kwargs)

    def _forward(self, x):
        if self.loggain:
            g = softplus(self.gain)
        else:
            g = self.gain
        vnorm = np.sqrt(self.in_dim * np.prod(self.kernel_size))  # all dims but out dim
        w = binarise(self.v) * (g / vnorm).view(self.out_dim, 1, 1, 1)
        # Note we can scale after the op for fast binary implementation
        return F.conv_transpose2d(x, w.transpose(0, 1), self.b, self.stride, self.padding, self.output_padding)


class Linear(WnModule):
    def __init__(self, in_features, out_features, init_scale=1.0, init_stdv=0.05):
        super().__init__()
        self.in_features, self.out_features, self.init_scale = in_features, out_features, init_scale

        self.w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b = nn.Parameter(torch.Tensor(out_features))

        nn.init.normal_(self.w, 0, init_stdv)
        nn.init.zeros_(self.b)

    def _init(self, x):
        y = self._forward(x)
        m = y.mean(dim=0)
        s = self.init_scale / (y.std(dim=0) + _SMALL)
        assert m.shape == s.shape == self.b.shape
        self.w.copy_(self.w * s[:, None])
        self.b.copy_(-m * s)
        return self._forward(x)

    def _forward(self, x):
        return F.linear(x, self.w, self.b[None, :])


class LearnedNorm(WnModule):
    def __init__(self, shape, init_scale=1.0):
        super().__init__()
        self.init_scale = init_scale
        self.g = nn.Parameter(torch.ones(*shape))
        self.b = nn.Parameter(torch.zeros(*shape))

    def _init(self, x, *, inverse):
        assert not inverse
        assert x.shape[1:] == self.g.shape == self.b.shape
        m_init = x.mean(dim=0)
        scale_init = self.init_scale / (x.std(dim=0) + _SMALL)
        self.g.copy_(scale_init)
        self.b.copy_(-m_init * scale_init)
        return self._forward(x, inverse=inverse)

    def get_gain(self):
        return torch.clamp(self.g, min=1e-10)

    def _forward(self, x, *, inverse):
        """
        inverse == False to normalize; inverse == True to unnormalize
        """
        assert x.shape[1:] == self.g.shape == self.b.shape
        assert x.dtype == self.g.dtype == self.b.dtype
        g = self.get_gain()
        if not inverse:
            return x * g[None] + self.b[None]
        else:
            return (x - self.b[None]) / g[None]


class _Nin(WnModule):
    def __init__(self, in_features, out_features, wn: bool, init_scale: float, binarised=False):
        super().__init__()
        if binarised and wn:
            # need wn with binarised
            base_module = BinarisedWnLinear
        else:
            base_module = WnLinear if wn else Linear
        self.dense = base_module(in_features=in_features, out_features=out_features, init_scale=init_scale)
        self.height, self.width = None, None

    def _preprocess(self, x):
        """(b,c,h,w) -> (b*h*w,c)"""
        B, C, H, W = x.shape
        if self.height is None or self.width is None:
            self.height, self.width = H, W
        else:
            assert self.height == H and self.width == W, 'nin input image shape changed!'
        assert C == self.dense.in_features
        return x.permute(0, 2, 3, 1).reshape(B * H * W, C)

    def _postprocess(self, x):
        """(b*h*w,c) -> (b,c,h,w)"""
        BHW, C = x.shape
        out = x.reshape(-1, self.height, self.width, C).permute(0, 3, 1, 2)
        assert out.shape[1:] == (self.dense.out_features, self.height, self.width)
        return out

    def _init(self, x):
        return self._postprocess(self.dense._init(self._preprocess(x)))

    def _forward(self, x):
        return self._postprocess(self.dense._forward(self._preprocess(x)))


class Nin(_Nin):
    def __init__(self, in_features, out_features, init_scale=1.0,):
        super().__init__(in_features=in_features, out_features=out_features, wn=False, init_scale=init_scale)


class WnNin(_Nin):
    def __init__(self, in_features, out_features, init_scale=1.0):
        super().__init__(in_features=in_features, out_features=out_features, wn=True, init_scale=init_scale)


class BinarisedWnNin(_Nin, BinarisedWnLayer):
    def __init__(self, in_features, out_features, init_scale=1.0):
        super().__init__(in_features=in_features, out_features=out_features, wn=True, init_scale=init_scale,
                         binarised=True)
