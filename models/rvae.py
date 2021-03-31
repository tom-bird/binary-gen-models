from functools import partial

import numpy as np

import torch
import torch.nn as nn

from models.modules import BinarisedWnConv2d, WnConv2d, softplus, WnConvTranpose2d, \
    BinarisedWnLayer, Sign, Pass

# potential improvements
# - move resnet layers to same location as in bit-swap
# - make "scale" for latent layers Sigmoid
# - use EMA
# - use dropout
# - smooth lr schedule
from utils import pack_bits, unpack_bits


def discretized_logistic_logpdf(mu, logscale, x):
    scale = softplus(logscale)
    invscale = 1. / scale
    x_centered = x - mu

    plus_in = invscale * (x_centered + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = invscale * (x_centered - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)

    # log-probability for edge case of 0
    log_cdf_plus = plus_in - softplus(plus_in)

    # log-probability for edge case of 255
    log_one_minus_cdf_min = - softplus(min_in)

    # other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = invscale * x_centered

    # log-probability in the center of the bin, to be used in extreme cases
    log_pdf_mid = mid_in - torch.log(scale) - 2. * softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal case, extremely low-probability case
    cond1 = torch.where(cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-12, max=None)),
                        log_pdf_mid - np.log(127.5))
    cond2 = torch.where(x > .999, log_one_minus_cdf_min, cond1)
    logps = torch.where(x < -.999, log_cdf_plus, cond2)

    return logps


class Logistic:
    def __init__(self, mu, scale):
        self.mu = mu
        self.scale = scale

    def sample(self):
        # sample from a Gaussian
        u = torch.rand(self.mu.shape, device=self.mu.device)

        # clamp between two bounds to ensure numerical stability
        u = torch.clamp(u, min=1e-5, max=1 - 1e-5)

        # transform to a sample from the Logistic distribution
        eps = torch.log(u) - torch.log1p(-u)

        # reparam trick
        sample = self.mu + self.scale * eps
        return sample

    def log_prob(self, x):
        _y = (x - self.mu) / self.scale
        logp = _y - torch.log(self.scale) - 2 * softplus(_y)
        return logp


class ResBlock(nn.Module):
    def __init__(self, z_size, h_size, kl_min, nreslayers = 1,
                 binarised=False, **kwargs):
        super().__init__()
        self.z_size = z_size
        self.h_size = h_size
        self.kl_min = kl_min

        conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        z_conv = partial(WnConv2d, **conv_kwargs)
        self.fp_act = nn.ELU()
        self.bin_act = Sign() if binarised else nn.Tanh()

        self.us_conv_z = z_conv(h_size, 2 * z_size)
        self.u_conv_h = NResLayers(h_size, h_size, 3, 1, 1, binarised, nreslayers) if nreslayers > 0 else Pass()

        self.ds_conv_z = z_conv(h_size, 4 * z_size)
        self.dm_conv_z = z_conv(z_size, h_size)
        self.d_conv_h = NResLayers(h_size, h_size, 3, 1, 1, binarised, nreslayers) if nreslayers > 0 else Pass()

        # saving these is memory intensive - can streamline if necessary
        self.prior = None
        self.posterior = None
        self.qz_mean = None
        self.qz_logstd = None

    def up_split(self, inputs):
        out_z = self.us_conv_z(self.fp_act(inputs))
        self.qz_mean, self.qz_logstd = torch.split(out_z, self.z_size, dim=1)

    def up(self, inputs):
        self.up_split(inputs)
        return self.u_conv_h(inputs)

    def down_split(self, inputs, sample=False):
        out_z = self.ds_conv_z(self.fp_act(inputs))
        pz_mean, pz_logstd, rz_mean, rz_logstd = torch.split(out_z, self.z_size, dim=1)
        scale = lambda s: (1e-2 + softplus(s))
        self.posterior = None if sample else Logistic(rz_mean + self.qz_mean, scale(rz_logstd + self.qz_logstd))
        self.prior = Logistic(pz_mean, scale(pz_logstd))

    def down_merge(self, inputs, z):
        z_down = self.dm_conv_z(self.fp_act(z))
        return self.d_conv_h(inputs) + z_down

    def down(self, inputs, sample=False):
        self.down_split(inputs, sample)
        if sample:
            z = self.prior.sample()
            kl = 0
        else:
            z = self.posterior.sample()
            kl = self.posterior.log_prob(z) - self.prior.log_prob(z)
            b = kl.shape[0]
            kl = torch.sum(kl, dim=[2, 3])
            kl = torch.mean(kl, dim=0)
            kl = torch.clamp(kl, min=self.kl_min)  # clamp the batch kl for each channel
            kl = torch.sum(kl) * b  # undo the divide in the mean
        return self.down_merge(inputs, z), kl


# PyTorch module used to build a sequence of ResNet layers
class NResLayers(nn.Sequential):
    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1,
                 binarised=False, nlayers=1):
        super(NResLayers, self).__init__()
        for i in range(nlayers):
            layer = ResLayer(inchannels, outchannels, kernel_size, stride, padding, binarised)
            self.add_module('res{}layer{}'.format(inchannels, i + 1), layer)


class ResLayer(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1,
                 binarised=False):
        super(ResLayer, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.stride = stride
        self.act = Sign() if binarised else nn.Tanh()

        conv = BinarisedWnConv2d if binarised else WnConv2d
        self.res_conv1 = conv(inchannels, outchannels, kernel_size=kernel_size, stride=1,
                              padding=padding, init_scale=1.0, loggain=True)
        self.res_conv2 = conv(outchannels,  outchannels, kernel_size=kernel_size,
                              stride=1, padding=padding, init_scale=0., loggain=False)

    def forward(self, x):
        c1 = self.act(self.res_conv1(self.act(x)))
        c2 = self.res_conv2(c1)
        return x + c2


class RVAE(nn.Module):
    def __init__(self, num_layers, z_size, h_size, kl_min, nproclayers, nreslayers,
                 binarised=False, x_channels=3, kl_min_gamma=None, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.z_size = z_size
        self.h_size = h_size
        self.kl_min = kl_min
        self.nreslayers = nreslayers
        self.nproclayers = nproclayers
        self.kl_min_gamma = kl_min_gamma
        self.binarised = binarised

        self.anneal_kl_min = False
        if isinstance(kl_min, (tuple, list)):
            self.anneal_kl_min = True
            kl_min = kl_min[0]  # start at the upper bound

        self.image_shape = None
        self.x_logsigma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.h_init = nn.Parameter(torch.zeros(h_size), requires_grad=False)

        self.act = nn.ELU()

        # distribute ResNet layers over stochastic layers (blocks)
        nreslayers = [0] * (self.num_layers)
        i = 0
        for _ in range(self.nreslayers):
            i = 0 if i == (self.num_layers) else i
            nreslayers[i] += 1
            i += 1

        self.layers = nn.ModuleList([ResBlock(z_size, h_size, kl_min, nreslayers[i], binarised)
                                     for i in range(num_layers)])
        ds = WnConv2d(x_channels, self.h_size, kernel_size=5, stride=2, padding=2)
        ds_res = NResLayers(self.h_size, self.h_size, kernel_size=5, padding=2,
                            binarised=binarised, nlayers=self.nproclayers) if self.nproclayers > 0 else Pass()
        self.downsample = nn.Sequential(ds, self.act, ds_res)

        us_res = NResLayers(self.h_size, self.h_size, kernel_size=5,
                            padding=2, binarised=binarised, nlayers=self.nproclayers) if self.nproclayers > 0 else Pass()
        us = WnConvTranpose2d(self.h_size, x_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.upsample = nn.Sequential(us_res, self.act, us)
        self.best_loss = np.inf

        self.binary_conv_names = ['res_conv1.v', 'res_conv2.v']
        binary_params = [p for n, p in self.named_parameters() if any(s in n for s in self.binary_conv_names)]
        all_parameters = [p for p in self.parameters() if p.requires_grad]
        num_params = sum([np.prod(p.size()) for p in all_parameters])
        num_binary_params = sum([np.prod(p.size()) for p in binary_params]) if binarised else 0
        num_fp_params = num_params - num_binary_params
        print('{} trainable params'.format(num_params))
        print('{} binary params ({:.2f}%)'.format(num_binary_params, 100 * num_binary_params/num_params))
        print('{} FP params ({:.2f}%)'.format(num_fp_params, 100 * num_fp_params/num_params))

        # transfer all the v params when initialising binary with pre-trained FP
        self.transfer_selection = [n for n, p in self.named_parameters() if n[-2:] == '.v']

    def preprocess(self, x):
        return (x-127.5) / 127.5  # into [-1, 1]

    def up_pass(self, x):
        self.image_shape = x.shape
        u = self.downsample(x)

        for layer in self.layers:
            u = layer.up(u)

    def initial_input_down(self):
        b, _, h, w = self.image_shape
        d_shape = (b, h // 2, w // 2)
        d = self.h_init.repeat(int(np.prod(d_shape)))
        return d.reshape(self.h_size, *d_shape).transpose(0, 1)

    def down_pass(self, sample=False):
        d = self.initial_input_down()
        kl = torch.zeros(len(self.layers), device=d.device)
        for l, layer in enumerate(reversed(self.layers)):
            d, cur_kl = layer.down(d, sample)
            kl[len(self.layers) - 1 - l] = cur_kl

        x_loc = self.upsample_and_postprocess(d)
        return x_loc, kl

    def loss(self, x):
        x = self.preprocess(x)
        self.up_pass(x)
        x_loc, kl = self.down_pass()
        log_pxz = torch.sum(discretized_logistic_logpdf(x_loc, self.x_logsigma, x))
        get_bpd = lambda t: -np.log2(np.e) * t / np.prod(x.shape)
        elbo = get_bpd(log_pxz - kl.sum())
        breakdown = {'log_pxz': get_bpd(log_pxz),
                     'kl': get_bpd(-kl.sum()),
                     }
        kl_list = {'kl' + str(l): get_bpd(-kl[l]) for l in range(len(self.layers))}
        breakdown = {**breakdown, **kl_list}
        return elbo, breakdown

    def upsample_and_postprocess(self, d):
        d = self.act(d)
        x = self.upsample(d)
        return x

    def sample(self, n, device):
        self.image_shape = (n, *self.image_shape[1:])
        with torch.no_grad():
            return self.down_pass(sample=True)[0]

    def reconstruct(self, x):
        with torch.no_grad():
            x = self.preprocess(x)
            self.up_pass(x)
            return self.down_pass()[0]

    def clamp_weights(self):
        for module in self.modules():
            if isinstance(module, BinarisedWnLayer):
                module.clamp_weights()

    def post_epoch(self):
        if self.anneal_kl_min:
            for block in self.layers:
                block.kl_min = max(self.kl_min[1], block.kl_min * self.kl_min_gamma)

    def get_latent_shape(self, x_shape):
        return x_shape[0], self.z_size, x_shape[2] // 2, x_shape[3] // 2

    def binary_state_dict(self, **kwargs):
        state_dict = super().state_dict(**kwargs)
        for name, p in state_dict.items():
            if any(s in name for s in self.binary_conv_names):
                b = torch.sign(p)
                b = ((b + 1) / 2).byte()  # to {0, 1}
                state_dict[name] = pack_bits(b)
        return state_dict

    def load_binary_state_dict(self, state_dict, **kwargs):
        for name, p in state_dict.items():
            if any(s in name for s in self.binary_conv_names):
                b = unpack_bits(p)
                b = 2*b - 1  # to {-1, 1}
                state_dict[name] = b
        super().load_state_dict(state_dict, **kwargs)
