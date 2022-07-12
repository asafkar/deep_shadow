import torch
from torch import nn
import numpy as np


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    @torch.jit.script  # JIT decorator - element-wise fusion
    def sin_(inp_):
        return torch.sin(30 * inp_)

    def forward(self, inp):
        return self.sin_(inp)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)  # according to paper


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class DepthNerf(torch.nn.Module):
    def __init__(self, num_encoding_functions, sine_activation=True, args=None, out_ch=1, factor=10):
        super(DepthNerf, self).__init__()
        filter_size = args.filter_size
        self.num_encoding = num_encoding_functions
        self.factor = nn.Parameter(torch.ones(1) * factor)
        lin_func = nn.Linear
        act_fn = Sine() if sine_activation else nn.LeakyReLU(0.1)

        dimension_factor = 2
        self.depth = nn.Sequential(
            nn.Linear(dimension_factor * (2 * self.num_encoding + 1), filter_size),
            act_fn,
            lin_func(filter_size, filter_size),
            act_fn,
            lin_func(filter_size, filter_size),
            act_fn,
            lin_func(filter_size, filter_size),
            act_fn,
            lin_func(filter_size, filter_size),
            act_fn,
            nn.Linear(filter_size, out_ch, bias=True),
            act_fn,
        )

        self.depth.apply(sine_init)
        self.depth[0].apply(first_layer_sine_init)

    def forward(self, pos_enc):
        res = self.depth(pos_enc).squeeze() * self.factor.abs()
        return res.abs()
