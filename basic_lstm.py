import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
from iternorm import IterNorm
import numbers


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

        self.bn_i = IterNorm(4 * hidden_size, num_groups=10, dim=2, T=10)
        self.bn_h = IterNorm(4 * hidden_size, num_groups=10, dim=2, T=10)

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (self.bn_i(torch.mm(input, self.weight_ih.t())) + self.bias_ih +
                 self.bn_h(torch.mm(hx, self.weight_hh.t())) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)
        self.h0 = Parameter(torch.zeros(self.cell.hidden_size).cuda(), requires_grad=True)
        self.c0 = Parameter(torch.zeros(self.cell.hidden_size).cuda(), requires_grad=True)

    def forward(self, input, state=None):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        h, c = (self.h0.repeat(input.shape[1], 1) + self.h0.data.new(input.shape[1], self.cell.hidden_size).normal_(0, 0.10),
                self.c0.repeat(input.shape[1], 1) + self.c0.data.new(input.shape[1], self.cell.hidden_size).normal_(0, 0.10))
        state = (h, c)
        # inputs = input.unbind(0)
        outputs = []
        for i in range(len(input)):
            out, state = self.cell(input[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class mnistModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, args, quantize=False):
        super(mnistModel, self).__init__()
        self.args = args
        self.quantize = quantize
        self.norm = args.norm
        self.rnns = [LSTMLayer(LSTMCell, ninp, nhid)]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, 10)  # there are as a total of 10 digits
        self.init_weights()
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, optimizer, return_h=False):
        _, output = self.rnns[0](input)
        result = output[0]
        result = self.decoder(result)
        return result
