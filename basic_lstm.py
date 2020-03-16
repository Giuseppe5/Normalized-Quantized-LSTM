import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Tuple
from torch import Tensor
from batchrenorm import BatchRenorm1d

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        temp = torch.empty(4*hidden_size, input_size)
        torch.nn.init.orthogonal_(temp)
        self.weight_ih = Parameter(temp, requires_grad=True)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(4, 1)
        self.weight_hh = Parameter(weight_hh_data, requires_grad=True)

        self.bias_ih = Parameter(torch.zeros(4 * hidden_size), requires_grad=True)
        self.bias_hh = Parameter(torch.zeros(4 * hidden_size), requires_grad=True)

        self.bn_i = BatchRenorm1d(4 * hidden_size)
        self.bn_h = BatchRenorm1d(4 * hidden_size)
        self.bn_c = BatchRenorm1d(hidden_size)

    def forward(self, input, state, last):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (self.bn_i(torch.mm(input, self.weight_ih.t()), last) + self.bias_ih +
                 self.bn_h(torch.mm(hx, self.weight_hh.t()), last) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(self.bn_c(cy, last))

        return hy, (hy, cy)


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)
        self.h0 = Parameter(torch.zeros(self.cell.hidden_size).cuda(), requires_grad=True)
        self.c0 = Parameter(torch.zeros(self.cell.hidden_size).cuda(), requires_grad=True)

    def forward(self, input, state=None):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        # h, c = 0.1*torch.randn((input.shape[1], self.cell.hidden_size)), 0.1*torch.randn((input.shape[1], self.cell.hidden_size))
        h, c = (
        self.h0.repeat(input.shape[1], 1) + self.h0.data.new(input.shape[1], self.cell.hidden_size).normal_(0, 0.10),
        self.c0.repeat(input.shape[1], 1) + self.c0.data.new(input.shape[1], self.cell.hidden_size).normal_(0, 0.10))
        state = (h, c)
        # inputs = input.unbind(0)
        outputs = []
        for i in range(len(input)):
            last = i == len(input)-1
            out, state = self.cell(input[i], state, last)
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
