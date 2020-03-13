import torch
import torch.nn as nn
from torch.nn import Parameter


class DBN_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        sizes = x.size()
        x = x.t()
        mean = x.mean(1, keepdim=True)
        # self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
        x_mean = x - mean
        sigma = x_mean.matmul(x_mean.t()) / x.size(1) + 1e-5 * torch.eye(x.size(0), device=x.device)
        # print('sigma size {}'.format(sigma.size()))
        d, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        u = (scale.diag()).matmul(d.t())
        wm = d.matmul(u)
        # wm = u.matmul(scale.diag()).matmul(u.t())
        # self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm
        y = wm.matmul(x_mean)
        ctx.save_for_backward(eig, u.matmul(x_mean), d)
        return y, wm

    @staticmethod
    def backward(ctx, grad_y, grad_wm):
        eig, x_tilde, d = ctx.saved_tensors

        dl_dx = grad_y.t().matmul(d)
        f = dl_dx.t().mean(1, keepdim=True)
        K = eig.view(-1, 1) - eig
        K = K.reciprocal()
        K[K!=K] = 0
        eig_m = eig.diag()
        F_c = (dl_dx.t().matmul(x_tilde.t())).mean(1)
        M = F_c.diag().diag()
        sqrt_eig_m = eig_m.sqrt()
        intermidiate_mul = (sqrt_eig_m.matmul(F_c)).matmul(sqrt_eig_m)
        S = 2 * (K.t() * (eig_m.matmul(F_c.t() + intermidiate_mul)))
        final = dl_dx - f + x_tilde.t().matmul(S) - x_tilde.t().matmul(M)
        final = (final.matmul(sqrt_eig_m)).matmul((d.t()))

        return final, None


class DBN(nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=0, dim=2, eps=1e-5, momentum=0.1, affine=True, mode=0,
                 *args, **kwargs):
        super(DBN, self).__init__()
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim
        self.shape[0] = num_features

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, 1))
        self.register_buffer('running_projection', torch.eye(num_groups))
        self.reset_parameters()

    # def reset_running_stats(self):
    #     self.running_mean.zero_()
    #     self.running_var.eye_(1)

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        size = input.size()
        # assert input.dim() == self.dim and size[1] == self.num_features
        x = input
        # # x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        # x = input.view(size[0] * size[1], 1)
        training = self.mode > 0 or (self.mode == 0 and self.training)
        # x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        if training:
            mean = x.mean(0, keepdim=True)
            # self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            x_mean = x - mean
            sigma = x_mean.t().matmul(x_mean) / x.size(0) + 1e-3 * torch.eye(x.size(1), device=x.device)
            d, eig, _ = sigma.svd()
            scale = eig.rsqrt()
            u = (scale.diag()).matmul(d.t())
            wm = d.matmul(u)
            y = x_mean.matmul(wm.t())
            # y = y.t()
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm.detach()
            # y = wm.matmul(x_mean)
        else:
            x_mean = x - self.running_mean
            y = self.running_projection.matmul(x_mean)
        # output = y.view(1, size[0] * size[1]).transpose(0,1)
        # output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        # output = output.contiguous().view_as(input)
        if self.affine:
            output = y * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)


class DBN2(DBN):
    """
    when evaluation phase, sigma using running average.
    """

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        mean = x.mean(1, keepdim=True) if training else self.running_mean
        x_mean = x - mean
        if training:
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * sigma
        else:
            sigma = self.running_projection
        u, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        wm = u.matmul(scale.diag()).matmul(u.t())
        y = wm.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output
