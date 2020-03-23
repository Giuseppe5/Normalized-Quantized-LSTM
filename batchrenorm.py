import torch


__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]


class BatchRenorm(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_var", torch.ones(num_features, dtype=torch.float)
        )

        self.register_buffer(
            "running_mean_global", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_var_global", torch.ones(num_features, dtype=torch.float)
        )

        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        self.weight = torch.nn.Parameter(
            0.1 * torch.ones(num_features, dtype=torch.float), requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float), requires_grad=True
        )
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        return (3 / 500 * self.num_batches_tracked).clamp_(
            1.0, 3.0
        )

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 500 * self.num_batches_tracked - 1).clamp_(
            0.0, 5.0
        )

    def forward(self, x: torch.Tensor, last: bool) -> torch.Tensor:
        batchsize, channels = x.size()
        numel = batchsize

        x = x.permute(1, 0).contiguous().view(channels, numel)
        sum_ = x.sum(1)
        sum_of_square = x.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean
        unbias_var = sumvar / (numel - 1)

        bias_var = sumvar / numel
        inv_std = 1 / (bias_var + self.eps).pow(0.5)
        if self.training:
            self.running_mean = self.running_mean*(1-self.momentum) + self.momentum * mean.detach()
            self.running_var = self.running_var * (1-self.momentum) + self.momentum * unbias_var.detach()
            self.num_batches_tracked += 1
            if last:
                self.running_mean_global = self.running_mean_global*(1-self.momentum) + self.momentum * self.running_mean.detach()
                self.running_var_global = self.running_var_global*(1-self.momentum) + self.momentum * self.running_var.detach()

            r = (inv_std.detach() / self.running_var.view_as(inv_std)).clamp_(1 / self.rmax, self.rmax)
            d = ((mean.detach() - self.running_mean.view_as(mean)) / self.running_var.view_as(inv_std)).clamp_(-self.dmax, self.dmax)
            x = (x - mean.unsqueeze(1)) * inv_std.unsqueeze(1)* r.unsqueeze(1) + d.unsqueeze(1)

        else:
            inv_std = 1/(self.running_var_global+self.eps).pow(0.5)
            x = (x - self.running_mean_global.unsqueeze(1)) * inv_std.unsqueeze(1)

        x = self.weight.unsqueeze(1) * x + self.bias.unsqueeze(1)

        return x.view(channels, batchsize).permute(1, 0).contiguous()


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")