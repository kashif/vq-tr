import torch
from gluonts.torch.modules.loss import DistributionLoss


class QuantileLoss(DistributionLoss):
    def __call__(
        self, input: torch.distributions.Distribution, target: torch.Tensor
    ) -> torch.Tensor:
        return input.quantile_loss(target)
