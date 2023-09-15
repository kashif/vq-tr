import pytorch_lightning as pl
import torch
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average

from .adan import Adan
from .module import VQTrModel


class VQTrLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: VQTrModel,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 0.02,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx: int):
        """Execute training step"""
        train_loss, train_perplexity = self(batch)
        self.log(
            "train_loss",
            train_loss.item(),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train_perplexity",
            train_perplexity,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        """Execute validation step"""
        with torch.inference_mode():
            val_loss, _ = self(batch)
        self.log(
            "val_loss", val_loss.item(), on_epoch=True, on_step=False, prog_bar=True
        )
        return val_loss

    def configure_optimizers(self):
        """Returns the optimizer to use"""
        # return torch.optim.Adam(
        return Adan(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            # capturable=False,
        )

    def forward(self, batch):
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]

        transformer_inputs, scale, _ = self.model.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )
        params, vq_loss, vq_perplexity = self.model.output_params(transformer_inputs)

        distr = self.model.output_distribution(params, scale)

        loss_values = self.loss(distr, future_target)

        if len(self.model.target_shape) == 0:
            loss_weights = future_observed_values
        else:
            loss_weights = future_observed_values.min(dim=-1, keepdim=False)

        return (
            weighted_average(loss_values, weights=loss_weights) + vq_loss,
            vq_perplexity,
        )
