from functools import partial

# +
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Beta

from gluonts.core.component import validated
from gluonts.torch.distributions import DistributionOutput
from gluonts.torch.modules.lambda_layer import LambdaLayer


class QuantileLayer(nn.Module):
    def __init__(self, num_output, cos_embedding_dim=128):
        super().__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(cos_embedding_dim, cos_embedding_dim),
            nn.PReLU(),
            nn.Linear(cos_embedding_dim, num_output),
        )

        self.register_buffer("integers", torch.arange(0, cos_embedding_dim))

    def forward(self, tau):  # tau: [B, T]
        cos_emb_tau = torch.cos(tau.unsqueeze(-1) * self.integers * torch.pi)
        return self.output_layer(cos_emb_tau)


class ImplicitQuantileModule(nn.Module):
    def __init__(
        self,
        in_features,
        args_dim,
        domain_map,
        concentration1=1.0,
        concentration0=1.0,
        output_domain_map=None,
        cos_embedding_dim=64,
    ):
        super().__init__()
        self.output_domain_map = output_domain_map
        self.domain_map = domain_map

        self.beta = Beta(concentration1=concentration1, concentration0=concentration0)

        self.quantile_layer = QuantileLayer(
            in_features, cos_embedding_dim=cos_embedding_dim
        )
        self.output_layer = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.PReLU(),
        )

        self.proj = nn.ModuleList(
            [nn.Linear(in_features, dim) for dim in args_dim.values()]
        )

    def forward(self, inputs):
        if self.training:
            taus = self.beta.sample(sample_shape=inputs.shape[:-1]).to(inputs.device)
        else:
            taus = torch.rand(size=inputs.shape[:-1], device=inputs.device)

        emb_taus = self.quantile_layer(taus)
        emb_inputs = inputs * (1.0 + emb_taus)

        emb_outputs = self.output_layer(emb_inputs)
        outputs = [proj(emb_outputs).squeeze(-1) for proj in self.proj]
        if self.output_domain_map is not None:
            outputs = [self.output_domain_map(output) for output in outputs]
        return (*self.domain_map(*outputs), taus)


class ImplicitQuantile(Distribution):
    arg_constraints = {}

    def __init__(self, outputs, taus, validate_args=None):
        self.taus = taus
        self.outputs = outputs

        super().__init__(
            batch_shape=outputs.shape,
            validate_args=validate_args,
        )

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        return self.outputs

    def quantile_loss(self, value):
        return torch.abs(
            (self.outputs - value) * ((value <= self.outputs).float() - self.taus)
        )


class ImplicitQuantileNetworkOutput(DistributionOutput):
    distr_cls = ImplicitQuantile
    args_dim = {"quantile_function": 1}

    @validated()
    def __init__(
        self, output_domain: str = None, concentration1=1.0, concentration0=1.0
    ) -> None:
        super().__init__()

        self.concentration1 = concentration1
        self.concentration0 = concentration0

        if output_domain in ["Positive", "Unit"]:
            output_domain_map_func = {
                "Positive": F.softplus,
                "Unit": partial(F.softmax, dim=-1),
            }
            self.output_domain_map = output_domain_map_func[output_domain]
        else:
            self.output_domain_map = None

    def get_args_proj(self, in_features: int) -> nn.Module:
        return ImplicitQuantileModule(
            in_features=in_features,
            args_dim=self.args_dim,
            output_domain_map=self.output_domain_map,
            domain_map=LambdaLayer(self.domain_map),
            concentration1=self.concentration1,
            concentration0=self.concentration0,
        )

    @classmethod
    def domain_map(cls, *args):
        return args

    def distribution(self, distr_args, loc=0, scale=None):
        (outputs, taus) = distr_args

        if scale is None:
            return self.distr_cls(
                outputs=outputs,
                taus=taus,
            )
        else:
            return self.distr_cls(
                outputs=loc + outputs * scale,
                taus=taus,
            )

    @property
    def event_shape(self):
        return ()
