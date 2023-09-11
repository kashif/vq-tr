from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops import rearrange, repeat
from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.feature import FeatureEmbedder
from vector_quantize_pytorch import VectorQuantize
from gluonts.torch.scaler import MeanScaler, NOPScaler


def FeedForward(dim, hidden_dim, dropout=0.0):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim, bias=False),
    )


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, dropout=0.0):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class VQAttention(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        dim_feedforward=16,
        dim_head=16,
        heads=2,
        max_heads_process=2,
        dropout=0.0,
        cross_attn_dropout=0.0,
        depth=1,
        decay=0.8,
        commitment_weight=1.0,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.max_heads_process = max_heads_process

        inner_dim = heads * dim_head
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # they drop out a percentage of the prefix during training, shown to help prevent overfitting
        self.cross_attn_dropout = cross_attn_dropout

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.vq = VectorQuantize(
            dim=dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight,
            threshold_ema_dead_code=2,
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim, dim_head=dim_head, heads=heads, dropout=dropout
                        ),
                        FeedForward(dim, hidden_dim=dim_feedforward, dropout=dropout),
                    ]
                )
            )

    def forward(self, context_input, context_mask=None):
        batch, context_len, device = (
            context_input.shape[0],
            context_input.shape[-2],
            context_input.device,
        )

        # take care of cross attention dropout
        if self.training and self.cross_attn_dropout > 0.0:
            rand = torch.zeros((batch, context_len), device=device).uniform_()
            keep_context_len = context_len - int(context_len * self.cross_attn_dropout)
            keep_indices = rand.topk(keep_context_len, dim=-1).indices
            keep_mask = torch.zeros_like(rand).scatter_(1, keep_indices, 1).bool()

            context_input = rearrange(
                context_input[keep_mask], "(b n) d -> b n d", b=batch
            )

            if context_mask is not None:
                context_mask = rearrange(
                    context_mask[keep_mask], "(b n) -> b n", b=batch
                )

        _, indices, commit_loss, perplexity = self.vq(context_input)

        x = repeat(self.vq.codebook, "m d -> b m d", b=batch)  # [B, M, D]

        # normalization
        x = self.norm(x)
        context = self.context_norm(context_input)

        # derive queries, keys, values
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        q = q * self.scale

        # take care of masking
        i, j = q.shape[-2], k.shape[-2]
        mask_value = -torch.finfo(q.dtype).max

        if context_mask is not None:
            mask_len = context_mask.shape[-1]
            context_mask = F.pad(context_mask, (0, max(j - mask_len, 0)), value=True)
            context_mask = rearrange(context_mask, "b j -> b 1 1 j")

        # process in chunks of heads
        out = []
        max_heads = self.max_heads_process
        for q_chunk, k_chunk, v_chunk in zip(
            q.split(max_heads, dim=1),
            k.split(max_heads, dim=1),
            v.split(max_heads, dim=1),
        ):
            sim = einsum("b h i d, b h j d -> b h i j", q_chunk, k_chunk)

            if context_mask is not None:
                sim = sim.masked_fill(~context_mask, mask_value)

            attn = sim.softmax(dim=-1)
            attn = self.dropout(attn)

            out_chunk = einsum("b h i j, b h j d -> b h i d", attn, v_chunk)
            out.append(out_chunk)

        # concat all the heads together
        out = torch.cat(out, dim=1)

        # merge heads and then combine with linear
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        # self-attention on latents
        for attn, ff in self.layers:
            out = attn(out) + out
            out = ff(out) + out

        expanded_indices = indices.unsqueeze(-1).expand(*indices.shape, self.dim)
        outputs = torch.gather(out, 1, expanded_indices)

        return outputs, commit_loss, perplexity


class VQTrModel(nn.Module):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int,
        num_feat_static_real: int,
        num_feat_static_cat: int,
        cardinality: List[int],
        # VQ arguments
        codebook_size: int,
        decay: float,
        commitment_weight: float,
        # transformer arguments
        nhead: int,
        depth: int,
        dim_head: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        # univariate input
        input_size: int = 1,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        freq: Optional[str] = None,
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()

        assert (freq is not None) or (
            lags_seq is not None
        ), "either freq or lags_seq must be given"

        self.input_size = input_size

        self.target_shape = distr_output.event_shape
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )

        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)

        self.num_parallel_samples = num_parallel_samples
        self.history_length = context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        # total feature size
        d_model = self.input_size * len(self.lags_seq) + self._number_of_features

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(d_model)

        # transformer enc-decoder and mask initializer
        self.encoder = nn.ModuleList([])
        for _ in range(num_encoder_layers):
            self.encoder.append(
                VQAttention(
                    codebook_size=codebook_size,
                    decay=decay,
                    commitment_weight=commitment_weight,
                    dim=d_model,
                    depth=depth,
                    heads=nhead,
                    dim_head=dim_head,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    cross_attn_dropout=dropout,
                ),
            )

        decoder_norm = nn.LayerNorm(d_model, eps=1e-5)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )

        # causal decoder tgt mask
        self.register_buffer(
            "tgt_mask",
            nn.Transformer.generate_square_subsequent_mask(prediction_length),
        )

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + 1  # the log(scale)
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        subsequences_length : int
            length of the subsequences to be extracted.
        shift: int
            shift the lags by this amount back.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        sequence_length = sequence.shape[1]
        indices = [l - shift for l in self.lags_seq]

        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def _check_shapes(
        self,
        prior_input: torch.Tensor,
        inputs: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> None:
        assert len(prior_input.shape) == len(inputs.shape)
        assert (
            len(prior_input.shape) == 2 and self.input_size == 1
        ) or prior_input.shape[2] == self.input_size
        assert (len(inputs.shape) == 2 and self.input_size == 1) or inputs.shape[
            -1
        ] == self.input_size
        assert (
            features is None or features.shape[2] == self._number_of_features
        ), f"{features.shape[2]}, expected {self._number_of_features}"

    def create_network_inputs(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ):
        # time feature
        time_feat = (
            torch.cat(
                (
                    past_time_feat[:, self._past_length - self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            if future_target is not None
            else past_time_feat[:, self._past_length - self.context_length :, ...]
        )

        # target
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        _, _, scale = self.scaler(context, observed_context)

        inputs = (
            torch.cat((past_target, future_target), dim=1) / scale
            if future_target is not None
            else past_target / scale
        )

        inputs_length = (
            self._past_length + self.prediction_length
            if future_target is not None
            else self._past_length
        )
        assert inputs.shape[1] == inputs_length

        subsequences_length = (
            self.context_length + self.prediction_length
            if future_target is not None
            else self.context_length
        )

        # embeddings
        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, scale.log()),
            dim=1,
        )
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, time_feat.shape[1], -1
        )

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        # self._check_shapes(prior_input, inputs, features)

        # sequence = torch.cat((prior_input, inputs), dim=1)
        lagged_sequence = self.get_lagged_subsequences(
            sequence=inputs,
            subsequences_length=subsequences_length,
        )

        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(
            lags_shape[0], lags_shape[1], -1
        )

        transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)

        return transformer_inputs, scale, static_feat

    def output_params(self, transformer_inputs):
        x = transformer_inputs[:, : self.context_length, ...]
        dec_input = transformer_inputs[:, self.context_length :, ...]

        vq_loss = 0
        vq_perplexity = 0
        for vq_attn in self.encoder:
            x, loss, perplexity = vq_attn(x)
            vq_loss = vq_loss + loss
            vq_perplexity = vq_perplexity + perplexity

        dec_output = self.decoder(dec_input, x, tgt_mask=self.tgt_mask)

        return self.param_proj(dec_output), vq_loss, vq_perplexity

    @torch.jit.ignore
    def output_distribution(
        self, params, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, scale=scale)

    # for prediction
    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:

        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        x, scale, static_feat = self.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
        )

        for vq_attn in self.encoder:
            x, _, _ = vq_attn(x)
        enc_out = x

        repeated_scale = scale.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_past_target = (
            past_target.repeat_interleave(repeats=self.num_parallel_samples, dim=0)
            / repeated_scale
        )

        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, future_time_feat.shape[1], -1
        )
        features = torch.cat((expanded_static_feat, future_time_feat), dim=-1)
        repeated_features = features.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        repeated_enc_out = enc_out.repeat_interleave(
            repeats=self.num_parallel_samples, dim=0
        )

        future_samples = []

        # greedy decoding
        for k in range(self.prediction_length):
            # self._check_shapes(repeated_past_target, next_sample, next_features)
            # sequence = torch.cat((repeated_past_target, next_sample), dim=1)

            lagged_sequence = self.get_lagged_subsequences(
                sequence=repeated_past_target,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(
                lags_shape[0], lags_shape[1], -1
            )

            decoder_input = torch.cat(
                (reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1
            )

            output = self.decoder(decoder_input, repeated_enc_out)

            params = self.param_proj(output[:, -1:])
            distr = self.output_distribution(params, scale=repeated_scale)
            next_sample = distr.sample()

            repeated_past_target = torch.cat(
                (repeated_past_target, next_sample / repeated_scale), dim=1
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)
        return concat_future_samples.reshape(
            (-1, self.num_parallel_samples, self.prediction_length) + self.target_shape,
        )
