from __future__ import annotations

import torch
from torch import nn

from .cnn1d import SharedHRRPEncoder1D


class DeepSetsClassifier(nn.Module):
    architecture_id = "deep_sets_mean_v1"

    def __init__(self, known_class_count: int = 7) -> None:
        super().__init__()
        if known_class_count != 7:
            raise ValueError("the frozen first-round classifier has seven known classes")
        self.encoder = SharedHRRPEncoder1D()
        self.classifier = nn.Linear(self.encoder.feature_dim, known_class_count)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.encoder.input_length:
            raise ValueError("Deep Sets input must have shape [batch, views, 601]")
        batch, views, length = inputs.shape
        encoded = self.encoder(inputs.reshape(batch * views, length))
        return encoded.reshape(batch, views, -1).mean(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class SetAttentionBlock(nn.Module):
    def __init__(self, feature_dim: int, heads: int, feedforward_dim: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            feature_dim,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(feature_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(feature_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, feature_dim),
        )
        self.norm_feedforward = nn.LayerNorm(feature_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(inputs, inputs, inputs, need_weights=False)
        hidden = self.norm_attention(inputs + attended)
        return self.norm_feedforward(hidden + self.feedforward(hidden))


class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, feature_dim: int, heads: int, feedforward_dim: int) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, 1, feature_dim))
        self.attention = nn.MultiheadAttention(
            feature_dim,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(feature_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(feature_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, feature_dim),
        )
        self.norm_feedforward = nn.LayerNorm(feature_dim)
        nn.init.normal_(self.seed, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self.seed.expand(inputs.shape[0], -1, -1)
        attended, _ = self.attention(query, inputs, inputs, need_weights=False)
        hidden = self.norm_attention(query + attended)
        return self.norm_feedforward(hidden + self.feedforward(hidden)).squeeze(1)


class SetTransformerClassifier(nn.Module):
    architecture_id = "set_transformer_sab2_pma_v1"

    def __init__(
        self,
        known_class_count: int = 7,
        heads: int = 4,
        feedforward_dim: int = 256,
        sab_layers: int = 2,
    ) -> None:
        super().__init__()
        if known_class_count != 7:
            raise ValueError("the frozen first-round classifier has seven known classes")
        if (heads, feedforward_dim, sab_layers) != (4, 256, 2):
            raise ValueError("the frozen B3 architecture is 2 SAB, 4 heads, FFN 256")
        self.encoder = SharedHRRPEncoder1D()
        self.blocks = nn.ModuleList(
            SetAttentionBlock(self.encoder.feature_dim, heads, feedforward_dim)
            for _ in range(sab_layers)
        )
        self.pool = PoolingByMultiheadAttention(
            self.encoder.feature_dim,
            heads,
            feedforward_dim,
        )
        self.classifier = nn.Linear(self.encoder.feature_dim, known_class_count)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.encoder.input_length:
            raise ValueError("Set Transformer input must have shape [batch, views, 601]")
        batch, views, length = inputs.shape
        hidden = self.encoder(inputs.reshape(batch * views, length)).reshape(
            batch, views, -1
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.pool(hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
