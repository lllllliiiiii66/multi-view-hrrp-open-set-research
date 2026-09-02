from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .arpl import ARPLReciprocalHead
from .cnn1d import SharedHRRPEncoder1D
from .hrrp_ms_resnet import HRRPMultiScaleResNet1D


METHODS = (
    "M0_CURRENT_CE_MEAN",
    "M1_CURRENT_ARPL_MEAN",
    "M2_MS_MEAN_ARPL",
    "M3_MS_SET_GLOBAL_ARPL",
    "M4_MS_SET_HIER_ARPL",
    "M5_MV_RPFORMER_MISMATCH",
    "M6_MV_RPFORMER_FULL",
    "M7_MV_CEFORMER_FULL",
)
SET_METHODS = frozenset(METHODS[3:])
HIERARCHICAL_METHODS = frozenset(METHODS[4:])
REJECTOR_METHODS = frozenset(METHODS[5:])
CE_METHODS = frozenset((METHODS[0], METHODS[7]))


@dataclass(frozen=True)
class MVModelOutput:
    raw_view_tokens: torch.Tensor
    contextual_view_tokens: torch.Tensor
    global_class_token: torch.Tensor
    global_reject_token: torch.Tensor
    per_view_logits: torch.Tensor
    global_logits: torch.Tensor
    sab_attention: torch.Tensor
    pma_attention: torch.Tensor
    reject_evidence: torch.Tensor | None
    unknown_probability: torch.Tensor | None


class PreNormSAB(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        ffn_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.attention_norm(tokens)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = tokens + self.attention_dropout(attended)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens, weights


class PoolingByMultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        seed_count: int = 2,
        embedding_dim: int = 128,
        num_heads: int = 4,
        ffn_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if seed_count != 2:
            raise ValueError("MV-RPFormer requires exactly two PMA seeds")
        self.seed_count = seed_count
        self.seeds = nn.Parameter(torch.empty(seed_count, embedding_dim))
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.key_value_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        nn.init.xavier_uniform_(self.seeds)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.seeds.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, weights = self.attention(
            self.query_norm(queries),
            self.key_value_norm(tokens),
            self.key_value_norm(tokens),
            need_weights=True,
            average_attn_weights=False,
        )
        outputs = queries + self.attention_dropout(attended)
        outputs = outputs + self.ffn(self.ffn_norm(outputs))
        return outputs, weights


class SmallRejector(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(evidence).squeeze(1))


def _head_logits(head: nn.Module, features: torch.Tensor) -> torch.Tensor:
    if isinstance(head, ARPLReciprocalHead):
        return head.logits(features)
    return head(features)


def _jensen_shannon(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1).clamp_min(1e-12)
    first, second = probabilities[:, 0], probabilities[:, 1]
    mixture = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.log() - mixture.log())).sum(dim=1)
        + (second * (second.log() - mixture.log())).sum(dim=1)
    )


def build_rejector_evidence(
    *,
    global_reject_token: torch.Tensor,
    global_logits: torch.Tensor,
    per_view_logits: torch.Tensor,
    contextual_view_tokens: torch.Tensor,
    pma_attention: torch.Tensor,
) -> torch.Tensor:
    """Build category-conditioned evidence without retaining slot order."""

    predicted = global_logits.argmax(dim=1)
    gather_index = predicted[:, None, None].expand(-1, 2, 1)
    support = per_view_logits.gather(dim=2, index=gather_index).squeeze(2)
    sorted_support = support.sort(dim=1, descending=True).values
    support_mean = support.mean(dim=1, keepdim=True)
    support_min = support.min(dim=1, keepdim=True).values
    support_std = support.std(dim=1, unbiased=False, keepdim=True)
    global_unknown = -global_logits.max(dim=1, keepdim=True).values
    js = _jensen_shannon(per_view_logits).unsqueeze(1)
    view_distance = (
        (contextual_view_tokens[:, 0] - contextual_view_tokens[:, 1])
        .pow(2)
        .mean(dim=1, keepdim=True)
    )
    if pma_attention.ndim != 4 or pma_attention.shape[2:] != (2, 2):
        raise ValueError("PMA attention must have shape [batch, heads, 2, 2]")
    classification_attention = pma_attention[:, :, 0, :].clamp_min(1e-12)
    attention_entropy = -(
        classification_attention * classification_attention.log()
    ).sum(dim=2).mean(dim=1, keepdim=True)
    top_two = global_logits.topk(k=2, dim=1).values
    top_margin = (top_two[:, 0] - top_two[:, 1]).unsqueeze(1)
    scalars = torch.cat(
        [
            global_unknown,
            sorted_support,
            support_mean,
            support_min,
            support_std,
            js,
            view_distance,
            attention_entropy,
            top_margin,
        ],
        dim=1,
    )
    return torch.cat([global_reject_token, scalars], dim=1)


class MVRPFormer(nn.Module):
    architecture_id = "dual_path_set_transformer_mv_reciprocal_point_v1"

    def __init__(
        self,
        method: str,
        known_class_count: int,
        *,
        feature_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 1.0,
        weight_pl: float = 0.1,
        margin: float = 1.0,
        reciprocal_init_std: float = 0.1,
        initial_radius: float = 0.0,
    ) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"unknown MV-RPFormer method: {method}")
        if known_class_count < 2:
            raise ValueError("known_class_count must be at least two")
        self.method = method
        self.known_class_count = int(known_class_count)
        self.feature_dim = int(feature_dim)
        self.encoder = (
            SharedHRRPEncoder1D()
            if method in METHODS[:2]
            else HRRPMultiScaleResNet1D(feature_dim=feature_dim, dropout=dropout)
        )
        if self.encoder.feature_dim != feature_dim:
            raise ValueError("encoder feature dimension changed")
        if method in SET_METHODS:
            self.sab: PreNormSAB | None = PreNormSAB(
                feature_dim, 4, 256, dropout
            )
            self.pma: PoolingByMultiheadAttention | None = PoolingByMultiheadAttention(
                seed_count=2,
                embedding_dim=feature_dim,
                num_heads=4,
                ffn_hidden_dim=256,
                dropout=dropout,
            )
        else:
            self.sab = None
            self.pma = None
        if method in CE_METHODS:
            self.global_head: nn.Module = nn.Linear(feature_dim, known_class_count)
            nn.init.normal_(self.global_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.global_head.bias)
        else:
            self.global_head = ARPLReciprocalHead(
                known_class_count=known_class_count,
                feature_dim=feature_dim,
                temperature=temperature,
                weight_pl=weight_pl,
                margin=margin,
                reciprocal_init_std=reciprocal_init_std,
                initial_radius=initial_radius,
            )
        if method in HIERARCHICAL_METHODS:
            if method in CE_METHODS:
                self.view_head: nn.Module | None = nn.Linear(feature_dim, known_class_count)
                nn.init.normal_(self.view_head.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.view_head.bias)
            else:
                self.view_head = ARPLReciprocalHead(
                    known_class_count=known_class_count,
                    feature_dim=feature_dim,
                    temperature=temperature,
                    weight_pl=weight_pl,
                    margin=margin,
                    reciprocal_init_std=reciprocal_init_std,
                    initial_radius=initial_radius,
                )
        else:
            self.view_head = None
        self.rejector = (
            SmallRejector(feature_dim + 10, dropout=dropout)
            if method in REJECTOR_METHODS
            else None
        )

    @property
    def uses_set_transformer(self) -> bool:
        return self.sab is not None

    @property
    def uses_hierarchical_head(self) -> bool:
        return self.view_head is not None

    @property
    def uses_rejector(self) -> bool:
        return self.rejector is not None

    @property
    def uses_arpl(self) -> bool:
        return isinstance(self.global_head, ARPLReciprocalHead)

    def encode_views(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (2, 601):
            raise ValueError("two-view HRRP input must have shape [batch, 2, 601]")
        batch = inputs.shape[0]
        return self.encoder(inputs.reshape(batch * 2, 601)).reshape(
            batch, 2, self.feature_dim
        )

    def forward_encoded(
        self, raw_view_tokens: torch.Tensor, *, compute_rejector: bool = True
    ) -> MVModelOutput:
        if raw_view_tokens.ndim != 3 or raw_view_tokens.shape[1:] != (
            2,
            self.feature_dim,
        ):
            raise ValueError("encoded views must have shape [batch, 2, feature_dim]")
        batch = raw_view_tokens.shape[0]
        if self.sab is None or self.pma is None:
            contextual = raw_view_tokens
            global_class = contextual.mean(dim=1)
            global_reject = global_class
            sab_attention = raw_view_tokens.new_empty((batch, 0, 2, 2))
            pma_attention = raw_view_tokens.new_empty((batch, 0, 2, 2))
        else:
            contextual, sab_attention = self.sab(raw_view_tokens)
            pooled, pma_attention = self.pma(contextual)
            global_class, global_reject = pooled[:, 0], pooled[:, 1]
        global_logits = _head_logits(self.global_head, global_class)
        per_view_head = self.view_head if self.view_head is not None else self.global_head
        per_view_logits = _head_logits(
            per_view_head, contextual.reshape(-1, self.feature_dim)
        ).reshape(batch, 2, self.known_class_count)
        reject_evidence: torch.Tensor | None = None
        unknown_probability: torch.Tensor | None = None
        if self.rejector is not None and compute_rejector:
            reject_evidence = build_rejector_evidence(
                global_reject_token=global_reject,
                global_logits=global_logits,
                per_view_logits=per_view_logits,
                contextual_view_tokens=contextual,
                pma_attention=pma_attention,
            )
            unknown_probability = self.rejector(reject_evidence)
        return MVModelOutput(
            raw_view_tokens=raw_view_tokens,
            contextual_view_tokens=contextual,
            global_class_token=global_class,
            global_reject_token=global_reject,
            per_view_logits=per_view_logits,
            global_logits=global_logits,
            sab_attention=sab_attention,
            pma_attention=pma_attention,
            reject_evidence=reject_evidence,
            unknown_probability=unknown_probability,
        )

    def forward(self, inputs: torch.Tensor, *, compute_rejector: bool = True) -> MVModelOutput:
        return self.forward_encoded(
            self.encode_views(inputs), compute_rejector=compute_rejector
        )

    def representation_loss(
        self, output: MVModelOutput, labels: torch.Tensor, *, lambda_view: float = 0.5
    ) -> dict[str, torch.Tensor]:
        if isinstance(self.global_head, ARPLReciprocalHead):
            global_loss = self.global_head.loss(output.global_class_token, labels)
            global_total = global_loss.total_loss
            global_classification = global_loss.classification_loss
            global_margin = global_loss.margin_loss
        else:
            global_total = F.cross_entropy(output.global_logits, labels)
            global_classification = global_total
            global_margin = global_total.new_zeros(())
        view_total = global_total.new_zeros(())
        view_classification = global_total.new_zeros(())
        view_margin = global_total.new_zeros(())
        if self.view_head is not None:
            repeated = labels[:, None].expand(-1, 2).reshape(-1)
            flat_tokens = output.contextual_view_tokens.reshape(-1, self.feature_dim)
            flat_logits = output.per_view_logits.reshape(-1, self.known_class_count)
            if isinstance(self.view_head, ARPLReciprocalHead):
                view_loss = self.view_head.loss(flat_tokens, repeated)
                view_total = view_loss.total_loss
                view_classification = view_loss.classification_loss
                view_margin = view_loss.margin_loss
            else:
                view_total = F.cross_entropy(flat_logits, repeated)
                view_classification = view_total
            total = global_total + float(lambda_view) * view_total
        else:
            total = global_total
        return {
            "total": total,
            "global_total": global_total,
            "global_classification": global_classification,
            "global_margin": global_margin,
            "view_total": view_total,
            "view_classification": view_classification,
            "view_margin": view_margin,
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
