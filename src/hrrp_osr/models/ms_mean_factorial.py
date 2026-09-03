from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .arpl import ARPLReciprocalHead
from .mv_rpformer import MVRPFormer


METHODS = (
    "R0_SHALLOW_MEAN_CE",
    "R1_SHALLOW_MEAN_ARPL",
    "R2_MS_MEAN_CE",
    "R3_MS_MEAN_ARPL",
)
SHALLOW_METHODS = frozenset(METHODS[:2])
MULTISCALE_METHODS = frozenset(METHODS[2:])
CE_METHODS = frozenset((METHODS[0], METHODS[2]))
ARPL_METHODS = frozenset((METHODS[1], METHODS[3]))


class MSMeanHeadFactorialModel(MVRPFormer):
    """Frozen mean-fusion 2x2 backbone/head factorial model.

    The implementation deliberately reuses the already audited M0, M1 and M2
    forward/loss paths.  R2 is the M2 mean-fusion path with only its global
    ARPL head replaced by the same linear CE head initialization used by M0.
    """

    architecture_id = "ms_mean_head_factorial_v1"

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
        ce_weight_init_std: float = 0.01,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown factorial method: {method}")
        base_method = {
            "R0_SHALLOW_MEAN_CE": "M0_CURRENT_CE_MEAN",
            "R1_SHALLOW_MEAN_ARPL": "M1_CURRENT_ARPL_MEAN",
            "R2_MS_MEAN_CE": "M2_MS_MEAN_ARPL",
            "R3_MS_MEAN_ARPL": "M2_MS_MEAN_ARPL",
        }[method]
        super().__init__(
            base_method,
            known_class_count,
            feature_dim=feature_dim,
            dropout=dropout,
            temperature=temperature,
            weight_pl=weight_pl,
            margin=margin,
            reciprocal_init_std=reciprocal_init_std,
            initial_radius=initial_radius,
        )
        if method == "R2_MS_MEAN_CE":
            self.global_head = nn.Linear(feature_dim, known_class_count)
            nn.init.normal_(self.global_head.weight, mean=0.0, std=ce_weight_init_std)
            nn.init.zeros_(self.global_head.bias)
        self.method = method
        self.factorial_method = method
        self._assert_frozen_structure()

    def _assert_frozen_structure(self) -> None:
        if any(
            component is not None
            for component in (self.sab, self.pma, self.view_head, self.rejector)
        ):
            raise RuntimeError("factorial model created a forbidden component")
        if self.method in CE_METHODS and not isinstance(self.global_head, nn.Linear):
            raise RuntimeError("CE factorial method lacks a linear head")
        if self.method in ARPL_METHODS and not isinstance(
            self.global_head, ARPLReciprocalHead
        ):
            raise RuntimeError("ARPL factorial method lacks the frozen ARPL head")

    @property
    def forbidden_component_status(self) -> Mapping[str, bool]:
        return {
            "sab_created": self.sab is not None,
            "pma_created": self.pma is not None,
            "view_head_created": self.view_head is not None,
            "rejector_created": self.rejector is not None,
            "pseudo_unknown_supported": False,
        }

    @property
    def head_type(self) -> str:
        return "arpl" if isinstance(self.global_head, ARPLReciprocalHead) else "ce"


def clone_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}
