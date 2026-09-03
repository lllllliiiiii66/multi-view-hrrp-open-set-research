from __future__ import annotations

import torch
from torch import nn


def _branch_widths(total: int, count: int) -> tuple[int, ...]:
    if total < count:
        raise ValueError("stage channels must be at least the branch count")
    base, remainder = divmod(total, count)
    return tuple(base + (index < remainder) for index in range(count))


class MultiScaleResidualStage1D(nn.Module):
    """A compact three-scale residual stage with standard Conv1d branches."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_sizes: tuple[int, int, int] = (3, 7, 15),
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("multi-scale kernels must be positive odd integers")
        widths = _branch_widths(out_channels, len(kernel_sizes))
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        width,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                )
                for kernel, width in zip(kernel_sizes, widths, strict=True)
            ]
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([branch(inputs) for branch in self.branches], dim=1)
        return self.dropout(self.activation(combined + self.residual(inputs)))


class DeterministicGlobalMaxPool1D(nn.Module):
    """Global max pooling with deterministic CUDA backward semantics."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("global max pooling expects [batch, channels, length]")
        return torch.max(inputs, dim=-1, keepdim=True).values


class HRRPMultiScaleResNet1D(nn.Module):
    """Shared, angle-free multi-scale residual encoder for 601-bin HRRP."""

    architecture_id = "hrrp_ms_resnet_1d_v1"

    def __init__(
        self,
        *,
        input_length: int = 601,
        stem_channels: int = 32,
        stem_kernel_size: int = 31,
        stage_channels: tuple[int, int, int] = (32, 64, 128),
        branch_kernel_sizes: tuple[int, int, int] = (3, 7, 15),
        feature_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_length != 601:
            raise ValueError("MV-RPFormer HRRP input length must be 601")
        if stem_kernel_size <= 1 or stem_kernel_size % 2 == 0:
            raise ValueError("stem kernel must be an odd value greater than one")
        if tuple(stage_channels) != (32, 64, 128):
            raise ValueError("MV-RPFormer stages must remain 32, 64, 128")
        if tuple(branch_kernel_sizes) != (3, 7, 15):
            raise ValueError("MV-RPFormer branch kernels must remain 3, 7, 15")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_length = int(input_length)
        self.feature_dim = int(feature_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(
                1,
                stem_channels,
                kernel_size=stem_kernel_size,
                stride=2,
                padding=stem_kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
        )
        channels = (stem_channels, *stage_channels)
        strides = (1, 2, 2)
        self.stages = nn.Sequential(
            *[
                MultiScaleResidualStage1D(
                    channels[index],
                    channels[index + 1],
                    kernel_sizes=branch_kernel_sizes,
                    stride=strides[index],
                    dropout=dropout,
                )
                for index in range(3)
            ]
        )
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.maximum_pool = DeterministicGlobalMaxPool1D()
        self.projection = nn.Sequential(
            nn.Linear(2 * stage_channels[-1], feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3 or inputs.shape[1:] != (1, self.input_length):
            raise ValueError("HRRP input must have shape [batch, 601] or [batch, 1, 601]")
        features = self.stages(self.stem(inputs))
        pooled = torch.cat(
            [self.average_pool(features).flatten(1), self.maximum_pool(features).flatten(1)],
            dim=1,
        )
        return self.projection(pooled)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
