from __future__ import annotations

import torch
from torch import nn


class ConvBlock1D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


class SharedHRRPEncoder1D(nn.Module):
    """Angle-free shared encoder used by B0 and later neural baselines."""

    architecture_id = "shared_hrrp_encoder_1d_v1"

    def __init__(
        self,
        channels: tuple[int, int, int] = (32, 64, 128),
        kernels: tuple[int, int, int] = (7, 5, 3),
        input_length: int = 601,
    ) -> None:
        super().__init__()
        if len(channels) != 3 or len(kernels) != 3:
            raise ValueError("the frozen encoder requires exactly three convolution blocks")
        if any(channel <= 0 for channel in channels):
            raise ValueError("encoder channels must be positive")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("encoder kernels must be positive odd integers")
        if input_length != 601:
            raise ValueError("the frozen P0 encoder input length must be 601")
        self.input_length = input_length
        self.feature_dim = channels[-1]
        self.network = nn.Sequential(
            ConvBlock1D(1, channels[0], kernels[0]),
            ConvBlock1D(channels[0], channels[1], kernels[1]),
            ConvBlock1D(channels[1], channels[2], kernels[2]),
            nn.AdaptiveAvgPool1d(1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("HRRP encoder input must have shape [batch, 601] or [batch, 1, 601]")
        if inputs.shape[-1] != self.input_length:
            raise ValueError(
                f"HRRP encoder input length is {inputs.shape[-1]}, expected {self.input_length}"
            )
        return self.network(inputs).flatten(1)


class HRRPClassifier1D(nn.Module):
    def __init__(
        self,
        known_class_count: int = 7,
        channels: tuple[int, int, int] = (32, 64, 128),
        kernels: tuple[int, int, int] = (7, 5, 3),
        input_length: int = 601,
    ) -> None:
        super().__init__()
        if known_class_count != 7:
            raise ValueError("the frozen first-round classifier must have seven known classes")
        self.encoder = SharedHRRPEncoder1D(
            channels=channels,
            kernels=kernels,
            input_length=input_length,
        )
        self.classifier = nn.Linear(self.encoder.feature_dim, known_class_count)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
