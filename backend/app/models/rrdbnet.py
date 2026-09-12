"""Minimal RRDBNet inference architecture adapted from BasicSR.

Source: BasicSR commit 8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a
License: Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_features: int = 64, growth_channels: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_features + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_features + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_features + 4 * growth_channels, num_features, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        feature1 = self.lrelu(self.conv1(inputs))
        feature2 = self.lrelu(self.conv2(torch.cat((inputs, feature1), dim=1)))
        feature3 = self.lrelu(self.conv3(torch.cat((inputs, feature1, feature2), dim=1)))
        feature4 = self.lrelu(
            self.conv4(torch.cat((inputs, feature1, feature2, feature3), dim=1))
        )
        feature5 = self.conv5(
            torch.cat((inputs, feature1, feature2, feature3, feature4), dim=1)
        )
        return feature5 * 0.2 + inputs


class RRDB(nn.Module):
    def __init__(self, num_features: int, growth_channels: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_features, growth_channels)
        self.rdb2 = ResidualDenseBlock(num_features, growth_channels)
        self.rdb3 = ResidualDenseBlock(num_features, growth_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.rdb1(inputs)
        output = self.rdb2(output)
        output = self.rdb3(output)
        return output * 0.2 + inputs


class RRDBNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        scale: int = 4,
        num_features: int = 64,
        num_blocks: int = 23,
        growth_channels: int = 32,
    ) -> None:
        super().__init__()
        if scale != 4:
            raise ValueError("the ClearFrame baseline RRDBNet supports scale 4 only")

        self.scale = scale
        self.conv_first = nn.Conv2d(input_channels, num_features, 3, 1, 1)
        self.body = nn.Sequential(
            *(RRDB(num_features, growth_channels) for _ in range(num_blocks))
        )
        self.conv_body = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, output_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.conv_first(inputs)
        body_features = self.conv_body(self.body(features))
        features = features + body_features
        features = self.lrelu(
            self.conv_up1(functional.interpolate(features, scale_factor=2, mode="nearest"))
        )
        features = self.lrelu(
            self.conv_up2(functional.interpolate(features, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(features)))
