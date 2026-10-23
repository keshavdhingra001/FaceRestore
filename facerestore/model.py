"""Partial-convolution U-Net for face inpainting.

Partial convolutions: Liu et al., 2018 (https://arxiv.org/abs/1804.07723).
Self-attention: Zhang et al., 2018 (https://arxiv.org/abs/1805.08318).

Parameter names match karfly/inpaint so its released weights load as-is.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    bn: bool = False
    # Liu et al. rescale each window by (window size / valid pixels). The
    # karfly weights were trained without it, so it must match the weights.
    renorm: bool = False
    # Index of the up block after which self-attention runs, or None.
    # Up block i outputs a 2^(i+2) x 2^(i+2) map for 256 inputs, so 3 -> 32x32.
    attention_after_up: int | None = None
    upsample_mode: str = 'nearest'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, renorm=False):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.renorm = renorm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.conv_bias = nn.Parameter(torch.zeros(out_channels))
        self.sum_conv = nn.Conv2d(in_channels, 1, kernel_size, stride=stride, padding=padding, bias=False)
        self.sum_conv.weight.data.fill_(1)
        self.sum_conv.weight.requires_grad_(False)

    def forward(self, x, mask):
        out = self.conv(x * mask)
        # Count valid pixels in float32: under bf16/fp16 autocast the sums lose
        # precision and the renorm ratio blows up.
        with torch.autocast(device_type=x.device.type, enabled=False):
            mask_sum = self.sum_conv(mask.float())
        valid = (mask_sum > 0).to(out.dtype)
        if self.renorm:
            scale = (self.in_channels * self.kernel_size ** 2) / mask_sum.clamp(min=1)
            out = out * (scale * valid).to(out.dtype)
        out = out + self.conv_bias.view(1, -1, 1, 1).to(out.dtype)
        new_mask = valid.expand(-1, out.shape[1], -1, -1)
        return out, new_mask


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn, renorm):
        super().__init__()
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size, 2, (kernel_size - 1) // 2, renorm)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x), mask


class UpBlock(nn.Module):
    def __init__(self, in_channels, bridge_channels, out_channels, bn, renorm, upsample_mode, activation=True):
        super().__init__()
        self.upsample_mode = upsample_mode
        self.pconv = PartialConv2d(in_channels + bridge_channels, out_channels, 3, 1, 1, renorm)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.activation = activation

    def forward(self, x, mask, x_bridge, mask_bridge):
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x, mask = self.pconv(torch.cat([x, x_bridge], 1), torch.cat([mask, mask_bridge], 1))
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            x = F.leaky_relu(x, 0.2)
        return x, mask


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        # Starts at zero so a pretrained network is unchanged until it learns to use it.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).flatten(2).transpose(1, 2)
        k = self.key(x).flatten(2).transpose(1, 2)
        v = self.value(x).flatten(2).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.gamma * out.transpose(1, 2).reshape(b, c, h, w)


_DOWN = [(3, 64, 7), (64, 128, 5), (128, 256, 5), (256, 512, 3), (512, 512, 3), (512, 512, 3), (512, 512, 3)]
_UP = [(512, 512, 512), (512, 512, 512), (512, 512, 512), (512, 256, 256), (256, 128, 128), (128, 64, 64), (64, 3, 3)]


class InpaintNet(nn.Module):
    """Takes an image with holes zeroed and a mask (1 = known pixel), both N x 3 x H x W."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        # The first two levels never use batch norm, as in Liu et al.
        self.down_blocks = nn.ModuleList([
            DownBlock(i, o, k, config.bn and n >= 2, config.renorm) for n, (i, o, k) in enumerate(_DOWN)
        ])
        last = len(_UP) - 1
        self.up_blocks = nn.ModuleList([
            UpBlock(i, b, o, config.bn and n < last, config.renorm, config.upsample_mode, activation=n < last)
            for n, (i, b, o) in enumerate(_UP)
        ])
        self.attention = None
        if config.attention_after_up is not None:
            self.attention = SelfAttention(_UP[config.attention_after_up][2])

    def forward(self, x, mask):
        bridges = []
        for block in self.down_blocks:
            bridges.append((x, mask))
            x, mask = block(x, mask)
        for n, block in enumerate(self.up_blocks):
            x_bridge, mask_bridge = bridges[-n - 1]
            x, mask = block(x, mask, x_bridge, mask_bridge)
            if n == self.config.attention_after_up:
                x = self.attention(x)
        return x
