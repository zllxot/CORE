import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm_a = LayerNorm(dim*2, eps=1e-6, data_format="channels_first")
        self.norm_v = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.a = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, a, v, mask):
        a = self.norm_a(a)
        a = self.a(a)

        v = self.norm_v(v)
        v = self.v(v)

        if mask is not None:
            att = self.proj(a * v * mask)
        else:
            att = self.proj(a * v)

        return att


class LayerNorm(nn.Module):
    """
    From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
