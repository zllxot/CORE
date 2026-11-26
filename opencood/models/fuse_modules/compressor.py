from typing import Dict, Tuple, Optional

import torch
from torch import nn


def sparse_message(features: torch.Tensor, topk_ratio: float, keep_ratio: float,
                   generator: Optional[torch.Generator] = None
                   ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Top K% by channel sum, then uniform sampling without replacement of R%."""
    if not 0 <= topk_ratio <= 1 or not 0 <= keep_ratio <= 1:
        raise ValueError('K and R must be fractions in [0,1]')
    batch, channels, height, width = features.shape
    count = height * width
    k = int(count * topk_ratio)
    n = int(k * keep_ratio)
    candidates = features.sum(1).flatten(1).topk(k, dim=1, sorted=True).indices
    # Sampling is independent per sender and independent of labels/features
    # inside the selected Top-K set. Reproducibility is controlled by generator.
    ranks = torch.rand((batch, k), device=features.device, generator=generator)
    selected = candidates.gather(1, ranks.argsort(1)[:, :n])
    values = features.flatten(2).gather(2, selected[:, None].expand(-1, channels, -1))
    dense = torch.zeros_like(features).flatten(2)
    dense = dense.scatter(2, selected[:, None].expand(-1, channels, -1), values)
    return dense.view_as(features), {'indices': selected, 'values': values}


def conv_bn_relu(cin: int, cout: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride, kernel // 2, bias=False),
                         nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01), nn.ReLU())


class Compressor(nn.Module):
    def __init__(self, channels: int, topk_ratio: float, keep_ratio: float):
        super().__init__()
        middle = max(32, channels // 2)
        self.encoder = nn.Sequential(conv_bn_relu(channels, middle, 1),
                                     conv_bn_relu(middle, 16, 1))
        self.decoder = nn.Sequential(conv_bn_relu(16, middle, 1),
                                     conv_bn_relu(middle, channels, 1))
        self.topk_ratio = topk_ratio
        self.keep_ratio = keep_ratio

    def forward(self, x: torch.Tensor, generator=None):
        compressed = self.encoder(x)
        sparse, packet = sparse_message(compressed, self.topk_ratio, self.keep_ratio, generator)
        restored = self.decoder(sparse)
        # Receiver reconstruction uses only transmitted values and positions.
        return restored, packet
