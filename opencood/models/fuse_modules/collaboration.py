"""CORE multiscale fusion of aligned vehicle features."""
import torch
from torch import nn


class AttentiveCollaboration(nn.Module):
    """Aggregate aligned vehicle features with ego-conditioned attention."""

    def __init__(self, channels, hidden=16, bound=2.):
        super().__init__()
        if hidden <= 0 or hidden % 4 or bound <= 0:
            raise ValueError('attention hidden size must be a positive multiple of four and bound must be positive')
        self.bound = float(bound)
        self.correction = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=False),
            nn.GroupNorm(4, hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.ReLU(), nn.Conv2d(hidden, 1, 1, bias=False))
        nn.init.zeros_(self.correction[-1].weight)

    def forward(self, agents):
        ego = agents[:1]
        if len(agents) == 1:
            return ego[0]
        pairs = torch.cat((ego.expand_as(agents), agents - ego), dim=1)
        bias = self.correction(pairs).tanh() * self.bound
        similarity = (agents * ego).sum(dim=1, keepdim=True) * agents.size(1) ** -0.5
        weights = (similarity + bias).softmax(dim=0)
        return (weights * agents).sum(dim=0)
