"""Height compression for a ten-bin VoxelNet grid."""
from collections import OrderedDict

from torch import nn


def height_block(stride, padding):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv3d(64, 64, kernel_size=3, stride=(stride, 1, 1), padding=(padding, 1, 1))),
        ('bn', nn.BatchNorm3d(64)),
        ('relu', nn.ReLU(inplace=True)),
    ]))


class VoxelMiddle(nn.Sequential):
    def __init__(self):
        super().__init__(OrderedDict([
            ('conv3d_1', height_block(2, 1)),
            ('conv3d_2', height_block(1, 0)),
            ('conv3d_3', height_block(2, 1)),
        ]))
