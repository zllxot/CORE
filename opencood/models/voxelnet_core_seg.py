from opencood.models.core import CORE


class VoxelNetCORESeg(CORE):
    """VoxelNet CORE for BEV semantic segmentation."""

    variant = ('voxelnet', 'segmentation')
