from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor


def build_postprocessor(anchor_cfg, train):
    if anchor_cfg['core_method'] != 'VoxelPostprocessor':
        raise ValueError('CORE requires VoxelPostprocessor')
    return VoxelPostprocessor(anchor_cfg, train)
