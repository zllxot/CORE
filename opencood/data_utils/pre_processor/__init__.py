from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor


def build_preprocessor(preprocess_cfg, train):
    if preprocess_cfg['core_method'] != 'SpVoxelPreprocessor':
        raise ValueError('CORE requires SpVoxelPreprocessor')
    return SpVoxelPreprocessor(preprocess_cfg, train)
