GT_RANGE = [-140, -40, -3, 140, 40, 1]
COM_RANGE = 70


def build_dataset(dataset_cfg, train=True):
    from opencood.data_utils.datasets.cooperative_scenes import CooperativeScenes

    if dataset_cfg['fusion']['core_method'] not in ('CooperativeScenes', 'core_ext'):
        raise ValueError('CORE requires the CooperativeScenes dataset')
    return CooperativeScenes(dataset_cfg, train=train, inference=dataset_cfg.get('inference', False))
