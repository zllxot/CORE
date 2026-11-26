"""OPV2V scenes for aligned communication and joint-cloud supervision."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.datasets import COM_RANGE, GT_RANGE
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils import box_utils, pcd_utils
from opencood.utils.transformation_utils import x1_to_x2

from opencood.utils.core_metadata import read_metadata
from opencood.utils.core_segmentation import load_segmentation


def index_scenes(directory, max_agents):
    frames = []
    for scene in sorted(directory.iterdir()):
        if not scene.is_dir():
            continue
        agents = sorted((d for d in scene.iterdir() if d.is_dir() and d.name.lstrip('-').isdigit()),
                        key=lambda d: (int(d.name) < 0, d.name))[:max_agents]
        if agents:
            frames.extend((agents, p.stem) for p in sorted(agents[0].glob('*.yaml')) if p.stem.isdigit())
    if not frames:
        raise ValueError('No timestamped OPV2V frames found in %s' % directory)
    return frames


def read_scene(agents, timestamp, train):
    clouds, objects = [], {}
    ego_pose = read_metadata(agents[0] / (timestamp + '.yaml'))['lidar_pose']
    for agent in agents:
        metadata = read_metadata(agent / (timestamp + '.yaml'))
        pose = metadata['lidar_pose']
        if np.linalg.norm(np.subtract(ego_pose[:2], pose[:2])) > COM_RANGE:
            continue
        points = pcd_utils.pcd_to_np(str(agent / (timestamp + '.pcd')))
        points = pcd_utils.mask_ego_points(pcd_utils.shuffle_points(points)).astype(np.float32)
        transform = x1_to_x2(pose, ego_pose)
        points[:, :3] = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        clouds.append(points)
        if train:
            for key, annotation in metadata['vehicles'].items():
                objects.setdefault(key, annotation)
        else:
            objects.update(metadata['vehicles'])
    projected = {}
    box_utils.project_world_objects(objects, projected, ego_pose, [-np.inf] * 3 + [np.inf] * 3, 'hwl')
    boxes = np.concatenate(list(projected.values())) if projected else np.empty((0, 7))
    return clouds, boxes


def augment_scene(clouds, boxes, augmentor):
    # The three extra points track the sampled linear transform for raster labels.
    basis = np.pad(np.eye(3, dtype=np.float32), ((0, 0), (0, 1)))
    lengths = np.cumsum([len(cloud) for cloud in clouds])
    combined = np.concatenate(clouds + [basis])
    augmented = augmentor.forward(dict(lidar_np=combined, object_bbx_center=boxes,
                                       object_bbx_mask=np.ones(len(boxes))))
    *clouds, axes = np.split(augmented['lidar_np'], lengths)
    matrix = np.eye(4)
    matrix[:3, :3] = axes[:, :3].T
    return clouds, augmented['object_bbx_center'], matrix


class CooperativeScenes(Dataset):
    def __init__(self, cfg, train, inference=False, test_dir=None):
        self.args = cfg['model']['args']
        self.workers = cfg['train_params']['num_workers']
        directory = cfg['root_dir'] if train else cfg['validate_dir']
        if inference:
            directory = test_dir or cfg['test_dir']
        self.frames = index_scenes(Path(directory), cfg['train_params']['max_cav'])
        self.augmentor = DataAugmentor(cfg['data_augment'], train)
        self.pre = build_preprocessor(cfg['preprocess'], train)
        self.post_processor = build_postprocessor(cfg['postprocess'], train)
        self.anchors = self.post_processor.generate_anchor_box()
        self.training = train
        self.reconstruct = not inference

    def __len__(self):
        return len(self.frames)

    def detection_labels(self, boxes):
        result = self.post_processor.generate_label(gt_box_center=boxes, mask=np.ones(len(boxes)),
                                                    anchors=self.anchors)
        return {key: torch.as_tensor(value[None], dtype=torch.float32) for key, value in result.items()}

    def __getitem__(self, index):
        agents, timestamp = self.frames[index]
        clouds, boxes = read_scene(agents, timestamp, self.training)
        matrix = np.eye(4)
        if self.training:
            clouds, boxes, matrix = augment_scene(clouds, boxes, self.augmentor)
        bounds = self.args['lidar_range']
        clouds = [pcd_utils.mask_points_by_range(cloud, bounds) for cloud in clouds]
        if len(boxes):
            boxes = box_utils.mask_boxes_outside_range_numpy(boxes, bounds if self.training else GT_RANGE, 'hwl')
        item = dict(processed_lidar=self.pre.collate_batch([self.pre.preprocess(cloud) for cloud in clouds]),
                    record_len=torch.tensor([len(clouds)]),
                    transformation_matrix=torch.eye(4), anchor_box=torch.as_tensor(self.anchors).float(),
                    gt_boxes=torch.as_tensor(boxes).float())
        if self.args['task'] == 'segmentation':
            masks = load_segmentation(agents[0], timestamp, matrix, self.args)
            item['label_dict'] = {key: torch.from_numpy(value[None]) for key, value in masks.items()}
        else:
            item['label_dict'] = self.detection_labels(boxes)
        if self.reconstruct:
            item['holistic_voxels'] = self.pre.collate_batch([self.pre.preprocess(np.concatenate(clouds))])
        return {'ego': item}

    @staticmethod
    def collate_batch_train(batch):
        if len(batch) != 1:
            raise ValueError('Use one scene per process for CORE')
        return batch[0]

    collate_batch_test = collate_batch_train

    def post_process(self, batch, outputs):
        predicted, scores = self.post_processor.post_process(batch, outputs)
        targets = box_utils.boxes_to_corners_3d(batch['ego']['gt_boxes'], 'hwl')
        return predicted, scores, targets
