"""PointPillar and VoxelNet with cooperative reconstruction."""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.voxel_middle import VoxelMiddle
from opencood.models.sub_modules.pillar_vfe import PillarVFE

from .fuse_modules.collaboration import AttentiveCollaboration
from .fuse_modules.compressor import Compressor
from .fuse_modules.multiscale import fuse_multiscale


@contextmanager
def batch_statistics(*modules):
    states = []
    try:
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    states.append((layer, layer.training, layer.track_running_stats))
                    layer.training = True
                    layer.track_running_stats = False
        yield
    finally:
        for layer, training, tracking in states:
            layer.training, layer.track_running_stats = training, tracking


class CORE(nn.Module):
    # Older checkpoint configurations select both values in model.args.
    variant = None

    def __init__(self, args):
        super().__init__()
        if self.variant is not None and self.variant != (args['backbone'], args['task']):
            raise ValueError('%s requires backbone=%s and task=%s' %
                             (type(self).__name__, *self.variant))
        self.width, self.height, self.depth = map(int, args['point_pillar_scatter']['grid_size'])
        self.pillar_vfe = PillarVFE(args['pillar_vfe'], 4, args['voxel_size'], args['lidar_range'])
        self.channels = self.pillar_vfe.get_output_feature_dim()
        is_voxel = args['backbone'] == 'voxelnet'
        if self.depth != (10 if is_voxel else 1) or self.channels != 64:
            raise ValueError('Expected 64 encoder channels and one pillar bin or ten VoxelNet height bins')
        self.middle = nn.Identity()
        if is_voxel:
            self.middle = VoxelMiddle()
        input_channels = 128 if is_voxel else self.channels
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], input_channels)
        widths = args['base_bev_backbone']['num_filters']
        channels = self.backbone.num_bev_features
        self.segment = args['task'] == 'segmentation'
        if self.segment:
            self.dynamic_head = nn.ConvTranspose2d(channels, 2, 2, stride=2)
            self.static_head = nn.ConvTranspose2d(channels, 3, 2, stride=2)
        else:
            self.cls_head = nn.Conv2d(channels, args['anchor_num'], 1)
            self.reg_head = nn.Conv2d(channels, args['anchor_num'] * 7, 1)
        if args.get('fusion', 'CORE') not in ('CORE', 'core'):
            raise ValueError('This model requires CORE fusion')
        self.attentive_collaboration = nn.ModuleList(
            AttentiveCollaboration(c, args['attention_hidden'], args['attention_logit_bound']) for c in widths)
        self.message_codecs = nn.ModuleList(Compressor(c, **args['joint_compression']) for c in widths)
        self.reconstruction = nn.ConvTranspose2d(channels, input_channels, 2, stride=2)

    def encode(self, voxels, count):
        coords = voxels['voxel_coords'].long()
        features = voxels['voxel_features'].new_empty((0, self.channels))
        if len(coords):
            features = self.pillar_vfe(dict(voxels))['pillar_features'].reshape(-1, self.channels)
        planes = []
        for agent in range(count):
            selected = coords[:, 0] == agent
            _, z, y, x = coords[selected].unbind(dim=1)
            index = (z * self.height + y) * self.width + x
            volume = features.new_zeros((self.channels, self.depth * self.height * self.width))
            volume[:, index] = features[selected].T
            volume = volume.reshape(1, self.channels, self.depth, self.height, self.width)
            planes.append(self.middle(volume).flatten(1, 2))
        return torch.cat(planes)

    def forward(self, data):
        counts = data['record_len']
        features = self.encode(data['processed_lidar'], int(counts.sum()))
        spatial, budget, dense = fuse_multiscale(features, counts, self.backbone,
                                                self.attentive_collaboration, self.message_codecs)
        if self.segment:
            outputs = dict(dynamic_logits=self.dynamic_head(spatial), static_logits=self.static_head(spatial))
        else:
            outputs = dict(psm=self.cls_head(spatial), rm=self.reg_head(spatial))
        outputs.update(communication_bytes=budget, dense_multiscale_sender_bytes=dense)
        if 'holistic_voxels' in data:
            with torch.no_grad(), batch_statistics(self.pillar_vfe, self.middle):
                target = self.encode(data['holistic_voxels'], len(counts))
            prediction = self.reconstruction(spatial)
            if prediction.shape != target.shape:
                raise ValueError('CORE target and reconstruction grids must have identical shapes')
            outputs['reconstruction_loss'] = F.mse_loss(prediction, target)
        return outputs
