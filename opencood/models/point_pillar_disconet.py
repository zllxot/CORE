# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.coperception_backbone import CoperceptionBackbone
from opencood.models.sub_modules.torch_transformation_utils import project_cav_features_to_ego
from opencood.models.fuse_modules.self_attn import AttFusion


class PointPillarDiscoNet(nn.Module):
    def __init__(self, args):
        super(PointPillarDiscoNet, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = CoperceptionBackbone()

        self.fusion_net = AttFusion(64)

        self.cls_head = nn.Conv2d(64, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(64, 7 * args['anchor_num'],
                                  kernel_size=1)

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'pairwise_t_matrix': pairwise_t_matrix,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        spatial_features_2d = self.backbone(batch_dict)

        # project all cav's feature to ego cord
        spatial_features_2d = project_cav_features_to_ego(spatial_features_2d, record_len, pairwise_t_matrix)

        fused_features = self.fusion_net(spatial_features_2d, record_len)

        psm = self.cls_head(fused_features)
        rm = self.reg_head(fused_features)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
