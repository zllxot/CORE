# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone_core import AttBEVBackboneCORE


class PointPillarCORE(nn.Module):
    def __init__(self, args):
        super(PointPillarCORE, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackboneCORE(args['base_bev_backbone'], 64)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)

    def forward(self, data_dict):
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        if self.training:
            multi_voxel_feats = data_dict['multi_voxel']['voxel_features']
            multi_voxel_coords = data_dict['multi_voxel']['voxel_coords']
            multi_voxel_num_points = data_dict['multi_voxel']['voxel_num_points']
            multi_batch_dict = {'voxel_features': multi_voxel_feats,
                                 'voxel_coords': multi_voxel_coords,
                                 'voxel_num_points': multi_voxel_num_points,
                                 'record_len': record_len}

            multi_batch_dict = self.pillar_vfe(multi_batch_dict)
            multi_batch_dict = self.scatter(multi_batch_dict)

        single_voxel_feats = data_dict['single_voxel']['voxel_features']
        single_voxel_coords = data_dict['single_voxel']['voxel_coords']
        single_voxel_num_points = data_dict['single_voxel']['voxel_num_points']
        single_batch_dict = {'voxel_features': single_voxel_feats,
                             'voxel_coords': single_voxel_coords,
                             'voxel_num_points': single_voxel_num_points,
                             'record_len': record_len,
                             'pairwise_t_matrix': pairwise_t_matrix}

        single_batch_dict = self.pillar_vfe(single_batch_dict)
        single_batch_dict = self.scatter(single_batch_dict)
        single_batch_dict = self.backbone(single_batch_dict)

        spatial_features_2d = single_batch_dict['spatial_features_2d']

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'psm': psm,
                       'rm': rm}

        if self.training:
            output_dict.update({'x_ideal': multi_batch_dict['spatial_features'],
                                'x_rec': single_batch_dict['spatial_features_2d_rec']})

        return output_dict