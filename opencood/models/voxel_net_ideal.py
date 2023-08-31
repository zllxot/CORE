# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from einops import rearrange

from opencood.models.voxel_net import RPN, CML
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.sub_modules.torch_transformation_utils import project_cav_features_to_ego


# conv2d + bn + relu
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, activation=True,
                 batch_norm=True, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p, bias=bias)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class NaiveFusion(nn.Module):
    def __init__(self):
        super(NaiveFusion, self).__init__()
        self.conv1 = Conv2d(128 * 5, 256, 3, 1, 1,
                            batch_norm=False, bias=False)
        self.conv2 = Conv2d(256, 128, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class VoxelNetIdeal(nn.Module):
    def __init__(self, args):
        super(VoxelNetIdeal, self).__init__()
        self.svfe = PillarVFE(args['pillar_vfe'],
                              num_point_features=4,
                              voxel_size=args['voxel_size'],
                              point_cloud_range=args['lidar_range'])
        self.cml = CML()
        self.fusion_net = AttFusion(128)
        self.rpn = RPN(args['anchor_num'])

        self.N = args['N']
        self.D = args['D']
        self.H = args['H']
        self.W = args['W']
        self.T = args['T']
        self.anchor_num = args['anchor_num']

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]

        dense_feature = Variable(
            torch.zeros(dim, self.N, self.D, self.H, self.W).cuda())

        dense_feature[:, coords[:, 0], coords[:, 1], coords[:, 2],
        coords[:, 3]] = sparse_features.transpose(0, 1)

        return dense_feature.transpose(0, 1)

    def regroup(self, dense_feature, record_len):
        """
        Regroup the data based on the record_len.

        Parameters
        ----------
        dense_feature : torch.Tensor
            N, C, H, W
        record_len : list
            [sample1_len, sample2_len, ...]

        Returns
        -------
        regroup_feature : torch.Tensor
            B, 5C, H, W
        """
        cum_sum_len = list(np.cumsum(record_len))
        split_features = torch.tensor_split(dense_feature,
                                            cum_sum_len[:-1])
        regroup_features = []

        for split_feature in split_features:
            # M, C, H, W
            feature_shape = split_feature.shape

            # the maximum M is 5 as most 5 cavs
            padding_len = 5 - feature_shape[0]
            padding_tensor = torch.zeros(padding_len, feature_shape[1],
                                         feature_shape[2], feature_shape[3])
            padding_tensor = padding_tensor.to(split_feature.device)

            split_feature = torch.cat([split_feature, padding_tensor],
                                      dim=0)

            # 1, 5C, H, W
            split_feature = split_feature.view(-1,
                                               feature_shape[2],
                                               feature_shape[3]).unsqueeze(0)
            regroup_features.append(split_feature)

        # B, 5C, H, W
        regroup_features = torch.cat(regroup_features, dim=0)

        return regroup_features

    def forward(self, data_dict):
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        multi_voxel_feats = data_dict['multi_voxel']['voxel_features']
        multi_voxel_coords = data_dict['multi_voxel']['voxel_coords']
        multi_voxel_num_points = data_dict['multi_voxel']['voxel_num_points']
        multi_batch_dict = {'voxel_features': multi_voxel_feats,
                            'voxel_coords': multi_voxel_coords,
                            'voxel_num_points': multi_voxel_num_points}

        record_len = data_dict['record_len']

        if multi_voxel_coords.is_cuda:
            record_len_tmp = record_len.cpu()

        record_len_tmp = list(record_len_tmp.numpy())

        self.N = sum(record_len_tmp)

        # feature learning network
        multi_vwfs = self.svfe(multi_batch_dict)['pillar_features']
        multi_voxel_coords = torch_tensor_to_numpy(multi_voxel_coords)
        multi_vwfs = self.voxel_indexing(multi_vwfs, multi_voxel_coords)

        # convolutional middle network
        x = self.cml(multi_vwfs)
        x = x.view(self.N, -1, self.H, self.W)

        # project all cav's feature to ego cord
        trans_x = project_cav_features_to_ego(x, record_len, pairwise_t_matrix)

        # information fusion
        x_fuse = self.fusion_net(trans_x, record_len)

        # region proposal network
        # merge the depth and feature dim into one, output probability score
        # map and regression map
        psm, rm = self.rpn(x_fuse)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
