import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.sub_modules.auto_encoder_core import AutoEncoderCORE
from opencood.models.sub_modules.auto_encoder_rec import AutoEncoderRec


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


class PointPillarIntermediateCORE(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediateCORE, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            # self.compression_layer = AutoEncoderRec(128, args['compression'])
            self.compression_layer = AutoEncoderCORE(128, args['compression'])

        self.fusion_net = AttFusion(128)

        self.cls_head = nn.Conv2d(128, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128, 7 * args['anchor_num'],
                                  kernel_size=1)

        self.down_conv1 = Conv2d(384, 128, 1, 1, 0)
        self.down_conv2 = Conv2d(128, 10, 1, 1, 0)

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'pairwise_t_matrix': pairwise_t_matrix}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = self.down_conv1(batch_dict['spatial_features_2d'])

        # compressor
        if self.compression:
            x_tsk, x_rec = self.compression_layer(spatial_features_2d)
        else:
            x_tsk = spatial_features_2d

        fused_feat = self.fusion_net(x_tsk, record_len)

        psm = self.cls_head(fused_feat)
        rm = self.reg_head(fused_feat)

        output_dict = {'psm': psm,
                       'rm': rm}

        if self.training and self.compression:
            output_dict['x_ideal'] = data_dict['holistic_bev']
            output_dict['x_rec'] = self.down_conv2(x_rec)

        return output_dict
