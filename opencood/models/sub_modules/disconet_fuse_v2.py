"""
Implementation of V2VNet Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from opencood.models.sub_modules.torch_transformation_utils import \
    get_discretized_transformation_matrix, get_transformation_matrix, \
    warp_affine


class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel * 2, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(32)

        self.conv1_2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(16)

        self.conv1_3 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=0)
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))
        return x_1


class DiscoNetFusion(nn.Module):
    def __init__(self, discrete_ratio, downsample_rate):
        super(DiscoNetFusion, self).__init__()
        self.discrete_ratio = discrete_ratio
        self.downsample_rate = downsample_rate
        self.mlp = nn.Linear(32, 32)
        self.pixel_weighted_fusion = PixelWeightedFusionSoftmax(32)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len, pairwise_t_matrix):
        # x: (B,C,H,W)
        # record_len: (B)
        # pairwise_t_matrix: (B,L,L,4,4)
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W)]
        split_x = self.regroup(x, record_len)
        # (B,L,L,2,3)
        pairwise_t_matrix = get_discretized_transformation_matrix(
            pairwise_t_matrix.reshape(-1, L, 4, 4), self.discrete_ratio,
            self.downsample_rate).reshape(B, L, L, 2, 3)

        batch_node_features = split_x

        batch_updated_node_features = []
        # iterate each batch
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            updated_node_features = []
            # update each node i
            for i in range(N):
                # (N,1,H,W)
                # flip the feature so the transformation is correct
                batch_node_feature = batch_node_features[b]

                current_t_matrix = t_matrix[:, i, :, :]
                current_t_matrix = get_transformation_matrix(current_t_matrix, (H, W))

                # (N,C,H,W)
                neighbor_feature = warp_affine(batch_node_feature, current_t_matrix, (H, W))
                # (N,C,H,W)
                ego_agent_feature = batch_node_feature[i].unsqueeze(0)
                tmp_agent_weight_list = list()
                sum_weight = 0
                for j in range(N):
                    cat_feat = torch.cat([ego_agent_feature[0], neighbor_feature[j]], dim=0)
                    cat_feat = cat_feat.unsqueeze(0)
                    # (N,1,H,W)
                    agent_weight = self.pixel_weighted_fusion(cat_feat)
                    tmp_agent_weight_list.append(torch.exp(agent_weight))
                    sum_weight = sum_weight + torch.exp(agent_weight)

                agent_weight_list = list()
                for j in range(N):
                    agent_weight = torch.div(tmp_agent_weight_list[j], sum_weight)
                    # agent_weight.expand([256, -1, -1])
                    agent_weight_list.append(agent_weight)

                agent_wise_weight_feat = 0
                for k in range(N):
                    agent_wise_weight_feat = (
                            agent_wise_weight_feat
                            + agent_weight_list[k] * neighbor_feature[k]
                    )
                updated_node_features.append(agent_wise_weight_feat)

            batch_updated_node_features = torch.cat(updated_node_features, dim=0)

        return batch_updated_node_features
