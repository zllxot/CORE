import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.fuse_modules.convmod_attention import ConvMod


class COREAttentiveCollaboration(nn.Module):
    def __init__(self, dim=16):
        super(COREAttentiveCollaboration, self).__init__()

        self.pixel_weighted = PixelWeightedFusionSoftmax(dim)
        self.convatt = ConvMod(dim)

    def forward(self, raw_feat):
        """
        Args:
            raw_feat: [num_agent, c, h, w]
        Returns:
            updated_feat: [num_agent, c, h, w]
        """
        num_agent = len(raw_feat)

        local_com_mat = list()
        for i in range(num_agent):
            local_com_mat.append(raw_feat[i, :, :, :])

        local_com_mat_update = local_com_mat.copy()

        conf_map_list = []

        for i in range(num_agent):
            for j in range(num_agent):
                # generate confidence map P and request map R
                conf_map = self.pixel_weighted(local_com_mat[j])
                conf_map_list.append(conf_map)

            ego_request = 1 - conf_map_list[i]

            for k in range(num_agent):
                att_map = ego_request * conf_map_list[k]

                cat_feats = torch.cat([local_com_mat[i], local_com_mat[k]], dim=0)
                q = cat_feats
                v = local_com_mat[k]

                att = self.convatt(q, v, att_map)

                local_com_mat_update[i] = local_com_mat_update[i] + att

        updated_feat = torch.cat(local_com_mat_update, dim=0)

        return updated_feat


class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel, 4, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(4)

        self.conv1_2 = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)
        # self.softmax = nn.Softmax(dim=1)

    def min_max_normalize(self, x):
        # Find the minimum and maximum values of the feature map
        min_val = x.min()
        max_val = x.max()
        # Normalize the feature map to have values between 0 and 1
        normalized = (x - min_val) / (max_val - min_val)
        return normalized

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.conv1_2(x_1))

        return self.min_max_normalize(x_1)


