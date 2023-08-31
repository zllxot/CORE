import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_features(features):
    # 获取特征图的数量
    N = len(features)

    x_list = []
    for i in range(N):
        x_list.append(features[i])

    # 使用torch.cat将特征图沿着通道维度（即维度1）连接起来
    cat_features = torch.cat(x_list, dim=0)

    # 创建一个卷积层，输入通道数为N*C，输出通道数为C
    conv = Conv2d(N*128, 128, 3, 1, 1, batch_norm=False, bias=False).to(features.device)

    # 将连接后的特征图通过卷积层
    output = conv(cat_features)

    return output.unsqueeze(0)


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