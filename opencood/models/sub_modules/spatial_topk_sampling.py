import torch


def spatial_sampling(feat, sample_ratio=0.8):
    N, _, H, W = feat.shape
    sample_pixels = int(H * W * sample_ratio)
    sampled_feat = feat.clone()

    for i in range(N):
        # mask = torch.cuda.FloatTensor(H, W).uniform_() > 0.7

        # mask = random_mask_generator(H, W, sample_pixels)
        mask = activate_mask_generator(sampled_feat[i], sample_pixels)

        sampled_feat[i] = sampled_feat[i] * mask

    return sampled_feat


def random_mask_generator(height, width, sample_pixels):
    zero_pixels = height*width - sample_pixels
    mask = torch.randperm(height*width).float().cuda()

    mask = torch.fmod(mask, height*width/zero_pixels)

    mask = torch.floor(torch.clamp(mask, 0, 1))

    mask = mask.view((height, width))

    return mask


def activate_mask_generator(x_updated, topk):
    h = x_updated.size(1)
    w = x_updated.size(2)
    # add all channels
    x_add = torch.sum(x_updated, dim=0)  # [5, 5]
    # reshape
    x_temp = x_add.reshape(-1)  # [25,]
    # initialize mask
    mask = torch.zeros_like(x_temp)  # [25,]
    # sort by descend
    x_sort = torch.sort(x_temp, descending=True)

    mask[x_sort.indices[:topk]] = 1

    mask = mask.reshape(h, w)

    return mask


if __name__ == '__main__':
    x = torch.rand(3, 16, 5, 5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)

    x_ = spatial_sampling(x, 5)

