import torch


def fuse_multiscale(features, record_len, backbone, collaboration, codecs):
    """Compress messages per sender, then fuse each scene at each backbone scale."""
    counts = record_len.tolist()
    levels = []
    byte_count = sent_cells = all_cells = dense_bytes = 0
    for level, block in enumerate(backbone.blocks):
        features = block(features)
        cells = features.size(-1) * features.size(-2)
        all_cells += cells
        dense_bytes += features[0].numel() * 4
        codec = codecs[level]
        selected = int(int(cells * codec.topk_ratio) * codec.keep_ratio)
        sent_cells += selected
        byte_count += selected * (16 * 4 + 4)
        scenes = []
        for agents in features.split(counts):
            if len(agents) > 1:
                decoded, _ = codec(agents[1:])
                agents = torch.cat((agents[:1], decoded))
            scenes.append(collaboration[level](agents))
        fused = torch.stack(scenes)
        levels.append(backbone.deblocks[level](fused))
    spatial = torch.cat(levels, dim=1)
    budget = torch.tensor([byte_count, sent_cells, all_cells], device=spatial.device, dtype=torch.long)
    return spatial, budget, dense_bytes
