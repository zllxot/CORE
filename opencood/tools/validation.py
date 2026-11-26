"""Accumulate metrics from the validation pass already performed by OpenCOOD."""
from statistics import mean

import torch
from opencood.tools.train_utils import to_device
import torch.distributed as dist
from opencood.utils import eval_utils

from opencood.utils.core_segmentation import segmentation_iou, segmentation_statistics


def detection_frame(dataset, batch, predictions):
    boxes, scores, gt = dataset.post_process(batch, predictions)
    records = {t: dict(fp=[], tp=[], score=[], gt=0) for t in (0.5, 0.7)}
    for threshold in records:
        eval_utils.caluclate_tp_fp(boxes, scores, gt, records, threshold)
    return records


def detection_ap(frames):
    metrics = {}
    for threshold, name in [(0.5, 'ap50'), (0.7, 'ap70')]:
        merged = dict(fp=[], tp=[], score=[], gt=0)
        for frame in frames:
            for key, value in frame[threshold].items():
                merged[key] += value
        metrics[name] = eval_utils.calculate_ap({threshold: merged}, threshold, False)[0] if merged['gt'] else 0.
    return metrics


@torch.no_grad()
def validate(model, criterion, loader, device):
    model.eval()
    criterion.eval()
    losses, frames = [], []
    dataset = loader.dataset
    for batch in loader:
        batch = to_device(batch, device)
        predictions = {key: model(item) for key, item in batch.items()}
        output, labels = predictions['ego'], batch['ego']['label_dict']
        losses.append(criterion(output, labels).item())
        if getattr(model, 'segment', False):
            frames.append(segmentation_statistics(output, labels, dataset.args))
        else:
            frames.append(detection_frame(dataset, batch, predictions))
    shard = (losses, frames)
    shards = [shard]
    if dist.is_initialized():
        shards = [None] * dist.get_world_size()
        dist.all_gather_object(shards, shard)
        if dist.get_rank() != 0:
            return {}
    losses = [value for local, _ in shards for value in local]
    if not losses:
        raise ValueError('Empty validation split')
    frames = [frame for _, local in shards for frame in local]
    metrics = {'loss': mean(losses)}
    if frames:
        metrics.update(segmentation_iou(frames) if getattr(model, 'segment', False) else detection_ap(frames))
    return metrics
