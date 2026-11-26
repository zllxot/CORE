"""Metric BEV and OPV2V raster coordinate transforms."""
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def decode_binary(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(str(path))
    return (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)


def metric_to_native(x, y, radius: float, shape):
    height, width = shape
    return y * width / (2 * radius) + width / 2, -x * height / (2 * radius) + height / 2


def load_segmentation(directory: Path, timestamp: str, augmentation: np.ndarray,
                      config: Dict) -> Dict[str, np.ndarray]:
    dynamic = decode_binary(directory / (timestamp + '_bev_dynamic.png'))
    road = decode_binary(directory / (timestamp + '_bev_static.png'))
    lane = decode_binary(directory / (timestamp + '_bev_lane.png'))
    static = road.copy()
    static[lane == 1] = 2
    bounds, voxel = config['lidar_range'], config['voxel_size']
    width = int(round((bounds[3] - bounds[0]) / voxel[0]))
    height = int(round((bounds[4] - bounds[1]) / voxel[1]))
    x, y = np.meshgrid(bounds[0] + (np.arange(width) + 0.5) * voxel[0],
                       bounds[1] + (np.arange(height) + 0.5) * voxel[1])
    plane = augmentation[np.ix_([0, 1, 3], [0, 1, 3])]
    original = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(plane).T
    u, v = metric_to_native(original[..., 0], original[..., 1], config.get('segmentation_radius', 50), dynamic.shape)
    result = {'native_dynamic': dynamic.astype(np.int64), 'native_static': static.astype(np.int64)}
    for key, source in [('dynamic', dynamic), ('static', static)]:
        result[key] = cv2.remap(source, u.astype(np.float32), v.astype(np.float32), cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=255).astype(np.int64)
    return result


def logits_to_native(logits, native_shape, bounds, voxel, radius):
    height, width = native_shape
    rows, cols = torch.meshgrid(torch.arange(height, device=logits.device, dtype=logits.dtype),
                                torch.arange(width, device=logits.device, dtype=logits.dtype), indexing='ij')
    x = (height / 2 - rows) * (2 * radius) / height
    y = (cols - width / 2) * (2 * radius) / width
    gx = 2 * (x - bounds[0]) / (bounds[3] - bounds[0]) - 1
    gy = 2 * (y - bounds[1]) / (bounds[4] - bounds[1]) - 1
    grid = torch.stack((gx, gy), -1).unsqueeze(0).expand(logits.shape[0], -1, -1, -1)
    return F.grid_sample(logits, grid, align_corners=False)


def segmentation_statistics(output, labels, config):
    predictions = {}
    for key in ('dynamic', 'static'):
        native = labels['native_' + key]
        predictions[key] = logits_to_native(output[key + '_logits'], native.shape[-2:],
                                             config['lidar_range'], config['voxel_size'],
                                             config.get('segmentation_radius', 50)).argmax(1)
    statistics = []
    for key, category in [('dynamic', 1), ('static', 1), ('static', 2)]:
        predicted = predictions[key] == category
        target = labels['native_' + key] == category
        intersection = int((predicted & target).sum())
        union = int((predicted | target).sum())
        statistics.append([intersection, union])
    return statistics


def segmentation_iou(frames):
    """Accumulate each class over the full split before computing its IoU."""
    if not frames:
        raise ValueError('segmentation dataset is empty')
    totals = np.sum(frames, axis=0, dtype=np.int64)
    values = totals[:, 0] / np.maximum(totals[:, 1], 1)
    result = dict(zip(('vehicle_iou', 'road_iou', 'lane_iou'), values.tolist()))
    result['miou'] = float(values.mean())
    return result
