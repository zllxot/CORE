"""Distributed setup, aligned checkpoint writes and scene-level validation."""
import os
import secrets
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


def initialize_device():
    if 'RANK' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1:
        if torch.cuda.is_available():
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            dist.init_process_group('nccl')
        else:
            dist.init_process_group('gloo')
    return torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')


def primary():
    return not dist.is_initialized() or dist.get_rank() == 0


def share(value):
    values = [value]
    if dist.is_initialized():
        dist.broadcast_object_list(values, src=0)
    return values[0]


def data_loader(dataset, settings, training):
    sampler = None
    if dist.is_initialized():
        if training:
            seed = share(secrets.randbits(31) if primary() else None)
            sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
        else:
            rank, size = dist.get_rank(), dist.get_world_size()
            sampler = range(len(dataset) * rank // size, len(dataset) * (rank + 1) // size)
    if training and len(dataset) < settings['batch_size'] and sampler is None:
        raise ValueError('Training split is smaller than batch_size')
    return DataLoader(dataset, batch_size=settings['batch_size'] if training else 1,
                      num_workers=settings.get('num_workers', 0), sampler=sampler,
                      shuffle=training and sampler is None, drop_last=training,
                      collate_fn=dataset.collate_batch_train if training else dataset.collate_batch_test)


def synchronize_buffers(model):
    if dist.is_initialized():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)


def save_weights(model, directory, epoch):
    if primary():
        path = Path(directory) / ('net_epoch%d.pth' % epoch)
        staging = path.with_suffix('.tmp')
        torch.save(model.state_dict(), staging)
        os.replace(staging, path)
