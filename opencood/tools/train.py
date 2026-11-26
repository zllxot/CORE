"""Train an OpenCOOD-style model on one device or with torchrun."""
import argparse
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
from tqdm import tqdm

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import runtime, train_utils
from opencood.tools.validation import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hypes_yaml', required=True)
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--half', action='store_true')
    options = parser.parse_args()
    cfg = load_yaml(options.hypes_yaml, options)
    device = runtime.initialize_device()
    writer = None
    try:
        train_set = build_dataset(cfg, train=True)
        val_set = build_dataset(cfg, train=False)
        settings = cfg['train_params']
        train_loader = runtime.data_loader(train_set, settings, training=True)
        val_loader = runtime.data_loader(val_set, settings, training=False)
        model = train_utils.create_model(cfg).to(device)
        criterion = train_utils.create_loss(cfg).to(device)
        if options.model_dir:
            directory = options.model_dir
            start, model = train_utils.load_saved_model(directory, model)
        else:
            directory = runtime.share(train_utils.setup_train(cfg) if runtime.primary() else None)
            start = 0
        if runtime.primary():
            writer = SummaryWriter(directory)
            print('Run directory:', directory, flush=True)
        optimizer = train_utils.setup_optimizer(cfg, model)
        scheduler = train_utils.setup_lr_schedular(cfg, optimizer)
        parallel = model
        if dist.is_initialized():
            devices = [device.index] if device.type == 'cuda' else None
            parallel = DistributedDataParallel(model, device_ids=devices, find_unused_parameters=True)
        amp = options.half and device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        for epoch in range(start, settings['epoches']):
            scheduler.step(epoch)
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            parallel.train()
            criterion.train()
            with tqdm(total=len(train_loader), disable=not runtime.primary()) as progress:
                for step, batch in enumerate(train_loader):
                    item = train_utils.to_device(batch['ego'], device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast() if amp else nullcontext():
                        output = parallel(item)
                        loss = criterion(output, item['label_dict'])
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if writer is not None:
                        criterion.logging(epoch, step, len(train_loader), writer, pbar=progress)
                    progress.update()
            if (epoch + 1) % settings['save_freq'] == 0:
                runtime.save_weights(model, directory, epoch + 1)
            if (epoch + 1) % settings['eval_freq'] == 0:
                runtime.synchronize_buffers(model)
                metrics = validate(model, criterion, val_loader, device)
                if writer is not None:
                    for name, value in metrics.items():
                        writer.add_scalar('val/' + name, value, epoch + 1)
                    writer.flush()
                    print('Validation epoch', epoch + 1, metrics, flush=True)
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
