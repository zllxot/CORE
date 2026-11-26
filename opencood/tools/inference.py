"""Evaluate a CORE checkpoint on an OPV2V test split."""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.utils import eval_utils
from opencood.utils.core_segmentation import segmentation_iou, segmentation_statistics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', required=True, help='Directory with config.yaml and net_epoch*.pth')
    parser.add_argument('--fusion_method', choices=['intermediate'], default='intermediate')
    parser.add_argument('--test_dir', help='Override the test split in config.yaml')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='Sort detections across all test frames when computing AP')
    opt = parser.parse_args()

    cfg = yaml_utils.load_yaml(None, opt)
    cfg['inference'] = True
    cfg['test_dir'] = opt.test_dir or cfg.get('test_dir') or cfg['validate_dir']
    dataset = build_dataset(cfg, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=cfg['train_params'].get('num_workers', 0),
                        collate_fn=dataset.collate_batch_test)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_utils.create_model(cfg).to(device)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    if model.segment:
        frames = []
        for batch in loader:
            item = train_utils.to_device(batch['ego'], device)
            frames.append(segmentation_statistics(model(item), item['label_dict'], dataset.args))
        metrics = segmentation_iou(frames)
        yaml_utils.save_yaml(metrics, os.path.join(opt.model_dir, 'eval_segmentation.yaml'))
        print('Segmentation IoU:', metrics)
    else:
        records = {threshold: dict(tp=[], fp=[], score=[], gt=0) for threshold in (0.3, 0.5, 0.7)}
        for batch in loader:
            batch = train_utils.to_device(batch, device)
            boxes, scores, gt = dataset.post_process(batch, {'ego': model(batch['ego'])})
            for threshold in records:
                eval_utils.caluclate_tp_fp(boxes, scores, gt, records, threshold)
        eval_utils.eval_final_results(records, opt.model_dir, opt.global_sort_detections)


if __name__ == '__main__':
    main()
