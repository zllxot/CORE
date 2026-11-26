"""Detection or raster segmentation with holistic BEV reconstruction."""
import torch.nn as nn
import torch.nn.functional as F
from opencood.loss.point_pillar_loss import PointPillarLoss


class CORELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = args['reconstruction_weight']
        self.detector = PointPillarLoss(args) if args['task'] == 'detection' else None
        self.loss_dict = {}

    def forward(self, outputs, labels):
        if self.detector is not None:
            task = self.detector(outputs, labels)
            terms = {'classification': self.detector.loss_dict['conf_loss'],
                     'regression': self.detector.loss_dict['reg_loss']}
        else:
            dynamic = F.cross_entropy(outputs['dynamic_logits'], labels['dynamic'], ignore_index=255)
            static = F.cross_entropy(outputs['static_logits'], labels['static'], ignore_index=255)
            task = dynamic + static
            terms = dict(dynamic=dynamic, static=static)
        if self.training and self.weight and 'reconstruction_loss' not in outputs:
            raise ValueError('training CORE requires a holistic reconstruction target')
        reconstruction = outputs.get('reconstruction_loss', task.new_zeros(()))
        total = task + self.weight * reconstruction
        self.loss_dict = dict(terms, task=task, reconstruction=reconstruction, total=total)
        return total

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        step = epoch * batch_len + batch_id
        for name, value in self.loss_dict.items():
            writer.add_scalar('loss/' + name, float(value.detach()), step)
        if pbar is not None:
            pbar.set_description('epoch %d loss %.4f' % (epoch, self.loss_dict['total'].item()))
