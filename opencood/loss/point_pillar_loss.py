# The weighted Smooth L1 loss adapts code from fvcore.
# Copyright 2019, Facebook, Inc. and its affiliates.
# The Apache License 2.0 is reproduced at the end of this file.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class PointPillarLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']
        self.loss_dict = {}

    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        rm = output_dict['rm']
        psm = output_dict['psm']
        targets = target_dict['targets']

        cls_preds = psm.permute(0, 2, 3, 1).contiguous()

        box_cls_labels = target_dict['pos_equal_one']
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), 2,
            dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.cls_loss_func(cls_preds,
                                          one_hot_targets,
                                          weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm,
                                                                 targets)
        loc_loss_src =\
            self.reg_loss_func(box_preds_sin,
                               reg_targets_sin,
                               weights=reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        total_loss = reg_loss + conf_loss

        self.loss_dict.update({'total_loss': total_loss,
                               'reg_loss': reg_loss,
                               'conf_loss': conf_loss})

        return total_loss

    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2


    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                " || Loc Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), conf_loss.item(), reg_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                  " || Loc Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), conf_loss.item(), reg_loss.item()))


        writer.add_scalar('Regression_loss', reg_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('Confidence_loss', conf_loss.item(),
                          epoch*batch_len + batch_id)


# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
#
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
# 1. Definitions.
#
# "License" shall mean the terms and conditions for use, reproduction,
# and distribution as defined by Sections 1 through 9 of this document.
#
# "Licensor" shall mean the copyright owner or entity authorized by
# the copyright owner that is granting the License.
#
# "Legal Entity" shall mean the union of the acting entity and all
# other entities that control, are controlled by, or are under common
# control with that entity. For the purposes of this definition,
# "control" means (i) the power, direct or indirect, to cause the
# direction or management of such entity, whether by contract or
# otherwise, or (ii) ownership of fifty percent (50%) or more of the
# outstanding shares, or (iii) beneficial ownership of such entity.
#
# "You" (or "Your") shall mean an individual or Legal Entity
# exercising permissions granted by this License.
#
# "Source" form shall mean the preferred form for making modifications,
# including but not limited to software source code, documentation
# source, and configuration files.
#
# "Object" form shall mean any form resulting from mechanical
# transformation or translation of a Source form, including but
# not limited to compiled object code, generated documentation,
# and conversions to other media types.
#
# "Work" shall mean the work of authorship, whether in Source or
# Object form, made available under the License, as indicated by a
# copyright notice that is included in or attached to the work
# (an example is provided in the Appendix below).
#
# "Derivative Works" shall mean any work, whether in Source or Object
# form, that is based on (or derived from) the Work and for which the
# editorial revisions, annotations, elaborations, or other modifications
# represent, as a whole, an original work of authorship. For the purposes
# of this License, Derivative Works shall not include works that remain
# separable from, or merely link (or bind by name) to the interfaces of,
# the Work and Derivative Works thereof.
#
# "Contribution" shall mean any work of authorship, including
# the original version of the Work and any modifications or additions
# to that Work or Derivative Works thereof, that is intentionally
# submitted to Licensor for inclusion in the Work by the copyright owner
# or by an individual or Legal Entity authorized to submit on behalf of
# the copyright owner. For the purposes of this definition, "submitted"
# means any form of electronic, verbal, or written communication sent
# to the Licensor or its representatives, including but not limited to
# communication on electronic mailing lists, source code control systems,
# and issue tracking systems that are managed by, or on behalf of, the
# Licensor for the purpose of discussing and improving the Work, but
# excluding communication that is conspicuously marked or otherwise
# designated in writing by the copyright owner as "Not a Contribution."
#
# "Contributor" shall mean Licensor and any individual or Legal Entity
# on behalf of whom a Contribution has been received by Licensor and
# subsequently incorporated within the Work.
#
# 2. Grant of Copyright License. Subject to the terms and conditions of
# this License, each Contributor hereby grants to You a perpetual,
# worldwide, non-exclusive, no-charge, royalty-free, irrevocable
# copyright license to reproduce, prepare Derivative Works of,
# publicly display, publicly perform, sublicense, and distribute the
# Work and such Derivative Works in Source or Object form.
#
# 3. Grant of Patent License. Subject to the terms and conditions of
# this License, each Contributor hereby grants to You a perpetual,
# worldwide, non-exclusive, no-charge, royalty-free, irrevocable
# (except as stated in this section) patent license to make, have made,
# use, offer to sell, sell, import, and otherwise transfer the Work,
# where such license applies only to those patent claims licensable
# by such Contributor that are necessarily infringed by their
# Contribution(s) alone or by combination of their Contribution(s)
# with the Work to which such Contribution(s) was submitted. If You
# institute patent litigation against any entity (including a
# cross-claim or counterclaim in a lawsuit) alleging that the Work
# or a Contribution incorporated within the Work constitutes direct
# or contributory patent infringement, then any patent licenses
# granted to You under this License for that Work shall terminate
# as of the date such litigation is filed.
#
# 4. Redistribution. You may reproduce and distribute copies of the
# Work or Derivative Works thereof in any medium, with or without
# modifications, and in Source or Object form, provided that You
# meet the following conditions:
#
# (a) You must give any other recipients of the Work or
# Derivative Works a copy of this License; and
#
# (b) You must cause any modified files to carry prominent notices
# stating that You changed the files; and
#
# (c) You must retain, in the Source form of any Derivative Works
# that You distribute, all copyright, patent, trademark, and
# attribution notices from the Source form of the Work,
# excluding those notices that do not pertain to any part of
# the Derivative Works; and
#
# (d) If the Work includes a "NOTICE" text file as part of its
# distribution, then any Derivative Works that You distribute must
# include a readable copy of the attribution notices contained
# within such NOTICE file, excluding those notices that do not
# pertain to any part of the Derivative Works, in at least one
# of the following places: within a NOTICE text file distributed
# as part of the Derivative Works; within the Source form or
# documentation, if provided along with the Derivative Works; or,
# within a display generated by the Derivative Works, if and
# wherever such third-party notices normally appear. The contents
# of the NOTICE file are for informational purposes only and
# do not modify the License. You may add Your own attribution
# notices within Derivative Works that You distribute, alongside
# or as an addendum to the NOTICE text from the Work, provided
# that such additional attribution notices cannot be construed
# as modifying the License.
#
# You may add Your own copyright statement to Your modifications and
# may provide additional or different license terms and conditions
# for use, reproduction, or distribution of Your modifications, or
# for any such Derivative Works as a whole, provided Your use,
# reproduction, and distribution of the Work otherwise complies with
# the conditions stated in this License.
#
# 5. Submission of Contributions. Unless You explicitly state otherwise,
# any Contribution intentionally submitted for inclusion in the Work
# by You to the Licensor shall be under the terms and conditions of
# this License, without any additional terms or conditions.
# Notwithstanding the above, nothing herein shall supersede or modify
# the terms of any separate license agreement you may have executed
# with Licensor regarding such Contributions.
#
# 6. Trademarks. This License does not grant permission to use the trade
# names, trademarks, service marks, or product names of the Licensor,
# except as required for reasonable and customary use in describing the
# origin of the Work and reproducing the content of the NOTICE file.
#
# 7. Disclaimer of Warranty. Unless required by applicable law or
# agreed to in writing, Licensor provides the Work (and each
# Contributor provides its Contributions) on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied, including, without limitation, any warranties or conditions
# of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
# PARTICULAR PURPOSE. You are solely responsible for determining the
# appropriateness of using or redistributing the Work and assume any
# risks associated with Your exercise of permissions under this License.
#
# 8. Limitation of Liability. In no event and under no legal theory,
# whether in tort (including negligence), contract, or otherwise,
# unless required by applicable law (such as deliberate and grossly
# negligent acts) or agreed to in writing, shall any Contributor be
# liable to You for damages, including any direct, indirect, special,
# incidental, or consequential damages of any character arising as a
# result of this License or out of the use or inability to use the
# Work (including but not limited to damages for loss of goodwill,
# work stoppage, computer failure or malfunction, or any and all
# other commercial damages or losses), even if such Contributor
# has been advised of the possibility of such damages.
#
# 9. Accepting Warranty or Additional Liability. While redistributing
# the Work or Derivative Works thereof, You may choose to offer,
# and charge a fee for, acceptance of support, warranty, indemnity,
# or other liability obligations and/or rights consistent with this
# License. However, in accepting such obligations, You may act only
# on Your own behalf and on Your sole responsibility, not on behalf
# of any other Contributor, and only if You agree to indemnify,
# defend, and hold each Contributor harmless for any liability
# incurred by, or claims asserted against, such Contributor by reason
# of your accepting any such warranty or additional liability.
#
# END OF TERMS AND CONDITIONS
#
# APPENDIX: How to apply the Apache License to your work.
#
# To apply the Apache License to your work, attach the following
# boilerplate notice, with the fields enclosed by brackets "[]"
# replaced with your own identifying information. (Don't include
# the brackets!)  The text should be enclosed in the appropriate
# comment syntax for the file format. We also recommend that a
# file or class name and description of purpose be included on the
# same "printed page" as the copyright notice for easier
# identification within third-party archives.
#
# Copyright 2019, Facebook, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
