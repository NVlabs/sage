# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Author: Wentao Yuan
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast
import numpy as np
import torch
import torch.nn.functional as F


def dice_loss_matrix(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs:  A float tensor with shape [N, ...].
                 Predicted logits for each query.
        targets: A float tensor with shape [M, ...].
                 Ground truth binary mask for each object.
    Returns:
        loss matrix of shape [N, M], averaged across all pixels/points
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def bce_loss_matrix(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs:  A float tensor with shape [N, ...].
                 Predicted logits for each query.
        targets: A float tensor with shape [M, ...].
                 Ground truth binary mask for each object.
    Returns:
        loss matrix of shape [N, M], averaged across all pixels/points
    """
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    num_points = inputs.shape[1]
    with autocast(enabled=False):
        loss = torch.einsum("nc,mc->nm", pos.float(), targets) \
             + torch.einsum("nc,mc->nm", neg.float(), (1 - targets))

    return loss / num_points


class HungarianMatcher(torch.nn.Module):
    """This class computes a 1-to-1 assignment between the targets and the
    network's predictions. The targets only include objects, so in general,
    there are more predictions than targets. The un-matched predictions are
    treated as non-objects).
    """
    def __init__(self, object_weight, bce_weight, dice_weight):
        super(HungarianMatcher, self).__init__()
        self.object_weight = object_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args['object_weight'] = cfg.object_weight
        args['bce_weight'] = cfg.bce_weight
        args['dice_weight'] = cfg.dice_weight
        return cls(**args)

    @torch.no_grad()
    def forward(self, outputs, data):
        """Performs the matching
        Params:
            outputs: a dict that contains these entries:
                "objectness":     dim [batch_size, num_queries]
                                  logits for the objectness score
                "instance_masks": dim [batch_size, num_queries, ...]
                                  predicted object instance masks
                "contact_masks":  dim [batch_size, num_queries, ...]
                                  predicted grasp contact masks
            targets: a dict that contains these entries:
                "instance_masks": a list of batch_size tensors
                                  ground truth object instance masks
                "contact_masks":  a list of batch_size tensors
                                  ground truth grasp contact masks
        Returns:
            indices: a list of length batch_size, containing indices of the
                     predictions that match the best with each target
        """
        indices, cost_matrices = [], []
        for i in range(len(outputs['objectness'])):
            # We approximate objectness NLL loss with 1 - prob.
            # The 1 is a constant that can be ommitted.
            cost = self.object_weight * (
                -outputs['objectness'][i:i+1].T.sigmoid()
            ) + self.bce_weight * bce_loss_matrix(
                 outputs['grasping_masks'][i], data['grasping_masks'][i]
            ) + self.dice_weight * dice_loss_matrix(
                outputs['grasping_masks'][i], data['grasping_masks'][i]
            )
            output_idx, target_idx = linear_sum_assignment(cost.cpu().numpy())
            output_idx = output_idx[np.argsort(target_idx)]
            indices.append(torch.from_numpy(output_idx).long().to(cost.device))
            cost_matrices.append(cost)
        return indices, cost_matrices
