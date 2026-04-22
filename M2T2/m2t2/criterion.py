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
Modules for computing ADD-S, mask and parameter losses.
"""
from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_loss, cross_entropy
)
import torch
import torch.distributed as dist
import torch.nn as nn

from m2t2.model_utils import load_control_points, repeat_new_axis


def dice_loss(pred: torch.Tensor, target: torch.Tensor, num_objs: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        pred:     A float tensor of arbitrary shape.
                  Predicted logits for each query matched with a target.
        target:   A float tensor with the same shape as pred.
                  Binary label for each element in pred.
        num_objs: Average number of objects across the batch.
    """
    pred = pred.sigmoid()
    numerator = 2 * (pred * target).sum(-1)
    denominator = pred.sum(-1) + target.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_objs


def adds(pred_grasps, confidence, target_grasps, ctr_pts):
    """Compute ADD-S loss
    Params:
        pred_grasps: [num_pred_grasps, 4, 4]
        confidence: [num_pred_grasps]
        target_grasps: [num_gt_grasps, 4, 4]
        ctr_pts: [4, 5]
    """
    # [N, 4, 4] x [4, 5] -> [N, 4, 5] -> [N, 5, 4] -> [N, 5, 3]
    pred_pts = (pred_grasps @ ctr_pts).transpose(-2, -1)[..., :3]
    gt_pts = (target_grasps @ ctr_pts).transpose(-2, -1)[..., :3]

    # [N, 1, 5, 3] - [1, M, 5, 3] -> [N, M, 5, 3] -> [N, M]
    dist_1 = torch.clip(((
        pred_pts.unsqueeze(1) - gt_pts.unsqueeze(0)) ** 2
    ).sum(dim=-1), 0).sqrt().mean(dim=-1)
    dist_2 = torch.clip(((
        pred_pts.unsqueeze(1) - gt_pts.flip(1).unsqueeze(0)) ** 2
    ).sum(dim=-1), 0).sqrt().mean(dim=-1)
    dist = torch.minimum(dist_1, dist_2)

    pred2gt, _ = dist.min(dim=1)
    gt2pred, gt2pred_idx = dist.min(dim=0)
    adds_pred2gt = confidence * pred2gt
    adds_gt2pred = confidence[gt2pred_idx] * gt2pred
    return adds_pred2gt, adds_gt2pred


def average(inputs, num_objs, num_points, mask=False):
    if mask:
        inputs = (inputs > 0).float()
    if num_points is not None:
        inputs = [input.mean() for input in inputs.split(num_points)]
        if len(inputs) > 0:
            avg = torch.stack(inputs).nansum() / num_objs
        else:
            avg = torch.tensor(0.).to(num_objs.device)
    else:
        avg = inputs.mean(dim=-1).sum() / max(num_objs, 1)
    return avg


class MaskCriterion(nn.Module):
    def __init__(self, loss_weights, top_k):
        super(MaskCriterion, self).__init__()
        self.loss_weights = loss_weights
        self.top_k = top_k

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args['loss_weights'] = {'bce': cfg.bce_weight, 'dice': cfg.dice_weight}
        args['top_k'] = cfg.bce_topk
        return cls(**args)

    def forward(self, key, out_mask, tgt_mask, loss_mask=None):
        # Compute the average number of objects accross all nodes
        num_objs = torch.tensor(max(out_mask.shape[0], 1)).to(out_mask.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_objs)
            num_objs = num_objs / dist.get_world_size()

        bce = bce_loss(out_mask, tgt_mask, reduction='none')
        if loss_mask is not None:
            bce = bce * loss_mask
        bce_topk, topk_ids = bce.topk(self.top_k, dim=1)
        out_mask = out_mask.gather(1, topk_ids)
        tgt_mask = tgt_mask.gather(1, topk_ids)
        losses = {
            'bce': torch.div(bce_topk.sum(), num_objs * bce_topk.shape[1]),
            'dice': dice_loss(out_mask, tgt_mask, num_objs)
        }
        stats = {
            'topk_pred_pos_ratio': (out_mask > 0).float().mean(),
            'topk_gt_pos_ratio': (tgt_mask > 0).float().mean(),
            'topk_hard_neg_ratio': (
                ((out_mask > 0) | (tgt_mask > 0)).float().mean()
            )
        }

        losses = {
            f'{key}_{k}': (self.loss_weights[k], v) for k, v in losses.items()
        }
        stats = {f'{key}_{k}': v for k, v in stats.items()}
        return losses, stats


class SetCriterion(nn.Module):
    """This class computes the Hungarian matching loss.
    The process consists of two steps:
        1) compute 1-1 assignments between outputs of the model and ground
           truth targets (usually, there are more outputs than targets)
        2) supervise each matched prediction with the corresponding target
    """

    def __init__(
        self, matcher, deep_supervision, recompute_indices, mask_criterion,
        object_weight, not_object_weight, pseudo_ce_weight
    ):
        """Create the criterion.
        Parameters:
            matcher: module to compute 1-1 matching between targets and outputs
            sampler: sample a subset of points to compute mask loss
            deep_supervision: whether to supervise intermediate layer outputs
            recompute_indices: recompute matching for each intermediate layer
            object_weight: weight of the objectness classification loss
            not_object_weight: multiplier for the ce loss of unmatched outputs
            instance_weights: weights of the instance mask loss
            contact_weights: weights of the contact mask loss
            pseudo_ce: use cross entropy with pseudo labels from matcher
        """
        super(SetCriterion, self).__init__()
        self.matcher = matcher
        self.deep_supervision = deep_supervision
        self.recompute_indices = recompute_indices

        self.object_weight = object_weight
        self.not_object_weight = not_object_weight
        self.mask_criterion = mask_criterion
        self.pseudo_ce_weight = pseudo_ce_weight
        if pseudo_ce_weight > 0:
            self.pseudo_ce_loss = nn.CrossEntropyLoss()

    @classmethod
    def from_config(cls, cfg, matcher):
        args = {}
        args['deep_supervision'] = cfg.deep_supervision
        args['recompute_indices'] = cfg.recompute_indices
        args['object_weight'] = cfg.object_weight
        args['not_object_weight'] = cfg.not_object_weight
        args['mask_criterion'] = MaskCriterion.from_config(cfg)
        args['pseudo_ce_weight'] = cfg.pseudo_ce_weight
        return cls(matcher, **args)

    def get_pseudo_ce_loss(self, pred_masks, gt_masks, matched_idx):
        B, N, H, W = pred_masks.shape
        pseudo_label = torch.zeros(B, H, W).long()
        pseudo_label = pseudo_label.to(pred_masks.device)
        tgt_mask_any = []
        for i, (tgt_mask, idx) in enumerate(zip(gt_masks, matched_idx)):
            obj_id, y, x = torch.where(tgt_mask > 0)
            pseudo_label[i, y, x] = idx[obj_id]
            tgt_mask_any.append(tgt_mask.any(dim=0))
        tgt_mask_any = torch.stack(tgt_mask_any)
        loss = self.pseudo_ce_loss(
            pred_masks.permute(0, 2, 3, 1)[tgt_mask_any],
            pseudo_label[tgt_mask_any]
        )
        return loss

    def get_loss(self, pred, data, matched_idx, layer=None):
        obj_label = torch.zeros_like(pred['objectness'])
        for i, idx in enumerate(matched_idx):
            obj_label[i][idx] = 1
        pos_weight = torch.tensor(1 / self.not_object_weight).to(
            pred['objectness'].device
        )
        loss_obj = bce_loss(
            pred['objectness'], obj_label,
            pos_weight=pos_weight, reduction='none'
        ) * self.not_object_weight
        mask = data['task_is_pick'].unsqueeze(1).float()
        loss_obj = (loss_obj * mask).sum() / torch.clamp(mask.sum(), 1)
        losses = {'objectness': (self.object_weight, loss_obj)}

        if self.pseudo_ce_weight > 0:
            pseudo_ce = self.get_pseudo_ce_loss(
                pred['grasping_masks'], data['grasping_masks'], matched_idx
            )
            losses['pseudo_ce'] = (self.pseudo_ce_weight, pseudo_ce)

        matched_masks = [mask[idx] for mask, idx in zip(
            pred['grasping_masks'], matched_idx
        )]
        outputs = {'matched_grasping_masks': matched_masks}
        mask_loss, stats = self.mask_criterion(
            'grasping', torch.cat(matched_masks),
            torch.cat(data['grasping_masks'])
        )
        losses.update(mask_loss)
        outputs.update(stats)

        if layer is not None:
            losses = {
                f'layer{layer}/{key}': val for key, val in losses.items()
            }
        return losses, outputs

    def forward(self, pred, targets):
        outputs = pred[-1]

        # Compute matching between final prediction and the targets
        output_idx, cost_matrices = self.matcher(pred[-1], targets)
        outputs.update({
            'matched_idx': output_idx, 'cost_matrices': cost_matrices
        })

        # Compute losses for the final layer outputs
        losses, stats = self.get_loss(pred[-1], targets, output_idx)
        outputs.update(stats)

        if self.deep_supervision and self.training:
            # Compute losses for each intermediate layer outputs
            for i, p in enumerate(pred[:-1]):
                if self.recompute_indices:
                    output_idx, _ = self.matcher(p, targets)
                l_dict, _ = self.get_loss(p, targets, output_idx, i + 1)
                losses.update(l_dict)
                outputs[f'layer{i+1}/matched_idx'] = output_idx

        return losses, outputs


class ADDSCriterion(nn.Module):
    def __init__(
        self, pred2gt_weight, gt2pred_weight, adds_per_obj
    ):
        super().__init__()
        self.loss_weights = {
            'adds_pred2gt': pred2gt_weight,
            'adds_gt2pred': gt2pred_weight
        }
        self.adds_per_obj = adds_per_obj
        self.control_points = load_control_points()

    def forward(self, pred_grasps, confidence, gt_grasps, device):
        """Compute ADD-S loss for a batch of scenes
        Params:
            pred_grasps: a list of lists, each list contains num_objects
                         tensors of shape [num_pred_grasps, 4, 4]
            confidence:  a list of of lists, each list contains num_objects
                         tensors of shape [num_pred_grasps]
            gt_grasps:   a list of lists, each list contains num_objects
                         tensors of shape [num_gt_grasps, 4, 4]
        """
        adds_pred2gt, adds_gt2pred = [], []
        ctr_pts = self.control_points.to(device)
        zero = torch.tensor(0.).to(device)
        for pred_grasp, conf, gt_grasp in zip(
            pred_grasps, confidence, gt_grasps
        ):
            if self.adds_per_obj:
                for pred, c, gt in zip(pred_grasp, conf, gt_grasp):
                    if pred.shape[0] == 0 or gt.shape[0] == 0:
                        continue
                    pred2gt, gt2pred = adds(pred, c, gt, ctr_pts)
                    adds_pred2gt.append(pred2gt.mean())
                    adds_gt2pred.append(gt2pred.mean())
            else:
                if len(pred_grasp) == 0 or len(gt_grasp) == 0:
                    continue
                num_grasps = [g.shape[0] for g in pred_grasp]
                pred_grasp = torch.cat(pred_grasp)
                conf = torch.cat(conf)
                num_gt_grasps = [g.shape[0] for g in gt_grasp]
                gt_grasp = torch.cat(gt_grasp)
                if pred_grasp.shape[0] == 0 or gt_grasp.shape[0] == 0:
                    continue

                pred2gt, gt2pred = adds(pred_grasp, conf, gt_grasp, ctr_pts)
                pred2gt = [
                    val.mean() for val in pred2gt.split(num_grasps)
                    if val.shape[0] > 0
                ]
                gt2pred = [
                    val.mean() for val in gt2pred.split(num_gt_grasps)
                    if val.shape[0] > 0
                ]
                adds_pred2gt += pred2gt
                adds_gt2pred += gt2pred

        losses = {'adds_pred2gt': zero, 'adds_gt2pred': zero}
        if len(adds_pred2gt) > 0:
            losses['adds_pred2gt'] = torch.stack(adds_pred2gt).mean()
        if len(adds_gt2pred) > 0:
            losses['adds_gt2pred'] = torch.stack(adds_gt2pred).mean()
        losses = {key: (self.loss_weights[key], losses[key]) for key in losses}
        return losses


class GraspCriterion(nn.Module):
    def __init__(
        self, adds_criterion, contact_dir_weight, approach_dir_weight,
        offset_weight, param_weight, bin_weights
    ):
        super(GraspCriterion, self).__init__()
        self.adds_criterion = adds_criterion
        self.loss_weights = {
            'contact_dir': contact_dir_weight,
            'approach_dir': approach_dir_weight,
            'offset': offset_weight,
            'param': param_weight,
            'release': param_weight
        }
        self.bin_weights = torch.tensor(bin_weights)

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args['adds_criterion'] = ADDSCriterion(
            cfg.adds_pred2gt, cfg.adds_gt2pred, cfg.adds_per_obj
        )
        args['contact_dir_weight'] = cfg.contact_dir
        args['approach_dir_weight'] = cfg.approach_dir
        args['offset_weight'] = cfg.offset
        args['param_weight'] = cfg.param
        args['bin_weights'] = cfg.offset_bin_weights
        return cls(**args)

    def forward(self, pred, data):
        losses = {}
        losses['contact_dir'] = (1 - (
            pred['contact_dirs'] * data['contact_dirs']
        ).sum(dim=1))
        losses['approach_dir'] = (1 - (
            pred['approach_dirs'] * data['approach_dirs']
        ).sum(dim=1))
        losses['offset'] = cross_entropy(
            pred['offsets'], data['offsets'],
            self.bin_weights.to(pred['offsets'].device), reduction='none'
        )
        if 'params' in data:
            losses['param'] = ((pred['params'] - data['params']) ** 2).mean()
        if 'release' in data:
            losses['release'] = bce_loss(
                pred['release'], data['release'].float()
            ).mean()
        for key in ['contact_dir', 'approach_dir', 'offset']:
            losses[key] = losses[key].sum() / max(losses[key].numel(), 1)
        losses = {
            key: (self.loss_weights[key], losses[key]) for key in losses
        }
        losses.update(self.adds_criterion(
            pred['grasps'], pred['grasp_confidence'],
            data['grasps'], data['inputs'].device
        ))
        return losses


class PlaceCriterion(nn.Module):
    def __init__(self, mask_criterion, deep_supervision):
        super(PlaceCriterion, self).__init__()
        self.mask_criterion = mask_criterion
        self.deep_supervision = deep_supervision

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args['mask_criterion'] = MaskCriterion.from_config(cfg)
        args['deep_supervision'] = cfg.deep_supervision
        return cls(**args)

    def forward(self, pred, data):
        pred_masks = pred[-1]['placement_masks'][data['task_is_place']]
        target_masks = data['placement_masks'][data['task_is_place']]
        loss_masks = data['placement_region'][data['task_is_place']]
        loss_masks = repeat_new_axis(
            loss_masks, target_masks.shape[1], dim=1
        ) # (B, H, W) -> (B, Q, H, W)
        loss_masks = loss_masks.flatten(0, 1)
        target_masks = target_masks.flatten(0, 1)
        pred_masks = pred_masks.flatten(0, 1)
        losses, stats = self.mask_criterion(
            'placement', pred_masks, target_masks, loss_masks
        )

        if self.deep_supervision and self.training:
            # Compute losses for each intermediate layer outputs
            for i, p in enumerate(pred[:-1]):
                pred_masks = p['placement_masks'][data['task_is_place']]
                pred_masks = pred_masks.flatten(0, 1)
                mask_losses, _ = self.mask_criterion(
                    'placement', pred_masks, target_masks, loss_masks
                )
                mask_losses = {
                    f'layer{i+1}/{key}': val
                    for key, val in mask_losses.items()
                }
                losses.update(mask_losses)
        return losses, stats
