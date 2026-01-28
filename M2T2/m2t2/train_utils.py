# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Author: Wentao Yuan
'''
Utility functions for training M2T2.
'''
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import copy
import torch

from m2t2.dataset import PickPlaceDataset, collate


def get_data_loader(cfg, split, use_ddp, training):
    kwargs = PickPlaceDataset.from_config(cfg.data)
    kwargs['root_dir'] = f"{cfg.data.root_dir}/{split}"
    if not training:
        kwargs['jitter_scale'] = 0
        kwargs['robot_prob'] = 1
    dataset = PickPlaceDataset(**kwargs)
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=training, drop_last=True)
    else:
        sampler = RandomSampler(dataset) if training \
                  else SequentialSampler(dataset)
    loader = DataLoader(
        dataset, cfg.train.batch_size, sampler=sampler,
        num_workers=cfg.train.num_workers, collate_fn=collate,
        persistent_workers=True, pin_memory=True
    )
    return sampler, loader


def build_optimizer(cfg, model):
    defaults = {}
    defaults["lr"] = (
        cfg.optimizer.base_lr / cfg.optimizer.base_batch_size
        * cfg.train.batch_size * cfg.train.num_gpus
    )
    defaults["weight_decay"] = cfg.optimizer.weight_decay

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params = []
    memo = set()
    for module_name, module in model.named_modules():
        for param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if (
                "relative_position_bias_table" in param_name
                or "absolute_pos_embed" in param_name
            ):
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types) or isinstance(
                module, torch.nn.Embedding
            ):
                hyperparams["weight_decay"] = 0.0
            params.append({"params": [value], **hyperparams})

    if cfg.optimizer.type == "SGD":
        optimizer = torch.optim.SGD(
            params, defaults["lr"], momentum=cfg.optimizer.momentum
        )
    elif cfg.optimizer.type == "ADAMW":
        optimizer = torch.optim.AdamW(params, defaults["lr"])
    return optimizer


def save_model(epoch, model, optimizer, log_dir, use_ddp, name=None):
    if use_ddp:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    ckpt = {
        "epoch": epoch, "model": model_state,
        "optimizer": optimizer.state_dict()
    }
    if name is None:
        name = f"epoch_{epoch}"
    torch.save(ckpt, f"{log_dir}/{name}.pth")


def to_gpu(dic):
    for key in dic:
        if isinstance(dic[key], torch.Tensor):
            dic[key] = dic[key].cuda()
        elif isinstance(dic[key], list):
            if isinstance(dic[key][0], torch.Tensor):
                for i in range(len(dic[key])):
                    dic[key][i] = dic[key][i].cuda()
            elif isinstance(dic[key][0], list):
                for i in range(len(dic[key])):
                    for j in range(len(dic[key][i])):
                        if isinstance(dic[key][i][j], torch.Tensor):
                            dic[key][i][j] = dic[key][i][j].detach().cuda()


def to_cpu(dic):
    for key in dic:
        if isinstance(dic[key], torch.Tensor):
            dic[key] = dic[key].detach().cpu()
        elif isinstance(dic[key], list):
            if isinstance(dic[key][0], torch.Tensor):
                for i in range(len(dic[key])):
                    dic[key][i] = dic[key][i].detach().cpu()
            elif isinstance(dic[key][0], list):
                for i in range(len(dic[key])):
                    for j in range(len(dic[key][i])):
                        if isinstance(dic[key][i][j], torch.Tensor):
                            dic[key][i][j] = dic[key][i][j].detach().cpu()


def write_scalar_ddp(
    writer, key, value, step, rank, num, reduce=False, debug=False
):
    if debug:
        print(f"Rank {rank} Step {step} {key} {value.item()}")
    if reduce:
        try:
            torch.distributed.reduce(value, dst=0)
        except Exception as e:
            print(
                "Exception while reducing key", key,
                "Rank", rank, "Global step", step
            )
            print(e)
            return
    if rank == 0:
        val = torch.div(value, num)
        if not torch.isnan(val) and not torch.isinf(val):
            writer.add_scalar(key, val.item(), step)


def add_to_dict(dict, key, val):
    if key not in dict:
        dict[key] = 0
    dict[key] += val


def get_iou(out_masks, tgt_masks, reduce=True):
    intersect = (out_masks & tgt_masks)
    union = (out_masks | tgt_masks)
    iou = torch.nan_to_num(intersect.sum(dim=-1) / union.sum(dim=-1), nan=1)
    if reduce:
        iou = iou.mean()
    return iou


def compute_iou(out_masks, tgt_masks, thresh=0., loss_masks=None, reduce=True):
    iou_dict = {}
    if isinstance(out_masks, list):
        masks = {}
        mask_any = {}
        mask_list = [mask.flatten(start_dim=1) > thresh for mask in tgt_masks]
        # [batch_size, num_points]
        mask_any['target'] = torch.stack([
            mask.any(dim=0) for mask in mask_list
        ])
        # [num_objects, num_points]
        masks['target'] = torch.cat(mask_list)

        mask_list = [mask.flatten(start_dim=1) > thresh for mask in out_masks]
        # [batch_size, num_points]
        mask_any['output'] = torch.stack([
            mask.any(dim=0) for mask in mask_list
        ])
        # [num_objects, num_points]
        masks['output'] = torch.cat(mask_list)

        for key, mask_dict in zip(['scene', 'object'], [mask_any, masks]):
            if mask_dict['output'].shape[0] != mask_dict['target'].shape[0]:
                continue
            iou_dict[key] = get_iou(
                mask_dict['output'], mask_dict['target'], reduce
            )
    elif loss_masks is None:
        iou_dict['scene'] = get_iou(
            out_masks.flatten(start_dim=1) > thresh,
            tgt_masks.flatten(start_dim=1) > thresh, reduce
        )
    else:
        ious = []
        for out_mask, tgt_mask, loss_mask in zip(
            out_masks, tgt_masks, loss_masks
        ):
            if len(out_mask.shape) == len(loss_mask.shape):
                out_mask = out_mask[loss_mask] > thresh
                tgt_mask = tgt_mask[loss_mask] > thresh
            else:
                out_mask = out_mask[:, loss_mask] > thresh
                tgt_mask = tgt_mask[:, loss_mask] > thresh
            ious.append(get_iou(out_mask, tgt_mask))
        iou_dict['scene'] = torch.stack(ious)
    if reduce:
        for key in iou_dict:
            iou_dict[key] = iou_dict[key].mean()
    return iou_dict


def clip_grad_norm(parameters, max_norm, norm_type):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm.
            Can be ``'inf'`` for infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    norms = []
    for p in parameters:
        if p.grad is None:
            continue
        if norm_type == "inf":
            norm = p.grad.detach().abs().max()
        else:
            norm = torch.norm(p.grad.detach(), norm_type)
        norms.append(norm)
    if norm_type == "inf":
        total_norm = torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack(norms), norm_type)
    clip_coef = torch.clamp(max_norm / total_norm.nan_to_num(), max=1.0)
    for p in parameters:
        p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm
