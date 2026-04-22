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
Training script for M2T2.
'''
from datetime import timedelta
from functools import partial
from itertools import chain
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from time import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
import hydra
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from m2t2.m2t2 import M2T2
from m2t2.plot_utils import plot_3D
from m2t2.train_utils import (
    get_data_loader, build_optimizer, save_model, to_gpu,
    to_cpu, compute_iou, write_scalar_ddp, add_to_dict
)


def train_one_epoch(
    loader, model, optimizer, clip_grad,
    writer, epoch, global_step, plot_fn, cfg
):
    rank = 0
    ws = 1
    use_ddp = dist.is_available() and cfg.train.num_gpus > 1
    if use_ddp:
        rank = dist.get_rank()
        ws = dist.get_world_size()
    params = list(chain(*[x["params"] for x in optimizer.param_groups]))
    num_steps = len(loader)
    data_time = torch.tensor(0., device=rank)
    step_time = torch.tensor(0., device=rank)
    log_time = torch.tensor(0., device=rank)

    start = time()
    for i, data in enumerate(loader):
        global_step += 1
        to_gpu(data)
        data_time += time() - start
        start = time()

        optimizer.zero_grad()
        outputs, losses = model(data, cfg.train)
        loss = sum([w * v for w, v in losses.values()])
        loss.backward()

        grad_has_inf_nan = False
        if clip_grad is not None:
            grad_norm = clip_grad(params)
            grad_has_inf_nan = grad_norm.isinf() or grad_norm.isnan()
            if use_ddp:
                dist.reduce(grad_norm, dst=0)
            if rank == 0:
                if grad_norm.isinf():
                    print('Epoch', epoch, 'Step', i + 1, 'Gradient contains Inf')
                elif grad_norm.isnan():
                    print('Epoch', epoch, 'Step', i + 1, 'Gradient contains NaN')
                else:
                    writer.add_scalar(
                        'train/gradient_norm', grad_norm.item() / ws, global_step
                    )
        if not grad_has_inf_nan:
            optimizer.step()
        step_time += time() - start
        start = time()

        losses['all_loss'] = (1, loss.detach())
        for key in losses:
            val = losses[key][1]
            key = f'train_{key}' if '/' in key else f'train/{key}'
            write_scalar_ddp(
                writer, key, val, global_step, rank, ws, use_ddp
            )

        iou = {}
        if 'grasping_masks' in outputs:
            pred = outputs['matched_grasping_masks']
            target = data['grasping_masks']
            is_pick = torch.where(data['task_is_pick'])[0]
            pred = [pred[j] for j in is_pick]
            target = [target[j] for j in is_pick]
            if len(pred) > 0:
                iou['grasp'] = compute_iou(pred, target)
            else:
                iou['grasp'] = {
                    'scene': torch.tensor(torch.nan).to(rank),
                    'object': torch.tensor(torch.nan).to(rank)
                }
        if 'placement_masks' in outputs:
            pred = outputs['placement_masks'][data['task_is_place']]
            target = data['placement_masks'][data['task_is_place']]
            loss_masks = data['placement_region'][data['task_is_place']] > 0
            if len(pred) > 0:
                iou['place'] = compute_iou(pred, target, loss_masks=loss_masks)
            else:
                iou['place'] = {'scene': torch.tensor(torch.nan).to(rank)}
        for key in iou:
            for subkey in iou[key]:
                write_scalar_ddp(
                    writer, f'train/IoU_{key}_{subkey}', iou[key][subkey],
                    global_step, rank, ws, use_ddp
                )

        for key in ['pred', 'gt']:
            for task in ['grasps', 'placements']:
                if f'num_{key}_{task}' in outputs:
                    num = outputs[f'num_{key}_{task}']
                    if task == 'grasp' and 'task_is_pick' in data:
                        num = torch.div(
                            num, data['task_is_pick'].float().mean()
                        )
                    if task == 'placements' and 'task_is_place' in data:
                        num = torch.div(
                            num, data['task_is_place'].float().mean()
                        )
                    write_scalar_ddp(
                        writer, f'train/num_{key}_{task}', num,
                        global_step, rank, ws, use_ddp
                    )

        keys = [
            f'{key}_{subkey}_ratio' for key in [
                'grasping', 'placement'
            ] for subkey in [
                'topk_pred_pos', 'topk_gt_pos', 'topk_hard_neg'
            ]
        ]
        for key in keys:
            if key in outputs:
                write_scalar_ddp(
                    writer, f'train/{key}', outputs[key],
                    global_step, rank, ws, use_ddp
                )

        if 'grasps' in outputs:
            num_both, num_pred, num_gt = 0, 0, 0
            for masks, gt_grasps in zip(
                outputs['matched_grasping_masks'], data['grasps']
            ):
                for mask, gt_grasp in zip(masks, gt_grasps):
                    if (mask > 0).sum() > 0:
                        num_pred += 1
                        if gt_grasp.shape[0] > 0:
                            num_both += 1
                    if gt_grasp.shape[0] > 0:
                        num_gt += 1
            write_scalar_ddp(
                writer, f'train/object_precision',
                torch.div(num_both, num_pred).to(rank),
                global_step, rank, ws, use_ddp
            )
            write_scalar_ddp(
                writer, f'train/object_recall',
                torch.div(num_both, num_gt).to(rank),
                global_step, rank, ws, use_ddp
            )

        if (i + 1) % cfg.train.plot_freq == 0 and rank == 0:
            to_cpu(outputs)
            to_cpu(data)
            figs = plot_fn(outputs, data)
            for key in figs:
                writer.add_figure(f'train/{key}', figs[key], global_step)
                plt.close(figs[key])

        log_time += time() - start
        start = time()

        if (i + 1) % cfg.train.print_freq == 0:
            if use_ddp:
                dist.reduce(data_time, 0)
                dist.reduce(step_time, 0)
                dist.reduce(log_time, 0)
            data_time = data_time.item() / ws / cfg.train.print_freq
            step_time = step_time.item() / ws / cfg.train.print_freq
            log_time = log_time.item() / ws / cfg.train.print_freq
            if rank == 0:
                print(
                    f'Train Epoch {epoch:02d}  {(i+1):04d}/{num_steps:04d}  '
                    f'Data time {data_time:.4f}  Forward time {step_time:.4f}'
                    f'  Logging time {log_time:.4f}'
                )
            data_time = torch.tensor(0., device=rank)
            step_time = torch.tensor(0., device=rank)
            log_time = torch.tensor(0., device=rank)

    return global_step


def eval_one_epoch(loader, model, writer, epoch, global_step, plot_fn, cfg):
    rank = 0
    ws = 1
    use_ddp = dist.is_available() and cfg.train.num_gpus > 1
    if use_ddp:
        rank = dist.get_rank()
        ws = dist.get_world_size()
    num_steps = len(loader)
    num_plots = num_steps // cfg.train.plot_freq
    plot_ids = torch.randperm(num_steps)[:num_plots]
    data_time = torch.tensor(0., device=rank)
    step_time = torch.tensor(0., device=rank)
    log_time = torch.tensor(0., device=rank)
    total = {
        'objects': torch.tensor(0., device=rank),
        'task_is_pick': torch.tensor(0., device=rank),
        'task_is_place': torch.tensor(0., device=rank),
        'both_has_grasp': torch.tensor(0., device=rank),
        'pred_has_grasp': torch.tensor(0., device=rank),
        'gt_has_grasp': torch.tensor(0., device=rank)
    }
    loss_dict, stats = {}, {}
    start = time()
    for i, data in enumerate(loader):
        if 'grasping_masks' in data:
            num_obj = sum([len(masks) for masks in data['grasping_masks']])
            total['objects'] += num_obj
            if 'task_is_pick' in data:
                num_pick = data['task_is_pick'].sum()
                total['task_is_pick'] += num_pick
            else:
                num_pick = len(data['grasping_masks'])
                total['task_is_pick'] += num_pick
        if 'placement_masks' in data:
            if 'task_is_place' in data:
                num_place = data['task_is_place'].sum()
                total['task_is_place'] += num_place
            else:
                num_place = data['placement_masks'].shape[0]
                total['task_is_place'] += num_place
        to_gpu(data)
        data_time += time() - start
        start = time()

        with torch.no_grad():
            outputs, losses = model(data, cfg.train)
            loss = sum([w * v for w, v in losses.values()])
            losses['all_loss'] = (1, loss.detach())
        step_time += time() - start
        start = time()

        for key in losses:
            add_to_dict(loss_dict, key, losses[key][1])

        if 'grasping_masks' in outputs:
            pred = outputs['matched_grasping_masks']
            target = data['grasping_masks']
            is_pick = torch.where(data['task_is_pick'])[0]
            pred = [pred[j] for j in is_pick]
            target = [target[j] for j in is_pick]
            if len(pred) > 0:
                iou = compute_iou(pred, target)
                for key in iou:
                    num = num_obj if key == 'object' else num_pick
                    add_to_dict(stats, f'IoU_{key}', iou[key] * num)
        if 'placement_masks' in outputs:
            pred = outputs['placement_masks'][data['task_is_place']]
            target = data['placement_masks'][data['task_is_place']]
            loss_masks = data['placement_region'][data['task_is_place']] > 0
            if len(pred) > 0:
                iou = compute_iou(pred, target, loss_masks=loss_masks)
                add_to_dict(
                    stats, 'IoU_placement_scene', iou['scene'] * num_place
                )
        for task in ['grasps', 'placements']:
            for key in ['pred', 'gt']:
                if f'num_{key}_{task}' in outputs:
                    num = num_place if task == 'placements' else num_obj
                    add_to_dict(
                        stats, f'num_{key}_{task}',
                        outputs[f'num_{key}_{task}'] * num
                    )

        keys = [
            f'{key}_mask_{subkey}_ratio' for key in [
                'grasping', 'placement'
            ] for subkey in [
                'topk_pred_pos', 'topk_gt_pos', 'topk_true_neg'
            ]
        ]
        for key in keys:
            if key in outputs:
                num = num_place if 'placement' in key else num_obj
                add_to_dict(stats, key, outputs[key] * num)

        if 'grasps' in outputs:
            for masks, gt_grasps in zip(
                outputs['matched_grasping_masks'], data['grasps']
            ):
                for mask, gt_grasp in zip(masks, gt_grasps):
                    if (mask > 0).sum() > 0:
                        total['pred_has_grasp'] += 1
                        if gt_grasp.shape[0] > 0:
                            total['both_has_grasp'] += 1
                    if gt_grasp.shape[0] > 0:
                        total['gt_has_grasp'] += 1

        if i in plot_ids and rank == 0:
            to_cpu(outputs)
            to_cpu(data)
            figs = plot_fn(outputs, data)
            for key in figs:
                writer.add_figure(f'valid/{key}', figs[key], global_step + i)
                plt.close(figs[key])

        log_time += time() - start
        start = time()

        if (i + 1) % cfg.train.print_freq == 0:
            if use_ddp:
                dist.reduce(data_time, 0)
                dist.reduce(step_time, 0)
                dist.reduce(log_time, 0)
            if rank == 0:
                data_time = data_time.item() / ws / cfg.train.print_freq
                step_time = step_time.item() / ws / cfg.train.print_freq
                log_time = log_time.item() / ws / cfg.train.print_freq
                print(
                    f'Valid Epoch {epoch:02d}  {(i+1):04d}/{num_steps:04d}  '
                    f'Data time {data_time:.4f}  Forward time {step_time:.4f}'
                    f'  Logging time {log_time:.4f}'
                )
            data_time = torch.tensor(0., device=rank)
            step_time = torch.tensor(0., device=rank)
            log_time = torch.tensor(0., device=rank)

    total['steps'] = torch.tensor(i + 1, device=rank)
    if use_ddp:
        for key in total:
            dist.reduce(total[key], 0)
    if 'grasps' in outputs:
        write_scalar_ddp(
            writer, f'valid/object_precision', total['both_has_grasp'],
            global_step, rank, total['pred_has_grasp']
        )
        write_scalar_ddp(
            writer, f'valid/object_recall', total['both_has_grasp'],
            global_step, rank, total['gt_has_grasp']
        )
    for key, val in loss_dict.items():
        write_scalar_ddp(
            writer, f'valid/{key}', val, global_step,
            rank, total['steps'], use_ddp
        )
    for key, val in stats.items():
        if 'placement' in key:
            num = total['task_is_place']
        elif 'scene' in key:
            num = total['task_is_pick']
        else:
            num = total['objects']
        write_scalar_ddp(
            writer, f'valid/{key}', val, global_step, rank, num, use_ddp
        )


def train(rank, cfg):
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    use_ddp = dist.is_available() and cfg.train.num_gpus > 1
    if use_ddp:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = cfg.train.port
        dist.init_process_group(
            "nccl", rank=rank, world_size=cfg.train.num_gpus
        )
        torch.cuda.set_device(rank)

    train_sampler, train_loader = get_data_loader(
        cfg, 'train', use_ddp, training=True
    )
    valid_sampler, valid_loader = get_data_loader(
        cfg, 'test', use_ddp, training=False
    )
    plot_fn = partial(plot_3D, world_coord=cfg.data.world_coord)

    model = M2T2.from_config(cfg.m2t2).to(rank)
    optimizer = build_optimizer(cfg, model)

    init_epoch = 0
    if cfg.train.checkpoint is not None:
        ckpt = torch.load(cfg.train.checkpoint, map_location="cpu")
        init_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    if use_ddp:
        # https://github.com/Lightning-AI/lightning/issues/6789
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])

    clip_grad = None
    if cfg.optimizer.grad_clip > 0:
        clip_grad = partial(clip_grad_norm_, max_norm=cfg.optimizer.grad_clip)

    writer = None
    if rank == 0:
        writer = SummaryWriter(cfg.train.log_dir)
    if rank == 0:
        with open(f"{cfg.train.log_dir}/config.yaml", 'w') as f:
            OmegaConf.save(cfg, f)

    global_step = init_epoch * len(train_loader)
    start = time()
    for epoch in range(init_epoch, cfg.train.num_epochs):
        model.train()
        if use_ddp:
            train_sampler.set_epoch(epoch)
        global_step = train_one_epoch(
            train_loader, model, optimizer, clip_grad, writer,
            epoch + 1, global_step, plot_fn, cfg
        )

        if (epoch + 1) % cfg.train.save_freq == 0 and rank == 0:
            save_model(
                epoch + 1, model, optimizer, cfg.train.log_dir, use_ddp
            )

        model.eval()
        if use_ddp:
            valid_sampler.set_epoch(epoch)
        eval_one_epoch(
            valid_loader, model, writer, epoch + 1, global_step, plot_fn, cfg
        )

    total_time = torch.tensor(time() - start).to(rank)
    if use_ddp:
        dist.reduce(total_time, 0)
    total_time = total_time.item() / cfg.train.num_gpus
    if rank == 0:
        print('Total training time', timedelta(seconds=total_time))


@hydra.main(config_path='.', config_name='config', version_base='1.3')
def main(cfg: DictConfig) -> None:
    mp.spawn(
        train,
        args=(cfg,),
        nprocs=cfg.train.num_gpus,
        join=True
    )

if __name__ == '__main__':
    main()
