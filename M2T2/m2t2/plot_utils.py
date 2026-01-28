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
Utility functions for plotting during training.
'''
from matplotlib.cm import get_cmap
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import numpy as np
import torch

from m2t2.dataset_utils import denormalize_rgb


def get_set_colors():
    n_colors = [9, 8, 12]
    colors = []
    for i, n in enumerate(n_colors):
        cmap = get_cmap(f'Set{i+1}')
        for j in range(n):
            colors.append([int(c * 255) for c in cmap(j / n)[:3]])
    return colors


def plot_mask_3D(
    objectness, names, out_masks, out_argmax, tgt_masks,
    num_targets, obj_thresh, pts, rgb, depth, size
):
    seg_pred = np.zeros((size, size, 3), dtype='uint8')
    seg = np.zeros((size, size, 3), dtype='uint8')

    colors = get_set_colors()
    num_rows = int(np.ceil(out_masks.shape[0] / 4))
    fig = plt.figure(figsize=((8, num_rows * 2 + 8)))
    gs = GridSpec(num_rows + 4, 4)
    plt.subplot(gs[:2, :2])
    plt.imshow(rgb)
    plt.title('RGB')
    plt.axis('off')
    plt.subplot(gs[:2, 2:])
    plt.imshow(depth, cmap='gray')
    plt.title('Depth')
    plt.axis('off')
    for i, (out, tgt) in enumerate(zip(out_masks, tgt_masks)):
        mask = out & (out_argmax == i)
        seg_pred[pts[1, mask], pts[0, mask]] = colors[i]
        if i < num_targets:
            seg[pts[1, tgt], pts[0, tgt]] = colors[i]
        plt.subplot(gs[4 + i // 4, i % 4])
        mask_plot = np.zeros((size, size, 3))
        mask_plot[pts[1], pts[0], 0] = tgt
        mask_plot[pts[1], pts[0], 1] = out
        intersect = (out & tgt).sum()
        union = (out | tgt).sum()
        iou = intersect / union if union > 0 else 1
        c = 'g'
        if objectness[i] <= obj_thresh:
            c = 'r'
        if i >= num_targets:
            c = 'b'
        title = f'{names[i]}\nObjectness {objectness[i]:.4f}\nIoU {iou:.4f}'
        plt.title(title, color=c)
        plt.imshow(mask_plot)
        plt.axis('off')
    plt.subplot(gs[2:4, :2])
    plt.imshow(seg_pred)
    plt.title('Predicted')
    plt.axis('off')
    plt.subplot(gs[2:4, 2:])
    plt.imshow(seg)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.tight_layout()
    return fig


def plot_place_mask_3D(pts, rgb, depth, seg, out_mask, tgt_mask, size):
    colors = get_set_colors()
    seg_plot = np.zeros((size, size, 3), dtype='uint8')
    for i, j in enumerate(np.unique(seg)):
        seg_plot[pts[1, seg == j], pts[0, seg == j]] = colors[i]

    fig = plt.figure(figsize=((12, 8)))
    plt.subplot(2, 3, 1)
    plt.imshow(rgb)
    plt.title('RGB')
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.imshow(depth, cmap='gray')
    plt.title('Depth')
    plt.axis('off')
    plt.subplot(2, 3, 3)
    plt.imshow(seg_plot)
    plt.title('Segmentation')
    plt.axis('off')

    out_any = out_mask.any(axis=0)
    out_all = out_mask.all(axis=0)
    colors = np.zeros((pts.shape[1], 3))
    colors[out_any, :2] = 1
    colors[out_all, 0] = 0
    plot = np.zeros((size, size, 3))
    plot[pts[1], pts[0]] = colors
    plt.subplot(2, 3, 4)
    plt.imshow(plot)
    plt.title('Predicted')
    plt.axis('off')

    tgt_any = tgt_mask.any(axis=0)
    tgt_all = tgt_mask.all(axis=0)
    colors = np.zeros((pts.shape[1], 3))
    colors[tgt_any, :2] = 1
    colors[tgt_all, 0] = 0
    plot = np.zeros((size, size, 3))
    plot[pts[1], pts[0]] = colors
    plt.subplot(2, 3, 5)
    plt.imshow(plot)
    plt.title('Ground truth')
    plt.axis('off')

    correct = out_mask == tgt_mask
    correct_all = correct.all(axis=0)
    colors = np.zeros((pts.shape[1], 3))
    colors[tgt_any & ~out_any, 0] = 1
    colors[out_any & ~tgt_any, 2] = 1
    colors[out_any & tgt_any, 1] = 1
    colors[correct_all & tgt_any, 0] = 1
    plot = np.zeros((size, size, 3))
    plot[pts[1], pts[0]] = colors
    inter = (out_mask & tgt_mask).sum(axis=1)
    union = (out_mask | tgt_mask).sum(axis=1)
    iou = np.nan_to_num(np.divide(inter, union), nan=1).mean()
    plt.subplot(2, 3, 6)
    plt.imshow(plot)
    plt.title(f'Intersection IoU {iou:.4f}')
    plt.axis('off')
    plt.tight_layout()
    return fig


def plot_3D(
    outputs, data, obj_thresh=0.5, mask_thresh=0, world_coord=False,
    scale=0.25, size=128, f=443.404968, c=256
):
    def plot_rgb_depth(batch_idx):
        pts = data['points'][batch_idx]
        if world_coord:
            world2cam = data['cam_pose'][batch_idx].inverse()
            pts = pts @ world2cam[:3, :3].T + world2cam[:3, 3]
        depth = pts[:, 2]
        pts = ((pts.T[:2] / pts.T[2:] * f + c) * scale).int()
        pts = torch.clamp(pts, 0, size - 1).numpy()
        rgb = data['inputs'][batch_idx, :, 3:]
        rgb = denormalize_rgb(rgb.T.unsqueeze(1)).squeeze(1).T.numpy()
        rgb_plot = np.zeros((size, size, 3))
        rgb_plot[pts[1], pts[0]] = np.clip(rgb, 0, 1)
        depth_plot = np.zeros((size, size))
        depth_plot[pts[1], pts[0]] = depth
        return pts, rgb_plot, depth_plot

    figs = {}
    batch_idx = 0
    while not data['task_is_place'][batch_idx]:
        batch_idx += 1
        if batch_idx == outputs['placement_masks'].shape[0]:
            break
    if batch_idx < outputs['placement_masks'].shape[0]:
        pts, rgb_plot, depth_plot = plot_rgb_depth(batch_idx)
        seg = data['seg'][batch_idx]
        out_mask = outputs['placement_masks'][batch_idx]
        out_mask = out_mask.numpy() > mask_thresh
        tgt_mask = data['placement_masks'][batch_idx]
        tgt_mask = tgt_mask.numpy() > 0
        figs['placement'] = plot_place_mask_3D(
            pts, rgb_plot, depth_plot, seg, out_mask, tgt_mask, size
        )

    batch_idx = 0
    while not data['task_is_pick'][batch_idx]:
        batch_idx += 1
        if batch_idx == outputs['grasping_masks'].shape[0]:
            break
    if batch_idx < outputs['grasping_masks'].shape[0]:
        pts, rgb_plot, depth_plot = plot_rgb_depth(batch_idx)
        objness = outputs['objectness'][batch_idx]
        objness = objness[outputs['matched_idx'][batch_idx]]
        out_masks = outputs['matched_grasping_masks'][batch_idx]
        out_argmax = out_masks.argmax(dim=0).numpy()
        out_masks = out_masks.numpy() > mask_thresh
        tgt_masks = data['grasping_masks'][batch_idx]
        names = data['names'][batch_idx]
        num_targets = len(names)
        tgt_masks = tgt_masks.numpy() > 0
        figs['grasping'] = plot_mask_3D(
            objness, names, out_masks, out_argmax, tgt_masks,
            num_targets, obj_thresh, pts, rgb_plot, depth_plot, size
        )
    return figs
