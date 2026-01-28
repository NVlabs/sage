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
Utility functions for data preprocessing.
'''
from torchvision import transforms
import numpy as np
import torch


class NormalizeInverse(transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


normalize_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


denormalize_rgb = transforms.Compose([
    NormalizeInverse(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def depth_to_xyz(depth, intrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    Z = depth
    X = (u - cx) * (Z / fx)
    Y = (v - cy) * (Z / fy)
    xyz = np.stack((X, Y, Z), axis=-1)
    return xyz


def jitter_gaussian(xyz, std, clip):
    return xyz + torch.clip(
        torch.randn_like(xyz) * std, -clip, clip
    )


def sample_points(xyz, num_points):
    num_replica = num_points // xyz.shape[0]
    num_remain = num_points % xyz.shape[0]
    pt_idx = torch.randperm(xyz.shape[0])
    pt_idx = torch.cat(
        [pt_idx for _ in range(num_replica)] + [pt_idx[:num_remain]]
    )
    return pt_idx
