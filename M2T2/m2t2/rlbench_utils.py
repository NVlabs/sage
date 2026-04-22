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
Utility functions for loading RLBench data.
'''
from PIL import Image
import numpy as np
import trimesh.transformations as tra

from m2t2.dataset_utils import depth_to_xyz


def load_image(episode_dir, camera, meta_data, frame_id):
    rgb = np.array(
        Image.open(f"{episode_dir}/{camera}_rgb/{frame_id}.png")
    )
    seg = np.array(
        Image.open(f"{episode_dir}/{camera}_mask/{frame_id}.png")
    )[..., 0]
    depth = np.array(
        Image.open(f"{episode_dir}/{camera}_depth/{frame_id}.png")
    )
    depth = np.sum(depth * [65536, 256, 1], axis=2)
    near = meta_data[f'{camera}_camera_near']
    far = meta_data[f'{camera}_camera_far']
    depth = near + depth / (2**24 - 1) * (far - near)
    pcd = depth_to_xyz(depth, meta_data[f'{camera}_camera_intrinsics'])
    cam_pose = meta_data[f'{camera}_camera_extrinsics'][frame_id]
    pcd = pcd @ cam_pose[:3, :3].T + cam_pose[:3, 3]
    return rgb, pcd, seg


def within_bound(demo, cameras, bounds):
    pcds, rgbs, masks = [], [], []
    for camera in cameras:
        pcd = demo[f'{camera}_point_cloud']
        rgb = demo[f'{camera}_rgb']
        pcds.append(pcd.reshape(-1, 3))
        rgbs.append(rgb.reshape(-1, 3))
        masks.append(demo[f'{camera}_mask'].reshape(-1))
    pcd = np.concatenate(pcds)
    rgb = np.concatenate(rgbs)
    mask = np.concatenate(masks)
    within = (pcd[:, 0] > bounds[0]) & (pcd[:, 0] < bounds[3]) \
           & (pcd[:, 1] > bounds[1]) & (pcd[:, 1] < bounds[4]) \
           & (pcd[:, 2] > bounds[2]) & (pcd[:, 2] < bounds[5])
    return pcd[within], rgb[within], mask[within]


def gripper_pose_from_rlbench(pose, gripper_depth=0.1034):
    pose = pose @ tra.euler_matrix(0, 0, np.pi / 2)
    pose[:3, 3] -= gripper_depth * pose[:3, 2]
    return pose


def gripper_pose_to_rlbench(pose, gripper_depth=0.1034):
    pose_out = pose.copy()
    pose_out[:3, 3] += gripper_depth * pose[:3, 2]
    pose_out = pose_out @ tra.euler_matrix(0, 0, -np.pi / 2)
    return pose_out
