# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import sys
import os
import hydra
import numpy as np
import torch
import m2t2
from m2t2.dataset import load_rgb_xyz, collate
from m2t2.dataset_utils import denormalize_rgb, sample_points
from m2t2.meshcat_utils import (
    create_visualizer, make_frame, visualize_grasp, visualize_pointcloud
)
from m2t2.m2t2 import M2T2
from m2t2.plot_utils import get_set_colors
from m2t2.train_utils import to_cpu, to_gpu

from m2t2.dataset_utils import (
    depth_to_xyz, jitter_gaussian, normalize_rgb, sample_points
)

import open3d as o3d
import os
import pickle
from omegaconf import OmegaConf
from constants import M2T2_ROOT_DIR
cfg_path_default = os.path.join(M2T2_ROOT_DIR, "config.yaml")
m2t2_model_path = os.path.join(M2T2_ROOT_DIR, "m2t2.pth")

def load_m2t2(cfg_path=cfg_path_default):
    cfg = OmegaConf.load(cfg_path)
    cfg.eval.checkpoint = m2t2_model_path
    cfg.eval.mask_thresh = 0.4
    cfg.eval.num_runs = 5

    model = M2T2.from_config(cfg.m2t2)
    ckpt = torch.load(cfg.eval.checkpoint)
    model.load_state_dict(ckpt['model'])
    model = model.cuda().eval()

    return model, cfg

def load_grasp_points_dense():
    control_points = np.array([
        [ 0.05268743, -0.00005996, 0.05900000, 1.00000000],
        [-0.05268743,  0.00005996, 0.05900000, 1.00000000],
        [ 0.05268743, -0.00005996, 0.10527314, 1.00000000],
        [-0.05268743,  0.00005996, 0.10527314, 1.00000000]
    ])
    mid_point = (control_points[0] + control_points[1]) / 2

    grasp_pc = [
        control_points[-2], control_points[0], mid_point,
        [0, 0, 0, 1], mid_point, control_points[1], control_points[-1]
    ]
    
    # Convert to numpy array for easier manipulation
    grasp_pc = np.array(grasp_pc, dtype=np.float32)
    
    # Densify by interpolating between adjacent points
    dense_grasp_pc = []
    num_interpolations = 10  # Number of points to interpolate between each pair
    
    for i in range(len(grasp_pc)):
        # Add the current point
        dense_grasp_pc.append(grasp_pc[i])
        
        # Add interpolated points between current and next point (except for the last point)
        if i < len(grasp_pc) - 1:
            current_point = grasp_pc[i]
            next_point = grasp_pc[i + 1]
            
            # Create interpolated points
            for j in range(1, num_interpolations + 1):
                t = j / (num_interpolations + 1)  # Interpolation parameter from 0 to 1
                interpolated_point = (1 - t) * current_point + t * next_point
                dense_grasp_pc.append(interpolated_point)
    
    return np.array(dense_grasp_pc, dtype=np.float32)


def infer_m2t2(meta_data, vis_data, model, cfg, return_contacts=False):
    
    rgb = normalize_rgb(vis_data['rgb']).permute(1, 2, 0)
    depth = vis_data['depth']
    xyz = torch.from_numpy(
        depth_to_xyz(depth, meta_data['intrinsics'])
    ).float()
    seg = torch.from_numpy(np.array(vis_data['seg']))
    label_map = meta_data['label_map']

    xyz, rgb, seg = xyz[depth > 0], rgb[depth > 0], seg[depth > 0]
    cam_pose = torch.from_numpy(meta_data['camera_pose']).float()
    xyz_world = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]

    robot_prob = cfg.data.robot_prob
    world_coord = cfg.data.world_coord
    jitter_scale = cfg.data.jitter_scale
    grid_res = cfg.data.grid_resolution
    surface_range = cfg.eval.surface_range

    if 'scene_bounds' in meta_data:
        bounds = meta_data['scene_bounds']
        within = (xyz_world[:, 0] > bounds[0]) & (xyz_world[:, 0] < bounds[3]) \
            & (xyz_world[:, 1] > bounds[1]) & (xyz_world[:, 1] < bounds[4]) \
            & (xyz_world[:, 2] > bounds[2]) & (xyz_world[:, 2] < bounds[5])
        xyz_world, rgb, seg = xyz_world[within], rgb[within], seg[within]
        # Set z-coordinate of all points near table to 0
        xyz_world[np.abs(xyz_world[:, 2]) < surface_range, 2] = 0
        if not world_coord:
            world2cam = cam_pose.inverse()
            xyz = xyz_world @ world2cam[:3, :3].T + world2cam[:3, 3]
    if world_coord:
        xyz = xyz_world

    if jitter_scale > 0:
        table_mask = seg == label_map['table']
        if 'robot_table' in label_map:
            table_mask |= seg == label_map['robot_table']
        xyz[table_mask] = jitter_gaussian(
            xyz[table_mask], jitter_scale, jitter_scale
        )

    data = {
        'inputs': torch.cat([xyz - xyz.mean(dim=0), rgb], dim=1),
        'points': xyz,
        'seg': seg,
        'cam_pose': cam_pose
    }

    data.update({
        'object_inputs': torch.rand(1024, 6),
        'ee_pose': torch.eye(4),
        'bottom_center': torch.zeros(3),
        'object_center': torch.zeros(3)
    })

    inputs, xyz, seg = data['inputs'], data['points'], data['seg']
    obj_inputs = data['object_inputs']
    outputs = {
        'grasps': [],
        'grasp_confidence': [],
        'grasp_contacts': [],
        'placements': [],
        'placement_confidence': [],
        'placement_contacts': []
    }
    for _ in range(cfg.eval.num_runs):
        pt_idx = sample_points(xyz, cfg.data.num_points)
        data['inputs'] = inputs[pt_idx]
        data['points'] = xyz[pt_idx]
        data['seg'] = seg[pt_idx]
        pt_idx = sample_points(obj_inputs, cfg.data.num_object_points)
        data['object_inputs'] = obj_inputs[pt_idx]
        data_batch = collate([data])
        to_gpu(data_batch)

        with torch.no_grad():
            model_ouputs = model.infer(data_batch, cfg.eval)
        to_cpu(model_ouputs)
        for key in outputs:
            if 'place' in key and len(outputs[key]) > 0:
                outputs[key] = [
                    torch.cat([prev, cur])
                    for prev, cur in zip(outputs[key], model_ouputs[key][0])
                ]
            else:
                outputs[key].extend(model_ouputs[key][0])
    data['inputs'], data['points'], data['seg'] = inputs, xyz, seg
    data['object_inputs'] = obj_inputs

    xyz = data['points'].numpy()
    cam_pose = data['cam_pose'].double().numpy()
    if not cfg.eval.world_coord:
        xyz = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]

    all_grasps = []
    all_confs = []
    all_contacts = []

    # # debug:
    # debug_dir = os.path.join(SERVER_ROOT_DIR, "test/m2t2_data")


    # # debug: save xyz and rgb to ply file using open3d
    # rgb = denormalize_rgb(
    #     data['inputs'][:, 3:].T.unsqueeze(2)
    # ).squeeze(2).T

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    # o3d.io.write_point_cloud(os.path.join(debug_dir, "scene.ply"), pcd)

    for i, (grasps, conf, contacts, color) in enumerate(zip(
        outputs['grasps'],
        outputs['grasp_confidence'],
        outputs['grasp_contacts'],
        get_set_colors()
    )):
        conf_filter = conf > 0.1
        grasps = grasps[conf_filter]
        conf = conf[conf_filter]
        contacts = contacts[conf_filter]
        print(f"object_{i:02d} has {grasps.shape[0]} grasps")

        conf = conf.numpy()

        grasps = grasps.numpy()
        if not cfg.eval.world_coord:
            grasps = cam_pose @ grasps

        all_grasps.append(grasps)
        all_confs.append(conf)
        all_contacts.append(contacts)
        # # debug: transform grasps_vis_points according to the grasp
        # grasps_vis_points_list = []

        # for j, grasp in enumerate(grasps):

        #     grasps_vis_points = load_grasp_points_dense()[:, :3] # shape (N, 3)
        #     grasps_vis_points = grasps_vis_points @ grasp[:3, :3].T + grasp[:3, 3]
        #     grasps_vis_points = grasps_vis_points.reshape(-1, 3)
        #     grasps_vis_points_list.append(grasps_vis_points)

        # grasps_vis_points_all = np.concatenate(grasps_vis_points_list, axis=0)

        # # grasps_vis_points store to ply file with open3d with random color
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(grasps_vis_points_all)
        # color_i = [np.random.rand(), np.random.rand(), np.random.rand()]
        # pcd.colors = o3d.utility.Vector3dVector([color_i for _ in range(grasps_vis_points_all.shape[0])])
        # o3d.io.write_point_cloud(os.path.join(debug_dir, f"object_{i:02d}_grasps.ply"), pcd)
    
    if len(all_grasps) == 0:
        if return_contacts:
            return np.zeros((0, 4, 4)), np.zeros((0, 3))
        return np.zeros((0, 4, 4))
    

    all_grasps = np.concatenate(all_grasps, axis=0)
    all_confs = np.concatenate(all_confs, axis=0)
    all_contacts = np.concatenate(all_contacts, axis=0)
    all_grasps = all_grasps[np.argsort(all_confs)[::-1]]
    all_contacts = all_contacts[np.argsort(all_confs)[::-1]]
    
    # if number of grasps is over 10, select the top 10 conf grasps
    # if all_grasps.shape[0] > 10:
    #     top_10_conf_idx = np.argsort(all_confs)[-10:]
    #     all_grasps = all_grasps[top_10_conf_idx]
    #     all_confs = all_confs[top_10_conf_idx]

    if return_contacts:
        return all_grasps, all_contacts

    return all_grasps

if __name__ == "__main__":
    load_m2t2()