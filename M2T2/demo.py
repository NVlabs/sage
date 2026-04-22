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
Demo script that shows data loading and model inference.
'''
import hydra
import numpy as np
import torch

from m2t2.dataset import load_rgb_xyz, collate
from m2t2.dataset_utils import denormalize_rgb, sample_points
from m2t2.meshcat_utils import (
    create_visualizer, make_frame, visualize_grasp, visualize_pointcloud
)
from m2t2.m2t2 import M2T2
from m2t2.plot_utils import get_set_colors
from m2t2.train_utils import to_cpu, to_gpu
import open3d as o3d
import os
import pickle

def load_and_predict(data_dir, cfg):
    data, meta_data = load_rgb_xyz(
        data_dir, cfg.data.robot_prob,
        cfg.data.world_coord, cfg.data.jitter_scale,
        cfg.data.grid_resolution, cfg.eval.surface_range
    )
    if 'object_label' in meta_data:
        data['task'] = 'place'
    else:
        data['task'] = 'pick'

    model = M2T2.from_config(cfg.m2t2)
    ckpt = torch.load(cfg.eval.checkpoint)
    model.load_state_dict(ckpt['model'])
    model = model.cuda().eval()

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
    return data, outputs

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

@hydra.main(config_path='.', config_name='config', version_base='1.3')
def main(cfg):
    data, outputs = load_and_predict(cfg.eval.data_dir, cfg)

    vis = create_visualizer()
    rgb = denormalize_rgb(
        data['inputs'][:, 3:].T.unsqueeze(2)
    ).squeeze(2).T
    rgb = (rgb.numpy() * 255).astype('uint8')
    xyz = data['points'].numpy()
    cam_pose = data['cam_pose'].double().numpy()
    make_frame(vis, 'camera', T=cam_pose)
    if not cfg.eval.world_coord:
        xyz = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]

    # save xyz and rgb to ply file using open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    o3d.io.write_point_cloud(os.path.join(cfg.eval.data_dir, "scene.ply"), pcd)

    with open(os.path.join(cfg.eval.data_dir, "m2t2_outputs.pkl"), "wb") as f:
        pickle.dump(outputs, f)

    visualize_pointcloud(vis, 'scene', xyz, rgb, size=0.005)
    if data['task'] == 'pick':
        for i, (grasps, conf, contacts, color) in enumerate(zip(
            outputs['grasps'],
            outputs['grasp_confidence'],
            outputs['grasp_contacts'],
            get_set_colors()
        )):
            print(f"object_{i:02d} has {grasps.shape[0]} grasps")
            conf_filter = conf > 0.9
            grasps = grasps[conf_filter]
            conf = conf[conf_filter]
            contacts = contacts[conf_filter]

            conf = conf.numpy()

            conf_colors = (np.stack([
                1 - conf, conf, np.zeros_like(conf)
            ], axis=1) * 255).astype('uint8')
            visualize_pointcloud(
                vis, f"object_{i:02d}/contacts",
                contacts.numpy(), conf_colors, size=0.01
            )
            grasps = grasps.numpy()
            if not cfg.eval.world_coord:
                grasps = cam_pose @ grasps

            # transform grasps_vis_points according to the grasp

            grasps_vis_points_list = []

            for j, grasp in enumerate(grasps):
                visualize_grasp(
                    vis, f"object_{i:02d}/grasps/{j:03d}",
                    grasp, color, linewidth=0.2
                )
                grasps_vis_points = load_grasp_points_dense()[:, :3] # shape (N, 3)
                grasps_vis_points = grasps_vis_points @ grasp[:3, :3].T + grasp[:3, 3]
                grasps_vis_points = grasps_vis_points.reshape(-1, 3)
                grasps_vis_points_list.append(grasps_vis_points)

            grasps_vis_points_all = np.concatenate(grasps_vis_points_list, axis=0)

            # grasps_vis_points store to ply file with open3d with random color
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(grasps_vis_points_all)
            color_i = [np.random.rand(), np.random.rand(), np.random.rand()]
            pcd.colors = o3d.utility.Vector3dVector([color_i for _ in range(grasps_vis_points_all.shape[0])])
            o3d.io.write_point_cloud(os.path.join(cfg.eval.data_dir, f"object_{i:02d}_grasps.ply"), pcd)
    elif data['task'] == 'place':
        ee_pose = data['ee_pose'].double().numpy()
        make_frame(vis, 'ee', T=ee_pose)
        obj_xyz_ee, obj_rgb = data['object_inputs'].split([3, 3], dim=1)
        obj_xyz_ee = (obj_xyz_ee + data['object_center']).numpy()
        obj_xyz = obj_xyz_ee @ ee_pose[:3, :3].T + ee_pose[:3, 3]
        obj_rgb = denormalize_rgb(obj_rgb.T.unsqueeze(2)).squeeze(2).T
        obj_rgb = (obj_rgb.numpy() * 255).astype('uint8')
        visualize_pointcloud(vis, 'object', obj_xyz, obj_rgb, size=0.005)
        for i, (placements, conf, contacts) in enumerate(zip(
            outputs['placements'],
            outputs['placement_confidence'],
            outputs['placement_contacts'],
        )):
            print(f"orientation_{i:02d} has {placements.shape[0]} placements")
            conf = conf.numpy()
            conf_colors = (np.stack([
                1 - conf, conf, np.zeros_like(conf)
            ], axis=1) * 255).astype('uint8')
            visualize_pointcloud(
                vis, f"orientation_{i:02d}/contacts",
                contacts.numpy(), conf_colors, size=0.01
            )
            placements = placements.numpy()
            if not cfg.eval.world_coord:
                placements = cam_pose @ placements
            visited = np.zeros((0, 3))
            for j, k in enumerate(np.random.permutation(placements.shape[0])):
                if visited.shape[0] > 0:
                    dist = np.sqrt((
                        (placements[k, :3, 3] - visited) ** 2
                    ).sum(axis=1))
                    if dist.min() < cfg.eval.placement_vis_radius:
                        continue
                visited = np.concatenate([visited, placements[k:k+1, :3, 3]])
                visualize_grasp(
                    vis, f"orientation_{i:02d}/placements/{j:02d}/gripper",
                    placements[k], [0, 255, 0], linewidth=0.2
                )
                obj_xyz_placed = obj_xyz_ee @ placements[k, :3, :3].T \
                               + placements[k, :3, 3]
                visualize_pointcloud(
                    vis, f"orientation_{i:02d}/placements/{j:02d}/object",
                    obj_xyz_placed, obj_rgb, size=0.01
                )


if __name__ == '__main__':
    main()
