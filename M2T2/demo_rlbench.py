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
Demo script showing prediction for language-conditioned tasks.
'''
import hydra
import pickle
import torch

from m2t2.dataset import collate
from m2t2.dataset_utils import normalize_rgb, sample_points
from m2t2.meshcat_utils import (
    create_visualizer, visualize_grasp, visualize_pointcloud
)
from m2t2.m2t2 import M2T2
from m2t2.rlbench_utils import (
    load_image, within_bound, gripper_pose_from_rlbench
)
from m2t2.train_utils import to_cpu, to_gpu


def load_data(episode_dir, cfg):
    with open(f"{episode_dir}/meta_data.pkl", 'rb') as f:
        meta_data = pickle.load(f)
    data = {}
    for camera in cfg.rlbench.cameras:
        rgb, xyz, mask = load_image(
            episode_dir, camera, meta_data, cfg.rlbench.frame_id
        )
        data[f"{camera}_rgb"] = rgb
        data[f"{camera}_point_cloud"] = xyz
        data[f"{camera}_mask"] = mask
    pcd_raw, rgb_raw, seg_raw = within_bound(
        data, cfg.rlbench.cameras, cfg.rlbench.scene_bounds
    )
    rgb = normalize_rgb(rgb_raw[:, None]).squeeze(2).T
    pcd = torch.from_numpy(pcd_raw).float()
    pt_idx = sample_points(pcd_raw, cfg.data.num_points)
    pcd, rgb = pcd[pt_idx], rgb[pt_idx]
    with open(cfg.rlbench.lang_emb_path, 'rb') as f:
        lang_emb = pickle.load(f)
    model_inputs = {
        'inputs': torch.cat([pcd - pcd.mean(dim=0), rgb], dim=1),
        'points': pcd,
        'lang_tokens': torch.from_numpy(
            lang_emb[meta_data['goal_description']]
        ).float()
    }
    obj_label = meta_data['object_label'][cfg.rlbench.frame_id]
    if obj_label == 0:
        model_inputs.update({
            'object_inputs': torch.rand(1024, 6),
            'ee_pose': torch.eye(4),
            'bottom_center': torch.zeros(3),
            'object_center': torch.zeros(3),
            'task': 'pick'
        })
    else:
        obj_xyz = torch.from_numpy(pcd_raw[seg_raw == obj_label]).float()
        obj_rgb = torch.from_numpy(rgb_raw[seg_raw == obj_label]).float()
        obj_xyz_grid = torch.unique(
            (obj_xyz[:, :2] / cfg.data.grid_resolution).round(), dim=0
        ) * cfg.data.grid_resolution
        bottom_center = obj_xyz.min(dim=0)[0]
        bottom_center[:2] = obj_xyz_grid.mean(dim=0)
        ee_pose = torch.from_numpy(gripper_pose_from_rlbench(
            meta_data['gripper_matrix'][cfg.rlbench.frame_id]
        )).float()
        inv_ee_pose = ee_pose.inverse()
        obj_xyz = obj_xyz @ inv_ee_pose[:3, :3].T + inv_ee_pose[:3, 3]
        model_inputs.update({
            'object_inputs': torch.cat([
                obj_xyz - obj_xyz.mean(dim=0), obj_rgb
            ], dim=1),
            'ee_pose': ee_pose,
            'bottom_center': bottom_center,
            'object_center': obj_xyz.mean(dim=0),
            'task': 'place'
        })
        
    raw_data = meta_data
    raw_data.update({
        'pcd': pcd_raw, 'rgb': rgb_raw,
        'seg': seg_raw, 'object_label': obj_label
    })
    return model_inputs, raw_data


@hydra.main(config_path='.', config_name='rlbench', version_base='1.3')
def main(cfg):
    episode_dir = f"{cfg.rlbench.base_dir}/{cfg.rlbench.task_name}/episode{cfg.rlbench.episode}"
    data, raw = load_data(episode_dir, cfg)
    data_batch = collate([data])
    to_gpu(data_batch)

    model = M2T2.from_config(cfg.m2t2)
    ckpt = torch.load(cfg.eval.checkpoint)
    model.load_state_dict(ckpt['model'])
    model = model.cuda().eval()

    with torch.no_grad():
        outputs = model.infer(data_batch, cfg.eval)
    to_cpu(outputs)

    vis = create_visualizer()
    visualize_pointcloud(vis, 'scene', raw['pcd'], raw['rgb'], size=0.01)
    if raw['object_label'] != 0:
        obj_pcd = raw['pcd'][raw['seg'] == raw['object_label']]
        obj_rgb = raw['rgb'][raw['seg'] == raw['object_label']]
        visualize_pointcloud(vis, 'object', obj_pcd, obj_rgb, size=0.01)
    if data['task'] == 'pick':
        confidence = outputs['grasp_confidence'][0][0]
        conf, idx = confidence.max(dim=0)
        gripper_target = outputs['grasps'][0][0][idx]
    elif data['task'] == 'place':
        confidence = torch.cat(outputs['placement_confidence'][0])
        conf, idx = confidence.max(dim=0)
        gripper_target = torch.cat(outputs['placements'][0])[idx]
    visualize_grasp(
        vis, 'gripper_target', gripper_target.numpy(),
        [0, 255, 0], linewidth=0.2
    )
    print('Confidence', conf.item())


if __name__ == '__main__':
    main()
