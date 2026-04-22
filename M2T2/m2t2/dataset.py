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
Data loader for training M2T2.
'''
from PIL import Image
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import torch

from m2t2.dataset_utils import (
    depth_to_xyz, jitter_gaussian, normalize_rgb, sample_points
)


def load_rgb_xyz(
    data_dir, robot_prob, world_coord, jitter_scale, grid_res, surface_range=0
):
    with open(f'{data_dir}/meta_data.pkl', 'rb') as f:
        meta_data = pickle.load(f)
    print("meta_data: ", meta_data.keys())
    print("meta_data['scene_bounds']: ", meta_data['scene_bounds'])
    # print("meta_data['ee_pose']: ", meta_data['ee_pose'])
    rgb = normalize_rgb(Image.open(f'{data_dir}/rgb.png')).permute(1, 2, 0)
    print("rgb: ", rgb.shape, rgb.min(), rgb.max())
    depth = np.load(f'{data_dir}/depth.npy')
    print("depth: ", depth.shape, depth.min(), depth.max())
    print("meta_data['intrinsics']: ", meta_data['intrinsics'])
    xyz = torch.from_numpy(
        depth_to_xyz(depth, meta_data['intrinsics'])
    ).float()
    seg = torch.from_numpy(np.array(Image.open(f'{data_dir}/seg.png')))
    print("seg: ", seg.shape, seg.min(), seg.max(), seg.unique())
    label_map = meta_data['label_map']
    print("label_map: ", label_map)

    if False and torch.rand(()) > robot_prob:
        robot_mask = seg == label_map['robot']
        if 'robot_table' in label_map:
            robot_mask |= seg == label_map['robot_table']
        if 'object_label' in meta_data:
            robot_mask |= seg == label_map[meta_data['object_label']]
        depth[robot_mask] = 0
        seg[robot_mask] = 0
    xyz, rgb, seg = xyz[depth > 0], rgb[depth > 0], seg[depth > 0]
    cam_pose = torch.from_numpy(meta_data['camera_pose']).float()
    print("cam_pose: ", cam_pose.shape, cam_pose)
    xyz_world = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]

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

    outputs = {
        'inputs': torch.cat([xyz - xyz.mean(dim=0), rgb], dim=1),
        'points': xyz,
        'seg': seg,
        'cam_pose': cam_pose
    }

    if 'object_label' in meta_data:
        print("need to load object_label")
        obj_mask = seg == label_map[meta_data['object_label']]
        obj_xyz, obj_rgb = xyz_world[obj_mask], rgb[obj_mask]
        obj_xyz_grid = torch.unique(
            (obj_xyz[:, :2] / grid_res).round(), dim=0
        ) * grid_res
        bottom_center = obj_xyz.min(dim=0)[0]
        bottom_center[:2] = obj_xyz_grid.mean(dim=0)

        ee_pose = torch.from_numpy(meta_data['ee_pose']).float()
        inv_ee_pose = ee_pose.inverse()
        obj_xyz = obj_xyz @ inv_ee_pose[:3, :3].T + inv_ee_pose[:3, 3]
        outputs.update({
            'object_inputs': torch.cat([
                obj_xyz - obj_xyz.mean(dim=0), obj_rgb
            ], dim=1),
            'ee_pose': ee_pose,
            'bottom_center': bottom_center,
            'object_center': obj_xyz.mean(dim=0)
        })
    else:
        print("no object_label")
        outputs.update({
            'object_inputs': torch.rand(1024, 6),
            'ee_pose': torch.eye(4),
            'bottom_center': torch.zeros(3),
            'object_center': torch.zeros(3)
        })
    return outputs, meta_data


def load_grasps(
    data, pts, seg, label_map, world2cam, contact_radius, offset_bins
):
    grasping_masks, matched_grasps = [], []
    contact_dirs = torch.zeros_like(pts)
    approach_dirs = torch.zeros_like(pts)
    offsets = torch.zeros_like(pts[:, 0])
    names = sorted(list(data['grasps'].keys()))
    for name in names:
        contacts = torch.from_numpy(data['grasp_contacts'][name]).float()
        grasps = torch.from_numpy(data['grasps'][name]).float()
        if world2cam is not None:
            # convert contacts and grasps to camera coordinate
            contacts = contacts @ world2cam[:3, :3].T + world2cam[:3, 3]
            grasps = world2cam @ grasps

        contact_dir = contacts[:, 1] - contacts[:, 0]
        offset = contact_dir.norm(dim=1)
        contact_dir = contact_dir / offset.unsqueeze(1)
        approach_dir = grasps[:, :3, 2]

        # Mx2x3 -> 2Mx3
        contacts = contacts.transpose(0, 1).reshape(-1, 3)
        contact_dir = torch.cat([contact_dir, -contact_dir])
        approach_dir = torch.cat([approach_dir, approach_dir])
        offset = torch.cat([offset, offset])
        grasps = torch.cat([grasps, grasps])

        mask = seg == label_map[name]
        if mask.sum() == 0:
            continue
        tree = KDTree(contacts.numpy())
        dist, idx = tree.query(pts[mask].numpy())
        matched = dist < contact_radius
        idx = idx[matched]
        grasps = grasps[idx]

        if matched.sum() > 0:
            pt_i = torch.where(mask)[0]
            contact_mask = torch.zeros_like(mask)
            contact_mask[pt_i[matched[:, 0]]] = 1
            contact_dirs[contact_mask] = contact_dir[idx]
            approach_dirs[contact_mask] = approach_dir[idx]
            offsets[contact_mask] = offset[idx]
            grasping_masks.append(contact_mask)
            matched_grasps.append(grasps)

    if len(grasping_masks) > 0:
        grasping_masks = torch.stack(grasping_masks).float()
    else:
        # No grasp, skip
        return {'invalid': True}
    contact_any_obj = grasping_masks.any(dim=0)
    contact_dirs = contact_dirs[contact_any_obj]
    approach_dirs = approach_dirs[contact_any_obj]
    offsets = offsets[contact_any_obj]
    outputs = {
        'names': names,
        'grasping_masks': grasping_masks,
        'contact_dirs': contact_dirs,
        'approach_dirs': approach_dirs,
        'grasps': matched_grasps
    }
    labels = torch.bucketize(offsets, torch.tensor(offset_bins)) - 1
    outputs['offsets'] = torch.clip(labels, 0, len(offset_bins) - 1)
    return outputs


def load_placements(
    data, pts, seg, meta_data, cam_pose, num_rotations, contact_radius
):
    place_pos = data['placements'][...]
    place_pos = np.concatenate([
        place_pos, np.full((place_pos.shape[0], 1), 0)
    ], axis=1)
    tree = KDTree(place_pos)
    if cam_pose is not None:
        pts = pts @ cam_pose[:3, :3].T + cam_pose[:3, 3]
    dist, idx = tree.query(pts.numpy())
    matched = dist < contact_radius
    indices = idx[matched]
    if len(indices) == 0:
        # No placement, skip
        return {'invalid': True}
    placement_region = torch.from_numpy(matched[:, 0])
    placement_masks = torch.zeros(pts.shape[0], num_rotations)
    success = data['placement_success'][...]
    skip = success.shape[1] // num_rotations
    success = success[:, ::skip]
    placement_masks[placement_region] = torch.from_numpy(
        success[indices]
    ).float()

    # Mark robot as non-placable region
    label_map = meta_data['label_map']
    object_label = meta_data['object_label']
    not_placable = (seg == label_map['robot']) \
                 | (seg == label_map[object_label])
    placement_region[not_placable] = 1

    outputs = {
        'placement_masks': placement_masks.T,
        'placement_region': placement_region.float()
    }
    return outputs


class PickPlaceDataset(Dataset):
    def __init__(
        self, root_dir, num_points, num_obj_points, world_coord,
        num_rotations, grid_res, jitter_scale, contact_radius,
        offset_bins, robot_prob
    ):
        self.root_dir = root_dir
        self.scenes = sorted(os.listdir(root_dir))
        self.num_points = num_points
        self.num_obj_points = num_obj_points
        self.world_coord = world_coord
        self.num_rotations = num_rotations
        self.grid_res = grid_res
        self.jitter_scale = jitter_scale
        self.robot_prob = robot_prob
        self.contact_radius = contact_radius
        self.offset_bins = offset_bins

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args['num_points'] = cfg.num_points
        args['num_obj_points'] = cfg.num_object_points
        args['world_coord'] = cfg.world_coord
        args['num_rotations'] = cfg.num_rotations
        args['grid_res'] = cfg.grid_resolution
        args['jitter_scale'] = cfg.jitter_scale
        args['contact_radius'] = cfg.contact_radius
        args['offset_bins'] = cfg.offset_bins
        args['robot_prob'] = cfg.robot_prob
        return args

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        data_dir = f"{self.root_dir}/{self.scenes[idx]}"
        outputs, meta_data = load_rgb_xyz(
            data_dir, self.robot_prob, self.world_coord,
            self.jitter_scale, self.grid_res
        )
        pt_idx = sample_points(outputs['points'], self.num_points)
        outputs['inputs'] = outputs['inputs'][pt_idx]
        outputs['points'] = outputs['points'][pt_idx]
        outputs['seg'] = outputs['seg'][pt_idx]
        pt_idx = sample_points(outputs['object_inputs'], self.num_obj_points)
        outputs['object_inputs'] = outputs['object_inputs'][pt_idx]
        outputs['scene'] = self.scenes[idx]
        if 'object_label' in meta_data:
            outputs['task'] = 'place'
        else:
            outputs['task'] = 'pick'
        cam_pose = None if self.world_coord else outputs['cam_pose']
        world2cam = None if self.world_coord else outputs['cam_pose'].inverse()

        with open(f"{data_dir}/annotation.pkl", 'rb') as f:
            annotation = pickle.load(f)
        if outputs['task'] == 'pick':
            outputs.update(load_grasps(
                annotation, outputs['points'], outputs['seg'],
                meta_data['label_map'], world2cam,
                self.contact_radius, self.offset_bins
            ))
        else:
            outputs.update({
                'names': [],
                'grasping_masks': torch.zeros(0, self.num_points),
                'contact_dirs': torch.zeros(0, 3),
                'approach_dirs': torch.zeros(0, 3),
                'offsets': torch.zeros(0).long(),
                'grasps': []
            })
        if outputs['task'] == 'place':
            outputs.update(load_placements(
                annotation, outputs['points'], outputs['seg'], meta_data,
                cam_pose, self.num_rotations, self.contact_radius
            ))
        else:
            outputs.update({
                'placement_masks': torch.zeros(
                    self.num_rotations, self.num_points
                ),
                'placement_region': torch.zeros(self.num_points)
            })
        return outputs


def collate(batch):
    batch = [data for data in batch if not data.get('invalid', False)]
    batch = {key: [data[key] for data in batch] for key in batch[0]}
    if 'task' in batch:
        task = batch.pop('task')
        batch['task_is_pick'] = torch.stack([
            torch.tensor(t == 'pick') for t in task
        ])
        batch['task_is_place'] = torch.stack([
            torch.tensor(t == 'place') for t in task
        ])
    for key in batch:
        if key in [
            'inputs', 'points', 'seg', 'object_inputs', 'bottom_center',
            'cam_pose', 'ee_pose', 'placement_masks', 'placement_region',
            'lang_tokens'
        ]:
            batch[key] = torch.stack(batch[key])
        if key in [
            'contact_dirs', 'approach_dirs', 'offsets'
        ]:
            batch[key] = torch.cat(batch[key])
    return batch
