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
from PIL import Image
import numpy as np
import torch
import os
from constants import SERVER_ROOT_DIR
from nvdiffrast_rendering.camera import build_camera_matrix, get_intrinsic, get_camera_perspective_projection_matrix, get_mvp_matrix, get_intrinsic_matrix
from nvdiffrast_rendering.context import get_glctx
from nvdiffrast_rendering.render import rasterize_mesh_dict_list_with_uv_efficient
from nvdiffrast_rendering.mesh import get_mesh_dict_list_from_mesh_info_dict_with_id

def generate_m2t2_data(mesh_dict_list, target_object_name, base_pos):
    glctx = get_glctx()

    target_object_mesh = mesh_dict_list[target_object_name]["mesh"]

    # find the bounding box center of the target object
    bbox_center = target_object_mesh.bounds.mean(axis=0)
    bbox_size = target_object_mesh.bounds.max(axis=0) - target_object_mesh.bounds.min(axis=0)

    base_pos_np = np.array(base_pos).reshape(3)
    camera_pos_np = base_pos_np + np.array([0.0, 0.0, 0.8])

    scene_bounds = (bbox_center - bbox_size * 4.0).tolist() + \
        (bbox_center + bbox_size * 4.0).tolist()
        
    c2w = build_camera_matrix(
        camera_pos=torch.from_numpy(camera_pos_np).float(),
        camera_lookat=torch.from_numpy(bbox_center).float(),
        camera_up=torch.from_numpy(np.array([0.0, 0.0, 1.0])).float(),
    )

    H, W = 720, 1280
    resolution = (H, W)

    fx, fy, cx, cy = get_intrinsic(fov=60.0, H=H, W=W)

    projection = get_camera_perspective_projection_matrix(
        fx, fy, cx, cy, H=H, W=W, near=0.001, far=100.0)

    mvp_matrix = get_mvp_matrix(c2w, projection)

    mesh_dict_list, mesh_ids = get_mesh_dict_list_from_mesh_info_dict_with_id(mesh_dict_list)
    # mesh_dict_list = mesh_dict_list[0:1]
    # mesh_dict_list = mesh_dict_list[1:2]
    # for mesh_dict in mesh_dict_list:
    #     for key, value in mesh_dict.items():
    #         print(f"key: {key}, value: {value.shape}")

    valid, instance_id, rgb, depth = rasterize_mesh_dict_list_with_uv_efficient(mesh_dict_list, mvp_matrix, glctx, resolution, c2w)

    depth_np = depth.cpu().numpy()
    rgb_np = rgb.cpu().numpy()
    valid_np = valid.cpu().numpy()
    instance_id_np = instance_id.cpu().numpy()
    instance_id_np[np.logical_not(valid_np)] = -1

    seg_mask = np.zeros((H, W), dtype=np.uint8)
    target_object_idx = mesh_ids.index(target_object_name)
    seg_mask[instance_id_np == target_object_idx] = 101

    label_map = {
        'ground': 0, 'table': 2, 'robot': 3, 
    }
    label_map["obj_0"] = 101

    depth_np[np.logical_not(valid_np)] = 1e6

    meta_data = {
        'camera_pose': c2w.cpu().numpy(),
        'intrinsics': get_intrinsic_matrix(fx, fy, cx, cy).cpu().numpy(),
        'label_map': label_map,
        'scene_bounds': scene_bounds,
    }

    vis_data = {
        'depth': depth_np,
        'rgb': Image.fromarray((rgb_np * 255.0).astype(np.uint8)),
        'seg': Image.fromarray(seg_mask),
    }

    # debug
    debug_dir = os.path.join(SERVER_ROOT_DIR, "test/m2t2_data")
    os.makedirs(debug_dir, exist_ok=True)
    depth_vis = depth_np.copy()
    depth_vis = depth_vis / 2.0
    depth_vis = depth_vis.clip(0.0, 1.0)

    Image.fromarray((depth_vis * 255.0).astype(np.uint8)).save(f"{debug_dir}/depth_vis.png")
    Image.fromarray((rgb_np * 255.0).astype(np.uint8)).save(f"{debug_dir}/rgb_vis.png")
    Image.fromarray((valid_np.astype(np.float32) * 255.0).astype(np.uint8)).save(f"{debug_dir}/valid_vis.png")
    Image.fromarray((seg_mask).astype(np.uint8)).save(f"{debug_dir}/seg_vis.png")

    return meta_data, vis_data
