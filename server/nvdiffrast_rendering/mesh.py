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
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tex_utils import export_layout_to_mesh_dict_list, export_single_room_layout_to_mesh_dict_list
from models import FloorPlan


def build_mesh_dict(vertices, faces, vts, fts, texture_map, device="cuda", contiguous=True):

    """
    vertices: (N, 3)
    faces: (F, 3)
    vts: (N, 2)
    fts: (F, 2)
    texture_map: (H, W, 3), values in [0, 1]
    """

    # transform all input to torch tensor if its still numpy array
    vertices = torch.from_numpy(vertices) if isinstance(vertices, np.ndarray) else vertices
    faces = torch.from_numpy(faces) if isinstance(faces, np.ndarray) else faces
    vts = torch.from_numpy(vts) if isinstance(vts, np.ndarray) else vts
    fts = torch.from_numpy(fts) if isinstance(fts, np.ndarray) else fts
    texture_map = torch.from_numpy(texture_map) if isinstance(texture_map, np.ndarray) else texture_map

    # move to device
    vertices = vertices.to(device).float()
    faces = faces.to(device).int()
    vts = vts.to(device).float()
    fts = fts.to(device).int()
    texture_map = texture_map.to(device).float()

    if contiguous:
        vertices = vertices.contiguous()
        faces = faces.contiguous()
        vts = vts.contiguous()
        fts = fts.contiguous()
        texture_map = texture_map.contiguous()
    
    # pad the vertices
    vertices = F.pad(vertices, pad=(0, 1), value=1.0, mode='constant')

    # flip the vts
    vts = vts.clone()
    vts[:, 1] = 1 - vts[:, 1]

    mesh_dict = {
        "vertices": vertices,
        "pos_idx": faces,
        "vt": vts,
        "ft": fts,
        "texture_map": texture_map
    }

    return mesh_dict


def get_mesh_dict_list_from_layout(layout: FloorPlan):
    mesh_info_dict = export_layout_to_mesh_dict_list(layout)

    mesh_dict_list = []
    for mesh_info in mesh_info_dict.values():
        vertices = mesh_info["mesh"].vertices
        faces = mesh_info["mesh"].faces
        vts = mesh_info["texture"]["vts"]
        fts = mesh_info["texture"]["fts"]
        texture_map = np.array(Image.open(mesh_info["texture"]["texture_map_path"])) / 255.0

        mesh_dict = build_mesh_dict(vertices, faces, vts, fts, texture_map)
        mesh_dict_list.append(mesh_dict)


    return mesh_dict_list

def get_mesh_dict_list_from_layout_with_id(layout: FloorPlan):
    mesh_info_dict = export_layout_to_mesh_dict_list(layout)

    mesh_dict_list = []
    mesh_ids = []
    for mesh_id, mesh_info in mesh_info_dict.items():
        vertices = mesh_info["mesh"].vertices
        faces = mesh_info["mesh"].faces
        vts = mesh_info["texture"]["vts"]
        fts = mesh_info["texture"]["fts"]
        texture_map = np.array(Image.open(mesh_info["texture"]["texture_map_path"])) / 255.0

        mesh_dict = build_mesh_dict(vertices, faces, vts, fts, texture_map)
        mesh_dict_list.append(mesh_dict)
        mesh_ids.append(mesh_id)

    return mesh_dict_list, mesh_ids

def get_mesh_dict_list_from_mesh_info_dict_with_id(mesh_info_dict: dict):

    mesh_dict_list = []
    mesh_ids = []
    for mesh_id, mesh_info in mesh_info_dict.items():
        vertices = mesh_info["mesh"].vertices
        faces = mesh_info["mesh"].faces
        vts = mesh_info["texture"]["vts"]
        fts = mesh_info["texture"]["fts"]
        texture_map = np.array(Image.open(mesh_info["texture"]["texture_map_path"])) / 255.0

        mesh_dict = build_mesh_dict(vertices, faces, vts, fts, texture_map)
        mesh_dict_list.append(mesh_dict)
        mesh_ids.append(mesh_id)

    return mesh_dict_list, mesh_ids

def get_mesh_dict_list_from_single_room(layout: FloorPlan, room_id: str):
    mesh_info_dict = export_single_room_layout_to_mesh_dict_list(layout, room_id)

    mesh_dict_list = []
    for mesh_info in mesh_info_dict.values():
        vertices = mesh_info["mesh"].vertices
        faces = mesh_info["mesh"].faces
        vts = mesh_info["texture"]["vts"]
        fts = mesh_info["texture"]["fts"]
        texture_map = np.array(Image.open(mesh_info["texture"]["texture_map_path"])) / 255.0

        # print the bounding box of the mesh
        mesh_dict = build_mesh_dict(vertices, faces, vts, fts, texture_map)
        mesh_dict_list.append(mesh_dict)

    return mesh_dict_list