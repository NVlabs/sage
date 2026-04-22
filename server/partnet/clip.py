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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR, PARTNET_ROOT_DIR
import os
import sapien
import numpy as np
from PIL import Image
from tqdm import tqdm
from vlm import call_vlm
from foundation_models import get_clip_models
import base64
from utils import extract_json_from_response
import json
import torch
import torch.nn.functional as F
import pickle
partnet_root_dir = PARTNET_ROOT_DIR

def render_urdf(urdf_path, save_path):
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    asset = loader.load(urdf_path)
    assert asset, "failed to load URDF."

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1])
    scene.add_point_light([1, -2, 2], [1, 1, 1])
    scene.add_point_light([-1, 0, 1], [1, 1, 1])

    near, far = 0.1, 100
    width, height = 640, 480

    # Compute the camera pose by specifying forward(x), left(y) and up(z)
    cam_pos = np.array([-2, -2, 2])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos

    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )
    camera.entity.set_pose(sapien.Pose(mat44))

    scene.step()  # run a physical step
    scene.update_render()  # sync pose from SAPIEN to renderer
    camera.take_picture()  # submit rendering jobs to the GPU

    rgba = camera.get_picture("Color")  # [H, W, 4]
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    rgba_pil = Image.fromarray(rgba_img[..., :3])
    rgba_pil.save(save_path)

def clip_process_image(image_path, clip_model, clip_preprocess, clip_tokenizer):
    image = Image.open(image_path)
    image = clip_preprocess(image).unsqueeze(0).to("cuda")
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features = F.normalize(image_features, p=2, dim=-1)
    image_features = image_features.cpu()
    return image_features


if __name__ == "__main__":
    partnet_data_dir = os.path.join(partnet_root_dir, "dataset")
    partnet_render_dir = os.path.join(partnet_root_dir, "render")
    partnet_desc_dir = os.path.join(partnet_root_dir, "desc")
    os.makedirs(partnet_render_dir, exist_ok=True)
    os.makedirs(partnet_desc_dir, exist_ok=True)

    clip_model, clip_preprocess, clip_tokenizer = get_clip_models()

    id_list = sorted(os.listdir(partnet_data_dir))

    valid_ids = []
    clip_features = []
    for id in tqdm(id_list):
        id_dir = os.path.join(partnet_data_dir, id)
        urdf_path = os.path.join(id_dir, "mobility.urdf")
        meta_path = os.path.join(id_dir, "meta.json")
        save_im_path = os.path.join(partnet_render_dir, f"{id}.png")
        try:
            render_urdf(urdf_path, save_im_path)
        except:
            print(f"skip rendering {id}")
            continue 

        clip_feature = clip_process_image(save_im_path, clip_model, clip_preprocess, clip_tokenizer)
        clip_features.append(clip_feature)
        valid_ids.append(id)

    clip_features = torch.cat(clip_features, dim=0).cpu().numpy()
    print(clip_features.shape)
    valid_ids = np.array(valid_ids).astype(np.int32)
    print(valid_ids.shape)

    with open(os.path.join(partnet_root_dir, "clip_features.pkl"), "wb") as f:
        pickle.dump({
            "image_features": clip_features,
            "valid_ids": valid_ids,
        }, f)
