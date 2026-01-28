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
import argparse
import json
import sys
import os
import numpy as np
from PIL import Image

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from isaacsim.isaac_mcp.server import (
    get_room_layout_scene_usd_separate_from_layout
)

from objects.object_augmentation import (
    object_augmentation_pose_object_tree_with_reach_test_parallel
)

from utils import get_layout_from_scene_save_dir
from constants import RESULTS_DIR, SERVER_ROOT_DIR
from scipy.spatial.transform import Rotation


def augment_layout(layout_id, room_id, object_id, aug_num, aug_name, group_size=10):
    """Test loading layout from JSON file"""
    
    json_file_path = os.path.join(SERVER_ROOT_DIR, f"results/{layout_id}/{layout_id}.json")

    current_layout = get_layout_from_scene_save_dir(os.path.dirname(json_file_path))
    scene_save_dir = os.path.join(RESULTS_DIR, current_layout.id)

    room = next((r for r in current_layout.rooms if r.id == room_id), None)
    if room is None:
        raise ValueError(f"Room with ID {room_id} not found in layout")
    object = next((o for o in room.objects if o.id == object_id), None)
    if object is None:
        raise ValueError(f"Object with ID {object_id} not found in room {room_id}")

    augmented_layouts_info = object_augmentation_pose_object_tree_with_reach_test_parallel(
        current_layout, room, object, aug_num, aug_name, reach_threshold=0.8, group_size=group_size)
    
    print(f"augmented layouts length: {len(augmented_layouts_info)}")
    usd_collection_dir = os.path.join(scene_save_dir, f"usd_collection_{aug_name}")
    os.makedirs(usd_collection_dir, exist_ok=True)

    all_augmented_layouts_info = {
        "object_transform_dict": {},
    }
    mass_dict = {}

    for (augmented_layout_id, augmented_layout) in augmented_layouts_info:
        all_object_transform_dict = {}
        for room in augmented_layout.rooms:
            for object in room.objects:
                position = object.position
                rotation = object.rotation
                mass_dict[object.id] = object.mass
                quat = Rotation.from_euler("xyz", [rotation.x, rotation.y, rotation.z], degrees=True).as_quat(scalar_first=True)
                all_object_transform_dict[object.id] = {
                    "position": [position.x, position.y, position.z],
                    "rotation": [quat[0], quat[1], quat[2], quat[3]],
                }

        all_augmented_layouts_info["object_transform_dict"][augmented_layout_id] = all_object_transform_dict

    all_augmented_layouts_info["usd_collection_dir"] = usd_collection_dir
    all_augmented_layouts_info["mass_dict"] = mass_dict

    with open(os.path.join(scene_save_dir, f"all_augmented_layouts_info_{aug_name}.json"), "w") as f:
        json.dump(all_augmented_layouts_info, f, indent=4)
    
    get_room_layout_scene_usd_separate_from_layout(json_file_path, usd_collection_dir)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_id", type=str, required=True)
    parser.add_argument("--room_id", type=str, required=True)
    parser.add_argument("--object_id", type=str, required=True)
    parser.add_argument("--aug_name", type=str, required=True)
    parser.add_argument("--aug_num", type=int, default=20)
    parser.add_argument("--group_size", type=int, default=10)
    args = parser.parse_args()
    augment_layout(args.layout_id, args.room_id, args.object_id, args.aug_num, args.aug_name, args.group_size)
