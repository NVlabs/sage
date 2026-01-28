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
import asyncio
import json
import sys
import os
import numpy as np
from PIL import Image

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from models import Object, Point3D, Euler, Dimensions
from layout import (
    get_layout_from_json,
    add_one_object_with_condition_in_room,
    get_current_layout,
    generate_room_layout,
    move_one_object_with_condition_in_room,
    place_objects_in_room
)
from isaacsim.isaac_mcp.server import (
    get_isaac_connection,
    create_room_layout_scene,
    simulate_the_scene,
    create_robot,
    move_robot_to_target,
    create_physics_scene,
    get_room_layout_scene_usd,
    create_single_room_layout_scene,
    get_room_layout_scene_usd_separate_from_layout
)
from tex_utils import export_layout_to_mesh_dict_list
from glb_utils import (
    create_glb_scene,
    add_textured_mesh_to_glb_scene,
    save_glb_scene
)
from objects.object_augmentation import (
    object_augmentation_pose_object_tree,
    object_augmentation_type_single_object_leaf,
    object_augmentation_type_object_tree
)
from objects.object_on_top_placement import (
    get_random_placements_on_target_object, 
    filter_placements_by_physics_critic,
)
from utils import get_layout_from_scene_save_dir
from constants import RESULTS_DIR, SERVER_ROOT_DIR
from scipy.spatial.transform import Rotation

class MockContext:
    """Mock context for testing MCP tools"""
    async def info(self, message: str):
        print(f"INFO: {message}")

async def test_load_layout():
    """Test loading layout from JSON file"""
    
    try:
        json_file_path = os.path.join(SERVER_ROOT_DIR, "results/layout_625b9812/layout_625b9812.json")

        current_layout = get_layout_from_scene_save_dir(os.path.dirname(json_file_path))
        scene_save_dir = os.path.join(RESULTS_DIR, current_layout.id)
        print(f"scene_save_dir: {scene_save_dir}")

        room_id = "room_0736c934"

        # object_id = "room_0736c934_ceramic_mug_with_handle_dcaede53"
        object_id = "room_0736c934_wooden_rectangular_coffee_table_ad3b7c58"
        aug_num = 100
        aug_name = "aug_pose_v2"
        # aug_name = "aug_type"

        room = next((r for r in current_layout.rooms if r.id == room_id), None)
        if room is None:
            raise ValueError(f"Room with ID {room_id} not found in layout")
        object = next((o for o in room.objects if o.id == object_id), None)
        if object is None:
            raise ValueError(f"Object with ID {object_id} not found in room {room_id}")

        augmented_layouts_info = object_augmentation_pose_object_tree(current_layout, room, object, aug_num, aug_name)
        # augmented_layouts_info = object_augmentation_type_single_object_leaf(current_layout, room, object, aug_num, aug_name)
        # augmented_layouts_info = object_augmentation_type_object_tree(current_layout, room, object, aug_num, aug_name)
        
        augmented_layouts_info.append((current_layout.id, current_layout))
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
        
    except Exception as e:
        print(f"ERROR: Exception occurred during test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_load_layout())
