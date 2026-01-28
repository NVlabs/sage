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
import random
from datetime import datetime

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from models import Object, Point3D, Euler, Dimensions
from layout_wo_robot import (
    get_layout_from_json,
    get_current_layout,
    generate_room_layout,
    place_objects_in_room,
    move_one_object_with_condition_in_room
)
from isaacsim.isaac_mcp.server import (
    get_isaac_connection,
    create_room_layout_scene,
    simulate_the_scene,
    create_robot,
    move_robot_to_target,
    create_physics_scene,
    get_room_layout_scene_usd,
    create_single_room_layout_scene
)
from tex_utils import export_layout_to_mesh_dict_list
from glb_utils import (
    create_glb_scene,
    add_textured_mesh_to_glb_scene,
    save_glb_scene
)
from objects.object_on_top_placement import (
    get_random_placements_on_target_object, 
    filter_placements_by_physics_critic,
)
from floor_plan_materials.material_generator import generate_texture_map_from_prompt_and_color
from utils import dict_to_floor_plan
from constants import RESULTS_DIR
from dataclasses import asdict
import copy

# Add eval directory to path to import rendering functions
eval_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
sys.path.insert(0, eval_dir)

from visualize_room_bpy_frames_longest import render_room_circular_video


def extract_main_color(image_path):
    """
    Extract the main color from an image and apply randomization.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of [r, g, b] values in range (0, 1)
    """
    # Load the image
    img = Image.open(image_path).convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Reshape to (num_pixels, 3) for easier processing
    pixels = img_array.reshape(-1, 3)
    
    # Use quantization to find the most common color
    # Reduce to 8 colors and get the palette
    img_quantized = img.quantize(colors=8, method=2)  # method=2 is median cut
    palette = img_quantized.getpalette()
    
    # Get color counts
    color_counts = img_quantized.getcolors(maxcolors=8)
    
    # Sort by count and get the most common color index
    most_common_idx = max(color_counts, key=lambda x: x[0])[1]
    
    # Extract RGB from palette (palette is flattened, so RGB at indices [idx*3, idx*3+1, idx*3+2])
    r = palette[most_common_idx * 3]
    g = palette[most_common_idx * 3 + 1]
    b = palette[most_common_idx * 3 + 2]
    
    # Convert to (0, 1) range
    main_color = np.array([r / 255.0, g / 255.0, b / 255.0])
    
    # Apply randomization
    # Add small random variations to each channel
    randomization_strength = 0.15  # 15% variation
    random_offset = np.random.uniform(-randomization_strength, randomization_strength, 3)
    
    # Apply offset and clip to valid range
    randomized_color = main_color + random_offset
    randomized_color = np.clip(randomized_color, 0.0, 1.0)
    
    return randomized_color.tolist()


if __name__ == "__main__":

    for _ in range(50):
        layout_id="layout_3dea5ab1"
        room_id="room_2adee539"
        type_group_idx="00"
        pose_group_idx="00"

        layout_dir = os.path.join(RESULTS_DIR, layout_id)
        layout_json_path = os.path.join(layout_dir, f"{layout_id}.json")

        save_compose_dir = os.path.join(layout_dir, f"compose_type_augmentation_{type_group_idx}_{pose_group_idx}")
        os.makedirs(save_compose_dir, exist_ok=True)

        with open(layout_json_path, "r") as f:
            layout_data = json.load(f)

        base_layout = copy.deepcopy(dict_to_floor_plan(layout_data))
        base_room = base_layout.rooms[0]

        base_objects = base_room.objects

        new_objects = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"compose_{timestamp}.json"
        save_video_name = f"compose_{timestamp}_longest_edges.mp4"
        save_material_name = f"compose_{timestamp}_materials.png"

        # generate new materials of floor and wall
        floor_material_save_name = base_layout.rooms[0].floor_material
        wall_material_save_name = base_layout.rooms[0].walls[0].material

        floor_material_save_path = os.path.join(layout_dir, f"materials/{floor_material_save_name}.png")
        wall_material_save_path = os.path.join(layout_dir, f"materials/{wall_material_save_name}.png")

        # extract the main color of the floor and wall materials
        floor_color = extract_main_color(floor_material_save_path) # [r, g, b] (0, 1)
        wall_color = extract_main_color(wall_material_save_path) # [r, g, b] (0, 1)
        texture_res = 1024

        floor_texture_pil = generate_texture_map_from_prompt_and_color(base_layout.building_style, floor_color)
        floor_texture_pil = floor_texture_pil.resize((texture_res, texture_res), Image.LANCZOS)
        floor_texture_pil.save(os.path.join(layout_dir, f"materials/{save_material_name}_floor.png"))

        wall_texture_pil = generate_texture_map_from_prompt_and_color(base_layout.building_style, wall_color)
        wall_texture_pil = wall_texture_pil.resize((texture_res, texture_res), Image.LANCZOS)
        wall_texture_pil.save(os.path.join(layout_dir, f"materials/{save_material_name}_wall.png"))

        base_layout.rooms[0].floor_material = f"{save_material_name}_floor"
        for wall in base_layout.rooms[0].walls:
            wall.material = f"{save_material_name}_wall"
            wall.height = 2.85

        for base_object in base_objects:
            base_object_id = base_object.id

            type_aug_base_dir = f"results/{layout_id}/{base_object_id}_aug_test_{type_group_idx}/"

            available_type_aug_i = []

            for type_aug_i in range(4):

                type_aug_i_base_name = f"{layout_id}_{base_object_id}_aug_test_{type_group_idx}_linear{type_aug_i}"

                type_candidate_info_path = os.path.join(type_aug_base_dir, f"{type_aug_i_base_name}_type_candidate.json")
                print(f"Type candidate info path: {type_candidate_info_path}")

                if not os.path.exists(type_candidate_info_path):
                    continue

                with open(type_candidate_info_path, "r") as f:
                    type_candidate_info = json.load(f)

                old_id_to_new_id_map = type_candidate_info["old_id_to_new_id_map"]
                print(f"Old ID to new ID map: {old_id_to_new_id_map}")

                type_aug_layout_json_base_name = f"aug_pose_group_{pose_group_idx}/{layout_id}_{base_object_id}_aug_test_{type_group_idx}_linear{type_aug_i}_sim_corrected.json"

                type_aug_layout_json_path = os.path.join(type_aug_base_dir, type_aug_i_base_name, type_aug_layout_json_base_name)

                print(f"Type augmentation layout JSON path: {type_aug_layout_json_path}")

                if os.path.exists(type_aug_layout_json_path):
                    available_type_aug_i.append(type_aug_i)

            print(f"Available type augmentation indices for {base_object_id}: {available_type_aug_i}")

            if len(available_type_aug_i) > 0:
                type_aug_i = random.choice(available_type_aug_i)

                type_aug_i_base_name = f"{layout_id}_{base_object_id}_aug_test_{type_group_idx}_linear{type_aug_i}"

                type_candidate_info_path = os.path.join(type_aug_base_dir, f"{type_aug_i_base_name}_type_candidate.json")

                with open(type_candidate_info_path, "r") as f:
                    type_candidate_info = json.load(f)

                old_id_to_new_id_map = type_candidate_info["old_id_to_new_id_map"]

                type_aug_layout_json_base_name = f"aug_pose_group_{pose_group_idx}/{layout_id}_{base_object_id}_aug_test_{type_group_idx}_linear{type_aug_i}_sim_corrected.json"

                type_aug_layout_json_path = os.path.join(type_aug_base_dir, type_aug_i_base_name, type_aug_layout_json_base_name)

                with open(type_aug_layout_json_path, "r") as f:
                    type_aug_layout_json = json.load(f)

                type_aug_layout = dict_to_floor_plan(type_aug_layout_json)

                type_aug_room = type_aug_layout.rooms[0]

                type_aug_objects = type_aug_room.objects

                for old_id in old_id_to_new_id_map:
                    new_id = old_id_to_new_id_map[old_id]
                    new_object = next((o for o in type_aug_objects if o.id == new_id), None)
                    if new_object is not None:
                        new_objects.append(new_object)
            
            elif base_object.place_id == "wall" or base_object.place_id == "floor":
                new_objects.append(base_object)
                # Add all descendants of base_object as well
                # Build object map and children map for the base room
                object_map = {o.id: o for o in base_objects}
                children_map = {}
                for o in base_objects:
                    parent_id = o.place_id
                    if isinstance(parent_id, str) and parent_id in object_map:
                        children_map.setdefault(parent_id, []).append(o)
                
                # BFS to collect all descendants of base_object
                bfs_descendants = []
                queue = [base_object.id]
                visited = set()
                while queue:
                    current_id = queue.pop(0)
                    if current_id in visited:
                        continue
                    visited.add(current_id)
                    # Add children of current object
                    for child in children_map.get(current_id, []):
                        if child.id not in visited:
                            bfs_descendants.append(child)
                            queue.append(child.id)
                
                # Add all descendants to new_objects
                new_objects.extend(bfs_descendants)


        base_layout.rooms[0].objects = new_objects
        base_layout_dict = asdict(base_layout)

        with open(os.path.join(save_compose_dir, save_name), "w") as f:
            json.dump(base_layout_dict, f, indent=4)


        render_room_circular_video(
            layout=base_layout,
            room_id=base_room.id,
            output_path=os.path.join(save_compose_dir, save_video_name),
            resolution=1920,
            angle_step=45,  # Unused in longest edge mode
            fps=24
        )

        print(f"Composed layout saved to: {os.path.join(save_compose_dir, save_name)}")
        print(f"Composed video saved to: {os.path.join(save_compose_dir, save_video_name)}")
