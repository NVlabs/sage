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
import random
from datetime import datetime
import importlib.util

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from models import Object, Point3D, Euler, Dimensions
from floor_plan_materials.material_generator import generate_texture_map_from_prompt_and_color
from constants import RESULTS_DIR, SERVER_ROOT_DIR
# from utils import dict_to_floor_plan
utils_spec = importlib.util.spec_from_file_location("server_utils", os.path.join(SERVER_ROOT_DIR, "utils.py"))
server_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(server_utils)

# Import the specific functions from server utils
dict_to_floor_plan = server_utils.dict_to_floor_plan
dict_to_object = server_utils.dict_to_object

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


def get_object_tree(base_objects, parent_object_id):
    """
    Get the parent object and all its descendants from the base objects.
    
    Returns:
        List of objects including the parent and all descendants
    """
    object_map = {o.id: o for o in base_objects}
    
    # Build children map
    children_map = {}
    for o in base_objects:
        parent_id = o.place_id
        if isinstance(parent_id, str) and parent_id in object_map:
            children_map.setdefault(parent_id, []).append(o)
    
    # BFS to collect parent + descendants
    result = []
    parent_obj = object_map.get(parent_object_id)
    if parent_obj:
        result.append(copy.deepcopy(parent_obj))
        
        queue = [parent_object_id]
        visited = {parent_object_id}
        while queue:
            current_id = queue.pop(0)
            for child in children_map.get(current_id, []):
                if child.id not in visited:
                    visited.add(child.id)
                    result.append(copy.deepcopy(child))
                    queue.append(child.id)
    
    return result


def compose_scene_level_pose_augmentations(layout_id, room_id, pose_aug_name, compose_num):
    """
    Compose random combinations of pose-augmented objects into complete layouts.
    
    Args:
        layout_id: ID of the layout
        room_id: ID of the room
        pose_aug_name: Name of the pose augmentation group
        compose_num: Number of composed layouts to generate
    """
    
    # Load pose augmentation summary
    layout_dir = os.path.join(RESULTS_DIR, layout_id)
    pose_aug_dir = os.path.join(layout_dir, pose_aug_name)
    summary_path = os.path.join(pose_aug_dir, "summary.json")
    
    if not os.path.exists(summary_path):
        raise ValueError(f"Pose augmentation summary not found: {summary_path}")
    
    with open(summary_path, "r") as f:
        pose_aug_summary = json.load(f)
    
    print(f"{'='*80}")
    print(f"Scene-Level Pose Augmentation Composition")
    print(f"{'='*80}")
    print(f"Layout ID: {layout_id}")
    print(f"Room ID: {room_id}")
    print(f"Pose Aug Name: {pose_aug_name}")
    print(f"Compose Num: {compose_num}")
    print(f"Found {len(pose_aug_summary['processed_objects'])} processed objects")
    print(f"{'='*80}\n")
    
    # Load base layout
    layout_json_path = os.path.join(layout_dir, f"{layout_id}.json")
    with open(layout_json_path, "r") as f:
        layout_data = json.load(f)
    
    base_layout = dict_to_floor_plan(layout_data)
    base_room = next((r for r in base_layout.rooms if r.id == room_id), None)
    if base_room is None:
        raise ValueError(f"Room {room_id} not found in layout")
    
    base_objects = base_room.objects
    
    # Load all pose augmentation data for each processed object
    pose_aug_data = {}
    for obj_info in pose_aug_summary['processed_objects']:
        parent_object_id = obj_info['parent_object']
        pose_aug_file = os.path.join(pose_aug_dir, f"pose_aug_{parent_object_id}.json")
        
        if os.path.exists(pose_aug_file):
            with open(pose_aug_file, "r") as f:
                data = json.load(f)
            if len(data.get('pose_augmentations', [])) > 0:
                pose_aug_data[parent_object_id] = data['pose_augmentations']
                print(f"  Loaded {len(data['pose_augmentations'])} pose augmentations for {parent_object_id}")
    
    # Find all floor/wall objects (same logic as scene_level_pose_aug.py)
    floor_and_wall_objects = [obj for obj in base_objects if obj.place_id in ["floor", "wall"]]
    
    # Create output directories
    compose_base_dir = os.path.join(layout_dir, f"{pose_aug_name}_composed")
    compose_layouts_dir = os.path.join(compose_base_dir, "layouts")
    compose_videos_dir = os.path.join(compose_base_dir, "videos")
    materials_dir = os.path.join(layout_dir, "materials")  # Use shared materials directory
    
    os.makedirs(compose_layouts_dir, exist_ok=True)
    os.makedirs(compose_videos_dir, exist_ok=True)
    os.makedirs(materials_dir, exist_ok=True)
    
    # Track composition metadata
    composition_summary = {
        "layout_id": layout_id,
        "room_id": room_id,
        "pose_aug_name": pose_aug_name,
        "compose_num": compose_num,
        "compositions": []
    }
    
    successful_compositions = 0
    
    # Generate composed layouts
    for compose_idx in range(compose_num):
        print(f"\n{'='*80}")
        print(f"Generating composition {compose_idx + 1}/{compose_num}")
        print(f"{'='*80}")
        
        try:
            # Deep copy base layout for this composition
            composed_layout = copy.deepcopy(base_layout)
            composed_room = next((r for r in composed_layout.rooms if r.id == room_id), None)
            
            new_objects = []
            object_selections = {}
            
            # For each floor/wall object, either use a pose augmentation or original
            for parent_obj in floor_and_wall_objects:
                parent_object_id = parent_obj.id
                
                if parent_object_id in pose_aug_data:
                    # Has pose augmentations available - randomly select one
                    available_augs = pose_aug_data[parent_object_id]
                    selected_idx = random.randint(0, len(available_augs) - 1)
                    selected_aug = available_augs[selected_idx]
                    
                    original_tree = get_object_tree(base_objects, parent_object_id)
                    original_tree_all_ids = [obj.id for obj in original_tree]
                    
                    # Convert dict objects back to Object instances
                    for obj_dict in selected_aug:
                        obj = dict_to_object(obj_dict)
                        if obj.id in original_tree_all_ids:
                            new_objects.append(obj)
                            print(f"  {obj.id}: added to new objects")
                    
                    object_selections[parent_object_id] = f"pose_aug_{selected_idx}"
                    print(f"  {parent_object_id}: selected pose_aug_{selected_idx}")
                else:
                    # No pose augmentations - use original object tree
                    original_tree = get_object_tree(base_objects, parent_object_id)
                    new_objects.extend(original_tree)
                    for obj in original_tree:
                        print(f"  {obj.id}: added to new objects (no augmentations)")
                    object_selections[parent_object_id] = "original"
                    print(f"  {parent_object_id}: using original (no augmentations)")
            
            # Generate unique composition ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            compose_id = f"compose_{compose_idx:04d}_{timestamp}"
            
            # Generate randomized floor and wall materials
            floor_material_name = composed_layout.rooms[0].floor_material
            wall_material_name = composed_layout.rooms[0].walls[0].material
            
            floor_material_path = os.path.join(layout_dir, f"materials/{floor_material_name}.png")
            wall_material_path = os.path.join(layout_dir, f"materials/{wall_material_name}.png")
            
            if os.path.exists(floor_material_path) and os.path.exists(wall_material_path):
                floor_color = extract_main_color(floor_material_path)
                wall_color = extract_main_color(wall_material_path)
                texture_res = 1024
                
                floor_texture_pil = generate_texture_map_from_prompt_and_color(
                    composed_layout.building_style, floor_color)
                floor_texture_pil = floor_texture_pil.resize((texture_res, texture_res), Image.LANCZOS)
                floor_texture_pil.save(os.path.join(materials_dir, f"{compose_id}_floor.png"))
                
                wall_texture_pil = generate_texture_map_from_prompt_and_color(
                    composed_layout.building_style, wall_color)
                wall_texture_pil = wall_texture_pil.resize((texture_res, texture_res), Image.LANCZOS)
                wall_texture_pil.save(os.path.join(materials_dir, f"{compose_id}_wall.png"))
                
                # Update material references
                composed_layout.rooms[0].floor_material = f"{compose_id}_floor"
                for wall in composed_layout.rooms[0].walls:
                    wall.material = f"{compose_id}_wall"
                    wall.height = 2.85
            
            # Update room objects with composed objects
            composed_room.objects = new_objects
            composed_layout.rooms[0] = composed_room
            
            # Save composed layout
            composed_layout_dict = asdict(composed_layout)
            layout_save_path = os.path.join(compose_layouts_dir, f"{compose_id}.json")
            with open(layout_save_path, "w") as f:
                json.dump(composed_layout_dict, f, indent=4)
            
            print(f"  Saved layout: {layout_save_path}")
            
            # Render video
            video_save_path = os.path.join(compose_videos_dir, f"{compose_id}_longest_edges.mp4")
            try:
                render_room_circular_video(
                    layout=composed_layout,
                    room_id=room_id,
                    output_path=video_save_path,
                    resolution=1920,
                    angle_step=45,
                    fps=24
                )
                print(f"  Saved video: {video_save_path}")
                
                composition_summary["compositions"].append({
                    "compose_id": compose_id,
                    "layout_path": layout_save_path,
                    "video_path": video_save_path,
                    "object_selections": object_selections
                })
                successful_compositions += 1
                
            except Exception as e:
                print(f"  Warning: Failed to render video: {e}")
                composition_summary["compositions"].append({
                    "compose_id": compose_id,
                    "layout_path": layout_save_path,
                    "video_path": None,
                    "object_selections": object_selections,
                    "render_error": str(e)
                })
        
        except Exception as e:
            print(f"  Error generating composition {compose_idx}: {e}")
            import traceback
            traceback.print_exc()
            composition_summary["compositions"].append({
                "compose_id": f"compose_{compose_idx:04d}_failed",
                "error": str(e)
            })
    
    # Save composition summary
    summary_save_path = os.path.join(compose_base_dir, f"composition_summary.json")
    composition_summary["successful_compositions"] = successful_compositions
    composition_summary["failed_compositions"] = compose_num - successful_compositions
    
    with open(summary_save_path, "w") as f:
        json.dump(composition_summary, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"Composition Complete!")
    print(f"{'='*80}")
    print(f"Total compositions: {compose_num}")
    print(f"Successful: {successful_compositions}")
    print(f"Failed: {compose_num - successful_compositions}")
    print(f"Layouts saved to: {compose_layouts_dir}")
    print(f"Videos saved to: {compose_videos_dir}")
    print(f"Materials saved to: {materials_dir}")
    print(f"Summary saved to: {summary_save_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compose random combinations of pose-augmented objects from scene_level_pose_aug.py"
    )
    parser.add_argument("--layout_id", type=str, required=True,
                       help="ID of the layout")
    parser.add_argument("--room_id", type=str, required=True,
                       help="ID of the room")
    parser.add_argument("--pose_aug_name", type=str, required=True,
                       help="Name of the pose augmentation group")
    parser.add_argument("--compose_num", type=int, default=50,
                       help="Number of composed layouts to generate (default: 50)")
    args = parser.parse_args()
    
    compose_scene_level_pose_augmentations(
        layout_id=args.layout_id,
        room_id=args.room_id,
        pose_aug_name=args.pose_aug_name,
        compose_num=args.compose_num
    )
