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


def compose_scene_level_type_augmentations(layout_id, room_id, type_aug_name, pose_aug_name, compose_num):
    """
    Compose random combinations of type-augmented objects into complete layouts.
    
    Args:
        layout_id: ID of the layout
        room_id: ID of the room
        type_aug_name: Name of the type augmentation (from scene_level_type_aug.py)
        pose_aug_name: Name of the pose augmentation group
        compose_num: Number of composed layouts to generate
    """
    
    # Load scene-level metadata
    layout_dir = os.path.join(RESULTS_DIR, layout_id)
    scene_metadata_path = os.path.join(layout_dir, f"{type_aug_name}_scene_level_type_augmentation_metadata.json")
    
    if not os.path.exists(scene_metadata_path):
        raise ValueError(f"Scene-level metadata not found: {scene_metadata_path}")
    
    with open(scene_metadata_path, "r") as f:
        scene_metadata = json.load(f)
    
    print(f"{'='*80}")
    print(f"Scene-Level Type Augmentation Composition")
    print(f"{'='*80}")
    print(f"Layout ID: {layout_id}")
    print(f"Room ID: {room_id}")
    print(f"Type Aug Name: {type_aug_name}")
    print(f"Pose Aug Name: {pose_aug_name}")
    print(f"Compose Num: {compose_num}")
    print(f"Found {len(scene_metadata['objects_augmented'])} augmented objects")
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
    
    # Create output directories
    compose_base_dir = os.path.join(layout_dir, f"{type_aug_name}_composed_{pose_aug_name}")
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
        "type_aug_name": type_aug_name,
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
            
            # For each augmented object, randomly select a type variant
            for obj_info in scene_metadata['objects_augmented']:
                # Skip objects with errors
                if 'error' in obj_info:
                    print(f"  Skipping {obj_info['object_id']} (had error during augmentation)")
                    # Find the original object and add it
                    original_obj = next((o for o in base_objects if o.id == obj_info['object_id']), None)
                    if original_obj:
                        new_objects.append(copy.deepcopy(original_obj))
                    continue
                
                base_object_id = obj_info['object_id']
                object_aug_name = obj_info['aug_name']
                
                # Find available type candidates that have been settled
                object_aug_dir = os.path.join(layout_dir, object_aug_name)
                available_variants = []
                
                # Check each type candidate for settled layouts
                metadata_path = os.path.join(object_aug_dir, f"{object_aug_name}_type_candidates_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        obj_metadata = json.load(f)
                    
                    for candidate_info in obj_metadata['candidates']:
                        candidate_id = candidate_info['layout_id']
                        settled_layout_path = os.path.join(
                            object_aug_dir,
                            candidate_id,
                            pose_aug_name,
                            f"{candidate_id}_sim_corrected.json"
                        )
                        
                        if os.path.exists(settled_layout_path):
                            available_variants.append({
                                'candidate_id': candidate_id,
                                'layout_path': settled_layout_path,
                                'type_candidate_path': os.path.join(object_aug_dir, candidate_info['candidate_file'])
                            })
                
                print(f"  Object {base_object_id}: {len(available_variants)} available variants")
                
                if len(available_variants) > 0:
                    # Randomly select a variant
                    selected_variant = random.choice(available_variants)
                    
                    # Load the type candidate info to get ID mapping
                    with open(selected_variant['type_candidate_path'], "r") as f:
                        type_candidate_info = json.load(f)
                    old_id_to_new_id_map = type_candidate_info['old_id_to_new_id_map']
                    
                    # Load the settled layout
                    with open(selected_variant['layout_path'], "r") as f:
                        variant_layout_data = json.load(f)
                    variant_layout = dict_to_floor_plan(variant_layout_data)
                    variant_room = next((r for r in variant_layout.rooms if r.id == room_id), None)
                    
                    if variant_room:
                        # Add the new objects from this variant
                        for old_id, new_id in old_id_to_new_id_map.items():
                            new_object = next((o for o in variant_room.objects if o.id == new_id), None)
                            if new_object:
                                new_objects.append(copy.deepcopy(new_object))
                        
                        object_selections[base_object_id] = selected_variant['candidate_id']
                        print(f"    Selected: {selected_variant['candidate_id']}")
                else:
                    # No variants available, use original object and its descendants
                    print(f"    No variants available, using original")
                    original_obj = next((o for o in base_objects if o.id == base_object_id), None)
                    if original_obj and (original_obj.place_id == "wall" or original_obj.place_id == "floor"):
                        new_objects.append(copy.deepcopy(original_obj))
                        
                        # Add descendants
                        object_map = {o.id: o for o in base_objects}
                        children_map = {}
                        for o in base_objects:
                            parent_id = o.place_id
                            if isinstance(parent_id, str) and parent_id in object_map:
                                children_map.setdefault(parent_id, []).append(o)
                        
                        # BFS to collect descendants
                        queue = [base_object_id]
                        visited = set()
                        while queue:
                            current_id = queue.pop(0)
                            if current_id in visited:
                                continue
                            visited.add(current_id)
                            for child in children_map.get(current_id, []):
                                if child.id not in visited:
                                    new_objects.append(copy.deepcopy(child))
                                    queue.append(child.id)
                        
                        object_selections[base_object_id] = "original"
            
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
    summary_path = os.path.join(compose_base_dir, f"composition_summary.json")
    composition_summary["successful_compositions"] = successful_compositions
    composition_summary["failed_compositions"] = compose_num - successful_compositions
    
    with open(summary_path, "w") as f:
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
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compose random combinations of type-augmented objects from scene_level_type_aug.py"
    )
    parser.add_argument("--layout_id", type=str, required=True,
                       help="ID of the layout")
    parser.add_argument("--room_id", type=str, required=True,
                       help="ID of the room")
    parser.add_argument("--type_aug_name", type=str, required=True,
                       help="Name of the type augmentation (same as --aug_name from scene_level_type_aug.py)")
    parser.add_argument("--pose_aug_name", type=str, required=True,
                       help="Name of the pose augmentation group")
    parser.add_argument("--compose_num", type=int, default=50,
                       help="Number of composed layouts to generate (default: 50)")
    args = parser.parse_args()
    
    compose_scene_level_type_augmentations(
        layout_id=args.layout_id,
        room_id=args.room_id,
        type_aug_name=args.type_aug_name,
        pose_aug_name=args.pose_aug_name,
        compose_num=args.compose_num
    )
