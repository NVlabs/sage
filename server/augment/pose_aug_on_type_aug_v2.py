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
import pdb

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from isaacsim.isaac_mcp.server import (
    get_room_layout_scene_usd_separate_from_layout
)

from objects.object_augmentation import (
    object_augmentation_pose_object_tree_with_reach_test_parallel_on_type_augmentation,
    object_augmentation_pose_support_tree_with_reach_test_parallel_on_type_augmentation
)

from utils import get_layout_from_scene_save_dir
from constants import RESULTS_DIR, SERVER_ROOT_DIR
from scipy.spatial.transform import Rotation
from models import FloorPlan, Room, Object, Point3D, Euler, Dimensions
from dataclasses import asdict
from typing import Dict, List, Any


def _dict_to_dataclass(cls, data):
    """Recursively convert dictionary to dataclass instances"""
    if isinstance(data, dict):
        # Get the field types of the dataclass
        field_types = {field.name: field.type for field in cls.__dataclass_fields__.values()}
        kwargs = {}
        
        for key, value in data.items():
            if key in field_types:
                field_type = field_types[key]
                # Handle nested dataclasses and lists
                if hasattr(field_type, '__dataclass_fields__'):
                    kwargs[key] = _dict_to_dataclass(field_type, value)
                elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                    # Handle List[SomeDataclass]
                    inner_type = field_type.__args__[0]
                    if hasattr(inner_type, '__dataclass_fields__'):
                        kwargs[key] = [_dict_to_dataclass(inner_type, item) for item in value]
                    else:
                        kwargs[key] = value
                else:
                    kwargs[key] = value
        return cls(**kwargs)
    return data


def pose_augment_type_candidate(layout_id, room_id, type_candidate_id, object_id, aug_num, type_aug_name, pose_aug_name, group_size=10):
    """Apply pose augmentation to a specific type augmentation candidate's support tree"""
    
    # Load the type candidate
    candidate_file_path = os.path.join(RESULTS_DIR, layout_id, type_aug_name, f"{type_candidate_id}_type_candidate.json")
    
    if not os.path.exists(candidate_file_path):
        raise ValueError(f"Type candidate file not found: {candidate_file_path}")
    
    with open(candidate_file_path, "r") as f:
        candidate_data = json.load(f)
    
    # Reconstruct the layout from the candidate data
    layout_dict = candidate_data["layout"]
    layout_copy = _dict_to_dataclass(FloorPlan, layout_dict)
    
    room_copy = next((r for r in layout_copy.rooms if r.id == room_id), None)
    if room_copy is None:
        raise ValueError(f"Room with ID {room_id} not found in type candidate")
    
    # Get the old_id to new_id mapping
    old_id_to_new_id_map = candidate_data["old_id_to_new_id_map"]
    
    print(f"Applying pose augmentation to support tree starting from object_id: {object_id}")
    print(f"Available mappings: {old_id_to_new_id_map}")

    # Create the pose augmentation subdirectory structure
    pose_aug_dir = os.path.join(RESULTS_DIR, layout_id, type_aug_name, type_candidate_id, pose_aug_name)
    os.makedirs(pose_aug_dir, exist_ok=True)

    augmented_layouts_info = object_augmentation_pose_support_tree_with_reach_test_parallel_on_type_augmentation(
        layout_copy, room_copy, object_id, old_id_to_new_id_map, layout_id, aug_num, 
        f"{pose_aug_name}", reach_threshold=0.8, group_size=group_size,
        custom_save_dir=pose_aug_dir)
    
    print(f"Generated {len(augmented_layouts_info)} pose-augmented layouts from type candidate {type_candidate_id}")
    
    # Create output metadata similar to pose_aug_v1_reach.py  
    scene_save_dir = os.path.join(RESULTS_DIR, layout_id)
    type_aug_base_dir = os.path.join(scene_save_dir, type_aug_name, type_candidate_id)
    usd_collection_dir = os.path.join(type_aug_base_dir, f"usd_collection_{pose_aug_name}")
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

    with open(os.path.join(type_aug_base_dir, f"all_augmented_layouts_info_{pose_aug_name}.json"), "w") as f:
        json.dump(all_augmented_layouts_info, f, indent=4)
    
    # Use the first pose-augmented layout file for USD generation since it has the correct object IDs
    if len(augmented_layouts_info) > 0:
        first_layout_id, first_layout = augmented_layouts_info[0]
        # The individual layout JSON files are already saved in pose_aug_dir
        first_layout_json_path = os.path.join(pose_aug_dir, f"{first_layout_id}.json")
        get_room_layout_scene_usd_separate_from_layout(first_layout_json_path, usd_collection_dir)
    else:
        print("Warning: No augmented layouts generated, skipping USD generation")

    # pdb.set_trace()


def pose_augment_all_type_candidates(layout_id, room_id, object_id, aug_num, type_aug_name, pose_aug_name, group_size=10):
    """Apply pose augmentation to all type augmentation candidates from metadata file"""
    
    # Load the type candidates metadata
    metadata_path = os.path.join(RESULTS_DIR, layout_id, type_aug_name, f"{type_aug_name}_type_candidates_metadata.json")
    
    if not os.path.exists(metadata_path):
        raise ValueError(f"Type candidates metadata file not found: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    print(f"Found {metadata['total_candidates']} type candidates to process")
    print(f"Original object_id: {metadata['object_id']}, provided object_id: {object_id}")
    
    # Verify the metadata matches our parameters
    if metadata["original_layout_id"] != layout_id:
        raise ValueError(f"Layout ID mismatch: metadata has {metadata['original_layout_id']}, provided {layout_id}")
    if metadata["room_id"] != room_id:
        raise ValueError(f"Room ID mismatch: metadata has {metadata['room_id']}, provided {room_id}")
    if metadata["object_id"] != object_id:
        raise ValueError(f"Object ID mismatch: metadata has {metadata['object_id']}, provided {object_id}")
    
    successful_candidates = 0
    failed_candidates = 0
    
    # Process each type candidate
    # print(metadata["candidates"])
    for candidate_info in metadata["candidates"]:
        type_candidate_id = candidate_info["layout_id"]
        print(f"\n--- Processing type candidate: {type_candidate_id} ---")
        
        try:
            pose_augment_type_candidate(
                layout_id=layout_id,
                room_id=room_id, 
                type_candidate_id=type_candidate_id,
                object_id=object_id,
                aug_num=aug_num,
                type_aug_name=type_aug_name,
                pose_aug_name=pose_aug_name,
                group_size=group_size
            )
            successful_candidates += 1
            print(f"✓ Successfully processed type candidate: {type_candidate_id}")
            
        except Exception as e:
            failed_candidates += 1
            print(f"✗ Failed to process type candidate {type_candidate_id}: {str(e)}")
            continue
    
    print(f"\n=== Summary ===")
    print(f"Total candidates: {metadata['total_candidates']}")
    print(f"Successful: {successful_candidates}")
    print(f"Failed: {failed_candidates}")
    
    # Save overall summary
    summary_path = os.path.join(RESULTS_DIR, layout_id, type_aug_name, f"pose_aug_summary_{pose_aug_name}.json")
    summary = {
        "type_aug_name": type_aug_name,
        "pose_aug_name": pose_aug_name,
        "total_type_candidates": metadata['total_candidates'],
        "successful_pose_augs": successful_candidates,
        "failed_pose_augs": failed_candidates,
        "processed_candidates": [candidate["layout_id"] for candidate in metadata["candidates"]]
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_id", type=str, required=True)
    parser.add_argument("--room_id", type=str, required=True)
    parser.add_argument("--object_id", type=str, required=True,
                       help="Original object ID to start the support tree pose augmentation from")
    parser.add_argument("--type_aug_name", type=str, required=True)
    parser.add_argument("--pose_aug_name", type=str, required=True)
    parser.add_argument("--aug_num", type=int, default=10, 
                       help="Number of pose augmentations to perform on each type candidate")
    parser.add_argument("--group_size", type=int, default=10)
    args = parser.parse_args()
    pose_augment_all_type_candidates(args.layout_id, args.room_id, 
                                   args.object_id, args.aug_num, args.type_aug_name, args.pose_aug_name, args.group_size)
