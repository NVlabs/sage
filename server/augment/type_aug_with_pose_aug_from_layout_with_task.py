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

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from isaacsim.isaac_mcp.server import (
    get_room_layout_scene_usd_separate_from_layout
)

from objects.object_augmentation import (
    object_augmentation_pose_support_tree_with_reach_test_parallel_on_type_augmentation,
    object_augmentation_pose_object_tree_sim_correction
)
from dataclasses import asdict

from tex_utils import dict_to_floor_plan
from constants import RESULTS_DIR
import copy
from scipy.spatial.transform import Rotation
from models import FloorPlan


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


def pose_augment_type_candidate(
    layout_id, 
    room_id, 
    type_candidate_id, 
    object_id, 
    aug_num, 
    type_aug_name, 
    pose_aug_name,
    reachable_object_ids
):
    """Apply pose augmentation to a specific type augmentation candidate's support tree"""
    
    # Load the type candidate
    candidate_file_path = os.path.join(RESULTS_DIR, layout_id, type_aug_name, f"{type_candidate_id}_type_candidate.json")
    
    if not os.path.exists(candidate_file_path):
        raise ValueError(f"Type candidate file not found: {candidate_file_path}")
    
    with open(candidate_file_path, "r") as f:
        candidate_data = json.load(f)
    
    # Reconstruct the layout from the candidate data
    layout_dict = candidate_data["layout"]
    layout_copy = dict_to_floor_plan(layout_dict)
    
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


    # Map the object_id using old_id_to_new_id_map to get the new object ID
    new_object_id = old_id_to_new_id_map.get(object_id, None)
    assert new_object_id is not None, f"Object ID {object_id} not found in old_id_to_new_id_map"
    print(f"Mapped object ID: {object_id} -> {new_object_id}")

    # Map the reachable_object_ids as well
    print(f"Mapping reachable object IDs: {reachable_object_ids}")
    reachable_object_ids = [old_id_to_new_id_map.get(obj_id, None) for obj_id in reachable_object_ids]
    print(f"Mapped reachable object IDs: {reachable_object_ids}")
    
    # Find the parent object in the room
    parent_object = next((obj for obj in room_copy.objects if obj.id == new_object_id), None)
    if parent_object is None:
        raise ValueError(f"Object with ID {new_object_id} not found in room {room_id}")
    
    augmented_layouts_info = []

    logs = []
    
    for aug_iter in range(aug_num):
        # Create a deep copy of layout and room for this iteration
        layout_iter = copy.deepcopy(layout_copy)
        room_iter = next((r for r in layout_iter.rooms if r.id == room_id), None)
        parent_object_iter = next((obj for obj in room_iter.objects if obj.id == new_object_id), None)
        
        # Apply pose augmentation to the support tree with sim correction
        new_room_objects, success = object_augmentation_pose_object_tree_sim_correction(
            layout_iter, room_iter, parent_object_iter,
            reachable_object_ids=reachable_object_ids,
            layout_file_name=layout_id
        )
        
        if not success:
            print(f"Warning: Sim correction failed for iteration {aug_iter}")
            logs.append({
                "aug_iter": aug_iter,
                "success": False,
            })
            continue
        
        # Update the room with the corrected objects
        room_iter.objects = new_room_objects

        layout_iter.rooms[0] = room_iter
        
        # Create unique layout ID for this augmentation
        augmented_layout_id = f"{type_candidate_id}_{pose_aug_name}_{aug_iter}"
        
        # Save the augmented layout
        augmented_layout_path = os.path.join(pose_aug_dir, f"{augmented_layout_id}.json")
        with open(augmented_layout_path, "w") as f:
            json.dump(asdict(layout_iter), f, indent=4)
        
        augmented_layouts_info.append((augmented_layout_id, layout_iter))
        print(f"Generated augmented layout: {augmented_layout_id}")
        logs.append({
            "aug_iter": aug_iter,
            "success": True,
            "augmented_layout_id": augmented_layout_id,
        })
    
    print(f"Generated {len(augmented_layouts_info)} pose-augmented layouts from type candidate {type_candidate_id}")
    
    # Create output metadata similar to pose_aug_v1_reach.py  
    scene_save_dir = os.path.join(RESULTS_DIR, layout_id)
    type_aug_base_dir = os.path.join(scene_save_dir, type_aug_name, type_candidate_id)
    usd_collection_dir = os.path.join(type_aug_base_dir, f"usd_collection_{pose_aug_name}")
    os.makedirs(usd_collection_dir, exist_ok=True)
    logs_path = os.path.join(type_aug_base_dir, f"logs_{pose_aug_name}.json")
    layout_template_save_path = os.path.join(type_aug_base_dir, f"layout_template_{pose_aug_name}.json")

    with open(layout_template_save_path, "w") as f:
        json.dump(asdict(layout_copy), f, indent=4)
    print(f"Layout template saved to: {layout_template_save_path}")

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
        # first_layout_id, first_layout = augmented_layouts_info[0]
        # # The individual layout JSON files are already saved in pose_aug_dir
        # first_layout_json_path = os.path.join(pose_aug_dir, f"{first_layout_id}.json")
        # get_room_layout_scene_usd_separate_from_layout(first_layout_json_path, usd_collection_dir)
        get_room_layout_scene_usd_separate_from_layout(layout_template_save_path, usd_collection_dir)
        logs.append({
            "info": f"USD generation completed with layout template at path: {layout_template_save_path}",
        })
    else:
        print("Warning: No augmented layouts generated, skipping USD generation")
        logs.append({
            "warning": "No augmented layouts generated, skipping USD generation",
        })

    # save logs of the pose augmentation process
    with open(logs_path, "w") as f:
        json.dump(logs, f, indent=4)
    print(f"Logs saved to: {logs_path}")

    # pdb.set_trace()


def pose_augment_all_type_candidates(
    layout_id, 
    aug_num, 
    type_aug_name, 
    pose_aug_name 
):
    """Apply pose augmentation to all type augmentation candidates from metadata file"""

    json_file_path = os.path.join(RESULTS_DIR, layout_id, f"{layout_id}.json")

    current_layout = dict_to_floor_plan(json.load(open(json_file_path)))
    room = current_layout.rooms[0] # default to only one room in the layout

    policy_analysis = current_layout.policy_analysis
    minimum_required_objects = policy_analysis["minimum_required_objects"]
    all_matched_objects = []
    for obj_info in minimum_required_objects:
        all_matched_objects.extend(obj_info["matched_object_ids"])
    
    
    # find the parent object of every matched object
    parent_object_ids = []
    for matched_object_id in all_matched_objects:
        matched_object = copy.deepcopy(next((o for o in room.objects if o.id == matched_object_id), None))
        assert matched_object is not None, f"Object with ID {matched_object_id} not found in room"
        while True:
            if matched_object.place_id in ["floor", "wall"]:
                break
            matched_object_parent_id = matched_object.place_id
            matched_object = next((o for o in room.objects if o.id == matched_object_parent_id), None)
            assert matched_object is not None, f"Object with ID {matched_object_parent_id} not found in room"
        parent_object_ids.append(matched_object.id)
    
    parent_object_ids = list(set(parent_object_ids))
    assert len(parent_object_ids) == 1, "currently only support one parent object for the matched objects"
    object_id = parent_object_ids[0]
    room_id = room.id

    all_matched_objects_not_parent = [matched_object_id for matched_object_id in all_matched_objects if matched_object_id != object_id]
    print(f"All matched objects not parent: {all_matched_objects_not_parent}")
    
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
                reachable_object_ids=all_matched_objects_not_parent,
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
    parser.add_argument("--type_aug_name", type=str, required=True)
    parser.add_argument("--pose_aug_name", type=str, required=True)
    parser.add_argument("--aug_num", type=int, default=10, 
                       help="Number of pose augmentations to perform on each type candidate")
    args = parser.parse_args()
    pose_augment_all_type_candidates(
        args.layout_id, 
        args.aug_num, 
        args.type_aug_name, 
        args.pose_aug_name
    )
