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

from tex_utils import dict_to_floor_plan

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from objects.object_augmentation import (
    object_augmentation_pose_object_tree_sim_correction
)

from constants import RESULTS_DIR
from dataclasses import asdict

import copy

def pose_augment_from_parent_object(
    layout,
    room,
    parent_object,
    aug_num,
    pose_aug_name,
):

    # Reconstruct the layout from the candidate data
    layout_copy = copy.deepcopy(layout)
    room_copy = copy.deepcopy(room)

    layout_id = layout.id
    room_id = room.id
    parent_object_id = parent_object.id

    print(f"Applying sim correction pose placement to object tree starting from object_id: {parent_object_id}")

    # Create the pose augmentation subdirectory structure
    pose_aug_dir = os.path.join(RESULTS_DIR, layout_id, pose_aug_name)
    os.makedirs(pose_aug_dir, exist_ok=True)

    augmented_layouts_info = {
        "parent_object_id": parent_object_id,
        "pose_augmentations": [],
    }

    for aug_iter in range(max(0, int(aug_num))):

        layout_iter = copy.deepcopy(layout_copy)
        room_iter = next((r for r in layout_iter.rooms if r.id == room_id), None)
        parent_object_iter = next((obj for obj in room_iter.objects if obj.id == parent_object_id), None)

        # Apply sim correction to place all descendants of the parent object
        new_room_objects, success = object_augmentation_pose_object_tree_sim_correction(
            layout_iter, room_iter, parent_object_iter,
            reachable_object_ids=[],
            layout_file_name=layout_id,
            always_return_room=True
        )

        if success:
            augmented_layouts_info["pose_augmentations"].append([asdict(obj) for obj in new_room_objects])
    
    # Save the pose augmentations
    pose_aug_file_path = os.path.join(pose_aug_dir, f"pose_aug_{parent_object_id}.json")

    with open(pose_aug_file_path, "w") as f:
        json.dump(augmented_layouts_info, f, indent=4)
    
    print(f"Generated pose augmentations for object: {parent_object_id}")

    return augmented_layouts_info

def pose_augment_scene_level_all_objects(layout_id, room_id, aug_num, pose_aug_name):
    """Apply pose augmentation to all floor and wall objects processed by scene_level_type_aug"""
    
    print(f"{'='*80}")
    print(f"Scene-Level Pose Augmentation")
    print(f"{'='*80}")
    print(f"Layout ID: {layout_id}")
    print(f"Room ID: {room_id}")
    print(f"{'='*80}\n")

    layout_json_path = os.path.join(RESULTS_DIR, layout_id, f"{layout_id}.json")
    layout_dict = json.load(open(layout_json_path, "r"))
    layout = dict_to_floor_plan(layout_dict)
    target_room = next((r for r in layout.rooms if r.id == room_id), None)
    
    overall_summary = {
        "layout_id": layout_id,
        "room_id": room_id,
        "pose_aug_name": pose_aug_name,
        "total_objects": len(target_room.objects),
        "processed_objects": []
    }
    
    floor_and_wall_objects = [obj for obj in target_room.objects if obj.place_id in ["floor", "wall"]]
    print(f"Floor and wall objects: {[obj.id for obj in floor_and_wall_objects]}")
    # assert False
    # debug
    # floor_and_wall_objects = [obj for obj in target_room.objects if obj.id == "room_2adee539_dresser_fff38e04"]
    
    # Process each object that was augmented
    for parent_object in floor_and_wall_objects:

        # check whether the object has children objects
        # if not, skip
        children_objects = [obj for obj in target_room.objects if obj.place_id == parent_object.id]
        if len(children_objects) == 0:
            print(f"Object {parent_object.id} has no children objects, skipping")
            continue
        
        object_id = parent_object.id
        
        print(f"\n{'='*80}")
        print(f"Processing Object: {object_id}")
        print(f"{'='*80}\n")
        
        augmented_layouts_info = pose_augment_from_parent_object(
            layout=layout,
            room=target_room,
            parent_object=parent_object,
            aug_num=aug_num,
            pose_aug_name=pose_aug_name,
        )
        
        overall_summary["processed_objects"].append({
            "parent_object": object_id,
            "num_augmented_layouts": len(augmented_layouts_info["pose_augmentations"]),
        })
        print(f"\n✓ Successfully completed pose augmentation for object: {object_id}\n")
            
    # Save overall scene-level summary
    scene_summary_path = os.path.join(RESULTS_DIR, layout_id, pose_aug_name, "summary.json")
    with open(scene_summary_path, "w") as f:
        json.dump(overall_summary, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"Scene-Level Pose Augmentation Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply settling (pose augmentation with sim correction) to all floor/wall objects "
                    "processed by scene_level_type_aug.py"
    )
    parser.add_argument("--layout_id", type=str, required=True,
                       help="ID of the layout")
    parser.add_argument("--room_id", type=str, required=True,
                       help="ID of the room")
    parser.add_argument("--pose_aug_name", type=str, required=True,
                       help="Name for the pose augmentation group")
    parser.add_argument("--aug_num", type=int, default=5, 
                       help="Number of pose augmentations to perform on each parent object (default: 5)")
    args = parser.parse_args()
    
    pose_augment_scene_level_all_objects(
        layout_id=args.layout_id,
        room_id=args.room_id,
        aug_num=args.aug_num,
        pose_aug_name=args.pose_aug_name,
    )
