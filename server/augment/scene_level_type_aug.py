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

from objects.object_augmentation import (
    generate_type_augmentation_candidates,
    generate_type_augmentation_candidates_linear
)

from utils import get_layout_from_scene_save_dir
from constants import RESULTS_DIR, SERVER_ROOT_DIR


def generate_scene_level_type_candidates(layout_id, room_id, aug_name, aug_num):
    """Generate type augmentation candidates for all floor and wall objects in a room"""
    
    json_file_path = os.path.join(SERVER_ROOT_DIR, f"results/{layout_id}/{layout_id}.json")

    current_layout = get_layout_from_scene_save_dir(os.path.dirname(json_file_path))
    scene_save_dir = os.path.join(RESULTS_DIR, current_layout.id)

    room = next((r for r in current_layout.rooms if r.id == room_id), None)
    if room is None:
        raise ValueError(f"Room with ID {room_id} not found in layout")
    
    # Filter objects that are on floor or wall
    floor_and_wall_objects = [
        obj for obj in room.objects 
        if obj.place_id == "floor" or obj.place_id == "wall"
    ]
    
    if not floor_and_wall_objects:
        print(f"No floor or wall objects found in room {room_id}")
        return
    
    print(f"Found {len(floor_and_wall_objects)} floor and wall objects to augment:")
    for obj in floor_and_wall_objects:
        print(f"  - {obj.id} (type: {obj.type}, place: {obj.place_id})")
    
    # Metadata for all objects
    scene_metadata = {
        "original_layout_id": layout_id,
        "room_id": room_id,
        "aug_name": aug_name,
        "aug_num": aug_num,
        "objects_augmented": []
    }
    
    # Iterate through each floor and wall object
    for object_to_aug in floor_and_wall_objects:
        print(f"\n{'='*80}")
        print(f"Processing object: {object_to_aug.id}")
        print(f"{'='*80}")
        
        object_aug_name = f"{object_to_aug.id}_{aug_name}"
        
        try:
            # Generate type augmentation candidates for this object
            type_candidates = generate_type_augmentation_candidates_linear(
                current_layout, room, object_to_aug, aug_num, object_aug_name)
            
            print(f"Generated {len(type_candidates)} type augmentation candidates for {object_to_aug.id}")
            
            # Save metadata about candidates for this object
            object_candidates_metadata = {
                "object_id": object_to_aug.id,
                "object_type": object_to_aug.type,
                "object_place_id": object_to_aug.place_id,
                "aug_name": object_aug_name,
                "total_candidates": len(type_candidates),
                "candidates": []
            }
            
            for layout_copy_id, layout_copy, room_copy, new_objects in type_candidates:
                object_candidates_metadata["candidates"].append({
                    "layout_id": layout_copy_id,
                    "candidate_file": f"{layout_copy_id}_type_candidate.json",
                    "new_object_count": len(new_objects),
                    "new_object_ids": [obj.id for obj in new_objects]
                })
            
            # Save the metadata for this object
            object_metadata_path = os.path.join(
                RESULTS_DIR, layout_id, object_aug_name, 
                f"{object_aug_name}_type_candidates_metadata.json"
            )
            with open(object_metadata_path, "w") as f:
                json.dump(object_candidates_metadata, f, indent=4)
            
            print(f"Type candidates metadata saved to: {object_metadata_path}")
            
            # Add to scene-level metadata
            scene_metadata["objects_augmented"].append({
                "object_id": object_to_aug.id,
                "object_type": object_to_aug.type,
                "aug_name": object_aug_name,
                "candidates_count": len(type_candidates),
                "metadata_file": object_metadata_path
            })
            
        except Exception as e:
            print(f"Error processing object {object_to_aug.id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Add error to metadata
            scene_metadata["objects_augmented"].append({
                "object_id": object_to_aug.id,
                "object_type": object_to_aug.type,
                "aug_name": object_aug_name,
                "error": str(e)
            })
    
    # Save scene-level metadata
    scene_metadata_path = os.path.join(
        RESULTS_DIR, layout_id, 
        f"{aug_name}_scene_level_type_augmentation_metadata.json"
    )
    with open(scene_metadata_path, "w") as f:
        json.dump(scene_metadata, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"Scene-level type augmentation complete!")
    print(f"Scene metadata saved to: {scene_metadata_path}")
    print(f"Total objects processed: {len(floor_and_wall_objects)}")
    print(f"Successful augmentations: {sum(1 for obj in scene_metadata['objects_augmented'] if 'error' not in obj)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate type augmentation candidates for all floor and wall objects in a room"
    )
    parser.add_argument("--layout_id", type=str, required=True, 
                       help="ID of the layout to process")
    parser.add_argument("--room_id", type=str, required=True,
                       help="ID of the room to process")
    parser.add_argument("--aug_name", type=str, required=True,
                       help="Name for this augmentation batch")
    parser.add_argument("--aug_num", type=int, default=5,
                       help="Number of augmentation candidates to generate per object")
    args = parser.parse_args()
    
    generate_scene_level_type_candidates(
        args.layout_id, 
        args.room_id, 
        args.aug_name, 
        args.aug_num
    )

