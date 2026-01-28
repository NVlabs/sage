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

from objects.object_augmentation import (
    generate_type_augmentation_candidates,
    generate_type_augmentation_candidates_mixed
)

from utils import get_layout_from_scene_save_dir
from constants import RESULTS_DIR, SERVER_ROOT_DIR


def generate_type_candidates(layout_id, room_id, object_id, aug_num, aug_name):
    """Generate type augmentation candidates for an object tree"""
    
    json_file_path = os.path.join(SERVER_ROOT_DIR, f"results/{layout_id}/{layout_id}.json")

    current_layout = get_layout_from_scene_save_dir(os.path.dirname(json_file_path))
    scene_save_dir = os.path.join(RESULTS_DIR, current_layout.id)

    room = next((r for r in current_layout.rooms if r.id == room_id), None)
    if room is None:
        raise ValueError(f"Room with ID {room_id} not found in layout")
    object = next((o for o in room.objects if o.id == object_id), None)
    if object is None:
        raise ValueError(f"Object with ID {object_id} not found in room {room_id}")

    type_candidates = generate_type_augmentation_candidates_mixed(
        current_layout, room, object, aug_num, aug_name)
    
    print(f"Generated {len(type_candidates)} type augmentation candidates")
    
    # Save metadata about all candidates
    candidates_metadata = {
        "original_layout_id": layout_id,
        "room_id": room_id,
        "object_id": object_id,
        "aug_name": aug_name,
        "total_candidates": len(type_candidates),
        "candidates": []
    }
    
    for layout_copy_id, layout_copy, room_copy, new_objects in type_candidates:
        candidates_metadata["candidates"].append({
            "layout_id": layout_copy_id,
            "candidate_file": f"{layout_copy_id}_type_candidate.json",
            "new_object_count": len(new_objects),
            "new_object_ids": [obj.id for obj in new_objects]
        })
    
    # Save the metadata
    metadata_path = os.path.join(RESULTS_DIR, layout_id, aug_name, f"{aug_name}_type_candidates_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(candidates_metadata, f, indent=4)
    
    print(f"Type candidates metadata saved to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_id", type=str, required=True)
    parser.add_argument("--room_id", type=str, required=True)
    parser.add_argument("--object_id", type=str, required=True)
    parser.add_argument("--aug_name", type=str, required=True)
    parser.add_argument("--aug_num", type=int, default=5)
    args = parser.parse_args()
    generate_type_candidates(args.layout_id, args.room_id, args.object_id, args.aug_num, args.aug_name)
