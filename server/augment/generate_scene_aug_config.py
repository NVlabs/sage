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
import os
import json
import argparse
from constants import RESULTS_DIR

def generate_scene_aug_config(layout_id, room_id, selected_object_ids):
    """Generate scene augmentation configuration"""
    with open(f"{RESULTS_DIR}/{layout_id}/{layout_id}.json", "r") as f:
        layout_data = json.load(f)
    room = next((room for room in layout_data["rooms"] if room["id"] == room_id), None)
    if not room:
        raise ValueError(f"Room with ID {room_id} not found in layout")
    
    selected_objects = []
    for object_id in selected_object_ids:
        object = next((object for object in room["objects"] if object["id"] == object_id), None)
        if not object:
            raise ValueError(f"Object with ID {object_id} not found in room")
        selected_objects.append(object)
    
    scene_aug_dict = {
        "id": layout_id,
        "selected_objects": selected_objects
    }

    return scene_aug_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_id", type=str, required=True)
    parser.add_argument("--room_id", type=str, required=True)
    parser.add_argument("--selected_object_ids", type=str, nargs="+", required=True)
    parser.add_argument("--save_name", type=str, required=True)
    args = parser.parse_args()

    scene_aug_dict = generate_scene_aug_config(args.layout_id, args.room_id, args.selected_object_ids)

    with open(f"{RESULTS_DIR}/{args.layout_id}/{args.save_name}.json", "w") as f:
        json.dump(scene_aug_dict, f, indent=4)

    print(f"Scene augmentation configuration saved to {RESULTS_DIR}/{args.layout_id}/{args.save_name}.json")