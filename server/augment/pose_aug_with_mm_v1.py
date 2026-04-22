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
import sys 
import os
import shutil
import json
from layout import policy_analysis
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import RESULTS_DIR
from objects.object_mobile_manipulation_utils import (
    sample_pick_object_pose,
    sample_robot_location,
    sample_robot_place_location,
    plan_robot_traj
)
from utils import dict_to_floor_plan
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir_name", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--layout_id", type=str, required=True)

    args = parser.parse_args()
    save_dir_name = args.save_dir_name
    num_samples = args.num_samples

    layout_id = args.layout_id
    save_dir = os.path.join(RESULTS_DIR, layout_id, f"{save_dir_name}")
    os.makedirs(save_dir, exist_ok=True)
    # clean save dir if exists
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    layout_json_path = os.path.join(RESULTS_DIR, layout_id, f"{layout_id}.json")
    current_layout = dict_to_floor_plan(json.load(open(layout_json_path, "r")))

    room_id = current_layout.rooms[0].id

    policy_analysis = current_layout.policy_analysis

    for step_dict in policy_analysis["updated_task_decomposition"]:
        if step_dict["action"] == "pick":
            pick_object_id = step_dict["target_object_id"]
            pick_table_id = step_dict["location_object_id"]
        elif step_dict["action"] == "place":
            place_table_id = step_dict["location_object_id"]

    debug_dir = os.path.join(RESULTS_DIR, layout_id, f"{save_dir_name}/debug")
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)

    sample_pick_object_pose(
        os.path.join(RESULTS_DIR, layout_id),
        layout_id,
        room_id,
        pick_object_id,
        pick_table_id,
        num_samples=num_samples,
        save_dir=os.path.join(RESULTS_DIR, layout_id, f"{save_dir_name}"),
        debug_dir=os.path.join(RESULTS_DIR, layout_id, f"{save_dir_name}/debug")
    )

    all_layout_names = [
        os.path.join(save_dir_name, f"{layout_i:0>2d}") for layout_i in range(num_samples)
    ]

    reachable_pick_ids = []
    reachable_place_ids = []
    successful_plan_ids = []
    
    for layout_i in range(num_samples):
        
        layout_json_path = os.path.join(RESULTS_DIR, layout_id, all_layout_names[layout_i]+".json")
        if not os.path.exists(layout_json_path):
            print(f"layout json path not found: {layout_json_path}; skip layout {layout_i}")
            continue
        
        target_layout = dict_to_floor_plan(json.load(open(layout_json_path, "r")))
        target_room = next(room for room in target_layout.rooms if room.id == room_id)

        pick_object = next(obj for obj in target_room.objects if obj.id == pick_object_id)

        robot_base_pos_pick, robot_base_quat_pick, _ = sample_robot_location(
            os.path.join(RESULTS_DIR, layout_id),
            all_layout_names[layout_i],
            room_id,
            pick_object_id,
            pick_table_id,
            1,
            os.path.join(RESULTS_DIR, layout_id, f"{save_dir_name}/debug")
        )

        pick_object_xy = np.array([pick_object.position.x, pick_object.position.y]).reshape(2)
        robot_base_pos_pick_xy = robot_base_pos_pick.cpu().numpy()[0].reshape(3)[:2]

        dist_to_pick_object = np.linalg.norm(robot_base_pos_pick_xy - pick_object_xy)
        if dist_to_pick_object > 0.8:
            print(f"dist to pick object too far: {dist_to_pick_object}; skip layout {layout_i}")
            continue
        
        reachable_pick_ids.append(layout_i)

        robot_base_pos_place, robot_base_quat_place, place_locations = sample_robot_place_location(
            os.path.join(RESULTS_DIR, layout_id),
            all_layout_names[layout_i],
            room_id,
            place_table_id,
            1,
            os.path.join(RESULTS_DIR, layout_id, f"{save_dir_name}/debug")
        )

        robot_base_pos_place_xy = robot_base_pos_place.cpu().numpy()[0].reshape(3)[:2]
        place_location_xy = place_locations.cpu().numpy()[0].reshape(3)[:2]

        dist_to_place_location = np.linalg.norm(robot_base_pos_place_xy - place_location_xy)
        if dist_to_place_location > 0.8:
            print(f"dist to place location too far: {dist_to_place_location}; skip layout {layout_i}")
            continue
        
        reachable_place_ids.append(layout_i)

        # Example of using the new return_plan_status parameter
        trajectory, plan_successful = plan_robot_traj(
            robot_base_pos_pick.reshape(3),
            robot_base_quat_pick.reshape(4),
            robot_base_pos_place.reshape(3),
            robot_base_quat_place.reshape(4),
            os.path.join(RESULTS_DIR, layout_id),
            all_layout_names[layout_i],
            room_id,
            os.path.join(RESULTS_DIR, layout_id, f"{save_dir_name}/debug"),
            return_plan_status=True
        )
        
        print(f"Layout {layout_i}: Trajectory planned successfully: {plan_successful}")
        if not plan_successful:
            print(f"Layout {layout_i}: Using fallback trajectory due to RRT failure")
            continue
        else:
            successful_plan_ids.append(layout_i)

    print(f"reachable pick ids: {reachable_pick_ids}")
    print(f"reachable place ids: {reachable_place_ids}")
    print(f"successful plan ids: {successful_plan_ids}")

    meta_dict = {
        "layouts": []
    }


    for layout_i in successful_plan_ids:

        layout_json_path = os.path.join(RESULTS_DIR, layout_id, all_layout_names[layout_i]+".json")
        target_layout = dict_to_floor_plan(json.load(open(layout_json_path, "r")))
        target_room = next(room for room in target_layout.rooms if room.id == room_id)

        rigid_object_transform_dict = {
            obj.id: {
                "position": {
                    "x": obj.position.x,
                    "y": obj.position.y,
                    "z": obj.position.z
                },
                "rotation": {
                    "x": obj.rotation.x,
                    "y": obj.rotation.y,
                    "z": obj.rotation.z
                }
            } for obj in target_room.objects
        }

        meta_dict["layouts"].append({
            "layout_id": all_layout_names[layout_i],
            "rigid_object_transform_dict": rigid_object_transform_dict
        })

    with open(os.path.join(RESULTS_DIR, layout_id, f"{save_dir_name}/meta.json"), "w") as f:
        json.dump(meta_dict, f, indent=4)



