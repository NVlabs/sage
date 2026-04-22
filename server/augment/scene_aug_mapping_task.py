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
# Import utils functions using importlib to avoid conflicts with cv2.utils
import os
import json
import sys
import copy
from dataclasses import asdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR
sys.path.insert(0, SERVER_ROOT_DIR)

import importlib.util
utils_spec = importlib.util.spec_from_file_location("server_utils", os.path.join(SERVER_ROOT_DIR, "utils.py"))
server_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(server_utils)

# Import the specific functions from server utils
dict_to_floor_plan = server_utils.dict_to_floor_plan

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_id", type=str, required=True)
    parser.add_argument("--layout_id_original", type=str, required=True)
    args = parser.parse_args()

    layout_id_original = args.layout_id_original
    layout_id = args.layout_id

    layout_current = dict_to_floor_plan(json.load(open(os.path.join(SERVER_ROOT_DIR, f"results/{layout_id}/{layout_id}.json"), "r")))
    layout_original = dict_to_floor_plan(json.load(open(os.path.join(SERVER_ROOT_DIR, f"results/{layout_id_original}/{layout_id_original}.json"), "r")))

    policy_analysis_original = layout_original.policy_analysis

    usage_log_dir = os.path.join(SERVER_ROOT_DIR, "results", layout_id_original, "selected_objects_usage_log")
    usage_log_file_path = os.path.join(usage_log_dir, f"{layout_id}.json")
    usage_log = json.load(open(usage_log_file_path, "r"))

    policy_analysis_current = copy.deepcopy(policy_analysis_original)
    object_id_mapping = usage_log["object_id_mapping"]

    for req_obj_i in range(len(policy_analysis_current["minimum_required_objects"])):
        policy_analysis_current["minimum_required_objects"][req_obj_i]["matched_object_ids"] = [
            object_id_mapping[obj_id] for obj_id in policy_analysis_current["minimum_required_objects"][req_obj_i]["matched_object_ids"]
        ]

    for req_obj_i in range(len(policy_analysis_current["task_decomposition"])):
        policy_analysis_current["task_decomposition"][req_obj_i]["actual_object_ids"] = [
            object_id_mapping[obj_id] for obj_id in policy_analysis_current["task_decomposition"][req_obj_i]["actual_object_ids"]
        ]

    for req_obj_i in range(len(policy_analysis_current["updated_task_decomposition"])):
        policy_analysis_current["updated_task_decomposition"][req_obj_i]["target_object_id"] = \
            object_id_mapping[policy_analysis_current["updated_task_decomposition"][req_obj_i]["target_object_id"]]
        policy_analysis_current["updated_task_decomposition"][req_obj_i]["location_object_id"] = \
            object_id_mapping[policy_analysis_current["updated_task_decomposition"][req_obj_i]["location_object_id"]]

    layout_current.policy_analysis = policy_analysis_current

    # save the layout_current to the results directory
    os.makedirs(os.path.join(SERVER_ROOT_DIR, "results", layout_id), exist_ok=True)
    with open(os.path.join(SERVER_ROOT_DIR, "results", layout_id, f"{layout_id}.json"), "w") as f:
        json.dump(asdict(layout_current), f, indent=4)