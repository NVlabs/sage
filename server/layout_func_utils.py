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
sys.path.append("objects")
import pdb
import uuid
from datetime import datetime
mcp_init_id = str(datetime.now().strftime("%Y%m%d_%H%M%S"))+"_"+str(uuid.uuid4())[:8]
def get_mcp_init_id():
    global mcp_init_id
    return mcp_init_id

from vlm import call_vlm

from typing import Optional, List, Dict, Any
import json
import os
import anthropic
from dataclasses import asdict
from mcp.server.fastmcp import FastMCP, Context, Image
from key import ANTHROPIC_API_KEY
from constants import RESULTS_DIR
from pathlib import Path
# Import modules
from models import FloorPlan, Room, Object, Point3D, Dimensions, Euler
from llm_client import (
    call_llm_for_rooms_with_validation,
    call_llm_for_doors_windows,
    call_llm_for_doors_windows_graph_based
)
from validation import (
    validate_room_layout, 
    validate_llm_response_structure, 
    validate_door_window_placement, 
    validate_room_structure_issues,
    validate_door_window_issues,
    check_door_window_integrity,
    check_room_overlap,
)
from layout_parser import (
    parse_llm_response_to_floor_plan, 
    parse_rooms_only_to_floor_plan,
    add_doors_windows_to_floor_plan
)
from correction import (
    generate_detailed_correction_prompt, 
    correct_door_window_integrity,
    correct_layout_issues,
    aggressive_layout_correction
)
from utils import (
    extract_wall_side_from_id,
    get_room_priorities_from_claude,
    calculate_room_reduction,
    export_layout_to_json,
    export_layout_to_mesh,
    dict_to_floor_plan,
    generate_unique_id,
    export_layout_to_mesh_dict_list,
)
from objects.object_selection_planner import (
    select_objects
)
from objects.object_placement_planner import (
    place_objects
)
from isaacsim.isaac_mcp.server import (
    get_isaac_connection,
    create_room_layout_scene,
    simulate_the_scene,
    create_robot,
    move_robot_to_target,
    create_physics_scene,
    get_room_layout_scene_usd,
    create_single_room_layout_scene,
    get_room_layout_scene_usd_separate_from_layout
)
import copy
from floor_plan_materials.room_material import MaterialSelector
from foundation_models import get_clip_models
import shutil
from floor_plan_materials.door_material import DoorMaterialSelector
import glob
import numpy as np
from constants import SERVER_ROOT_DIR
from utils import extract_json_from_response
from floor_plan_materials.material_generator import material_generate_from_prompt
from isaaclab.correct_mobile_franka import (
    correct_mobile_franka_standalone,
    robot_task_feasibility_correction_for_room_standalone
)
import hashlib

