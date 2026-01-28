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
sys.path.append("./physics")
sys.path.append(".")
from typing import Optional, List, Dict, Any
import json
import os
import anthropic
from dataclasses import asdict
from mcp.server.fastmcp import FastMCP
from utils import (
    dict_to_floor_plan,
    export_layout_to_mesh_dict_list
)
from phys_env import (
    sim_scene_eval
)

# Initialize FastMCP server
mcp = FastMCP("physics")


@mcp.tool()
async def physics_simulation_scene(scene_save_dir: str = "") -> str:
    """
    Simulate the scene with physics.

    Args:
        scene_save_dir: The directory to save the scene layout.
    
    Returns:
        A dictionary containing the simulation trace data.
    """

    # global current_layout

    # if current_layout is None:
    #     return json.dumps({
    #         "success": False,
    #         "error": "No layout has been generated yet. Use 'generate_room_layout()' first."
    #     })
    
    
    # try:
        
    #     mesh_dict_list, mesh_idx_to_object_id = export_layout_to_mesh_dict_list(current_layout)
    #     sim_trace_data = sim_scene_eval(mesh_dict_list, mesh_idx_to_object_id)

    #     return json.dumps({
    #         "success": True,
    #         "sim_trace_data": sim_trace_data,
    #         "mesh_idx_to_object_id": mesh_idx_to_object_id,
    #     }, indent=4)
    # except Exception as e:
    #     return json.dumps({
    #         "success": False,
    #         "error": f"Physics simulation failed: {str(e)}",
    #     })

    try:
        # Validate input parameters
        if not scene_save_dir:
            return json.dumps({
                "success": False,
                "error": "scene_save_dir must be provided"
            })
        
        # Load JSON data
        current_layout_id = os.path.basename(scene_save_dir)
        json_file_path = os.path.join(scene_save_dir, f"{current_layout_id}.json")
        
        # Load from file
        try:
            with open(json_file_path, 'r') as f:
                layout_data = json.load(f)
        except FileNotFoundError:
            return json.dumps({
                "success": False,
                "error": f"JSON file not found: {json_file_path}"
            })
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON format in file: {str(e)}"
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Error reading file: {str(e)}"
            })
        
        try:
            floor_plan = dict_to_floor_plan(layout_data)
            current_layout = floor_plan
        except ValueError as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to convert JSON data to FloorPlan: {str(e)}"
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Unexpected error during conversion: {str(e)}"
            })
        
        try:
            mesh_dict_list, mesh_idx_to_object_id = export_layout_to_mesh_dict_list(current_layout)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Unexpected error during export_layout_to_mesh_dict_list: {str(e)}"
            })
        
        try:
            sim_trace_data = sim_scene_eval(mesh_dict_list, mesh_idx_to_object_id)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Unexpected error during physics simulation: {str(e)}"
            })

        return json.dumps({
            "success": True,
            "sim_trace_data": sim_trace_data,
        }, indent=4)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Unexpected error in physics simulation scene: {str(e)}"
        })
    
    
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')