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
"""
MIT License

Copyright (c) 2023-2025 omni-mcp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import asyncio
import numpy as np
from omni.isaac.core import World, SimulationContext
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.nucleus import get_assets_root_path

# Get the path to the Franka robot asset
assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"

# Create a 3x3 grid of Franka robots
# The grid will span from [-3, -3, 0] to [3, 3, 0]
def create_franka_grid():
    # Calculate spacing between robots
    num_robots = 3  # 3x3 grid
    x_min, y_min = -3, -3
    x_max, y_max = 3, 3
    x_spacing = (x_max - x_min) / (num_robots - 1) if num_robots > 1 else 0
    y_spacing = (y_max - y_min) / (num_robots - 1) if num_robots > 1 else 0
    
    # Create each robot in the grid
    for row in range(num_robots):
        for col in range(num_robots):
            # Calculate position for this robot
            x_pos = x_min + row * x_spacing
            y_pos = y_min + col * y_spacing
            
            # Create a unique path for each robot
            object_path = f"/World/Franka_{row}_{col}"
            
            # Add robot to the stage
            add_reference_to_stage(asset_path, object_path)
            
            # Set position
            robot_prim = XFormPrim(prim_path=object_path)
            robot_prim.set_world_pose(position=np.array([x_pos, y_pos, 0.0]))
            
            print(f"Created Franka robot at position [{x_pos}, {y_pos}, 0.0]")

async def main():
    # Create the robots
    create_franka_grid()
    
    # Wait for robots to be fully loaded
    await asyncio.sleep(2)
    
    # Create world and initialize physics
    my_world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, stage_units_in_meters=1.0)
    simulation_context = SimulationContext(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, stage_units_in_meters=1.0)
    
    # Initialize physics
    my_world.initialize_physics()
    
    # Make sure the world is playing
    if not my_world.is_playing():
        my_world.play()
    
    # Run simulation for some time
    for _ in range(500):
        my_world.step_async()
        await asyncio.sleep(0.01)  # Small delay to prevent blocking
    
    # Initialize all robots
    for row in range(3):
        for col in range(3):
            robot_path = f"/World/Franka_{row}_{col}"
            try:
                robot = Articulation(prim_path=robot_path, name=f"Franka_{row}_{col}")
                robot.initialize(simulation_context.physics_sim_view)
                print(f"Initialized robot at {robot_path}")
            except Exception as e:
                print(f"Error initializing robot at {robot_path}: {e}")

# Launch the async function
asyncio.ensure_future(main())
print("Franka robot grid simulation launched") 