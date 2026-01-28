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
from time import sleep
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation
import numpy as np
from omni.isaac.core import World



assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
franka_robot = add_reference_to_stage(asset_path, "/World/Franka")
#cre_prim("/DistantLight", "DistantLight")

# set the position of the robot
# robot_prim = XFormPrim(prim_path="/World/Franka")
# robot_prim.set_world_pose(position=np.array([2.0, 1.5, 0.0])) 
for row in range(5):
    for col in range(5):
        object_path = f"/World/Franka_{row}_{col}"
        add_reference_to_stage(asset_path, object_path)
        robot_prim = XFormPrim(prim_path=object_path)
        robot_prim.set_world_pose(position=np.array([2.0 + row * 1 - 4, 1.5 + col * 1 - 4, 0.0])) 
    

physics_initialized = False
simulation_context = None

# async def initialize_physics(physics_initialized, simulation_context):
#         """Initializes the physics simulation, handling potential errors."""

#         if physics_initialized:
#             print("Physics already initialized, skipping.")
#             return

#         try:
#             print("Initializing physics...")
#             # Configure simulation parameters
#             sim_config = {
#                 "physics_engine": "PhysX",
#                 "physx": {
#                     "use_gpu": True,  # Enable GPU acceleration
#                     "solver_type": 1,
#                     "gpu_found_lost_pairs_capacity": 2048,
#                     "gpu_found_lost_aggregate_pairs_capacity": 2048
#                 },
#                 "stage_units_in_meters": 1.0,
#             }

#             simulation_context = SimulationContext()
#             simulation_context.initialize_physics(**sim_config)  # Pass sim_config

#             if not simulation_context.is_physics_running():
#                 print("Physics is not running!  Playing the simulation.")
#                 simulation_context.play()


#             physics_initialized = True
#             print("Physics initialization complete.")

#         except Exception as e:
#             print(f"Error during physics initialization: {e}")
#             raise  # Re-raise the exception to halt the simulation setup
# Create and run the simulation
async def main():
    
    
    
    await asyncio.sleep(3)
    # need to initialize physics getting any articulation..etc
    # simulation_context = SimulationContext()
    # simulation_context.initialize_physics()
    # await initialize_physics(physics_initialized, simulation_context)
    # Create world
    my_world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
        
    # Get the stage
    # stage = my_world.get_stage()
    simulation_context = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    my_world.initialize_physics()

    # Make sure the world is playing before initializing the robot
    if not my_world.is_playing():
        my_world.play()
        # Wait a few frames for physics to stabilize
    for _ in range(1000):
        my_world.step_async()


    
    # Now initialize the robot
    franka_robot_art = Articulation(prim_path="/World/Franka", name="Franka")
    franka_robot_art.initialize(simulation_context.physics_sim_view)
    # Now get the controller after initialization
    franka_controller = franka_robot_art.get_articulation_controller()
    
        # art = Articulation(prim_path="/World/G1")

    # simulation_context.stop()

# Launch the async function
# sleep(1)
asyncio.ensure_future(main())
print("Franka simulation launched in background")