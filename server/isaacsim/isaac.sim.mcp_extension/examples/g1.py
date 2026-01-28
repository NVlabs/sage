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
import omni
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.nucleus import get_assets_root_path
import numpy as np
from omni.isaac.core import SimulationContext
from omni.isaac.core import PhysicsContext
from omni.isaac.core.prims import XFormPrim

class G1Simulation:
    """Simulation of G1 robot in a factory environment using the BaseSample pattern."""
    
    def __init__(self):
        self.my_world = None
        self.g1_robot = None
        self.simulation_context = None
        self.g1_controller = None
        self.num_joints = 0
        self.assets_root_path = None
        self.factory_elements = []
        self.trot_sequence = []
        
    async def setup_scene(self):
        """Set up the initial scene with ground plane and environment."""
        print("Setting up scene...")
        
        # Create world
        self.my_world = World(stage_units_in_meters=1.0)
        
        # Get the stage
        stage = self.my_world.stage
        
        # Check if ground plane exists before adding it
        if not stage.GetPrimAtPath("/World/groundPlane"):
            print("Adding default ground plane")
            self.my_world.scene.add_default_ground_plane()
        else:
            print("Ground plane already exists, skipping creation")
        
        # Get assets root path
        self.assets_root_path = get_assets_root_path()
        print(f"Assets root path: {self.assets_root_path}")
        
        # Add factory environment objects
        await self._create_factory_environment()

        # Mn
        physics_context = PhysicsContext()
        physics_context.set_physics_dt(0.0167)  
        physics_context.enable_gpu_dynamics(True)  # 

        #
        self.simulation_context = SimulationContext()
        # self.simulation_context.set_physics_context(physics_context)
        # Initialize physics directly from simulation_context
        # This method handles the physics initialization internally
        # without needing PhysicsContext
        # await self.simulation_context.initialize_physics_async()
        # await self.simulation_context.start_async()  # 
    
    async def _create_factory_environment(self):
        """Create the factory environment objects."""
        # Define factory obstacles
        factory_obstacles = [
            # Format: name, position, scale, color
            ("machine_1", [2.0, 1.0, 0.5], [1.0, 0.5, 1.0], [0.7, 0.7, 0.8]),
            ("machine_2", [-2.0, 1.5, 0.5], [0.8, 0.8, 1.0], [0.8, 0.7, 0.7]),
            ("workbench_1", [0.0, 2.0, 0.3], [2.0, 0.8, 0.6], [0.6, 0.5, 0.4]),
            ("storage_shelf", [-1.5, -1.5, 0.75], [0.5, 1.5, 1.5], [0.5, 0.5, 0.5]),
            ("conveyor_belt", [2.5, -1.0, 0.2], [3.0, 0.7, 0.4], [0.3, 0.3, 0.3])
        ]
        
        # Add factory objects (checking if they already exist)
        stage = self.my_world.stage
        for name, position, scale, color in factory_obstacles:
            object_path = f"/World/{name}"
            
            # Check if object already exists
            if not stage.GetPrimAtPath(object_path):
                print(f"Adding factory object: {name}")
                obj =DynamicCuboid(
                        prim_path=object_path,
                        name=name,
                        position=np.array(position),
                        scale=np.array(scale),
                        color=np.array(color)
                    )
                # obj = self.my_world.scene.add(
                    
                # )
                self.factory_elements.append(obj)
            else:
                print(f"Factory object {name} already exists, skipping creation")
    
    async def load_robot(self):
        """Load the G1 robot into the scene."""
        print("Loading G1 robot...")
        
        # Define potential G1 paths
        g1_paths = []
        if self.assets_root_path:
            g1_paths.extend([
                f"{self.assets_root_path}/Isaac/Robots/Unitree/G1/g1.usd"
                #f"{self.assets_root_path}/Isaac/Robots/Unitree/Go1/go1.usd"
                #f"{self.assets_root_path}/Robots/Unitree/G1/g1.usd"
            ])
        g1_paths.extend([
            "/Isaac/Robots/Unitree/G1/g1.usd"
            #"/Isaac/Robots/Unitree/Go1/go1.usd"
            #"/Robots/Unitree/G1/g1.usd"
        ])
        
        # Check if G1 robot already exists
        stage = self.my_world.stage
        g1_exists = stage.GetPrimAtPath("/World/G1").IsValid()
        robot_prim = XFormPrim(prim_path="/World/G1")
        robot_prim.set_world_pose(position=np.array([4.0 , 6, 0.0])) 
    
        
        if not g1_exists:
            print("G1 robot not found, adding it to stage...")
            # Add G1 robot reference to stage
            g1_added = False
            for path in g1_paths:
                try:
                    print(f"Trying to add G1 from: {path}")
                    add_reference_to_stage(usd_path=path, prim_path="/World/G1")
                    g1_added = True
                    print(f"Successfully added G1 from: {path}")
                    break
                except Exception as e:
                    print(f"Could not add G1 from {path}: {e}")
            
            if not g1_added:
                raise Exception("Could not add G1 from any known path")
        else:
            print("G1 robot already exists in stage, skipping creation")
            g1_added = True
        
        return g1_added
    
    async def post_load(self):
        """Setup after loading - initialize the robot and controllers."""
        print("Post-load setup...")
        
        # Add the G1 robot to the scene
        # self.g1_robot = self.my_world.stage.GetPrimAtPath("/World/G1")
        # from omni.isaac.articulations import Articulation
        # self.g1_robot_articulation = Articulation(prim_path=robot_prim_path)
        from omni.isaac.core.articulations import Articulation
        # art = Articulation(prim_path="/World/G1")

        self.g1_robot = Articulation(prim_path="/World/G1", name="G1")

        # self.simulation_context.add_articulation(art)  
        
        # await art.initialize()
        # self.g1_robot = self.my_world.stage.add(Robot(prim_path="/World/G2", name="g1_robot"))
        
        # Get joint names to understand the robot's structure
        # print(art.dof_properties["name"])
        # joint_names = art.dof_properties["name"]
        
        
        # Create articulation controller
        # Initialize the G1 robot as an articulation
       
        # self.g1_robot = self.my_world.scene.add(Robot(prim_path="/World/G1", name="g1_robot"))
        
        # Wait for the robot to be fully initialized
        # self.my_world.scene.initialize_physics()
        
        # Get joint information
        # joint_names = self.g1_robot_articulation.dof_properties["name"]
        # self.num_joints = len(joint_names)
        # print(f"G1 has {self.num_joints} joints: {joint_names}")
        
        # Create the articulation controller for the robot
        # Initialize the robot first to ensure it's properly set up
        # Initialize physics scene first to avoid "no active physics scene" error
        # from omni.isaac.core import get_physics_context
        physics_context = self.my_world.get_physics_context()
        if physics_context is None:
            print("Warning: Physics context is None")
            # Don't attempt to reset the world here as it causes the error
            # Instead, ensure physics is initialized before this point
            print("Make sure physics scene is created before initializing the robot")
        elif not physics_context.is_initialized():
            print("Physics context exists but not initialized")
            # Don't call reset() directly as it's causing the error
            
            # Check if physics scene exists
            if not self.my_world.physics_sim_view:
                print("No physics simulation view found - need to initialize physics first")
                # Try playing the simulation first
                if not self.my_world.is_playing():
                    self.my_world.play()
                    print("Started simulation to initialize physics")
        # Make sure the world is playing before initializing the robot
        if not self.my_world.is_playing():
            self.my_world.play()
            # Wait a few frames for physics to stabilize
            for _ in range(10):
                self.my_world.step_async()
        
        # Now initialize the robot
        self.g1_robot.initialize(self.my_world.physics_sim_view)
        # Now get the controller after initialization
        self.g1_controller = self.g1_robot.get_articulation_controller()

        print("joint_names", self.g1_robot.dof_names)
        joint_names = self.g1_robot.dof_names
        if joint_names is not None:
            self.num_joints = len(joint_names)
            print(f"G1 has {self.num_joints} joints: {joint_names}")
        
        # Set control parameters for better stability
        self.g1_controller.set_gains(kps=[100.0] * self.num_joints, 
                                    kds=[10.0] * self.num_joints)
        
        # print("G1 articulation controller created successfully")
        # from omni.isaac.core_nodes.IsaacArticulationController import IsaacArticulationController
      

        # self.g1_controller = IsaacArticulationController(self.g1_robot)
        
        # Define trot sequence for quadruped walking
        if self.num_joints >= 12:
            # Create full joint position arrays with zeros for all joints
            # The error shows we need arrays of shape (1,37) instead of (1,12)
            base_positions = np.zeros(self.num_joints)
            
            # Create two different walking poses
            trot_pose_1 = base_positions.copy()
            trot_pose_2 = base_positions.copy()
            
            # Only modify the leg joint positions (assuming first 12 are leg joints)
            # Front right leg
            trot_pose_1[0:3] = [0.1, 0.4, -0.8]  # up and forward
            trot_pose_2[0:3] = [-0.1, 0.4, -0.6]  # down and back
            
            # Front left leg
            trot_pose_1[3:6] = [-0.1, 0.4, -0.6]  # down and back
            trot_pose_2[3:6] = [0.1, 0.4, -0.8]  # up and forward
            
            # Rear right leg
            trot_pose_1[6:9] = [-0.1, 0.4, -0.6]  # down and back
            trot_pose_2[6:9] = [0.1, 0.4, -0.8]  # up and forward
            
            # Rear left leg
            trot_pose_1[9:12] = [0.1, 0.4, -0.8]  # up and forward
            trot_pose_2[9:12] = [-0.1, 0.4, -0.6]  # down and back
            
            self.trot_sequence = [trot_pose_1, trot_pose_2]
    
    async def initialize_simulation(self):
        """Initialize the simulation with proper robot pose."""
        # Define standing pose
        standing_pose = np.zeros(self.num_joints)
        if self.num_joints >= 12:  # Safety check in case joint structure is different
            # Set knee joints to a slight bend for stability
            standing_pose[2] = -0.5  # front right knee
            standing_pose[5] = -0.5  # front left knee
            standing_pose[8] = -0.5  # rear right knee
            standing_pose[11] = -0.5  # rear left knee
        
        # Move to standing pose
        print("Moving to standing pose...")
        self.g1_controller.apply_action(ArticulationAction(joint_positions=standing_pose))
        # self.g1_robot._articulation_view.set_joint_positions(standing_pose)
        # Allow time to stabilize
        for _ in range(60):
            self.my_world.step_async()
    
    async def run_simulation(self):
        """Run the main simulation sequence."""
        if self.num_joints < 12:
            print("Not enough joints for walking animation, simulation will be static")
            return
            
        # Perform a walking movement in a circle
        print("Starting walking pattern...")
        steps = 120
        radius = 1.0
        
        # Initial position
        initial_position = self.g1_robot.get_world_pose()[0]
        center = initial_position + np.array([radius, 0, 0])
        
        for i in range(steps):
            # Calculate angle around circle
            angle = i * 2 * np.pi / steps
            
            # Calculate new position on circle
            new_x = center[0] - radius * np.cos(angle)
            new_y = center[1] + radius * np.sin(angle)
            new_position = np.array([new_x, new_y, initial_position[2]])
            
            # Calculate direction (tangent to circle)
            direction_angle = angle + np.pi/2
            
            # Set rotation to face direction of travel
            # Convert angle to quaternion (w,x,y,z format)
            orientation = np.array([np.cos(direction_angle/2), 0, 0, np.sin(direction_angle/2)])
            
            # Alternate between trot poses
            pose_idx = i % 2
            self.g1_controller.apply_action(ArticulationAction(joint_positions=self.trot_sequence[pose_idx]))
            
            # Update position
            self.g1_robot.set_world_pose(position=new_position, orientation=orientation)
            
            # Step the simulation
            for _ in range(3):
                self.my_world.step_async()
        
        # Return to standing pose
        print("Returning to standing pose...")
        standing_pose = np.zeros(self.num_joints)
        if self.num_joints >= 12:
            standing_pose[2] = -0.5
            standing_pose[5] = -0.5
            standing_pose[8] = -0.5
            standing_pose[11] = -0.5
        
        self.g1_controller.apply_action(ArticulationAction(joint_positions=standing_pose))
        
        # Allow time to stabilize
        for _ in range(60):
            self.my_world.step_async()
    
    async def cleanup(self):
        """Clean up resources after simulation."""
        print("Cleaning up simulation resources...")
        # Any specific cleanup if needed
    
    async def clear_async(self):
        """Clear the simulation completely."""
        print("Clearing simulation...")
        # Stop physics simulation if running
        if self.my_world and self.my_world.is_playing():
            await self.my_world.stop_async()
        
        # Additional cleanup as needed
        self.g1_robot = None
        self.g1_controller = None
        self.factory_elements = []
    
    async def run(self):
        """Run the complete simulation sequence."""
        try:
            print("Starting G1 robot simulation using asyncio...")
            
            # Setup scene
            await self.setup_scene()
            
            # Load robot
            robot_loaded = await self.load_robot()
            if not robot_loaded:
                print("Failed to load G1 robot, aborting simulation")
                return
            
            # Reset physics with async method
            print("Resetting world with async method...")
            # Simply use reset_async without checking for physics_context
            try:
                self.my_world.reset()
                
            except Exception as e:
                print(f"Error during reset: {str(e)}")
                # Initialize physics if needed
                try:
                    await self.my_world.initialize_physics_async()
                except Exception as e:
                    print(f"Error initializing physics: {str(e)}")
            
            physics_context = self.my_world.get_physics_context()
            print("physics_context", physics_context)
            # Make sure physics is properly set up
            if  physics_context is not None and not physics_context.is_initialized():
                print("Warning: Unable to initialize physics context")
                
            # Post-load setup
            await self.post_load()
            
            # Start simulation using async method
            print("Starting simulation with async method...")
            # Simply use play_async without checking for physics_context
            try:
                self.my_world.play()
            except Exception as e:
                print(f"Error during play: {str(e)}")
                try:
                    await self.my_world.initialize_physics_async()
                    await self.my_world.play_async()
                except Exception as e:
                    print(f"Error initializing and playing physics: {str(e)}")
            
            # Initialize the simulation
            await self.initialize_simulation()
            
            # Run simulation sequence
            await self.run_simulation()
            
            print("G1 robot simulation completed successfully!")
            
        except Exception as e:
            print(f"Error in G1 simulation: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            # Cleanup
            await self.cleanup()

# Create and run the simulation
async def main():
    sim = G1Simulation()
    await sim.run()

# Launch the async function
asyncio.ensure_future(main())
print("G1 simulation launched in background")