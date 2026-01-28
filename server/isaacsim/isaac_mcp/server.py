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

# isaac_sim_mcp_server.py
import time
from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import asyncio
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List
import os
from pathlib import Path
import base64
from urllib.parse import urlparse
import sys
# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IsaacMCPServer")

@dataclass
class IsaacConnection:
    host: str
    port: int
    sock: socket.socket = None  # Changed from 'socket' to 'sock' to avoid naming conflict
    
    def connect(self) -> bool:
        """Connect to the Isaac addon socket server"""
        if self.sock:
            return True
            
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Isaac at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Isaac: {str(e)}")
            self.sock = None
            return False
    
    def disconnect(self):
        """Disconnect from the Isaac addon"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Isaac: {str(e)}")
            finally:
                self.sock = None

    def receive_full_response(self, sock, buffer_size=16384):
        """Receive the complete response, potentially in multiple chunks"""
        chunks = []
        # Use a consistent timeout value that matches the addon's timeout
        sock.settimeout(300.0)  # Match the extension's timeout
        
        try:
            while True:
                try:
                    logger.info("Waiting for data from Isaac")
                    time.sleep(1.0)
                    chunk = sock.recv(buffer_size)
                    if not chunk:
                        # If we get an empty chunk, the connection might be closed
                        if not chunks:  # If we haven't received anything yet, this is an error
                            raise Exception("Connection closed before receiving any data")
                        break
                    
                    chunks.append(chunk)
                    
                    # Check if we've received a complete JSON object
                    try:
                        data = b''.join(chunks)
                        json.loads(data.decode('utf-8'))
                        # If we get here, it parsed successfully
                        logger.info(f"Received complete response ({len(data)} bytes)")
                        return data
                    except json.JSONDecodeError:
                        # Incomplete JSON, continue receiving
                        continue
                except socket.timeout:
                    # If we hit a timeout during receiving, break the loop and try to use what we have
                    logger.warning("Socket timeout during chunked receive")
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    logger.error(f"Socket connection error during receive: {str(e)}")
                    raise  # Re-raise to be handled by the caller
        except socket.timeout:
            logger.warning("Socket timeout during chunked receive")
        except Exception as e:
            logger.error(f"Error during receive: {str(e)}")
            raise
            
        # If we get here, we either timed out or broke out of the loop
        # Try to use what we have
        if chunks:
            data = b''.join(chunks)
            logger.info(f"Returning data after receive completion ({len(data)} bytes)")
            try:
                # Try to parse what we have
                json.loads(data.decode('utf-8'))
                return data
            except json.JSONDecodeError:
                # If we can't parse it, it's incomplete
                raise Exception("Incomplete JSON response received")
        else:
            raise Exception("No data received")

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to Isaac and return the response"""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Isaac")
        
        command = {
            "type": command_type,
            "params": params or {}
        }
        
        try:
            # Log the command being sent
            logger.info(f"Sending command: {command_type} with params: {params}")
            
            # Send the command
            self.sock.sendall(json.dumps(command).encode('utf-8'))
            logger.info(f"Command sent, waiting for response...")
            
            # Set a timeout for receiving - use the same timeout as in receive_full_response
            self.sock.settimeout(300.0)  # Match the extension's timeout
            
            # Receive the response using the improved receive_full_response method
            response_data = self.receive_full_response(self.sock)
            logger.info(f"Received {len(response_data)} bytes of data")
            
            response = json.loads(response_data.decode('utf-8'))
            logger.info(f"Response parsed, status: {response.get('status', 'unknown')}")
            
            if response.get("status") == "error":
                logger.error(f"Isaac error: {response.get('message')}")
                raise Exception(response.get("message", "Unknown error from Isaac"))
            
            return response.get("result", {})
        except socket.timeout:
            logger.error("Socket timeout while waiting for response from Isaac")
            # Don't try to reconnect here - let the get_isaac_connection handle reconnection
            # Just invalidate the current socket so it will be recreated next time
            self.sock = None
            raise Exception("Timeout waiting for Isaac response - try simplifying your request")
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.error(f"Socket connection error: {str(e)}")
            self.sock = None
            raise Exception(f"Connection to Isaac lost: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Isaac: {str(e)}")
            # Try to log what was received
            if 'response_data' in locals() and response_data:
                logger.error(f"Raw response (first 200 bytes): {response_data[:200]}")
            raise Exception(f"Invalid response from Isaac: {str(e)}")
        except Exception as e:
            logger.error(f"Error communicating with Isaac: {str(e)}")
            # Don't try to reconnect here - let the get_isaac_connection handle reconnection
            self.sock = None
            raise Exception(f"Communication error with Isaac: {str(e)}")

# @asynccontextmanager
# async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
#     """Manage server startup and shutdown lifecycle"""
#     # We don't need to create a connection here since we're using the global connection
#     # for resources and tools
    
#     try:
#         # Just log that we're starting up
#         logger.info("IsaacMCP server starting up")
        
#         # Try to connect to Isaac on startup to verify it's available
#         try:
#             # This will initialize the global connection if needed
#             isaac = get_isaac_connection()
#             logger.info("Successfully connected to Isaac on startup")
#         except Exception as e:
#             logger.warning(f"Could not connect to Isaac on startup: {str(e)}")
#             logger.warning("Make sure the Isaac addon is running before using Isaac resources or tools")
        
#         # Return an empty context - we're using the global connection
#         yield {}
#     finally:
#         # Clean up the global connection on shutdown
#         global _isaac_connection
#         if _isaac_connection:
#             logger.info("Disconnecting from Isaac Sim on shutdown")
#             _isaac_connection.disconnect()
#             _isaac_connection = None
#         logger.info("Isaac SimMCP server shut down")

# # Create the MCP server with lifespan support
# mcp = FastMCP(
#     "IsaacSimMCP",
#     description="Isaac Sim integration through the Model Context Protocol",
#     lifespan=server_lifespan
# )

# Resource endpoints

# Global connection for resources (since resources can't access context)
_isaac_connection = None

import hashlib

def slurm_job_id_to_port(job_id, port_start=8080, port_end=40000):
    """
    Hash-based mapping function to convert SLURM job ID to a port number.
    
    Args:
        job_id (str or int): SLURM job ID
        port_start (int): Starting port number (default: 8080)
        port_end (int): Ending port number (default: 40000)
    
    Returns:
        int: Mapped port number within the specified range
    """
    # Convert job_id to string if it's an integer
    job_id_str = str(job_id)
    
    # Create a hash of the job ID
    hash_obj = hashlib.md5(job_id_str.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    
    # Map to port range
    port_range = port_end - port_start + 1
    mapped_port = port_start + (hash_int % port_range)
    
    return mapped_port


def get_port():
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    port = slurm_job_id_to_port(slurm_job_id)
    print(f"Isaacsim MCP server port: {port}", file=sys.stderr)
    return port

def get_isaac_connection():
    """Get or create a persistent Isaac connection"""
    global _isaac_connection
    
    # If we have an existing connection, check if it's still valid
    if _isaac_connection is not None:
        try:
            
            return _isaac_connection
        except Exception as e:
            # Connection is dead, close it and create a new one
            logger.warning(f"Existing connection is no longer valid: {str(e)}")
            try:
                _isaac_connection.disconnect()
            except:
                pass
            _isaac_connection = None
    
    # Create a new connection if needed
    if _isaac_connection is None:
        _isaac_connection = IsaacConnection(host="localhost", port=get_port())
        if not _isaac_connection.connect():
            logger.error("Failed to connect to Isaac")
            _isaac_connection = None
            raise Exception("Could not connect to Isaac. Make sure the Isaac addon is running.")
        logger.info("Created new persistent connection to Isaac")
    
    return _isaac_connection


def get_scene_info() -> str:
    """Ping status of Isaac Sim Extension Server"""
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("get_scene_info")
        print("result: ", result)
        
        # Just return the JSON representation of what Isaac sent us
        return json.dumps(result, indent=2)
        # return json.dumps(result)
        # return result
    except Exception as e:
        logger.error(f"Error getting scene info from Isaac: {str(e)}")
        # return f"Error getting scene info: {str(e)}"
        return {"status": "error", "error": str(e), "message": "Error getting scene info"}

def create_physics_scene(
    objects: List[Dict[str, Any]] = [],
    floor: bool = True,
    gravity: List[float] = [0,  -0.981, 0],
    scene_name: str = "physics_scene"
) -> Dict[str, Any]:
    """Create a physics scene with multiple objects. Before create physics scene, you need to call get_scene_info() first to verify availability of connection.
    
    Args:
        objects: List of objects to create. Each object should have at least 'type' and 'position'. 
        objects  = [
        {"path": "/World/Cube", "type": "Cube", "size": 20, "position": [0, 100, 0]},
        {"path": "/World/Sphere", "type": "Sphere", "radius": 5, "position": [5, 200, 0]},
        {"path": "/World/Cone", "type": "Cone", "height": 8, "radius": 3, "position": [-5, 150, 0]}
         ]
        floor: Whether to create a floor. deafult is True
        gravity: The gravity vector. Default is [0, 0, -981.0] (cm/s^2).
        scene_name: The name of the scene. deafult is "physics_scene"
        
    Returns:
        Dictionary with result information.
    """
    params = {"objects": objects, "floor": floor}
    
    if gravity is not None:
        params["gravity"] = gravity
    if scene_name is not None:
        params["scene_name"] = scene_name
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        
        result = isaac.send_command("create_physics_scene", params)
        return f"create_physics_scene successfully: {result.get('result', '')}, {result.get('message', '')}"
    except Exception as e:
        logger.error(f"Error create_physics_scene: {str(e)}")
        # return f"Error create_physics_scene: {str(e)}"
        return {"status": "error", "error": str(e), "message": "Error create_physics_scene"}
    
def create_robot(robot_type: str = "g1", position: List[float] = [0, 0, 0]) -> str:
    """Create a robot in the Isaac scene. Directly create robot prim in stage at the right position. For any creation of robot, you need to call create_physics_scene() first. call create_robot() as first attmpt beofre call execute_script().
    
    Args:
        robot_type: The type of robot to create. Available options:
            - "franka": Franka Emika Panda robot
            - "jetbot": NVIDIA JetBot robot
            - "carter": Carter delivery robot
            - "g1": Unitree G1 quadruped robot (default)
            - "go1": Unitree Go1 quadruped robot
            - "ridgeback_franka": Clearpath Ridgeback with Franka arm
        
    Returns:
        String with result information.
    """
    isaac = get_isaac_connection()
    result = isaac.send_command("create_robot", {"robot_type": robot_type, "position": position})
    return f"create_robot successfully: {result.get('result', '')}, {result.get('message', '')}"

def move_robot_to_target() -> str:
    """Move the robot to a target position using RRT path planning.
    
    This function will:
    1. Create a target at position (1, 1, 0) on the floor
    2. Use RRT path planning to compute a path from the robot's current position to the target
    3. Execute the path by applying articulation actions to move the robot
    
    Note: A robot must be created first using create_robot() before calling this function.
    
    Returns:
        String with result information including final position and distance to target.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("move_robot_to_target")
        return f"move_robot_to_target result: {result}"
    except Exception as e:
        logger.error(f"Error moving robot to target: {str(e)}")
        return f"Error moving robot to target: {str(e)}"

def omni_kit_command(command: str = "CreatePrim", prim_type: str = "Sphere") -> str:
    """Execute an Omni Kit command.
    
    Args:
        command: The Omni Kit command to execute.
        prim_type: The primitive type for the command.
        
    Returns:
        String with result information.
    """
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        
        result = isaac.send_command("omini_kit_command", {
            "command": command,
            "prim_type": prim_type
        })
        return f"Omni Kit command executed successfully: {result.get('message', '')}"
    except Exception as e:
        logger.error(f"Error executing Omni Kit command: {str(e)}")
        # return f"Error executing Omni Kit command: {str(e)}"
        return {"status": "error", "error": str(e), "message": "Error executing Omni Kit command"}


def execute_script(code: str) -> str:
    """
    Before execute script pls check prompt from asset_creation_strategy() to ensure the scene is properly initialized.
    Execute arbitrary Python code in Isaac Sim. Before executing any code, first verify if get_scene_info() has been called to ensure the scene is properly initialized. Always print the formatted code into chat to confirm before execution to confirm its correctness. 
    Before execute script pls check if create_physics_scene() has been called to ensure the physics scene is properly initialized.
    When working with robots, always try using the create_robot() function first before resorting to execute_script(). The create_robot() function provides a simpler, more reliable way to add robots to your scene with proper initialization and positioning. Only use execute_script() for robot creation when you need custom configurations or behaviors not supported by create_robot().
    
    For physics simulation, avoid using simulation_context to run simulations in the main thread as this can cause blocking. Instead, use the World class with async methods for initializing physics and running simulations. For example, use my_world = World(physics_dt=1.0/60.0) and my_world.step_async() in a loop, which allows for better performance and responsiveness. If you need to wait for physics to stabilize, consider using my_world.play() followed by multiple step_async() calls.
    To create an simulation of Franka robot, the code should be like this:
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.nucleus import get_assets_root_path

assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
simulation_context = SimulationContext()
add_reference_to_stage(asset_path, "/Franka")
#create_prim("/DistantLight", "DistantLight")




    To control the Franka robot, the code should be like this:

from omni.isaac.core import SimulationContext
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.nucleus import get_assets_root_path

my_world = World(stage_units_in_meters=1.0)

assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"

simulation_context = SimulationContext()
add_reference_to_stage(asset_path, "/Franka")

# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()
art = Articulation("/Franka")
art.initialize(my_world.physics_sim_view)
dof_ptr = art.get_dof_index("panda_joint2")

simulation_context.play()
# NOTE: before interacting with dc directly you need to step physics for one step at least
# simulation_context.step(render=True) which happens inside .play()
for i in range(1000):
    art.set_joint_positions([-1.5], [dof_ptr])
    simulation_context.step(render=True)

simulation_context.stop()


    
    Parameters:
    - code: The Python code to execute, e.g. "omni.kit.commands.execute("CreatePrim", prim_type="Sphere")"
    """
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        print("code: ", code)
        
        result = isaac.send_command("execute_script", {"code": code})
        print("result: ", result)
        return result
        # return f"Code executed successfully: {result.get('result', '')}"
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        # return f"Error executing code: {str(e)}"
        return {"status": "error", "error": str(e), "message": "Error executing code"}

def create_room_layout_scene(scene_save_dir: str) -> str:
    """
    Create a room layout scene from a dictionary of mesh information.

    Args:
        scene_save_dir: The directory to save the scene layout.

    Returns:
        String with result information.
    """
    try:

        
        isaac = get_isaac_connection()
        
        result = isaac.send_command("create_room_layout_scene", {"scene_save_dir": scene_save_dir})
        return result
    except Exception as e:
        logger.error(f"Error create_room_layout_scene: {str(e)}")
        # return f"Error create_room_layout_scene: {str(e)}"
        return f'error: {"status": "error", "error": str(e), "message": "Error create_room_layout_scene"}'

def create_single_room_layout_scene(scene_save_dir: str, room_id: str) -> str:
    """
    Create a single room layout scene from a dictionary of mesh information.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("create_single_room_layout_scene", {"scene_save_dir": scene_save_dir, "room_id": room_id})
        return result
    except Exception as e:
        logger.error(f"Error create_single_room_layout_scene: {str(e)}")
        return {"status": "error", "error": str(e), "message": "Error create_single_room_layout_scene"}

def create_single_room_layout_scene_from_room(scene_save_dir: str, room_dict_save_path: str) -> str:
    """
    Create a single room layout scene from a dictionary of mesh information.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("create_single_room_layout_scene_from_room", {"scene_save_dir": scene_save_dir, "room_dict_save_path": room_dict_save_path})
        return result
    except Exception as e:
        logger.error(f"Error create_single_room_layout_scene: {str(e)}")
        return {"status": "error", "error": str(e), "message": "Error create_single_room_layout_scene"}

def create_room_groups_layouts(scene_save_dir: str, room_list_save_path: str) -> str:
    """
    Create a single room layout scene from a dictionary of mesh information.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("create_room_groups_layouts", {"scene_save_dir": scene_save_dir, "room_list_save_path": room_list_save_path})
        return result
    except Exception as e:
        logger.error(f"Error create_room_groups_layouts: {str(e)}")
        return {"status": "error", "error": str(e), "message": "Error create_room_groups_layouts"}



def get_room_layout_scene_usd(scene_save_dir: str, usd_file_path: str) -> str:
    """
    Get the USD file of the room layout scene.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("get_room_layout_scene_usd", {"scene_save_dir": scene_save_dir, "usd_file_path": usd_file_path})
        return result
    except Exception as e:
        logger.error(f"Error get_room_layout_scene_usd: {str(e)}")
        return f'error: {"status": "error", "error": str(e), "message": "Error get_room_layout_scene_usd"}'

def get_room_layout_scene_usd_separate(scene_save_dir: str, usd_collection_dir: str) -> str:
    """
    Get the USD file of the room layout scene.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("get_room_layout_scene_usd_separate", {"scene_save_dir": scene_save_dir, "usd_collection_dir": usd_collection_dir})
        return result
    except Exception as e:
        logger.error(f"Error get_room_layout_scene_usd_separate: {str(e)}")
        return f'error: {"status": "error", "error": str(e), "message": "Error get_room_layout_scene_usd_separate"}'

def get_room_layout_scene_usd_separate_from_layout(layout_json_path: str, usd_collection_dir: str) -> str:
    """
    Get the USD file of the room layout scene.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("get_room_layout_scene_usd_separate_from_layout", {"layout_json_path": layout_json_path, "usd_collection_dir": usd_collection_dir})
        return result
    except Exception as e:
        logger.error(f"Error get_room_layout_scene_usd_separate: {str(e)}")
        return f'error: {"status": "error", "error": str(e), "message": "Error get_room_layout_scene_usd_separate"}'


def simulate_the_scene() -> str:
    """
    Simulate the scene. Before simulate the scene, you need to call create_room_layout_scene(scene_save_dir) first to create the scene.


    Returns:
        String with the result of the simulation. Each item in the dictionary is a key-value pair.
        The key is the object id in the simulation.

        The value is a dictionary with the following keys:

    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("simulate_the_scene")
        # result = {
        #     "status": "success",
        #     "next_step": "stable",
        #     "message": "Simulation completed successfully"
        # }
        return result
    except Exception as e:
        logger.error(f"Error simulate_the_scene: {str(e)}")
        return str({"status": "error", "error": str(e), "message": "Error simulate_the_scene"})

def simulate_the_scene_groups() -> str:
    """
    Simulate the scene. Before simulate the scene, you need to call create_room_layout_scene(scene_save_dir) first to create the scene.
    """
    try:
        isaac = get_isaac_connection()
        result = isaac.send_command("simulate_the_scene_groups")
        return result
    except Exception as e:
        logger.error(f"Error simulate_the_scene_groups: {str(e)}")
        return str({"status": "error", "error": str(e), "message": "Error simulate_the_scene_groups"})


def test_object_placements_in_single_room(room_dict_save_path: str, placements_info_path: str, only_need_one: bool = False):
    """
    Test object placements in a single room.
    """
    try:
        isaac = get_isaac_connection()
        print(f"room_dict_save_path: {room_dict_save_path}, placements_info_path: {placements_info_path}")
        result = isaac.send_command("test_object_placements_in_single_room", {"room_dict_save_path": room_dict_save_path, "placements_info_path": placements_info_path, "only_need_one": only_need_one})
        return result
    except Exception as e:
        logger.error(f"Error test_object_placements_in_single_room: {str(e)}")
        return f'error: {"status": "error", "error": str(e), "message": "Error test_object_placements_in_single_room"}'

# @mcp.prompt()
# def asset_creation_strategy() -> str:
#     """Defines the preferred strategy for creating assets in Isaac Sim"""
#     return """
#     0. Before anything, always check the scene from get_scene_info(), retrive rool path of assset through return value of assets_root_path.
#     1. If the scene is empty, create a physics scene with create_physics_scene()
#     2. if execute script due to communication error, then retry 3 times at most

#     3. For Franka robot simulation, the code should be like this:
# from omni.isaac.core import SimulationContext
# from omni.isaac.core.utils.prims import create_prim
# from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
# from omni.isaac.nucleus import get_assets_root_path

# assets_root_path = get_assets_root_path()
# asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
# add_reference_to_stage(asset_path, "/Franka")
# #create_prim("/DistantLight", "DistantLight")


# # need to initialize physics getting any articulation..etc
# simulation_context = SimulationContext()
# simulation_context.initialize_physics()
# simulation_context.play()

# for i in range(1000):
#     simulation_context.step(render=True)

# simulation_context.stop()

#     4. For Franka robot control, the code should be like this:
    
# from omni.isaac.core import SimulationContext
# from omni.isaac.core.utils.prims import create_prim
# from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
# from omni.isaac.nucleus import get_assets_root_path
# from pxr import UsdPhysics

# def create_physics_scene(stage, scene_path="/World/PhysicsScene"):
#     if not stage.GetPrimAtPath(scene_path):
#         UsdPhysics.Scene.Define(stage, scene_path)
    
#     return stage.GetPrimAtPath(scene_path)

# stage = omni.usd.get_context().get_stage()
# physics_scene = create_physics_scene(stage)
# if not physics_scene:
#     raise RuntimeError("Failed to create or find physics scene")
# import omni.physics.tensors as physx

# def create_simulation_view(stage):
#     sim_view = physx.create_simulation_view(stage)
#     if not sim_view:
#         carb.log_error("Failed to create simulation view")
#         return None
    
#     return sim_view

# sim_view = create_simulation_view(stage)
# if not sim_view:
#     raise RuntimeError("Failed to create simulation view")

# simulation_context = SimulationContext()
# assets_root_path = get_assets_root_path()
# asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
# add_reference_to_stage(asset_path, "/Franka")
# #create_prim("/DistantLight", "DistantLight")

# # need to initialize physics getting any articulation..etc
# simulation_context.initialize_physics()
# art = Articulation("/Franka")
# art.initialize()
# dof_ptr = art.get_dof_index("panda_joint2")

# simulation_context.play()
# # NOTE: before interacting with dc directly you need to step physics for one step at least
# # simulation_context.step(render=True) which happens inside .play()
# for i in range(1000):
#     art.set_joint_positions([-1.5], [dof_ptr])
#     simulation_context.step(render=True)

# simulation_context.stop()

#     5. For Jetbot simulation, the code should be like this:
# import carb
# import numpy as np
# from omni.isaac.core import World
# from omni.isaac.core import SimulationContext
# from omni.isaac.core.utils.prims import create_prim
# from omni.isaac.nucleus import get_assets_root_path
# from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
# from omni.isaac.wheeled_robots.robots import WheeledRobot

# simulation_context = SimulationContext()
# simulation_context.initialize_physics()

# my_world = World(stage_units_in_meters=1.0)

# assets_root_path = get_assets_root_path()
# if assets_root_path is None:
#     carb.log_error("Could not find Isaac Sim assets folder")
# jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
# my_jetbot = my_world.scene.add(
#     WheeledRobot(
#         prim_path="/World/Jetbot",
#         name="my_jetbot",
#         wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
#         create_robot=True,
#         usd_path=jetbot_asset_path,
#         position=np.array([0, 0.0, 2.0]),
#     )
# )


# create_prim("/DistantLight", "DistantLight")
# # need to initialize physics getting any articulation..etc


# my_world.scene.add_default_ground_plane()
# my_controller = DifferentialController(name="simple_control", wheel_radius=0.03, wheel_base=0.1125)
# my_world.reset()

# simulation_context.play()
# for i in range(10):
#     simulation_context.step(render=True) 

# i = 0
# reset_needed = False
# while i < 2000:
#     my_world.step(render=True)
#     if my_world.is_stopped() and not reset_needed:
#         reset_needed = True
#     if my_world.is_playing():
#         if reset_needed:
#             my_world.reset()
#             my_controller.reset()
#             reset_needed = False
#         if i >= 0 and i < 1000:
#             # forward
#             my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
#             print(my_jetbot.get_linear_velocity())
#         elif i >= 1000 and i < 1300:
#             # rotate
#             my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.0, np.pi / 12]))
#             print(my_jetbot.get_angular_velocity())
#         elif i >= 1300 and i < 2000:
#             # forward
#             my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
#         elif i == 2000:
#             i = 0
#         i += 1
# simulation_context.stop()

# 6. For G1 simulation, the code should be like this see g1_ok.py


#     """



def transform(
    prim_path: str,
    position: List[float] = [0, 0, 50],
    scale: List[float] = [10, 10, 10]
) -> str:
    """
    Transform a USD model by applying position and scale.
    
    Args:
        prim_path (str): Path to the USD prim to transform
        position (list, optional): The position to set [x, y, z]
        scale (list, optional): The scale to set [x, y, z]
        
    Returns:
        String with transformation result
    """
    try:
        # Get the global connection
        isaac = get_isaac_connection()
        
        result = isaac.send_command("transform", {
            "prim_path": prim_path,
            "position": position,
            "scale": scale
        })
        
        if result.get("status") == "success":
            return f"Successfully transformed model at {prim_path} to position {position} and scale {scale}"
        else:
            return f"Error transforming model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error transforming model: {str(e)}")
        return f"Error transforming model: {str(e)}"

