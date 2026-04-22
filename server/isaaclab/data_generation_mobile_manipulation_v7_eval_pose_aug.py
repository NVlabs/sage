# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import torch

import argparse
from doctest import FAIL_FAST
import gc
from omni.isaac.lab.app import AppLauncher

num_envs = 1

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=num_envs, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Mobile-Manipulation-Obj-Scene-Franka-IK-Abs-v3", help="Name of the task.")
parser.add_argument("--num_demos", type=int, default=128, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
parser.add_argument("--visualize_figs", action="store_true", help="Generate visualization figures for debugging (default: False)")
parser.add_argument("--post_fix", type=str, default="", help="Postfix for the log directory")
parser.add_argument("--policy_path", type=str, default="", help="Path to the policy checkpoint")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import open3d as o3d
import imageio
import json
import random
import shutil
import igl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import time
# Add parent directory to Python path to import constants
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR, ROBOMIMIC_ROOT_DIR, M2T2_ROOT_DIR

sys.path.insert(0, SERVER_ROOT_DIR)
sys.path.insert(0, M2T2_ROOT_DIR)
# from utils import (
#     dict_to_floor_plan, 
#     get_layout_from_scene_save_dir,
#     get_layout_from_scene_json_path
# )

import importlib.util
utils_spec = importlib.util.spec_from_file_location("server_utils", os.path.join(SERVER_ROOT_DIR, "utils.py"))
server_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(server_utils)

# Import the specific functions from server utils
dict_to_floor_plan = server_utils.dict_to_floor_plan
get_layout_from_scene_save_dir = server_utils.get_layout_from_scene_save_dir
get_layout_from_scene_json_path = server_utils.get_layout_from_scene_json_path

# from tex_utils import (
#     export_layout_to_mesh_dict_list_tree_search_with_object_id,
#     export_layout_to_mesh_dict_object_id,
#     get_textured_object_mesh
# )

import importlib.util
utils_spec = importlib.util.spec_from_file_location("tex_utils", os.path.join(SERVER_ROOT_DIR, "tex_utils.py"))
tex_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(tex_utils)

# Import the specific functions from server utils
export_layout_to_mesh_dict_list_tree_search_with_object_id = tex_utils.export_layout_to_mesh_dict_list_tree_search_with_object_id
export_layout_to_mesh_dict_object_id = tex_utils.export_layout_to_mesh_dict_object_id
get_textured_object_mesh = tex_utils.get_textured_object_mesh

from m2t2_utils.data import generate_m2t2_data
from m2t2_utils.infer import load_m2t2, infer_m2t2
import trimesh
from models import FloorPlan
from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml, load_yaml, load_pickle

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.data_collector import RobomimicDataCollector
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, AssetBase
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from omni.isaac.lab_assets import RIDGEBACK_FRANKA_PANDA_CFG  # isort:skip
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg, RigidObjectCollectionCfg
from omni.isaac.lab.sensors import ContactSensorCfg
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.core.utils.prims as prim_utils
from isaaclab.curobo_tools_mobile_grasp.curobo_planner import MotionPlanner

from isaaclab.omron_franka_occupancy import occupancy_map, support_point, get_forward_side_from_support_point_and_yaw


from robomimic.algo import RolloutPolicy
import robomimic.utils.file_utils as FileUtils


##
# UNIFIED COLLISION CHECKING CONFIGURATION
##
class CollisionCheckingConfig:
    """Unified configuration for collision checking across all functions."""
    
    # Grid and distance parameters - use the most conservative values
    GRID_RES = 0.05  # Fine resolution for accuracy
    ROBOT_MIN_DIST_TO_ROOM_EDGE = 0.5  # Conservative room edge distance
    ROBOT_MIN_DIST_TO_OBJECT = 0.40  # Conservative object distance (max from all functions)
    
    # Robot occupancy parameters - use most conservative settings
    ROBOT_OCCUPANCY_OFFSET = 0.05  # Conservative robot size offset
    ROBOT_SPAWN_OCCUPANCY_OFFSET = 0.40  # Stricter offset for spawn collision checking
    
    # Collision checking parameters
    CHECK_RANGE = 0.5  # Range around robot center to check for collisions
    CHECK_RES = 0.02   # Fine resolution for collision checking points
    
    # Template-based checking (for trajectory planning optimization)
    NUM_ORIENTATION_SAMPLES = 36  # Sample orientations every 10 degrees
    TEMPLATE_CHECK_RANGE = 0.7    # Slightly larger range for template checking
    TEMPLATE_CHECK_RES = 0.02     # Fine resolution for template checking
    
    # Place location sampling parameters
    MIN_DIST_TO_BOUNDARY = 0.15   # Minimum distance from place location to table boundary
    MAX_DIST_TO_OBJECT = 0.8      # Maximum distance from robot to place location for feasibility


def create_unified_occupancy_grid(scene_save_dir, layout_name, room_id):
    """
    Create a unified occupancy grid that will be consistent across all functions.
    
    Returns:
        tuple: (occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, floor_plan, target_room)
    """
    layout_json_path = os.path.join(scene_save_dir, f"{layout_name}.json")
    with open(layout_json_path, "r") as f:
        layout_info = json.load(f)
    
    floor_plan: FloorPlan = dict_to_floor_plan(layout_info)
    target_room = next(room for room in floor_plan.rooms if room.id == room_id)
    assert target_room is not None, f"target_room {room_id} not found in floor_plan"

    # Get room rectangle bounds
    room_min_x = target_room.position.x
    room_min_y = target_room.position.y
    room_max_x = target_room.position.x + target_room.dimensions.width
    room_max_y = target_room.position.y + target_room.dimensions.length
    room_bounds = (room_min_x, room_min_y, room_max_x, room_max_y)
    
    # Get all object meshes in the room for occupancy calculation
    object_meshes = []
    for obj in target_room.objects:
        try:
            mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, obj.id)
            if mesh_info and mesh_info["mesh"] is not None:
                object_meshes.append(mesh_info["mesh"])
        except Exception as e:
            print(f"Warning: Could not load mesh for object {obj.id}: {e}")
            continue
    
    # Combine all object meshes for ray casting
    if object_meshes:
        combined_mesh = trimesh.util.concatenate(object_meshes)
    else:
        # If no meshes available, create an empty mesh
        combined_mesh = trimesh.Trimesh()
    
    # Create occupancy grid using ray casting with unified parameters
    grid_x = np.arange(room_min_x, room_max_x, CollisionCheckingConfig.GRID_RES)
    grid_y = np.arange(room_min_y, room_max_y, CollisionCheckingConfig.GRID_RES)
    
    # Create ray origins at grid centers, elevated above the room
    grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x + CollisionCheckingConfig.GRID_RES/2, 
                                          grid_y + CollisionCheckingConfig.GRID_RES/2, 
                                          indexing='ij')
    ray_origins = np.stack([
        grid_x_mesh.flatten(), 
        grid_y_mesh.flatten(), 
        np.full(grid_x_mesh.size, 10.0)
    ], axis=1).astype(np.float32)
    
    ray_directions = np.tile([0, 0, -1], (len(ray_origins), 1)).astype(np.float32)  # All pointing down
    
    # Perform ray casting to detect object occupancy
    occupancy_grid = np.zeros((len(grid_x), len(grid_y)), dtype=bool)
    
    if len(object_meshes) > 0:
        try:
            locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(combined_mesh).intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions
            )
            
            # Mark occupied grid cells using vectorized operations
            if len(index_ray) > 0:
                grid_i = index_ray // len(grid_y)
                grid_j = index_ray % len(grid_y)
                
                # Filter valid indices
                valid_mask = (grid_i < len(grid_x)) & (grid_j < len(grid_y))
                valid_i = grid_i[valid_mask]
                valid_j = grid_j[valid_mask]
                
                # Mark all valid cells as occupied at once
                occupancy_grid[valid_i, valid_j] = True
        except Exception as e:
            print(f"Warning: Ray casting failed: {e}")
    
    return occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, floor_plan, target_room


def create_unified_scene_occupancy_fn(occupancy_grid, grid_x, grid_y, room_bounds):
    """
    Create a unified scene occupancy function that's consistent across all collision checks.
    
    Args:
        occupancy_grid: 2D boolean array of scene occupancy
        grid_x, grid_y: Grid coordinate arrays
        room_bounds: Tuple of (min_x, min_y, max_x, max_y)
    
    Returns:
        function: Scene occupancy checker function
    """
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    def scene_occupancy_fn(points):
        """Check if points are in scene object occupancy using the unified grid."""
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        # Convert points to grid coordinates
        grid_coords_x = np.floor((points[:, 0] - room_min_x) / CollisionCheckingConfig.GRID_RES).astype(int)
        grid_coords_y = np.floor((points[:, 1] - room_min_y) / CollisionCheckingConfig.GRID_RES).astype(int)
        
        # Check bounds
        valid_coords = (
            (grid_coords_x >= 0) & (grid_coords_x < len(grid_x)) &
            (grid_coords_y >= 0) & (grid_coords_y < len(grid_y))
        )
        
        # Initialize result as False (not occupied)
        result = np.zeros(len(points), dtype=bool)
        
        # Check occupancy for valid coordinates
        if np.any(valid_coords):
            result[valid_coords] = occupancy_grid[
                grid_coords_x[valid_coords], 
                grid_coords_y[valid_coords]
            ]
        
        return result
    
    return scene_occupancy_fn


def check_unified_robot_collision(pos, quat, scene_occupancy_fn, room_bounds, robot_occupancy_offset=None):
    """
    Unified robot collision checking function used by all sampling and planning functions.
    
    Args:
        pos: Robot position [x, y, z] (only x,y used)
        quat: Robot quaternion [w, x, y, z] (scalar-first format)
        scene_occupancy_fn: Scene occupancy checking function
        room_bounds: Tuple of (min_x, min_y, max_x, max_y)
        robot_occupancy_offset: Optional custom offset for robot occupancy (defaults to CollisionCheckingConfig.ROBOT_OCCUPANCY_OFFSET)
    
    Returns:
        bool: True if collision detected, False otherwise
    """
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Check room bounds first (fastest check)
    if (pos[0] < room_min_x + CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE or 
        pos[0] > room_max_x - CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE or
        pos[1] < room_min_y + CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE or 
        pos[1] > room_max_y - CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE):
        return True
    
    # Convert quaternion to yaw
    if len(quat) == 4:
        # Handle both [w,x,y,z] and [x,y,z,w] formats
        if isinstance(quat, torch.Tensor):
            quat_np = quat.cpu().numpy() if hasattr(quat, 'cpu') else quat.numpy()
        else:
            quat_np = np.array(quat)
        
        # Assume scalar-first format [w,x,y,z], convert to [x,y,z,w] for scipy
        try:
            rotation = R.from_quat([quat_np[1], quat_np[2], quat_np[3], quat_np[0]])
            yaw = rotation.as_euler('xyz')[2]
        except:
            # Fallback: try [x,y,z,w] format
            rotation = R.from_quat(quat_np)
            yaw = rotation.as_euler('xyz')[2]
    else:
        raise ValueError(f"Invalid quaternion format: {quat}")
    
    # Get forward/side from support point and yaw
    forward, side = get_forward_side_from_support_point_and_yaw([pos[0], pos[1]], yaw)
    
    # Create robot occupancy checker with unified parameters
    # Use custom offset if provided, otherwise use default
    if robot_occupancy_offset is None:
        robot_occupancy_offset = CollisionCheckingConfig.ROBOT_OCCUPANCY_OFFSET
    robot_occupancy_fn = occupancy_map(forward, side, yaw, offset=robot_occupancy_offset)
    
    # Sample points within and around the robot's footprint to check for collisions
    # Use unified collision checking parameters
    robot_center_x = side + (-0.20)  # Robot center offset from occupancy_map
    robot_center_y = forward + 0.0
    
    check_x = np.arange(robot_center_x - CollisionCheckingConfig.CHECK_RANGE, 
                       robot_center_x + CollisionCheckingConfig.CHECK_RANGE, 
                       CollisionCheckingConfig.CHECK_RES)
    check_y = np.arange(robot_center_y - CollisionCheckingConfig.CHECK_RANGE, 
                       robot_center_y + CollisionCheckingConfig.CHECK_RANGE, 
                       CollisionCheckingConfig.CHECK_RES)
    check_x_mesh, check_y_mesh = np.meshgrid(check_x, check_y)
    check_points = np.column_stack([check_x_mesh.ravel(), check_y_mesh.ravel()])
    
    # Get robot occupancy for these points
    robot_occupied = robot_occupancy_fn(check_points)
    
    # Get scene occupancy for the same points
    scene_occupied = scene_occupancy_fn(check_points)
    
    # Check for collision: if any point is occupied by both robot and scene
    collision = np.any(robot_occupied & scene_occupied)
    
    return collision


def get_arm_action_relative(ee_frame_pos, ee_frame_quat, target_ee_pos, target_ee_quat):
    d_rotvec = math_utils.axis_angle_from_quat(math_utils.quat_unique(math_utils.quat_mul(target_ee_quat, math_utils.quat_inv(ee_frame_quat))))
    arm_action = torch.cat([target_ee_pos - ee_frame_pos, d_rotvec.reshape(-1, 3)], dim=-1)
    # print(f"arm_action: {arm_action.shape} {arm_action}")
    return arm_action

def get_default_camera_view():
    return [0., 0., 0.], [1., 0., 0.]

def get_default_base_pos():
    return [0., 0., 0.]

def get_action_relative(ee_frame_pos, ee_frame_quat, target_ee_pos, target_ee_quat):
    d_rotvec = math_utils.axis_angle_from_quat(math_utils.quat_unique(math_utils.quat_mul(target_ee_quat, math_utils.quat_inv(ee_frame_quat))))
    d_pos = target_ee_pos - ee_frame_pos
    if d_pos.abs().max() > 1.0:
        d_pos = d_pos / d_pos.abs().max()
    if d_rotvec.abs().max() > 1.0:
        d_rotvec = d_rotvec / d_rotvec.abs().max()
    arm_action = torch.cat([d_pos.reshape(-1, 3), d_rotvec.reshape(-1, 3)], dim=-1)
    if arm_action.abs().max() > 1.0:
        print(f"arm_action: {arm_action.shape} {arm_action}")
    return arm_action


def visualize_robot_place_planning_data(
    room_bounds, occupancy_grid, grid_x, grid_y, grid_res,
    table_bounds, table_occupancy_grid, 
    robot_base_positions, robot_base_quats, place_locations,
    table_object, valid_place_cells=None, valid_robot_points=None,
    layout_name="", room_id="", table_object_name="",
    save_path=None
):
    """
    Visualize robot place location planning data including room occupancy, table occupancy, 
    place locations, robot positions, and their relationships.
    
    Args:
        room_bounds: tuple of (min_x, min_y, max_x, max_y) for the room
        occupancy_grid: 2D numpy array of boolean occupancy for scene objects
        grid_x, grid_y: arrays defining room grid coordinates
        grid_res: room grid resolution
        table_bounds: tuple of (min_x, min_y, max_x, max_y) for the table
        table_occupancy_grid: 2D numpy array of boolean occupancy for table surface
        robot_base_positions: tensor/array of robot positions (N, 3)
        robot_base_quats: tensor/array of robot orientations (N, 4) - quaternions [w, x, y, z]
        place_locations: tensor/array of place locations (N, 3)
        table_object: object with position attributes
        valid_place_cells: optional list of valid place cells for debugging
        valid_robot_points: optional array of valid robot positions for debugging
        layout_name, room_id, table_object_name: strings for labeling
        save_path: path to save the PNG file
    """
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    table_min_x, table_min_y, table_max_x, table_max_y = table_bounds
    
    # Import required matplotlib components
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from scipy.spatial.transform import Rotation as R
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 1. Draw room boundaries
    room_rect = patches.Rectangle(
        (room_min_x, room_min_y), 
        room_max_x - room_min_x, 
        room_max_y - room_min_y,
        linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.2,
        label='Room Boundary'
    )
    ax.add_patch(room_rect)
    
    # 2. Visualize room occupancy grid
    if occupancy_grid is not None and occupancy_grid.size > 0:
        extent = [room_min_x, room_max_x, room_min_y, room_max_y]
        
        # Create custom colormap: white for free, red for occupied
        colors = ['white', 'red']
        cmap = ListedColormap(colors)
        
        # Display room occupancy grid
        im = ax.imshow(
            occupancy_grid.T,  # Transpose for correct orientation
            extent=extent,
            origin='lower',
            cmap=cmap,
            alpha=0.4,
            aspect='equal'
        )
    
    # 3. Draw table boundaries and occupancy
    table_rect = patches.Rectangle(
        (table_min_x, table_min_y), 
        table_max_x - table_min_x, 
        table_max_y - table_min_y,
        linewidth=2, edgecolor='brown', facecolor='none',
        label='Table Boundary'
    )
    ax.add_patch(table_rect)
    
    # 4. Visualize table occupancy grid (if available)
    if table_occupancy_grid is not None and table_occupancy_grid.size > 1:
        table_extent = [table_min_x, table_max_x, table_min_y, table_max_y]
        
        # Create custom colormap for table: transparent for no table, green for table surface
        table_colors = [(0, 0, 0, 0), (0.2, 0.8, 0.2, 0.6)]  # transparent, semi-transparent green
        table_cmap = ListedColormap(table_colors)
        
        # Display table occupancy grid
        table_im = ax.imshow(
            table_occupancy_grid.T,  # Transpose for correct orientation
            extent=table_extent,
            origin='lower',
            cmap=table_cmap,
            alpha=0.8,
            aspect='equal'
        )
    
    # 5. Draw valid place cells (if provided for debugging)
    if valid_place_cells is not None and len(valid_place_cells) > 0:
        place_cells_array = np.array(valid_place_cells)
        ax.scatter(
            place_cells_array[:, 0], place_cells_array[:, 1],
            c='lightgreen', s=20, alpha=0.6, marker='s',
            label=f'Valid Place Cells ({len(valid_place_cells)})'
        )
    
    # 6. Draw valid robot positions (if provided for debugging)
    if valid_robot_points is not None and len(valid_robot_points) > 0:
        ax.scatter(
            valid_robot_points[:, 0], valid_robot_points[:, 1],
            c='lightblue', s=8, alpha=0.4, marker='o',
            label=f'Valid Robot Positions ({len(valid_robot_points)})'
        )
    
    # 7. Convert tensors to numpy if needed
    if hasattr(robot_base_positions, 'cpu'):
        robot_positions_np = robot_base_positions.cpu().numpy()
        robot_quats_np = robot_base_quats.cpu().numpy()
        place_locations_np = place_locations.cpu().numpy()
    else:
        robot_positions_np = np.array(robot_base_positions)
        robot_quats_np = np.array(robot_base_quats)
        place_locations_np = np.array(place_locations)
    
    # 8. Create robot occupancy overlay for each robot
    robot_occupancy_combined = np.zeros_like(occupancy_grid, dtype=bool)
    
    for i, (pos, quat) in enumerate(zip(robot_positions_np, robot_quats_np)):
        # Convert quaternion to yaw angle
        # quaternion is [w, x, y, z] format (scalar-first)
        rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert to [x, y, z, w] for scipy
        yaw = rotation.as_euler('xyz')[2]  # Get Z-axis rotation (yaw)
        
        # Use support point (robot position) and yaw to get forward/side
        support_pos = [pos[0], pos[1]]
        forward, side = get_forward_side_from_support_point_and_yaw(support_pos, yaw)
        
        # Create robot occupancy function
        robot_occupancy_fn = occupancy_map(forward, side, yaw, offset=0.05)
        
        # Create points for robot occupancy grid
        robot_grid_x_mesh, robot_grid_y_mesh = np.meshgrid(grid_x + grid_res/2, grid_y + grid_res/2, indexing='ij')
        robot_grid_points = np.stack([
            robot_grid_x_mesh.flatten(), 
            robot_grid_y_mesh.flatten()
        ], axis=1)
        
        # Get robot occupancy for grid points
        robot_occupied = robot_occupancy_fn(robot_grid_points)
        robot_occupied_grid = robot_occupied.reshape(robot_grid_x_mesh.shape)
        
        # Add to combined robot occupancy
        robot_occupancy_combined |= robot_occupied_grid
    
    # 9. Visualize robot occupancy overlay
    if robot_occupancy_combined.size > 0:
        extent = [room_min_x, room_max_x, room_min_y, room_max_y]
        
        # Create custom colormap for robot occupancy: transparent for free, blue for occupied
        robot_colors = [(0, 0, 0, 0), (0, 0, 1, 0.3)]  # transparent, semi-transparent blue
        robot_cmap = ListedColormap(robot_colors)
        
        # Display robot occupancy grid
        robot_im = ax.imshow(
            robot_occupancy_combined.T,  # Transpose for correct orientation
            extent=extent,
            origin='lower',
            cmap=robot_cmap,
            alpha=0.7,
            aspect='equal'
        )
    
    # 10. Draw robot positions, orientations, place locations and connections
    for i, (robot_pos, robot_quat, place_pos) in enumerate(zip(robot_positions_np, robot_quats_np, place_locations_np)):
        robot_x, robot_y = robot_pos[0], robot_pos[1]
        place_x, place_y = place_pos[0], place_pos[1]
        
        # Convert quaternion to yaw for arrow display
        rotation = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])  # Convert to [x, y, z, w] for scipy
        yaw = rotation.as_euler('xyz')[2]  # Get Z-axis rotation (yaw)
        
        # Color scheme for different environments
        color = plt.cm.viridis(i / max(1, len(robot_positions_np) - 1))
        
        # Robot base position
        ax.scatter(robot_x, robot_y, c=[color], s=120, marker='o', 
                  edgecolors='black', linewidth=2,
                  label=f'Robot {i}' if i < 5 else '')  # Only label first 5 to avoid clutter
        
        # Robot orientation arrow
        arrow_length = 0.3
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        
        ax.arrow(robot_x, robot_y, dx, dy, head_width=0.05, head_length=0.05,
                fc=color, ec='black', linewidth=1, alpha=0.8)
        
        # Place location
        ax.scatter(place_x, place_y, c=[color], s=100, marker='*', 
                  edgecolors='darkred', linewidth=2, alpha=0.9,
                  label=f'Place {i}' if i < 5 else '')
        
        # Connection line from robot to place location
        ax.plot([robot_x, place_x], [robot_y, place_y], '--', color=color, alpha=0.6, linewidth=2)
        
        # Distance annotation
        distance = np.linalg.norm([place_x - robot_x, place_y - robot_y])
        mid_x, mid_y = (robot_x + place_x) / 2, (robot_y + place_y) / 2
        ax.annotate(f'{distance:.2f}m', (mid_x, mid_y), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7, color=color,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # 11. Draw table object center
    table_center_x = table_object.position.x
    table_center_y = table_object.position.y
    ax.scatter(table_center_x, table_center_y, c='brown', s=150, marker='s', 
              edgecolors='darkred', linewidth=2,
              label=f'Table Center ({table_object_name})')
    
    # 12. Add legend entries
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]
    
    # Add custom legend entries
    legend_elements.extend([
        Patch(facecolor='red', alpha=0.4, label='Scene Occupancy'),
        Patch(facecolor='green', alpha=0.6, label='Table Surface'),
        Patch(facecolor='blue', alpha=0.3, label='Robot Occupancy')
    ])
    legend_labels.extend(['Scene Occupancy', 'Table Surface', 'Robot Occupancy'])
    
    # 13. Add legend and labels
    ax.legend(handles=legend_elements, labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Robot Place Location Planning Visualization\n'
                f'Layout: {layout_name}, Room: {room_id}\n'
                f'Table: {table_object_name}', 
                fontsize=14, pad=20)
    
    # 14. Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 15. Add text annotations
    table_area = (table_max_x - table_min_x) * (table_max_y - table_min_y)
    room_area = (room_max_x - room_min_x) * (room_max_y - room_min_y)
    
    info_text = (f"Grid Resolution: {grid_res:.3f}m\n"
                f"Room Size: {room_max_x-room_min_x:.2f}×{room_max_y-room_min_y:.2f}m\n"
                f"Table Size: {table_max_x-table_min_x:.2f}×{table_max_y-table_min_y:.2f}m\n"
                f"Num Environments: {len(robot_positions_np)}\n"
                f"Min Dist to Boundary: {CollisionCheckingConfig.MIN_DIST_TO_BOUNDARY:.2f}m\n"
                f"Max Dist to Object: {CollisionCheckingConfig.MAX_DIST_TO_OBJECT:.2f}m")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
           verticalalignment='top', fontsize=10)
    
    # 16. Tight layout and save
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Robot place location planning visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_robot_planning_data(
    room_bounds, occupancy_grid, grid_x, grid_y, grid_res,
    robot_base_positions, robot_base_quats, camera_lookats,
    target_object, table_object, valid_points=None,
    layout_name="", room_id="", target_object_name="", table_object_name="",
    save_path=None
):
    """
    Visualize robot planning data including occupancy grid, robot positions, orientations, robot occupancy, and camera lookat points.
    
    Args:
        room_bounds: tuple of (min_x, min_y, max_x, max_y)
        occupancy_grid: 2D numpy array of boolean occupancy for scene objects
        grid_x, grid_y: arrays defining grid coordinates
        grid_res: grid resolution
        robot_base_positions: tensor/array of robot positions (N, 3)
        robot_base_quats: tensor/array of robot orientations (N, 4) - quaternions [w, x, y, z]
        camera_lookats: tensor/array of camera lookat positions (N, 3)
        target_object: object with position attributes
        table_object: object with position attributes
        valid_points: optional array of valid robot positions for debugging
        layout_name, room_id, target_object_name, table_object_name: strings for labeling
        save_path: path to save the PNG file
    """
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Import required matplotlib components
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from scipy.spatial.transform import Rotation as R
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. Draw room boundaries
    room_rect = patches.Rectangle(
        (room_min_x, room_min_y), 
        room_max_x - room_min_x, 
        room_max_y - room_min_y,
        linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.3,
        label='Room Boundary'
    )
    ax.add_patch(room_rect)
    
    # 2. Visualize occupancy grid
    if occupancy_grid is not None and occupancy_grid.size > 0:
        # Create grid coordinates for visualization
        extent = [room_min_x, room_max_x, room_min_y, room_max_y]
        
        # Create custom colormap: white for free, red for occupied
        colors = ['white', 'red']
        cmap = ListedColormap(colors)
        
        # Display occupancy grid
        im = ax.imshow(
            occupancy_grid.T,  # Transpose for correct orientation
            extent=extent,
            origin='lower',
            cmap=cmap,
            alpha=0.6,
            aspect='equal'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Occupancy (White=Free, Red=Occupied)', rotation=270, labelpad=20)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Free', 'Occupied'])
    
    # 3. Draw valid robot positions (if provided for debugging)
    if valid_points is not None and len(valid_points) > 0:
        ax.scatter(
            valid_points[:, 0], valid_points[:, 1],
            c='lightgreen', s=8, alpha=0.4,
            label=f'Valid Robot Positions ({len(valid_points)})'
        )
    
    # 4. Draw target and table objects
    target_pos = [target_object.position.x, target_object.position.y]
    table_pos = [table_object.position.x, table_object.position.y]
    
    ax.scatter(*target_pos, c='blue', s=200, marker='*', 
              edgecolors='darkblue', linewidth=2, 
              label=f'Target Object ({target_object_name})')
    ax.scatter(*table_pos, c='brown', s=150, marker='s', 
              edgecolors='darkred', linewidth=2,
              label=f'Table ({table_object_name})')
    
    # 5. Convert tensors to numpy if needed
    if hasattr(robot_base_positions, 'cpu'):
        robot_positions_np = robot_base_positions.cpu().numpy()
        robot_quats_np = robot_base_quats.cpu().numpy()
        camera_lookats_np = camera_lookats.cpu().numpy()
    else:
        robot_positions_np = np.array(robot_base_positions)
        robot_quats_np = np.array(robot_base_quats)
        camera_lookats_np = np.array(camera_lookats)
    
    # 6. Create robot occupancy overlay
    robot_occupancy_combined = np.zeros_like(occupancy_grid, dtype=bool)
    
    for i, (pos, quat) in enumerate(zip(robot_positions_np, robot_quats_np)):
        # Convert quaternion to yaw angle
        # quaternion is [w, x, y, z] format (scalar-first)
        rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert to [x, y, z, w] for scipy
        yaw = rotation.as_euler('xyz')[2]  # Get Z-axis rotation (yaw)
        
        # Use support point (robot position) and yaw to get forward/side
        support_pos = [pos[0], pos[1]]
        forward, side = get_forward_side_from_support_point_and_yaw(support_pos, yaw)
        
        # Create robot occupancy function
        robot_occupancy_fn = occupancy_map(forward, side, yaw, offset=0.05)
        
        # Create points for robot occupancy grid
        robot_grid_x_mesh, robot_grid_y_mesh = np.meshgrid(grid_x + grid_res/2, grid_y + grid_res/2, indexing='ij')
        robot_grid_points = np.stack([
            robot_grid_x_mesh.flatten(), 
            robot_grid_y_mesh.flatten()
        ], axis=1)
        
        # Get robot occupancy for grid points
        robot_occupied = robot_occupancy_fn(robot_grid_points)
        robot_occupied_grid = robot_occupied.reshape(robot_grid_x_mesh.shape)
        
        # Add to combined robot occupancy
        robot_occupancy_combined |= robot_occupied_grid
    
    # 7. Visualize robot occupancy overlay
    if robot_occupancy_combined.size > 0:
        extent = [room_min_x, room_max_x, room_min_y, room_max_y]
        
        # Create custom colormap for robot occupancy: transparent for free, blue for occupied
        robot_colors = [(0, 0, 0, 0), (0, 0, 1, 0.3)]  # transparent, semi-transparent blue
        robot_cmap = ListedColormap(robot_colors)
        
        # Display robot occupancy grid
        robot_im = ax.imshow(
            robot_occupancy_combined.T,  # Transpose for correct orientation
            extent=extent,
            origin='lower',
            cmap=robot_cmap,
            alpha=0.7,
            aspect='equal'
        )
    
    # 8. Draw robot positions and orientations
    for i, (pos, quat, cam_lookat) in enumerate(zip(robot_positions_np, robot_quats_np, camera_lookats_np)):
        x, y = pos[0], pos[1]
        
        # Convert quaternion to yaw for arrow display
        rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert to [x, y, z, w] for scipy
        yaw = rotation.as_euler('xyz')[2]  # Get Z-axis rotation (yaw)
        
        # Robot base position
        color = plt.cm.viridis(i / max(1, len(robot_positions_np) - 1))
        ax.scatter(x, y, c=[color], s=100, marker='o', 
                  edgecolors='black', linewidth=1,
                  label=f'Robot {i}' if i < 5 else '')  # Only label first 5 to avoid clutter
        
        # Robot orientation arrow
        arrow_length = 0.3
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        
        ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.05,
                fc=color, ec='black', linewidth=1, alpha=0.8)
        
        # Camera lookat position
        cam_x, cam_y = cam_lookat[0], cam_lookat[1]
        ax.scatter(cam_x, cam_y, c=[color], s=60, marker='^', 
                  edgecolors='black', linewidth=1, alpha=0.7)
        
        # Line from robot to camera lookat
        ax.plot([x, cam_x], [y, cam_y], '--', color=color, alpha=0.5, linewidth=1)
    
    # 9. Add legend entries for robot occupancy
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]
    
    # Add robot occupancy to legend
    legend_elements.append(Patch(facecolor='blue', alpha=0.3, label='Robot Occupancy'))
    legend_labels.append('Robot Occupancy')
    
    # 10. Add legend and labels
    ax.legend(handles=legend_elements, labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Robot Planning Visualization with Occupancy\n'
                f'Layout: {layout_name}, Room: {room_id}\n'
                f'Target: {target_object_name}, Table: {table_object_name}', 
                fontsize=14, pad=20)
    
    # 11. Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 12. Add text annotations
    info_text = (f"Grid Resolution: {grid_res:.3f}m\n"
                f"Room Size: {room_max_x-room_min_x:.2f}×{room_max_y-room_min_y:.2f}m\n"
                f"Num Robots: {len(robot_positions_np)}\n"
                f"Robot Base: 0.7×0.5m + 0.05m offset")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
           verticalalignment='top', fontsize=10)
    
    # 13. Tight layout and save
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Robot planning visualization with occupancy saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def sample_robot_location(
    scene_save_dir, layout_name, room_id,
    target_object_name, table_object_name, num_envs, debug_dir
):

    """
    the sampling is solving a math problem:
    the room is a rectangle, and we treat the whole room as a 2d rectangle space, divided by grid_res x grid_res small occupancy rectangles.
    each object take over the occupancy rectangles that it covers.

    Uses unified collision checking configuration for consistency with trajectory planning.

    the way to sample robot location is:
    1. sample 100k points in the room rectangle, remove the points that: 
        i. inside the object occupancy rectangles 
        ii. has a distance less than robot_min_dist_to_object to the object occupancy rectangles.
        iii. has a distance less than robot_min_dist_to_room_edge to the room rectangle edge.
    2. among the remaining points, find the point that minimizes the distance to the target_object, 
    choose the point as the robot pos x and y, robot z is max(the height of the table top - robot_height_offset, 0) (use the table object height);
    3. the robot 3d rotation quaternion is a z-axis rotation from the robot pos towards the target_object_name.

    extra for debugging: save a 2d vis image of the room rectangle, object occupancy rectangles, all valid points, and the final robot pos and quat.
    """

    # Use unified parameters for consistency
    robot_height_offset = 0.20
    num_sample_points = 10000
    
    # Create unified occupancy grid and scene occupancy function
    occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, floor_plan, target_room = create_unified_occupancy_grid(
        scene_save_dir, layout_name, room_id
    )
    scene_occupancy_fn = create_unified_scene_occupancy_fn(occupancy_grid, grid_x, grid_y, room_bounds)
    
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds

    target_object = next(obj for obj in target_room.objects if obj.id == target_object_name)
    table_object = next(obj for obj in target_room.objects if obj.id == table_object_name)
    
    assert target_object is not None, f"target_object {target_object_name} not found in floor_plan"
    assert table_object is not None, f"table_object {table_object_name} not found in floor_plan"
    
    # Sample random points in the room rectangle
    sample_points = np.random.uniform(
        low=[room_min_x, room_min_y], 
        high=[room_max_x, room_max_y], 
        size=(num_sample_points, 2)
    )
    
    # Filter points based on constraints using vectorized operations
    # Check distance to room edges for all points at once
    dist_to_edges = np.minimum.reduce([
        sample_points[:, 0] - room_min_x,  # distance to left edge
        room_max_x - sample_points[:, 0],  # distance to right edge
        sample_points[:, 1] - room_min_y,  # distance to bottom edge
        room_max_y - sample_points[:, 1]   # distance to top edge
    ])
    
    # Filter out points too close to room edges
    edge_valid_mask = dist_to_edges >= CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE
    edge_valid_points = sample_points[edge_valid_mask]
    
    if len(edge_valid_points) == 0:
        valid_points = np.array([])
    else:
        # Convert to grid coordinates
        grid_coords = np.floor((edge_valid_points - [room_min_x, room_min_y]) / CollisionCheckingConfig.GRID_RES).astype(int)
        
        # Check if points are within grid bounds and not in occupied cells
        grid_valid_mask = (
            (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < len(grid_x)) &
            (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < len(grid_y)) &
            (~occupancy_grid[grid_coords[:, 0], grid_coords[:, 1]])
        )
        
        grid_valid_points = edge_valid_points[grid_valid_mask]
        grid_valid_coords = grid_coords[grid_valid_mask]
        
        if len(grid_valid_points) == 0:
            valid_points = np.array([])
        else:
            # Check distance to occupied cells using vectorized operations
            search_radius = int(np.ceil(CollisionCheckingConfig.ROBOT_MIN_DIST_TO_OBJECT / CollisionCheckingConfig.GRID_RES))
            
            # Find all occupied cell positions
            occupied_indices = np.where(occupancy_grid)
            if len(occupied_indices[0]) > 0:
                occupied_positions = np.column_stack([
                    room_min_x + occupied_indices[0] * CollisionCheckingConfig.GRID_RES + CollisionCheckingConfig.GRID_RES/2,
                    room_min_y + occupied_indices[1] * CollisionCheckingConfig.GRID_RES + CollisionCheckingConfig.GRID_RES/2
                ])
                
                # OPTIMIZATION: Use batched processing to avoid memory explosion
                batch_size = 1000  # Process points in smaller batches
                valid_points_list = []
                
                print(f"Processing {len(grid_valid_points)} valid points in batches of {batch_size}")
                print(f"Checking distance to {len(occupied_positions)} occupied cells")
                
                for i in range(0, len(grid_valid_points), batch_size):
                    batch_end = min(i + batch_size, len(grid_valid_points))
                    batch_points = grid_valid_points[i:batch_end]
                    
                    if i % (batch_size * 10) == 0:  # Print progress every 10 batches
                        print(f"Processing batch {i//batch_size + 1}/{(len(grid_valid_points)-1)//batch_size + 1}")
                    
                    # Calculate distances from batch to all occupied cells
                    # Shape: (batch_size, n_occupied_cells)
                    distances = np.linalg.norm(
                        batch_points[:, np.newaxis, :] - occupied_positions[np.newaxis, :, :], 
                        axis=2
                    )
                    
                    # Find minimum distance to any occupied cell for each point in batch
                    min_distances = np.min(distances, axis=1)
                    
                    # Filter points that are far enough from occupied cells
                    distance_valid_mask = min_distances >= CollisionCheckingConfig.ROBOT_MIN_DIST_TO_OBJECT
                    valid_points_list.append(batch_points[distance_valid_mask])
                
                # Combine all valid points from batches
                if valid_points_list:
                    valid_points = np.concatenate(valid_points_list, axis=0)
                    print(f"Found {len(valid_points)} valid robot positions after distance filtering")
                else:
                    valid_points = np.array([])
                    print("No valid robot positions found after distance filtering")
            else:
                # No occupied cells, all grid-valid points are valid
                valid_points = grid_valid_points
    
    # Calculate robot z position based on table height
    table_mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, table_object_name)
    if table_mesh_info and table_mesh_info["mesh"] is not None:
        table_mesh = table_mesh_info["mesh"]
        table_height = np.max(table_mesh.vertices[:, 2])
    else:
        # Fallback to object position + dimensions
        table_height = table_object.position.z + table_object.dimensions.height
    
    robot_z = max(table_height - robot_height_offset, 0)
    if len(valid_points) == 0:
        print("Warning: No valid robot positions found, using room center for all environments")
        robot_positions = np.array([[(room_min_x + room_max_x) / 2, (room_min_y + room_max_y) / 2]] * num_envs)
    else:
        # Find optimal point based on minimum distance to target object
        target_pos = np.array([target_object.position.x, target_object.position.y])
        
        # Calculate distances for each valid point to target object
        target_distances = np.linalg.norm(valid_points - target_pos, axis=1)
        
        # Sort indices by distance to target object (closest first)
        sorted_indices = np.argsort(target_distances)
        
        # Check robot occupancy collision for each candidate position using unified collision checking
        collision_free_positions = []
        collision_free_indices = []
        
        print(f"Checking robot occupancy collision for {len(valid_points)} candidate positions...")
        
        for idx in sorted_indices:
            candidate_point = valid_points[idx]
            
            # Calculate yaw orientation towards target object
            direction_to_target = target_pos - candidate_point
            yaw = np.arctan2(direction_to_target[1], direction_to_target[0])
            
            # Create quaternion from yaw (scalar-first format [w, x, y, z])
            candidate_quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
            candidate_pos_3d = np.array([candidate_point[0], candidate_point[1], 0])
            
            # Use unified collision checking
            collision = check_unified_robot_collision(candidate_pos_3d, candidate_quat, scene_occupancy_fn, room_bounds)
            
            if not collision:
                collision_free_positions.append(candidate_point)
                collision_free_indices.append(idx)
                
                # Stop if we have enough collision-free positions
                if len(collision_free_positions) >= num_envs:
                    break
        
        print(f"Found {len(collision_free_positions)} collision-free robot positions out of {len(valid_points)} candidates")
        
        if len(collision_free_positions) == 0:
            print("Warning: No collision-free robot positions found, using original closest points")
            # Fall back to original method without collision checking
            selected_indices = sorted_indices[:num_envs]
            robot_positions = valid_points[selected_indices]
        else:
            # Use collision-free positions
            if num_envs == 1:
                robot_positions = np.array([collision_free_positions[0]])
            else:
                # Take up to num_envs collision-free positions
                num_to_take = min(num_envs, len(collision_free_positions))
                robot_positions = np.array(collision_free_positions[:num_to_take])
                
                # If we need more positions, repeat the collision-free ones
                if len(robot_positions) < num_envs:
                    print(f"Warning: Only {len(robot_positions)} collision-free positions available, repeating closest ones")
                    while len(robot_positions) < num_envs:
                        additional_needed = num_envs - len(robot_positions)
                        repeat_count = min(additional_needed, len(collision_free_positions))
                        robot_positions = np.concatenate([
                            robot_positions, 
                            np.array(collision_free_positions[:repeat_count])
                        ])
    
    # Calculate orientations and camera positions for all environments
    target_pos_2d = np.array([target_object.position.x, target_object.position.y])
    
    robot_base_positions = []
    robot_base_quats = []
    camera_lookats = []
    
    for i in range(num_envs):
        robot_x, robot_y = robot_positions[i]
        
        # Calculate robot orientation towards target object
        direction_to_target = target_pos_2d - np.array([robot_x, robot_y])
        yaw = np.arctan2(direction_to_target[1], direction_to_target[0])
        
        # Convert to torch tensors on CUDA
        robot_base_pos = torch.tensor([robot_x, robot_y, robot_z], dtype=torch.float, device="cuda")
        robot_base_quat = torch.tensor(
            R.from_euler('z', yaw).as_quat(scalar_first=True), 
            dtype=torch.float, device="cuda"
        )
        
        camera_lookat = torch.tensor([
            target_object.position.x, 
            target_object.position.y, 
            table_height
        ], dtype=torch.float, device="cuda")
        
        robot_base_positions.append(robot_base_pos)
        robot_base_quats.append(robot_base_quat)
        camera_lookats.append(camera_lookat)
    
    # Stack into tensors
    robot_base_pos = torch.stack(robot_base_positions)
    robot_base_quat = torch.stack(robot_base_quats)
    camera_lookat = torch.stack(camera_lookats)

    # Create visualization of robot planning data
    if args_cli.visualize_figs:
        try:
            # Create save directory if it doesn't exist
            robot_planning_debug_dir = os.path.join(debug_dir, "robot_planning")
            os.makedirs(robot_planning_debug_dir, exist_ok=True)
            
            # Generate filename with timestamp and identifiers
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            viz_filename = f"robot_planning_{layout_name}_{room_id}_{target_object_name}_{timestamp}.png"
            viz_path = os.path.join(robot_planning_debug_dir, viz_filename)
            
            # Call visualization function
            visualize_robot_planning_data(
                room_bounds=(room_min_x, room_min_y, room_max_x, room_max_y),
                occupancy_grid=occupancy_grid,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_res=CollisionCheckingConfig.GRID_RES,
                robot_base_positions=robot_base_pos,
                robot_base_quats=robot_base_quat,
                camera_lookats=camera_lookat,
                target_object=target_object,
                table_object=table_object,
                valid_points=valid_points if len(valid_points) > 0 else None,
                layout_name=layout_name,
                room_id=room_id,
                target_object_name=target_object_name,
                table_object_name=table_object_name,
                save_path=viz_path
            )
        except Exception as e:
            print(f"Warning: Failed to create robot planning visualization: {e}")

    return robot_base_pos, robot_base_quat, camera_lookat

def create_table_occupancy_grid(scene_save_dir, layout_name, room_id, table_object_name):
    """
    Create occupancy grid specifically for a table object for place location sampling.
    
    Returns:
        tuple: (table_occupancy_grid, table_bounds, table_mesh, floor_plan, target_room)
    """
    layout_json_path = os.path.join(scene_save_dir, f"{layout_name}.json")
    with open(layout_json_path, "r") as f:
        layout_info = json.load(f)
    
    floor_plan: FloorPlan = dict_to_floor_plan(layout_info)
    target_room = next(room for room in floor_plan.rooms if room.id == room_id)
    assert target_room is not None, f"target_room {room_id} not found in floor_plan"
    
    # Get table object
    table_object = next(obj for obj in target_room.objects if obj.id == table_object_name)
    assert table_object is not None, f"table_object {table_object_name} not found in room"
    
    # Get table mesh
    try:
        table_mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, table_object_name)
        if table_mesh_info and table_mesh_info["mesh"] is not None:
            table_mesh = table_mesh_info["mesh"]
        else:
            print(f"Warning: Could not load mesh for table {table_object_name}")
            table_mesh = trimesh.Trimesh()
    except Exception as e:
        print(f"Warning: Could not load mesh for table {table_object_name}: {e}")
        table_mesh = trimesh.Trimesh()
    
    if len(table_mesh.vertices) == 0:
        # Fallback: create bounding box from object dimensions
        table_bounds = (
            table_object.position.x,
            table_object.position.y,
            table_object.position.x + table_object.dimensions.width,
            table_object.position.y + table_object.dimensions.length
        )
        table_occupancy_grid = np.ones((1, 1), dtype=bool)  # Single cell representing table
        print(f"Using fallback bounding box for table {table_object_name}")
    else:
        # Get table bounds from mesh
        table_vertices_2d = table_mesh.vertices[:, :2]  # Only x, y coordinates
        table_min_x = np.min(table_vertices_2d[:, 0])
        table_min_y = np.min(table_vertices_2d[:, 1])
        table_max_x = np.max(table_vertices_2d[:, 0])
        table_max_y = np.max(table_vertices_2d[:, 1])
        table_bounds = (table_min_x, table_min_y, table_max_x, table_max_y)
        
        # Create fine-grained occupancy grid for table surface
        table_grid_res = CollisionCheckingConfig.GRID_RES / 2  # Finer resolution for table
        table_grid_x = np.arange(table_min_x, table_max_x, table_grid_res)
        table_grid_y = np.arange(table_min_y, table_max_y, table_grid_res)
        
        # Create ray origins at grid centers, elevated above the table
        table_grid_x_mesh, table_grid_y_mesh = np.meshgrid(
            table_grid_x + table_grid_res/2, 
            table_grid_y + table_grid_res/2, 
            indexing='ij'
        )
        ray_origins = np.stack([
            table_grid_x_mesh.flatten(), 
            table_grid_y_mesh.flatten(), 
            np.full(table_grid_x_mesh.size, 10.0)
        ], axis=1).astype(np.float32)
        
        ray_directions = np.tile([0, 0, -1], (len(ray_origins), 1)).astype(np.float32)
        
        # Perform ray casting to detect table occupancy
        table_occupancy_grid = np.zeros((len(table_grid_x), len(table_grid_y)), dtype=bool)
        
        try:
            locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(table_mesh).intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions
            )
            
            # Mark occupied grid cells
            if len(index_ray) > 0:
                grid_i = index_ray // len(table_grid_y)
                grid_j = index_ray % len(table_grid_y)
                
                # Filter valid indices
                valid_mask = (grid_i < len(table_grid_x)) & (grid_j < len(table_grid_y))
                valid_i = grid_i[valid_mask]
                valid_j = grid_j[valid_mask]
                
                # Mark all valid cells as occupied
                table_occupancy_grid[valid_i, valid_j] = True
        except Exception as e:
            print(f"Warning: Ray casting failed for table: {e}")
            # Fallback: mark all grid cells as occupied
            table_occupancy_grid[:, :] = True
    
    return table_occupancy_grid, table_bounds, table_mesh, floor_plan, target_room


def sample_robot_place_location(
    scene_save_dir, layout_name, room_id,
    table_object_name, num_envs, debug_dir
):
    """
    Sample robot place locations for placing objects on a table.
    
    This function:
    1. Samples valid place locations on the table surface (with min_dist_to_boundary from table edges)
    2. For each place location, finds feasible robot poses within max_dist_to_object
    3. Validates robot poses for collision-free placement
    4. Returns robot base poses and place locations for multiple environments
    
    Args:
        scene_save_dir: Directory containing scene data
        layout_name: Name of the layout
        room_id: ID of the room
        table_object_name: Name of the table object to place on
        num_envs: Number of environments to sample for
    
    Returns:
        tuple: (robot_base_pos, robot_base_quat, place_locations)
            - robot_base_pos: torch tensor (num_envs, 3) - robot base positions
            - robot_base_quat: torch tensor (num_envs, 4) - robot base quaternions
            - place_locations: torch tensor (num_envs, 3) - place locations on table
    """
    
    # Use unified parameters for consistency
    robot_height_offset = 0.20
    num_sample_robot_points = 10000  # For robot position sampling
    
    print(f"Sampling robot place locations for table {table_object_name} in room {room_id}")
    
    # Create unified occupancy grid for scene collision checking
    occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, floor_plan, target_room = create_unified_occupancy_grid(
        scene_save_dir, layout_name, room_id
    )
    scene_occupancy_fn = create_unified_scene_occupancy_fn(occupancy_grid, grid_x, grid_y, room_bounds)
    
    # Create table-specific occupancy grid for place location sampling
    table_occupancy_grid, table_bounds, table_mesh, _, _ = create_table_occupancy_grid(
        scene_save_dir, layout_name, room_id, table_object_name
    )
    
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    table_min_x, table_min_y, table_max_x, table_max_y = table_bounds
    
    table_object = next(obj for obj in target_room.objects if obj.id == table_object_name)
    assert table_object is not None, f"table_object {table_object_name} not found in floor_plan"
    
    # Calculate table height for place locations
    if table_mesh and len(table_mesh.vertices) > 0:
        table_height = np.max(table_mesh.vertices[:, 2])
    else:
        # Fallback to object position + dimensions
        table_height = table_object.position.z + table_object.dimensions.height
    
    robot_z = max(table_height - robot_height_offset, 0)
    
    # Step 1: Sample valid place locations on table surface
    print("Step 1: Sampling valid place locations on table surface...")
    
    # Find valid cells on table that are far enough from boundaries
    valid_place_cells = []
    table_grid_res = CollisionCheckingConfig.GRID_RES / 2  # Finer resolution used in table occupancy
    
    if table_occupancy_grid.size == 1:
        # Fallback case: single cell table
        table_center_x = (table_min_x + table_max_x) / 2
        table_center_y = (table_min_y + table_max_y) / 2
        valid_place_cells = [(table_center_x, table_center_y)]
        print("Using table center as single place location (fallback)")
    else:
        # Check each table cell for valid placement
        for i in range(table_occupancy_grid.shape[0]):
            for j in range(table_occupancy_grid.shape[1]):
                if table_occupancy_grid[i, j]:  # Cell is part of table
                    # Convert grid coordinates to world coordinates
                    cell_x = table_min_x + i * table_grid_res + table_grid_res / 2
                    cell_y = table_min_y + j * table_grid_res + table_grid_res / 2
                    
                    # Check distance to table boundaries
                    dist_to_table_edges = min(
                        cell_x - table_min_x,           # distance to left edge
                        table_max_x - cell_x,           # distance to right edge
                        cell_y - table_min_y,           # distance to bottom edge
                        table_max_y - cell_y            # distance to top edge
                    )
                    
                    if dist_to_table_edges >= CollisionCheckingConfig.MIN_DIST_TO_BOUNDARY:
                        valid_place_cells.append((cell_x, cell_y))
    
    print(f"Found {len(valid_place_cells)} valid place locations on table surface")
    
    if len(valid_place_cells) == 0:
        print("Warning: No valid place locations found on table, using table center")
        table_center_x = (table_min_x + table_max_x) / 2
        table_center_y = (table_min_y + table_max_y) / 2
        valid_place_cells = [(table_center_x, table_center_y)]
    
    # Step 2: Sample robot positions in the room (collision-free)
    print("Step 2: Sampling collision-free robot positions...")
    
    # Sample random points in the room rectangle
    sample_points = np.random.uniform(
        low=[room_min_x, room_min_y], 
        high=[room_max_x, room_max_y], 
        size=(num_sample_robot_points, 2)
    )
    
    # Filter robot positions using same logic as sample_robot_location
    # Check distance to room edges
    dist_to_edges = np.minimum.reduce([
        sample_points[:, 0] - room_min_x,
        room_max_x - sample_points[:, 0],
        sample_points[:, 1] - room_min_y,
        room_max_y - sample_points[:, 1]
    ])
    
    edge_valid_mask = dist_to_edges >= CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE
    edge_valid_points = sample_points[edge_valid_mask]
    
    if len(edge_valid_points) == 0:
        print("Warning: No edge-valid robot positions, using room center")
        valid_robot_points = np.array([[(room_min_x + room_max_x) / 2, (room_min_y + room_max_y) / 2]])
    else:
        # Check scene occupancy and distance to objects
        grid_coords = np.floor((edge_valid_points - [room_min_x, room_min_y]) / CollisionCheckingConfig.GRID_RES).astype(int)
        
        grid_valid_mask = (
            (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < len(grid_x)) &
            (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < len(grid_y)) &
            (~occupancy_grid[grid_coords[:, 0], grid_coords[:, 1]])
        )
        
        grid_valid_points = edge_valid_points[grid_valid_mask]
        
        if len(grid_valid_points) == 0:
            print("Warning: No grid-valid robot positions, using room center")
            valid_robot_points = np.array([[(room_min_x + room_max_x) / 2, (room_min_y + room_max_y) / 2]])
        else:
            # Check distance to occupied cells (same as sample_robot_location)
            occupied_indices = np.where(occupancy_grid)
            if len(occupied_indices[0]) > 0:
                occupied_positions = np.column_stack([
                    room_min_x + occupied_indices[0] * CollisionCheckingConfig.GRID_RES + CollisionCheckingConfig.GRID_RES/2,
                    room_min_y + occupied_indices[1] * CollisionCheckingConfig.GRID_RES + CollisionCheckingConfig.GRID_RES/2
                ])
                
                # Batched distance checking
                batch_size = 1000
                valid_points_list = []
                
                for i in range(0, len(grid_valid_points), batch_size):
                    batch_end = min(i + batch_size, len(grid_valid_points))
                    batch_points = grid_valid_points[i:batch_end]
                    
                    distances = np.linalg.norm(
                        batch_points[:, np.newaxis, :] - occupied_positions[np.newaxis, :, :], 
                        axis=2
                    )
                    min_distances = np.min(distances, axis=1)
                    distance_valid_mask = min_distances >= CollisionCheckingConfig.ROBOT_MIN_DIST_TO_OBJECT
                    valid_points_list.append(batch_points[distance_valid_mask])
                
                if valid_points_list:
                    valid_robot_points = np.concatenate(valid_points_list, axis=0)
                else:
                    valid_robot_points = np.array([[(room_min_x + room_max_x) / 2, (room_min_y + room_max_y) / 2]])
            else:
                valid_robot_points = grid_valid_points
    
    print(f"Found {len(valid_robot_points)} collision-free robot positions")
    
    # Step 3: For each environment, find feasible (place_location, robot_pose) pairs
    print("Step 3: Finding feasible place location and robot pose pairs...")
    
    final_robot_positions = []
    final_robot_quats = []
    final_place_locations = []
    
    for env_i in range(num_envs):
        print(f"Processing environment {env_i + 1}/{num_envs}")
        
        # Try different place locations until we find a feasible one
        place_found = False
        
        # Shuffle place locations for variety
        shuffled_place_cells = valid_place_cells.copy()
        np.random.shuffle(shuffled_place_cells)
        
        for place_x, place_y in shuffled_place_cells:
            place_pos_2d = np.array([place_x, place_y])
            
            # Find robot positions within max_dist_to_object of this place location
            distances_to_place = np.linalg.norm(valid_robot_points - place_pos_2d, axis=1)
            feasible_robot_mask = distances_to_place <= CollisionCheckingConfig.MAX_DIST_TO_OBJECT
            feasible_robot_points = valid_robot_points[feasible_robot_mask]
            
            if len(feasible_robot_points) == 0:
                continue  # No feasible robot positions for this place location
            
            # Sort by distance to place location (closest first)
            feasible_distances = distances_to_place[feasible_robot_mask]
            sorted_indices = np.argsort(feasible_distances)
            
            # Try robot positions until we find one without collision
            for idx in sorted_indices:
                robot_pos_2d = feasible_robot_points[idx]
                
                # Calculate robot orientation towards place location
                direction_to_place = place_pos_2d - robot_pos_2d
                yaw = np.arctan2(direction_to_place[1], direction_to_place[0])
                
                # Create quaternion from yaw (scalar-first format [w, x, y, z])
                robot_quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
                robot_pos_3d = np.array([robot_pos_2d[0], robot_pos_2d[1], 0])
                
                # Check robot collision using unified collision checking
                collision = check_unified_robot_collision(robot_pos_3d, robot_quat, scene_occupancy_fn, room_bounds)
                
                if not collision:
                    # Found feasible pair!
                    final_robot_positions.append([robot_pos_2d[0], robot_pos_2d[1], robot_z])
                    final_robot_quats.append(R.from_euler('z', yaw).as_quat(scalar_first=True))
                    final_place_locations.append([place_x, place_y, table_height])
                    place_found = True
                    print(f"  Found feasible pair: place=({place_x:.2f}, {place_y:.2f}), robot=({robot_pos_2d[0]:.2f}, {robot_pos_2d[1]:.2f})")
                    break
            
            if place_found:
                break
        
        if not place_found:
            print(f"  Warning: No feasible pair found for environment {env_i}, using fallback")
            # Fallback: use closest valid robot position and table center
            table_center = np.array([(table_min_x + table_max_x) / 2, (table_min_y + table_max_y) / 2])
            
            if len(valid_robot_points) > 0:
                distances_to_table_center = np.linalg.norm(valid_robot_points - table_center, axis=1)
                closest_robot_idx = np.argmin(distances_to_table_center)
                robot_pos_2d = valid_robot_points[closest_robot_idx]
            else:
                robot_pos_2d = np.array([(room_min_x + room_max_x) / 2, (room_min_y + room_max_y) / 2])
            
            direction_to_place = table_center - robot_pos_2d
            yaw = np.arctan2(direction_to_place[1], direction_to_place[0])
            
            final_robot_positions.append([robot_pos_2d[0], robot_pos_2d[1], robot_z])
            final_robot_quats.append(R.from_euler('z', yaw).as_quat(scalar_first=True))
            final_place_locations.append([table_center[0], table_center[1], table_height])
    
    # Convert to torch tensors on CUDA
    robot_base_pos = torch.tensor(final_robot_positions, dtype=torch.float, device="cuda")
    robot_base_quat = torch.tensor(final_robot_quats, dtype=torch.float, device="cuda")
    place_locations = torch.tensor(final_place_locations, dtype=torch.float, device="cuda")
    
    print(f"Successfully generated {len(final_robot_positions)} robot place locations")
    print(f"Sample robot position: {robot_base_pos[0]}")
    print(f"Sample place location: {place_locations[0]}")
    
    # Create visualization of robot place location planning data
    if args_cli.visualize_figs:
        try:
            # Create save directory if it doesn't exist
            robot_place_planning_debug_dir = os.path.join(debug_dir, "robot_place_planning")
            os.makedirs(robot_place_planning_debug_dir, exist_ok=True)
            
            # Generate filename with timestamp and identifiers
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            viz_filename = f"robot_place_planning_{layout_name}_{room_id}_{table_object_name}_{timestamp}.png"
            viz_path = os.path.join(robot_place_planning_debug_dir, viz_filename)
            
            # Call visualization function
            visualize_robot_place_planning_data(
                room_bounds=(room_min_x, room_min_y, room_max_x, room_max_y),
                occupancy_grid=occupancy_grid,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_res=CollisionCheckingConfig.GRID_RES,
                table_bounds=table_bounds,
                table_occupancy_grid=table_occupancy_grid,
                robot_base_positions=robot_base_pos,
                robot_base_quats=robot_base_quat,
                place_locations=place_locations,
                table_object=table_object,
                valid_place_cells=valid_place_cells if len(valid_place_cells) > 0 else None,
                valid_robot_points=valid_robot_points if len(valid_robot_points) > 0 else None,
                layout_name=layout_name,
                room_id=room_id,
                table_object_name=table_object_name,
                save_path=viz_path
            )
        except Exception as e:
            print(f"Warning: Failed to create robot place location planning visualization: {e}")
    
    return robot_base_pos, robot_base_quat, place_locations


def visualize_robot_spawn_planning_data(
    room_bounds, occupancy_grid, grid_x, grid_y, grid_res,
    spawn_positions, spawn_angles, valid_points=None,
    layout_name="", room_id="",
    save_path=None
):
    """
    Visualize robot spawn location planning data including room occupancy, spawn positions, orientations, and robot occupancy.
    
    Args:
        room_bounds: tuple of (min_x, min_y, max_x, max_y) for the room
        occupancy_grid: 2D numpy array of boolean occupancy for scene objects
        grid_x, grid_y: arrays defining room grid coordinates
        grid_res: room grid resolution
        spawn_positions: array of spawn positions (N, 2) - [x, y] world coordinates
        spawn_angles: array of spawn yaw angles (N,) - yaw angles in radians
        valid_points: optional array of valid spawn positions for debugging
        layout_name, room_id: strings for labeling
        save_path: path to save the PNG file
    """
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Import required matplotlib components
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from scipy.spatial.transform import Rotation as R
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. Draw room boundaries
    room_rect = patches.Rectangle(
        (room_min_x, room_min_y), 
        room_max_x - room_min_x, 
        room_max_y - room_min_y,
        linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.2,
        label='Room Boundary'
    )
    ax.add_patch(room_rect)
    
    # 2. Visualize room occupancy grid
    if occupancy_grid is not None and occupancy_grid.size > 0:
        extent = [room_min_x, room_max_x, room_min_y, room_max_y]
        
        # Create custom colormap: white for free, red for occupied
        colors = ['white', 'red']
        cmap = ListedColormap(colors)
        
        # Display room occupancy grid
        im = ax.imshow(
            occupancy_grid.T,  # Transpose for correct orientation
            extent=extent,
            origin='lower',
            cmap=cmap,
            alpha=0.4,
            aspect='equal'
        )
    
    # 3. Draw valid spawn positions (if provided for debugging)
    if valid_points is not None and len(valid_points) > 0:
        ax.scatter(
            valid_points[:, 0], valid_points[:, 1],
            c='lightblue', s=8, alpha=0.4, marker='o',
            label=f'Valid Spawn Positions ({len(valid_points)})'
        )
    
    # 4. Convert spawn data to numpy if needed
    if hasattr(spawn_positions, 'cpu'):
        spawn_positions_np = spawn_positions.cpu().numpy()
        spawn_angles_np = spawn_angles.cpu().numpy()
    else:
        spawn_positions_np = np.array(spawn_positions)
        spawn_angles_np = np.array(spawn_angles)
    
    # Ensure spawn_angles is 1D
    if spawn_angles_np.ndim > 1:
        spawn_angles_np = spawn_angles_np.flatten()
    
    # 5. Create robot occupancy overlay for spawn positions
    robot_occupancy_combined = np.zeros_like(occupancy_grid, dtype=bool)
    
    for i, (pos, yaw) in enumerate(zip(spawn_positions_np, spawn_angles_np)):
        # Use support point (spawn position) and yaw to get forward/side
        support_pos = [pos[0], pos[1]]
        forward, side = get_forward_side_from_support_point_and_yaw(support_pos, yaw)
        
        # Create robot occupancy function
        robot_occupancy_fn = occupancy_map(forward, side, yaw, offset=0.05)
        
        # Create points for robot occupancy grid
        robot_grid_x_mesh, robot_grid_y_mesh = np.meshgrid(grid_x + grid_res/2, grid_y + grid_res/2, indexing='ij')
        robot_grid_points = np.stack([
            robot_grid_x_mesh.flatten(), 
            robot_grid_y_mesh.flatten()
        ], axis=1)
        
        # Get robot occupancy for grid points
        robot_occupied = robot_occupancy_fn(robot_grid_points)
        robot_occupied_grid = robot_occupied.reshape(robot_grid_x_mesh.shape)
        
        # Add to combined robot occupancy
        robot_occupancy_combined |= robot_occupied_grid
    
    # 6. Visualize robot occupancy overlay
    if robot_occupancy_combined.size > 0:
        extent = [room_min_x, room_max_x, room_min_y, room_max_y]
        
        # Create custom colormap for robot occupancy: transparent for free, blue for occupied
        robot_colors = [(0, 0, 0, 0), (0, 0, 1, 0.3)]  # transparent, semi-transparent blue
        robot_cmap = ListedColormap(robot_colors)
        
        # Display robot occupancy grid
        robot_im = ax.imshow(
            robot_occupancy_combined.T,  # Transpose for correct orientation
            extent=extent,
            origin='lower',
            cmap=robot_cmap,
            alpha=0.7,
            aspect='equal'
        )
    
    # 7. Draw spawn positions and orientations
    for i, (pos, yaw) in enumerate(zip(spawn_positions_np, spawn_angles_np)):
        x, y = pos[0], pos[1]
        
        # Color scheme for different environments
        color = plt.cm.viridis(i / max(1, len(spawn_positions_np) - 1))
        
        # Robot spawn position
        ax.scatter(x, y, c=[color], s=120, marker='o', 
                  edgecolors='black', linewidth=2,
                  label=f'Spawn {i}' if i < 5 else '')  # Only label first 5 to avoid clutter
        
        # Robot orientation arrow
        arrow_length = 0.3
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        
        ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.05,
                fc=color, ec='black', linewidth=1, alpha=0.8)
        
        # Add spawn index annotation
        ax.annotate(f'{i}', (x, y), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.9, color='white',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
    
    # 8. Add legend entries
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]
    
    # Add custom legend entries
    legend_elements.extend([
        Patch(facecolor='red', alpha=0.4, label='Scene Occupancy'),
        Patch(facecolor='blue', alpha=0.3, label='Robot Occupancy')
    ])
    legend_labels.extend(['Scene Occupancy', 'Robot Occupancy'])
    
    # 9. Add legend and labels
    ax.legend(handles=legend_elements, labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Robot Spawn Location Planning Visualization\n'
                f'Layout: {layout_name}, Room: {room_id}', 
                fontsize=14, pad=20)
    
    # 10. Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 11. Add text annotations
    room_area = (room_max_x - room_min_x) * (room_max_y - room_min_y)
    
    info_text = (f"Grid Resolution: {grid_res:.3f}m\n"
                f"Room Size: {room_max_x-room_min_x:.2f}×{room_max_y-room_min_y:.2f}m\n"
                f"Num Spawn Positions: {len(spawn_positions_np)}\n"
                f"Min Dist to Room Edge: {CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE:.2f}m\n"
                f"Min Dist to Objects: {CollisionCheckingConfig.ROBOT_MIN_DIST_TO_OBJECT:.2f}m")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
           verticalalignment='top', fontsize=10)
    
    # 12. Tight layout and save
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Robot spawn location planning visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def sample_robot_spawn(
    scene_save_dir, layout_id, room_id, num_envs, debug_dir
):
    """
    sample collision-free robot spawn positions

    Uses unified collision checking configuration for consistency with trajectory planning.

    no need to consider the target object

    Args:
        scene_save_dir: Directory containing scene data
        layout_id: ID of the layout
        room_id: ID of the room
        num_envs: Number of environments to sample spawn positions for
        debug_dir: Directory to save debug visualizations

    Returns: 
        spawn_pos: torch tensor of shape (num_envs, 2) - joint values [side, forward]
        spawn_angles: torch tensor of shape (num_envs, 1) - joint yaw angles
    """
    # Use unified parameters for consistency
    num_sample_points = 5000  # Reduced since we don't need to optimize for specific target
    
    # Create unified occupancy grid and scene occupancy function
    occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, floor_plan, target_room = create_unified_occupancy_grid(
        scene_save_dir, layout_id, room_id
    )
    scene_occupancy_fn = create_unified_scene_occupancy_fn(occupancy_grid, grid_x, grid_y, room_bounds)
    
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Sample random points in the room rectangle
    sample_points = np.random.uniform(
        low=[room_min_x, room_min_y], 
        high=[room_max_x, room_max_y], 
        size=(num_sample_points, 2)
    )
    
    # Filter points based on constraints using vectorized operations
    # Check distance to room edges for all points at once
    dist_to_edges = np.minimum.reduce([
        sample_points[:, 0] - room_min_x,  # distance to left edge
        room_max_x - sample_points[:, 0],  # distance to right edge
        sample_points[:, 1] - room_min_y,  # distance to bottom edge
        room_max_y - sample_points[:, 1]   # distance to top edge
    ])
    
    # Filter out points too close to room edges
    edge_valid_mask = dist_to_edges >= CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE
    edge_valid_points = sample_points[edge_valid_mask]
    
    if len(edge_valid_points) == 0:
        valid_points = np.array([])
    else:
        # Convert to grid coordinates
        grid_coords = np.floor((edge_valid_points - [room_min_x, room_min_y]) / CollisionCheckingConfig.GRID_RES).astype(int)
        
        # Check if points are within grid bounds and not in occupied cells
        grid_valid_mask = (
            (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < len(grid_x)) &
            (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < len(grid_y)) &
            (~occupancy_grid[grid_coords[:, 0], grid_coords[:, 1]])
        )
        
        grid_valid_points = edge_valid_points[grid_valid_mask]
        grid_valid_coords = grid_coords[grid_valid_mask]
        
        if len(grid_valid_points) == 0:
            valid_points = np.array([])
        else:
            # Check distance to occupied cells using vectorized operations
            search_radius = int(np.ceil(CollisionCheckingConfig.ROBOT_MIN_DIST_TO_OBJECT / CollisionCheckingConfig.GRID_RES))
            
            # Find all occupied cell positions
            occupied_indices = np.where(occupancy_grid)
            if len(occupied_indices[0]) > 0:
                occupied_positions = np.column_stack([
                    room_min_x + occupied_indices[0] * CollisionCheckingConfig.GRID_RES + CollisionCheckingConfig.GRID_RES/2,
                    room_min_y + occupied_indices[1] * CollisionCheckingConfig.GRID_RES + CollisionCheckingConfig.GRID_RES/2
                ])
                
                # Use batched processing to avoid memory explosion
                batch_size = 1000  # Process points in smaller batches
                valid_points_list = []
                
                print(f"Processing {len(grid_valid_points)} valid spawn points in batches of {batch_size}")
                print(f"Checking distance to {len(occupied_positions)} occupied cells")
                
                for i in range(0, len(grid_valid_points), batch_size):
                    batch_end = min(i + batch_size, len(grid_valid_points))
                    batch_points = grid_valid_points[i:batch_end]
                    
                    if i % (batch_size * 10) == 0:  # Print progress every 10 batches
                        print(f"Processing batch {i//batch_size + 1}/{(len(grid_valid_points)-1)//batch_size + 1}")
                    
                    # Calculate distances from batch to all occupied cells
                    distances = np.linalg.norm(
                        batch_points[:, np.newaxis, :] - occupied_positions[np.newaxis, :, :], 
                        axis=2
                    )
                    
                    # Find minimum distance to any occupied cell for each point in batch
                    min_distances = np.min(distances, axis=1)
                    
                    # Filter points that are far enough from occupied cells
                    distance_valid_mask = min_distances >= CollisionCheckingConfig.ROBOT_MIN_DIST_TO_OBJECT
                    valid_points_list.append(batch_points[distance_valid_mask])
                
                # Combine all valid points from batches
                if valid_points_list:
                    valid_points = np.concatenate(valid_points_list, axis=0)
                    print(f"Found {len(valid_points)} valid robot spawn positions after distance filtering")
                else:
                    valid_points = np.array([])
                    print("No valid robot spawn positions found after distance filtering")
            else:
                # No occupied cells, all grid-valid points are valid
                valid_points = grid_valid_points
    
    if len(valid_points) == 0:
        print("Warning: No valid robot spawn positions found, using room center for all environments")
        room_center = [(room_min_x + room_max_x) / 2, (room_min_y + room_max_y) / 2]
        spawn_positions = np.array([room_center] * num_envs)
        spawn_angles = np.zeros((num_envs, 1))  # All facing same direction
    else:
        # Check robot occupancy collision for spawn positions using unified collision checking
        collision_free_positions = []
        collision_free_angles = []
        
        print(f"Checking robot occupancy collision for {len(valid_points)} candidate spawn positions...")
        
        # Randomly shuffle valid points for variety in spawn selection
        shuffled_indices = np.random.permutation(len(valid_points))
        
        for idx in shuffled_indices:
            candidate_point = valid_points[idx]
            
            # Generate random orientation for spawn (no need to face specific target)
            yaw = np.random.uniform(-np.pi, np.pi)
            
            # Create quaternion from yaw (scalar-first format [w, x, y, z])
            candidate_quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
            candidate_pos_3d = np.array([candidate_point[0], candidate_point[1], 0])
            
            # Use unified collision checking with stricter offset for spawn
            collision = check_unified_robot_collision(
                candidate_pos_3d, candidate_quat, scene_occupancy_fn, room_bounds,
                robot_occupancy_offset=CollisionCheckingConfig.ROBOT_SPAWN_OCCUPANCY_OFFSET
            )
            
            if not collision:
                collision_free_positions.append(candidate_point)
                collision_free_angles.append(yaw)
                
                # Stop if we have enough collision-free positions
                if len(collision_free_positions) >= num_envs:
                    break
        
        print(f"Found {len(collision_free_positions)} collision-free robot spawn positions out of {len(valid_points)} candidates")
        
        if len(collision_free_positions) == 0:
            print("Warning: No collision-free robot spawn positions found, using room center")
            room_center = [(room_min_x + room_max_x) / 2, (room_min_y + room_max_y) / 2]
            spawn_positions = np.array([room_center] * num_envs)
            spawn_angles = np.zeros((num_envs, 1))
        else:
            # Use collision-free positions
            if num_envs <= len(collision_free_positions):
                # We have enough positions
                selected_positions = collision_free_positions[:num_envs]
                selected_angles = collision_free_angles[:num_envs]
            else:
                # We need more positions, repeat the available ones
                print(f"Warning: Only {len(collision_free_positions)} collision-free positions available, repeating for {num_envs} environments")
                selected_positions = []
                selected_angles = []
                for i in range(num_envs):
                    idx = i % len(collision_free_positions)
                    selected_positions.append(collision_free_positions[idx])
                    selected_angles.append(collision_free_angles[idx])
            
            spawn_positions = np.array(selected_positions)
            spawn_angles = np.array(selected_angles).reshape(-1, 1)
    
    # Convert support point positions to joint values (forward/side)
    joint_forward_values = []
    joint_side_values = []
    
    for i in range(len(spawn_positions)):
        support_point_pos = spawn_positions[i]  # [x, y] position of base_support
        yaw_angle = spawn_angles[i, 0] if len(spawn_angles.shape) > 1 else spawn_angles[i]
        
        # Convert support point position to joint values
        forward, side = get_forward_side_from_support_point_and_yaw(support_point_pos, yaw_angle)
        joint_forward_values.append(forward)
        joint_side_values.append(side)
    
    # Create joint position array: [side, forward] (note the order!)
    joint_positions = np.column_stack([joint_side_values, joint_forward_values])
    
    # Convert to torch tensors on CUDA
    spawn_pos = torch.tensor(joint_positions, dtype=torch.float, device="cuda")  # shape: (num_envs, 2) [side, forward]
    spawn_angles_tensor = torch.tensor(spawn_angles, dtype=torch.float, device="cuda")  # shape: (num_envs, 1) [yaw]
    
    print(f"Generated joint positions (side/forward): {spawn_pos.shape} and yaw angles: {spawn_angles_tensor.shape}")
    print(f"Sample joint values - side: {spawn_pos[0, 0]:.3f}, forward: {spawn_pos[0, 1]:.3f}, yaw: {spawn_angles_tensor[0, 0]:.3f}")
    
    # Create visualization of robot spawn location planning data
    if args_cli.visualize_figs:
        try:
            # Create save directory if it doesn't exist
            robot_spawn_planning_debug_dir = os.path.join(debug_dir, "robot_spawn_planning")
            os.makedirs(robot_spawn_planning_debug_dir, exist_ok=True)
            
            # Generate filename with timestamp and identifiers
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            viz_filename = f"robot_spawn_planning_{layout_id}_{room_id}_{timestamp}.png"
            viz_path = os.path.join(robot_spawn_planning_debug_dir, viz_filename)
            
            # Call visualization function
            visualize_robot_spawn_planning_data(
                room_bounds=(room_min_x, room_min_y, room_max_x, room_max_y),
                occupancy_grid=occupancy_grid,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_res=CollisionCheckingConfig.GRID_RES,
                spawn_positions=spawn_positions,  # Use numpy array of world positions
                spawn_angles=spawn_angles.flatten(),  # Use numpy array of yaw angles
                valid_points=valid_points if len(valid_points) > 0 else None,
                layout_name=layout_id,
                room_id=room_id,
                save_path=viz_path
            )
        except Exception as e:
            print(f"Warning: Failed to create robot spawn location planning visualization: {e}")
    
    return spawn_pos, spawn_angles_tensor

def visualize_loaded_spawn_poses(
    eval_spawn_pos, eval_spawn_angles, scene_save_dir, layout_id, room_id, debug_dir,
    layout_name="", save_path=None
):
    """
    Visualize all the spawn poses loaded from JSON file.
    
    Args:
        eval_spawn_pos: torch tensor of shape (N, 2) - joint values [side, forward] loaded from JSON
        eval_spawn_angles: torch tensor of shape (N, 1) - joint yaw angles loaded from JSON
        scene_save_dir: Directory containing scene data
        layout_id: ID of the layout
        room_id: ID of the room
        debug_dir: Directory to save debug visualizations
        layout_name: Layout name for labeling
        save_path: Optional path to save the visualization
    """
    print(f"Visualizing {len(eval_spawn_pos)} loaded spawn poses...")
    
    # Create unified occupancy grid and scene occupancy function for room visualization
    occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, floor_plan, target_room = create_unified_occupancy_grid(
        scene_save_dir, layout_id, room_id
    )
    
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Convert joint positions back to world positions for visualization
    spawn_positions_world = []
    spawn_angles_np = eval_spawn_angles.cpu().numpy().flatten()
    
    for i in range(len(eval_spawn_pos)):
        # Extract joint values [side, forward] and yaw angle
        side = float(eval_spawn_pos[i, 0].cpu())
        forward = float(eval_spawn_pos[i, 1].cpu())  
        yaw = float(eval_spawn_angles[i, 0].cpu())
        
        # Convert joint values back to support point world position
        # This is the inverse of get_forward_side_from_support_point_and_yaw
        from isaaclab.omron_franka_occupancy import support_point
        support_pos = support_point(forward, side, yaw)
        spawn_positions_world.append(support_pos)
    
    spawn_positions_world = np.array(spawn_positions_world)
    
    # Generate save path if not provided
    if save_path is None:
        # Create save directory if it doesn't exist
        robot_spawn_eval_debug_dir = os.path.join(debug_dir, "robot_spawn_eval")
        os.makedirs(robot_spawn_eval_debug_dir, exist_ok=True)
        
        # Generate filename with timestamp and identifiers
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        viz_filename = f"loaded_spawn_poses_{layout_id}_{room_id}_{len(eval_spawn_pos)}poses_{timestamp}.png"
        save_path = os.path.join(robot_spawn_eval_debug_dir, viz_filename)
    
    # Call the existing visualization function
    visualize_robot_spawn_planning_data(
        room_bounds=(room_min_x, room_min_y, room_max_x, room_max_y),
        occupancy_grid=occupancy_grid,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_res=CollisionCheckingConfig.GRID_RES,
        spawn_positions=spawn_positions_world,  # World positions converted from joint values
        spawn_angles=spawn_angles_np,  # Yaw angles
        valid_points=None,  # No need to show valid points for loaded poses
        layout_name=layout_name if layout_name else layout_id,
        room_id=room_id,
        save_path=save_path
    )
    
    print(f"Saved loaded spawn poses visualization to: {save_path}")
    return save_path

def visualize_multi_env_trajectory_planning(
    room_bounds, occupancy_grid, grid_x, grid_y, grid_res,
    trajectories_data, layout_name="", room_id="",
    save_path=None
):
    """
    Visualize trajectory planning results for multiple environments in a single image.
    
    Args:
        room_bounds: tuple of (min_x, min_y, max_x, max_y)
        occupancy_grid: 2D numpy array of boolean occupancy for scene objects
        grid_x, grid_y: arrays defining grid coordinates
        grid_res: grid resolution
        trajectories_data: list of dicts, each containing:
            - 'trajectory': list/array of trajectory waypoints [(x, y, z, qw, qx, qy, qz), ...]
            - 'start_pos': start position [x, y, z]
            - 'end_pos': end position [x, y, z]
            - 'tree_nodes': optional list of RRT tree nodes
            - 'env_id': environment identifier
            - 'trajectory_type': string like "pick" or "place"
        layout_name, room_id: strings for labeling
        save_path: path to save the PNG file
    """
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Import required matplotlib components
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from scipy.spatial.transform import Rotation as R
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 1. Draw room boundaries
    room_rect = patches.Rectangle(
        (room_min_x, room_min_y), 
        room_max_x - room_min_x, 
        room_max_y - room_min_y,
        linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.3,
        label='Room Boundary'
    )
    ax.add_patch(room_rect)
    
    # 2. Visualize occupancy grid
    if occupancy_grid is not None and occupancy_grid.size > 0:
        # Create grid coordinates for visualization
        extent = [room_min_x, room_max_x, room_min_y, room_max_y]
        
        # Create custom colormap: white for free, red for occupied
        colors = ['white', 'red']
        cmap = ListedColormap(colors)
        
        # Display occupancy grid
        im = ax.imshow(
            occupancy_grid.T,  # Transpose for correct orientation
            extent=extent,
            origin='lower',
            cmap=cmap,
            alpha=0.6,
            aspect='equal'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Occupancy (White=Free, Red=Occupied)', rotation=270, labelpad=20)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Free', 'Occupied'])
    
    # 3. Create combined robot occupancy overlay for all trajectories
    robot_occupancy_combined = np.zeros_like(occupancy_grid, dtype=bool)
    
    # Define colors for different environments and trajectory types
    env_colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Up to 10 different colors
    trajectory_type_styles = {
        'pick': {'linestyle': '-', 'alpha': 0.8, 'linewidth': 3},
        'place': {'linestyle': '--', 'alpha': 0.7, 'linewidth': 2.5}
    }
    
    total_trajectory_length = 0
    total_waypoints = 0
    
    # Process each trajectory
    for traj_idx, traj_data in enumerate(trajectories_data):
        trajectory = traj_data['trajectory']
        start_pos = traj_data['start_pos']
        end_pos = traj_data['end_pos']
        tree_nodes = traj_data.get('tree_nodes', None)
        env_id = traj_data.get('env_id', traj_idx)
        traj_type = traj_data.get('trajectory_type', 'unknown')
        
        if len(trajectory) == 0:
            continue
            
        # Color for this environment
        color = env_colors[env_id % len(env_colors)]
        style = trajectory_type_styles.get(traj_type, trajectory_type_styles['pick'])
        
        # 4. Create robot occupancy overlay for this trajectory
        traj_array = np.array(trajectory)
        traj_positions = traj_array[:, :2]  # Extract x, y positions  
        traj_quats = traj_array[:, 3:7]     # Extract quaternions
        
        # Sample every few waypoints to avoid overcrowding the visualization
        sample_interval = max(1, len(trajectory) // 15)  # Sample ~15 waypoints max per trajectory
        
        for i in range(0, len(traj_positions), sample_interval):
            pos = traj_positions[i]
            quat = traj_quats[i]
            
            # Convert quaternion to yaw angle
            # quaternion is [w, x, y, z] format (scalar-first)
            rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert to [x, y, z, w] for scipy
            yaw = rotation.as_euler('xyz')[2]  # Get Z-axis rotation (yaw)
            
            # Use support point (robot position) and yaw to get forward/side
            support_pos = [pos[0], pos[1]]
            forward, side = get_forward_side_from_support_point_and_yaw(support_pos, yaw)
            
            # Create robot occupancy function
            robot_occupancy_fn = occupancy_map(forward, side, yaw, offset=0.05)
            
            # Create points for robot occupancy grid
            robot_grid_x_mesh, robot_grid_y_mesh = np.meshgrid(grid_x + grid_res/2, grid_y + grid_res/2, indexing='ij')
            robot_grid_points = np.stack([
                robot_grid_x_mesh.flatten(), 
                robot_grid_y_mesh.flatten()
            ], axis=1)
            
            # Get robot occupancy for grid points
            robot_occupied = robot_occupancy_fn(robot_grid_points)
            robot_occupied_grid = robot_occupied.reshape(robot_grid_x_mesh.shape)
            
            # Add to combined robot occupancy
            robot_occupancy_combined |= robot_occupied_grid
        
        # 5. Draw RRT tree (if provided) - make it lighter for multiple trajectories
        if tree_nodes is not None and len(tree_nodes) > 1:
            # Draw tree edges (lighter for multiple trajectories)
            for i, (pos, quat, parent_idx) in enumerate(tree_nodes):
                if parent_idx >= 0:  # Skip root node
                    parent_pos = tree_nodes[parent_idx][0]
                    ax.plot([parent_pos[0], pos[0]], [parent_pos[1], pos[1]], 
                           color=color, alpha=0.1, linewidth=0.3)
            
            # Draw tree nodes (smaller for multiple trajectories)
            tree_positions = np.array([node[0][:2] for node in tree_nodes])
            ax.scatter(tree_positions[:, 0], tree_positions[:, 1], 
                      c=[color], s=1, alpha=0.3)
        
        # 6. Draw start and end positions
        ax.scatter(*start_pos[:2], c=[color], s=120, marker='o', 
                  edgecolors='darkgreen', linewidth=2, alpha=0.8,
                  label=f'Env {env_id} Start ({traj_type})' if traj_idx < 5 else '')
        ax.scatter(*end_pos[:2], c=[color], s=120, marker='*', 
                  edgecolors='darkred', linewidth=2, alpha=0.8,
                  label=f'Env {env_id} Goal ({traj_type})' if traj_idx < 5 else '')
        
        # 7. Draw planned trajectory
        # Draw trajectory path
        trajectory_label = f'Env {env_id} {traj_type.title()} Traj ({len(trajectory)} steps)' if traj_idx < 5 else ''
        ax.plot(traj_positions[:, 0], traj_positions[:, 1], 
               color=color, label=trajectory_label, **style)
        
        # Draw waypoints with orientation arrows (less frequent for multiple trajectories)
        for i, (pos, quat) in enumerate(zip(traj_positions, traj_quats)):
            if i % max(1, len(trajectory) // 8) == 0:  # Show every 8th waypoint or so
                # Convert quaternion to yaw for arrow display
                rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert to [x, y, z, w] for scipy
                yaw = rotation.as_euler('xyz')[2]  # Get Z-axis rotation (yaw)
                
                # Draw waypoint
                ax.scatter(pos[0], pos[1], c=[color], s=15, marker='o', 
                          alpha=0.6, zorder=4)
                
                # Draw orientation arrow (smaller for multiple trajectories)
                arrow_length = 0.1
                dx = arrow_length * np.cos(yaw)
                dy = arrow_length * np.sin(yaw)
                
                ax.arrow(pos[0], pos[1], dx, dy, head_width=0.02, head_length=0.02,
                        fc=color, ec=color, linewidth=0.8, alpha=0.7, zorder=4)
        
        # Update statistics
        if len(trajectory) > 1:
            traj_array = np.array(trajectory)
            for i in range(1, len(traj_array)):
                total_trajectory_length += np.linalg.norm(traj_array[i][:2] - traj_array[i-1][:2])
        total_waypoints += len(trajectory)
    
    # 8. Visualize combined robot occupancy overlay
    if robot_occupancy_combined.size > 0:
        extent = [room_min_x, room_max_x, room_min_y, room_max_y]
        
        # Create custom colormap for robot occupancy: transparent for free, blue for occupied
        robot_colors = [(0, 0, 0, 0), (0, 0, 1, 0.2)]  # More transparent for multiple trajectories
        robot_cmap = ListedColormap(robot_colors)
        
        # Display robot occupancy grid
        robot_im = ax.imshow(
            robot_occupancy_combined.T,  # Transpose for correct orientation
            extent=extent,
            origin='lower',
            cmap=robot_cmap,
            alpha=0.6,
            aspect='equal'
        )
    
    # 9. Add legend entries for robot occupancy
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]
    
    # Add robot occupancy to legend
    legend_elements.append(Patch(facecolor='blue', alpha=0.2, label='Robot Occupancy'))
    legend_labels.append('Robot Occupancy')
    
    # 10. Add legend and labels
    ax.legend(handles=legend_elements, labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Multi-Environment Robot Trajectory Planning\n'
                f'Layout: {layout_name}, Room: {room_id}\n'
                f'Total Trajectories: {len(trajectories_data)}', 
                fontsize=14, pad=20)
    
    # 11. Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 12. Add text annotations
    info_text = (f"Grid Resolution: {grid_res:.3f}m\n"
                f"Room Size: {room_max_x-room_min_x:.2f}×{room_max_y-room_min_y:.2f}m\n"
                f"Total Trajectories: {len(trajectories_data)}\n"
                f"Total Trajectory Length: {total_trajectory_length:.2f}m\n"
                f"Total Waypoints: {total_waypoints}")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
           verticalalignment='top', fontsize=10)
    
    # 13. Tight layout and save
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Multi-environment trajectory planning visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_trajectory_planning(
    room_bounds, occupancy_grid, grid_x, grid_y, grid_res,
    trajectory, start_pos, end_pos, tree_nodes=None,
    layout_name="", room_id="",
    save_path=None
):
    """
    Visualize trajectory planning results including occupancy grid, robot occupancy, RRT tree, and final trajectory.
    
    Args:
        room_bounds: tuple of (min_x, min_y, max_x, max_y)
        occupancy_grid: 2D numpy array of boolean occupancy for scene objects
        grid_x, grid_y: arrays defining grid coordinates
        grid_res: grid resolution
        trajectory: list/array of trajectory waypoints [(x, y, z, qw, qx, qy, qz), ...]
        start_pos: start position [x, y, z]
        end_pos: end position [x, y, z]
        tree_nodes: optional list of RRT tree nodes for visualization
        layout_name, room_id: strings for labeling
        save_path: path to save the PNG file
    """
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Import required matplotlib components
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from scipy.spatial.transform import Rotation as R
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. Draw room boundaries
    room_rect = patches.Rectangle(
        (room_min_x, room_min_y), 
        room_max_x - room_min_x, 
        room_max_y - room_min_y,
        linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.3,
        label='Room Boundary'
    )
    ax.add_patch(room_rect)
    
    # 2. Visualize occupancy grid
    if occupancy_grid is not None and occupancy_grid.size > 0:
        # Create grid coordinates for visualization
        extent = [room_min_x, room_max_x, room_min_y, room_max_y]
        
        # Create custom colormap: white for free, red for occupied
        colors = ['white', 'red']
        cmap = ListedColormap(colors)
        
        # Display occupancy grid
        im = ax.imshow(
            occupancy_grid.T,  # Transpose for correct orientation
            extent=extent,
            origin='lower',
            cmap=cmap,
            alpha=0.6,
            aspect='equal'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Occupancy (White=Free, Red=Occupied)', rotation=270, labelpad=20)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Free', 'Occupied'])
    
    # 3. Create robot occupancy overlay for trajectory waypoints
    if len(trajectory) > 0:
        robot_occupancy_combined = np.zeros_like(occupancy_grid, dtype=bool)
        
        # Convert trajectory to numpy array
        traj_array = np.array(trajectory)
        traj_positions = traj_array[:, :2]  # Extract x, y positions  
        traj_quats = traj_array[:, 3:7]     # Extract quaternions
        
        # Sample every few waypoints to avoid overcrowding the visualization
        sample_interval = max(1, len(trajectory) // 20)  # Sample ~20 waypoints max
        
        for i in range(0, len(traj_positions), sample_interval):
            pos = traj_positions[i]
            quat = traj_quats[i]
            
            # Convert quaternion to yaw angle
            # quaternion is [w, x, y, z] format (scalar-first)
            rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert to [x, y, z, w] for scipy
            yaw = rotation.as_euler('xyz')[2]  # Get Z-axis rotation (yaw)
            
            # Use support point (robot position) and yaw to get forward/side
            support_pos = [pos[0], pos[1]]
            forward, side = get_forward_side_from_support_point_and_yaw(support_pos, yaw)
            
            # Create robot occupancy function
            robot_occupancy_fn = occupancy_map(forward, side, yaw, offset=0.05)
            
            # Create points for robot occupancy grid
            robot_grid_x_mesh, robot_grid_y_mesh = np.meshgrid(grid_x + grid_res/2, grid_y + grid_res/2, indexing='ij')
            robot_grid_points = np.stack([
                robot_grid_x_mesh.flatten(), 
                robot_grid_y_mesh.flatten()
            ], axis=1)
            
            # Get robot occupancy for grid points
            robot_occupied = robot_occupancy_fn(robot_grid_points)
            robot_occupied_grid = robot_occupied.reshape(robot_grid_x_mesh.shape)
            
            # Add to combined robot occupancy
            robot_occupancy_combined |= robot_occupied_grid
        
        # 4. Visualize robot occupancy overlay
        if robot_occupancy_combined.size > 0:
            extent = [room_min_x, room_max_x, room_min_y, room_max_y]
            
            # Create custom colormap for robot occupancy: transparent for free, blue for occupied
            robot_colors = [(0, 0, 0, 0), (0, 0, 1, 0.3)]  # transparent, semi-transparent blue
            robot_cmap = ListedColormap(robot_colors)
            
            # Display robot occupancy grid
            robot_im = ax.imshow(
                robot_occupancy_combined.T,  # Transpose for correct orientation
                extent=extent,
                origin='lower',
                cmap=robot_cmap,
                alpha=0.7,
                aspect='equal'
            )
    
    # 5. Draw RRT tree (if provided)
    if tree_nodes is not None and len(tree_nodes) > 1:
        # Draw tree edges
        for i, (pos, quat, parent_idx) in enumerate(tree_nodes):
            if parent_idx >= 0:  # Skip root node
                parent_pos = tree_nodes[parent_idx][0]
                ax.plot([parent_pos[0], pos[0]], [parent_pos[1], pos[1]], 
                       'c-', alpha=0.3, linewidth=0.5)
        
        # Draw tree nodes
        tree_positions = np.array([node[0][:2] for node in tree_nodes])
        ax.scatter(tree_positions[:, 0], tree_positions[:, 1], 
                  c='cyan', s=2, alpha=0.5, label=f'RRT Tree ({len(tree_nodes)} nodes)')
    
    # 6. Draw start and end positions
    ax.scatter(*start_pos[:2], c='green', s=200, marker='o', 
              edgecolors='darkgreen', linewidth=2, 
              label='Start Position', zorder=5)
    ax.scatter(*end_pos[:2], c='red', s=200, marker='*', 
              edgecolors='darkred', linewidth=2,
              label='Goal Position', zorder=5)
    
    # 7. Draw planned trajectory
    if len(trajectory) > 0:
        # Convert trajectory to numpy array
        traj_array = np.array(trajectory)
        traj_positions = traj_array[:, :2]  # Extract x, y positions
        traj_quats = traj_array[:, 3:7]     # Extract quaternions
        
        # Draw trajectory path
        ax.plot(traj_positions[:, 0], traj_positions[:, 1], 
               'b-', linewidth=3, alpha=0.8, label=f'Planned Trajectory ({len(trajectory)} steps)')
        
        # Draw waypoints with orientation arrows
        for i, (pos, quat) in enumerate(zip(traj_positions, traj_quats)):
            if i % max(1, len(trajectory) // 10) == 0:  # Show every 10th waypoint or so
                # Convert quaternion to yaw for arrow display
                rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert to [x, y, z, w] for scipy
                yaw = rotation.as_euler('xyz')[2]  # Get Z-axis rotation (yaw)
                
                # Draw waypoint
                ax.scatter(pos[0], pos[1], c='blue', s=30, marker='o', 
                          alpha=0.7, zorder=4)
                
                # Draw orientation arrow
                arrow_length = 0.15
                dx = arrow_length * np.cos(yaw)
                dy = arrow_length * np.sin(yaw)
                
                ax.arrow(pos[0], pos[1], dx, dy, head_width=0.03, head_length=0.03,
                        fc='blue', ec='darkblue', linewidth=1, alpha=0.8, zorder=4)
    
    # 8. Add legend entries for robot occupancy
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]
    
    # Add robot occupancy to legend
    legend_elements.append(Patch(facecolor='blue', alpha=0.3, label='Robot Occupancy'))
    legend_labels.append('Robot Occupancy')
    
    # 9. Add legend and labels
    ax.legend(handles=legend_elements, labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Robot Trajectory Planning Visualization\n'
                f'Layout: {layout_name}, Room: {room_id}', 
                fontsize=14, pad=20)
    
    # 10. Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 11. Add text annotations
    trajectory_length = 0.0
    if len(trajectory) > 1:
        traj_array = np.array(trajectory)
        for i in range(1, len(traj_array)):
            trajectory_length += np.linalg.norm(traj_array[i][:2] - traj_array[i-1][:2])
    
    info_text = (f"Grid Resolution: {grid_res:.3f}m\n"
                f"Room Size: {room_max_x-room_min_x:.2f}×{room_max_y-room_min_y:.2f}m\n"
                f"Trajectory Length: {trajectory_length:.2f}m\n"
                f"Waypoints: {len(trajectory)}")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
           verticalalignment='top', fontsize=10)
    
    # 12. Tight layout and save
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Trajectory planning visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plan_robot_traj(start_pos, start_quat, end_pos, end_quat,
    scene_save_dir, layout_name, room_id, debug_dir,
    max_move_distance_per_step = 0.02,
    max_rotate_degree_per_step = 2.0,
    min_move_distance_per_step = 0.01,
    min_rotate_degree_per_step = 1.0,
    max_length=30,
    return_visualization_data=False
):

    """
    plan the robot traj from start_pos, start_quat to end_pos, end_quat
    pos and quat are support_point_pos and support_point_quat

    plan the traj with RRT
    you need to ensure no collisions between the robot and the environment

    Uses unified collision checking configuration for consistency with sampling functions.

    you need to conside max_move_distance_per_step and max_rotate_degree_per_step

    return a torch tensor of shape (num_steps, 7)
    """
    
    # Convert inputs to numpy for easier manipulation
    start_pos_np = start_pos.cpu().numpy() if hasattr(start_pos, 'cpu') else np.array(start_pos)
    start_quat_np = start_quat.cpu().numpy() if hasattr(start_quat, 'cpu') else np.array(start_quat)
    end_pos_np = end_pos.cpu().numpy() if hasattr(end_pos, 'cpu') else np.array(end_pos)
    end_quat_np = end_quat.cpu().numpy() if hasattr(end_quat, 'cpu') else np.array(end_quat)
    
    # RRT parameters - optimized for speed but still accurate
    max_iterations = 10000
    goal_threshold_pos = 0.05  # Slightly more tolerant
    goal_threshold_rot = np.deg2rad(4.0)  # More tolerant
    step_size_pos = max_move_distance_per_step  # Larger steps
    step_size_rot = np.deg2rad(max_rotate_degree_per_step)  # Larger rotation steps

    min_step_size_pos = min_move_distance_per_step  # Larger steps
    min_step_size_rot = np.deg2rad(min_rotate_degree_per_step)  # Larger rotation steps
    
    # Create unified occupancy grid and scene occupancy function - ensures consistency with sampling
    occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, floor_plan, target_room = create_unified_occupancy_grid(
        scene_save_dir, layout_name, room_id
    )
    scene_occupancy_fn = create_unified_scene_occupancy_fn(occupancy_grid, grid_x, grid_y, room_bounds)
    
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Define unified collision checking function for trajectory planning
    def check_collision_unified(pos, quat):
        """Use the unified collision checking for consistency with sampling functions."""
        return check_unified_robot_collision(pos, quat, scene_occupancy_fn, room_bounds)
    
    # Fast distance calculation (avoiding scipy rotations in inner loop)
    def distance_pos_rot_fast(pos1, quat1, pos2, quat2):
        """Fast distance calculation with approximations."""
        pos_dist = np.linalg.norm(pos1 - pos2)
        
        # Fast quaternion distance approximation (much faster than full rotation)
        # Use dot product of quaternions as distance metric
        quat_dot = np.abs(np.dot(quat1, quat2))
        rot_dist = 2.0 * np.arccos(np.clip(quat_dot, 0, 1))
        
        return pos_dist, rot_dist
    
    # Fast SLERP implementation with fewer checks
    def slerp_fast(q1, q2, t):
        """Fast SLERP implementation."""
        dot = np.dot(q1, q2)
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        if dot > 0.995:  # Linear interpolation threshold
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
    def steer_fast(from_pos, from_quat, to_pos, to_quat):
        """Optimized steering function."""
        pos_dist, rot_dist = distance_pos_rot_fast(from_pos, from_quat, to_pos, to_quat)
        
        # Position steering
        if pos_dist > 0:
            step_pos = min(step_size_pos, pos_dist)
            new_pos = from_pos + (to_pos - from_pos) * (step_pos / pos_dist)
        else:
            new_pos = from_pos.copy()
        
        # Rotation steering (optimized)
        if rot_dist > 0:
            t = min(step_size_rot / rot_dist, 1.0)
            new_quat = slerp_fast(from_quat, to_quat, t)
        else:
            new_quat = from_quat.copy()
        
        return new_pos, new_quat
    
    def is_goal_reached_fast(pos, quat, goal_pos, goal_quat):
        """Fast goal checking."""
        pos_dist, rot_dist = distance_pos_rot_fast(pos, quat, goal_pos, goal_quat)
        return pos_dist < goal_threshold_pos and rot_dist < goal_threshold_rot
    
    # RRT Algorithm with optimizations
    print(f"Planning robot trajectory from {start_pos_np[:2]} to {end_pos_np[:2]} (OPTIMIZED)")
    
    # Tree structure: [pos, quat, parent_index] - use lists for faster appending
    tree_positions = [start_pos_np]
    tree_quaternions = [start_quat_np] 
    tree_parents = [-1]
    
    # Accurate collision checks for start/goal (critical)
    if check_collision_unified(start_pos_np, start_quat_np):
        print("Warning: Start pose is in collision!")
        fallback_trajectory = torch.tensor([[*start_pos_np, *start_quat_np]], dtype=torch.float, device="cuda")
        if return_visualization_data:
            return fallback_trajectory, None
        else:
            return fallback_trajectory
    
    if check_collision_unified(end_pos_np, end_quat_np):
        print("Warning: Goal pose is in collision!")
        fallback_trajectory = torch.tensor([[*start_pos_np, *start_quat_np]], dtype=torch.float, device="cuda")
        if return_visualization_data:
            return fallback_trajectory, None
        else:
            return fallback_trajectory
    
    if is_goal_reached_fast(start_pos_np, start_quat_np, end_pos_np, end_quat_np):
        print("Already at goal!")
        fallback_trajectory = torch.tensor([[*start_pos_np, *start_quat_np]], dtype=torch.float, device="cuda")
        if return_visualization_data:
            return fallback_trajectory, None
        else:
            return fallback_trajectory
    
    goal_found = False
    goal_node_idx = -1
    
    # Pre-allocate arrays for random sampling (avoid repeated allocation)
    room_bounds_sampling = np.array([
        room_min_x + CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE,
        room_min_y + CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE,
        room_max_x - CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE,
        room_max_y - CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE
    ])
    
    for iteration in range(max_iterations):
        if iteration % 200 == 0:  # Less frequent progress updates
            print(f"RRT iteration {iteration}/{max_iterations}, tree size: {len(tree_positions)}")
        
        # Optimized random sampling
        if random.random() < 0.15:  # Increased goal bias
            rand_pos = end_pos_np.copy()
            rand_quat = end_quat_np.copy()
        else:
            # Vectorized random sampling
            rand_pos = np.array([
                np.random.uniform(room_bounds_sampling[0], room_bounds_sampling[2]),
                np.random.uniform(room_bounds_sampling[1], room_bounds_sampling[3]),
                start_pos_np[2]
            ])
            rand_yaw = np.random.uniform(-np.pi, np.pi)
            rand_quat = np.array([np.cos(rand_yaw/2), 0, 0, np.sin(rand_yaw/2)])  # Fast quaternion creation
        
        # Find nearest node (this could be further optimized with KDTree for very large trees)
        min_dist = float('inf')
        nearest_idx = 0
        
        # Vectorized distance calculation for small trees
        if len(tree_positions) < 100:
            for i, (node_pos, node_quat) in enumerate(zip(tree_positions, tree_quaternions)):
                pos_dist, rot_dist = distance_pos_rot_fast(node_pos, node_quat, rand_pos, rand_quat)
                total_dist = pos_dist + 0.5 * rot_dist  # Weight position more
                if total_dist < min_dist:
                    min_dist = total_dist
                    nearest_idx = i
        else:
            # For larger trees, use position-only distance for initial filtering
            positions_array = np.array(tree_positions)
            pos_distances = np.linalg.norm(positions_array[:, :2] - rand_pos[:2], axis=1)
            nearest_candidates = np.argsort(pos_distances)[:5]  # Check top 5 candidates
            
            for i in nearest_candidates:
                pos_dist, rot_dist = distance_pos_rot_fast(tree_positions[i], tree_quaternions[i], rand_pos, rand_quat)
                total_dist = pos_dist + 0.5 * rot_dist
                if total_dist < min_dist:
                    min_dist = total_dist
                    nearest_idx = i
        
        nearest_pos = tree_positions[nearest_idx]
        nearest_quat = tree_quaternions[nearest_idx]
        
        # Steer towards random pose
        new_pos, new_quat = steer_fast(nearest_pos, nearest_quat, rand_pos, rand_quat)
        
        # Use unified collision checking for consistency
        pos_progress = np.linalg.norm(new_pos[:2] - nearest_pos[:2])
        if pos_progress > step_size_pos * 0.05:  # Check more frequently for accuracy
            # Use unified collision checking
            collision = check_collision_unified(new_pos, new_quat)
            
            if not collision:
                # Add new node to tree
                tree_positions.append(new_pos)
                tree_quaternions.append(new_quat)
                tree_parents.append(nearest_idx)
                new_node_idx = len(tree_positions) - 1
                
                # Check if we reached the goal
                if is_goal_reached_fast(new_pos, new_quat, end_pos_np, end_quat_np):
                    # Final verification with unified collision checking
                    if not check_collision_unified(new_pos, new_quat):
                        # Add exact target as final node if not already exact
                        pos_diff = np.linalg.norm(new_pos - end_pos_np)
                        quat_diff = np.abs(np.dot(new_quat, end_quat_np))
                        
                        if pos_diff > 1e-6 or quat_diff < 0.9999:
                            # Verify exact target is collision-free before adding
                            if not check_collision_unified(end_pos_np, end_quat_np):
                                # Add exact target node
                                tree_positions.append(end_pos_np)
                                tree_quaternions.append(end_quat_np)
                                tree_parents.append(new_node_idx)
                                goal_node_idx = len(tree_positions) - 1
                                print(f"Added exact target node (diff: pos={pos_diff:.6f}, quat={quat_diff:.6f})")
                            else:
                                print("Warning: Exact target is in collision, using closest valid pose")
                                goal_node_idx = new_node_idx
                        else:
                            goal_node_idx = new_node_idx
                        
                        goal_found = True
                        print(f"Goal reached at iteration {iteration}!")
                        break
    
    if not goal_found:
        print(f"Failed to find path to goal after {max_iterations} iterations")
        # Fast fallback trajectory
        distance = np.linalg.norm(end_pos_np - start_pos_np)
        num_steps = max(1, int(distance / step_size_pos))
        trajectory = []
        
        for i in range(num_steps + 1):
            alpha = i / num_steps
            interp_pos = start_pos_np + alpha * (end_pos_np - start_pos_np)
            interp_quat = slerp_fast(start_quat_np, end_quat_np, alpha)
            trajectory.append([*interp_pos, *interp_quat])
        
        # Ensure the last waypoint is exactly the target (fix floating point errors)
        trajectory[-1] = [*end_pos_np, *end_quat_np]
    else:
        # Reconstruct path
        path_indices = []
        current_idx = goal_node_idx
        while current_idx != -1:
            path_indices.append(current_idx)
            current_idx = tree_parents[current_idx]
        
        path_indices.reverse()
        
        trajectory = []
        for idx in path_indices:
            pos = tree_positions[idx]
            quat = tree_quaternions[idx]
            trajectory.append([*pos, *quat])
        
        # CRITICAL: Ensure trajectory ends exactly at the target
        last_waypoint = trajectory[-1]
        last_pos = np.array(last_waypoint[:3])
        last_quat = np.array(last_waypoint[3:7])
        
        # Check if last waypoint is exactly at target
        pos_diff = np.linalg.norm(last_pos - end_pos_np)
        quat_diff = np.abs(np.dot(last_quat, end_quat_np))
        
        if pos_diff > 1e-6 or quat_diff < 0.9999:  # Not exactly at target
            print(f"Adding exact target waypoint (pos_diff: {pos_diff:.6f}, quat_diff: {quat_diff:.6f})")
            trajectory.append([*end_pos_np, *end_quat_np])
        
        print(f"Found path with {len(trajectory)} waypoints (ending exactly at target)")
        
        # Validate final trajectory for collision-free path using unified collision checking
        print("Validating trajectory for collisions...")
        collision_found = False
        for i, waypoint in enumerate(trajectory):
            pos = waypoint[:3]
            quat = waypoint[3:7]
            if check_collision_unified(pos, quat):
                print(f"Warning: Collision detected at waypoint {i}")
                collision_found = True
                break
        
        if collision_found:
            print("Trajectory has collisions, using fallback straight-line path")
            # Fall back to straight line trajectory
            distance = np.linalg.norm(end_pos_np - start_pos_np)
            num_steps = max(1, int(distance / (step_size_pos / 3)))  # Smaller steps for safety
            trajectory = []
            
            for i in range(num_steps + 1):
                alpha = i / num_steps
                interp_pos = start_pos_np + alpha * (end_pos_np - start_pos_np)
                interp_quat = slerp_fast(start_quat_np, end_quat_np, alpha)
                trajectory.append([*interp_pos, *interp_quat])
            
            # Ensure the last waypoint is exactly the target
            trajectory[-1] = [*end_pos_np, *end_quat_np]
        else:
            print("Trajectory validation passed - path is collision-free")
    
    # Prepare visualization data if requested
    visualization_data = None
    if return_visualization_data:
        # Convert tree data for visualization
        tree_for_viz = [(tree_positions[i], tree_quaternions[i], tree_parents[i]) 
                        for i in range(len(tree_positions))] if goal_found else None
        
        visualization_data = {
            'room_bounds': (room_min_x, room_min_y, room_max_x, room_max_y),
            'occupancy_grid': occupancy_grid,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_res': CollisionCheckingConfig.GRID_RES,
            'trajectory': trajectory,
            'start_pos': start_pos_np,
            'end_pos': end_pos_np,
            'tree_nodes': tree_for_viz,
            'layout_name': layout_name,
            'room_id': room_id
        }
    
    # Create individual visualization (simplified for speed) - only if not returning visualization data
    if args_cli.visualize_figs and not return_visualization_data:
        try:
            trajectory_planning_debug_dir = os.path.join(debug_dir, "trajectory_planning")
            os.makedirs(trajectory_planning_debug_dir, exist_ok=True)
            
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            viz_filename = f"trajectory_planning_{layout_name}_{room_id}_{timestamp}.png"
            viz_path = os.path.join(trajectory_planning_debug_dir, viz_filename)
            
            # Convert tree data for visualization
            tree_for_viz = [(tree_positions[i], tree_quaternions[i], tree_parents[i]) 
                            for i in range(len(tree_positions))] if goal_found else None
            
            visualize_trajectory_planning(
                room_bounds=(room_min_x, room_min_y, room_max_x, room_max_y),
                occupancy_grid=occupancy_grid,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_res=CollisionCheckingConfig.GRID_RES,
                trajectory=trajectory,
                start_pos=start_pos_np,
                end_pos=end_pos_np,
                tree_nodes=tree_for_viz,
                layout_name=layout_name,
                room_id=room_id,
                save_path=viz_path
            )
        except Exception as e:
            print(f"Warning: Failed to create trajectory planning visualization: {e}")

    # min step pos and rot check
    # delete the waypoints that are too close to the previous waypoint
    # do it efficiently (not O(N^2))
    # so that all adjacent waypoints are at least min_step_size_pos and min_step_size_rot apart
    
    def filter_trajectory_min_steps(trajectory, min_step_pos, min_step_rot):
        """
        Filter trajectory to ensure minimum step sizes between adjacent waypoints.
        Preserves start and end poses.
        
        Args:
            trajectory: List of waypoints [x, y, z, qw, qx, qy, qz]
            min_step_pos: Minimum position distance between waypoints
            min_step_rot: Minimum rotation distance between waypoints (radians)
        
        Returns:
            List of filtered waypoints
        """
        if len(trajectory) <= 2:
            return trajectory  # Keep start and end if only 2 waypoints
        
        filtered_trajectory = [trajectory[0]]  # Always keep start pose
        
        for i in range(1, len(trajectory) - 1):  # Skip first and last waypoints
            current_waypoint = trajectory[i]
            last_kept_waypoint = filtered_trajectory[-1]
            
            # Calculate position distance
            pos_current = np.array(current_waypoint[:3])
            pos_last = np.array(last_kept_waypoint[:3])
            pos_dist = np.linalg.norm(pos_current - pos_last)
            
            # Calculate rotation distance using quaternion dot product
            quat_current = np.array(current_waypoint[3:7])
            quat_last = np.array(last_kept_waypoint[3:7])
            
            # Fast quaternion distance approximation
            quat_dot = np.abs(np.dot(quat_current, quat_last))
            quat_dot = np.clip(quat_dot, 0, 1)  # Ensure valid range for arccos
            rot_dist = 2.0 * np.arccos(quat_dot)
            
            # Keep waypoint if it meets minimum distance requirements
            if pos_dist >= min_step_pos or rot_dist >= min_step_rot:
                filtered_trajectory.append(current_waypoint)
        
        # Always keep end pose
        if len(trajectory) > 1:
            filtered_trajectory.append(trajectory[-1])
        
        return filtered_trajectory
    
    # Apply min step filtering to trajectory
    if len(trajectory) > 2:
        original_length = len(trajectory)
        trajectory = filter_trajectory_min_steps(
            trajectory, 
            min_step_size_pos, 
            min_step_size_rot
        )
        filtered_length = len(trajectory)
        
        if filtered_length < original_length:
            print(f"Trajectory filtered: {original_length} -> {filtered_length} waypoints "
                  f"(removed {original_length - filtered_length} waypoints too close to previous)")
    
    # Convert to torch tensor
    trajectory_tensor = torch.tensor(trajectory, dtype=torch.float, device="cuda")

    if len(trajectory_tensor) > max_length:
        original_length = len(trajectory_tensor)
        selected_indices = torch.from_numpy(np.linspace(0, len(trajectory_tensor) - 1, max_length).astype(np.int32)).to(trajectory_tensor.device)
        trajectory_tensor = trajectory_tensor[selected_indices]
        print(f"Trajectory truncated to {max_length} steps from {original_length} steps")
    
    if return_visualization_data:
        return trajectory_tensor, visualization_data
    else:
        return trajectory_tensor

def get_grasp_transforms(layout_json_path, target_object_name, base_pos):
    
    layout = get_layout_from_scene_json_path(layout_json_path)
    mesh_dict_list = export_layout_to_mesh_dict_list_tree_search_with_object_id(layout, target_object_name)
    print(f"mesh_dict_list: {mesh_dict_list.keys()}")
    target_mesh = mesh_dict_list[target_object_name]["mesh"]
    meta_data, vis_data = generate_m2t2_data(mesh_dict_list, target_object_name, base_pos)
    model, cfg = load_m2t2()
    total_trials = 0
    max_dist_to_mesh = 0.02
    while True:
        grasp_transforms, contacts = infer_m2t2(meta_data, vis_data, model, cfg, return_contacts=True)
        signed_dists = igl.signed_distance(contacts, target_mesh.vertices, target_mesh.faces)[0]
        dists = np.abs(signed_dists)
        z_dir = grasp_transforms[:, 2, 2]
        grasp_transforms = grasp_transforms[np.argsort(signed_dists + z_dir)]
        if grasp_transforms.shape[0] > 0:
            break
        total_trials += 1
        print(f"Total trials: {total_trials}")
        if total_trials > 10:
            break
    grasp_transforms_rotation = grasp_transforms[:, :3, :3]
    grasp_transforms_translation = grasp_transforms[:, :3, 3].reshape(-1, 3, 1)

    # create a rotation matrix which rotates along the z axis by 90 degrees
    # rotate_90 = R.from_euler('z', 90, degrees=True).as_matrix()
    # grasp_transforms_rotation = grasp_transforms_rotation @ rotate_90

    grasp_transforms_updated = np.concatenate([grasp_transforms_rotation, grasp_transforms_translation], axis=2)
    
    grasp_transforms[:, :3, :] = grasp_transforms_updated

    return grasp_transforms


def sample_grasp(layout_json_path, target_object_name, robot_base_pos, num_envs, device):
    """
    Sample grasp poses for the target object.
    
    Args:
        layout_json_path: Path to the layout JSON file
        target_object_name: Name of the target object to grasp
        robot_base_pos: Robot base position (first environment)
        num_envs: Number of environments to sample grasps for
        device: Device to create tensors on
    
    Returns:
        ee_goals: Tensor of shape (num_envs, 7) containing end-effector goals [pos, quat]
    """
    current_pose_cnt = 0
    all_ee_goals = []
    total_trials = 0

    while True:
        grasp_transforms = get_grasp_transforms(layout_json_path, target_object_name, robot_base_pos.tolist())
        
        ee_goals_quat_w = []
        ee_goals_translate_w = []
        for grasp_transforms_i in grasp_transforms:
            try:
                ee_goals_quat_w.append(R.from_matrix(grasp_transforms_i[:3, :3]).as_quat(scalar_first=True))
                ee_goals_translate_w.append(grasp_transforms_i[:3, 3].reshape(1, 3))
            except Exception as e:
                pass
        ee_goals_quat_w = np.array(ee_goals_quat_w).reshape(-1, 4)
        ee_goals_translate_w = np.concatenate(ee_goals_translate_w, axis=0).reshape(-1, 3)

        ee_goals_translate_w = torch.tensor(ee_goals_translate_w, device=device).float()
        ee_goals_quat_w = torch.tensor(ee_goals_quat_w, device=device).float()

        ee_goals = torch.cat([ee_goals_translate_w, ee_goals_quat_w], dim=1)
        current_pose_cnt += ee_goals.shape[0]
        all_ee_goals.append(ee_goals)
        print(f"out Total trials: {total_trials}")
        if current_pose_cnt >= num_envs or total_trials > 3:
            break
        total_trials += 1
    
    ee_goals = torch.cat(all_ee_goals, dim=0)
    ee_goals = ee_goals[:num_envs]
    
    return ee_goals


def curobo_plan_traj(motion_planner, robot_qpos, current_ee_pos, target_ee_pos, target_ee_quat,
    max_attempts=100,
    max_interpolation_step_distance = 0.01,  # Maximum distance per interpolation step
    interpolate=False,
    max_length=100,
):
    """
    robot_qpos: (1, num_joints)
    current_ee_pos: (1, 3)
    target_ee_pos: (1, 3)
    target_ee_quat: (1, 4)

    return:
        traj: (num_steps, 3+4),
        success: bool
    """
    if not interpolate:
        ee_pose, _ = motion_planner.plan_motion(
            robot_qpos,
            target_ee_pos,
            target_ee_quat,
            max_attempts=max_attempts
        )

    if not interpolate and ee_pose is not None:
        curobo_target_positions = ee_pose.ee_position
        curobo_target_quaternion = ee_pose.ee_quaternion

        curobo_target_ee_pose = torch.cat([
            curobo_target_positions, curobo_target_quaternion,
        ], dim=1).float()

        if curobo_target_ee_pose.shape[0] > max_length:
            selected_indices = torch.from_numpy(np.linspace(0, curobo_target_ee_pose.shape[0] - 1, max_length).astype(np.int32)).to(curobo_target_ee_pose.device)
            curobo_target_ee_pose = curobo_target_ee_pose[selected_indices]
            print(f"Trajectory truncated to {max_length} steps from {curobo_target_ee_pose.shape[0]} steps")

        curobo_target_ee_pose = torch.cat([
            curobo_target_ee_pose,
            torch.cat([target_ee_pos, target_ee_quat], dim=1).float()
        ], dim=0).float()

        return curobo_target_ee_pose, True
    
    else:
        # linear interpolation

        # Calculate total distance and number of steps needed
        total_distance = torch.norm(target_ee_pos - current_ee_pos).item()
        num_steps = max(1, int(torch.ceil(torch.tensor(total_distance / max_interpolation_step_distance)).item()))
        
        # Create interpolated trajectory
        interpolated_positions = []
        for i in range(num_steps + 1):
            alpha = i / num_steps
            interp_pos = current_ee_pos + alpha * (target_ee_pos - current_ee_pos)
            interpolated_positions.append(interp_pos.reshape(-1))
        
        # Stack positions and add orientation
        interpolated_positions = torch.stack(interpolated_positions)
        orientations = target_ee_quat.reshape(1, 4).repeat(len(interpolated_positions), 1)
        
        curobo_target_ee_pose = torch.cat([
            interpolated_positions,
            orientations,
        ], dim=1).float()

        if curobo_target_ee_pose.shape[0] > max_length:
            selected_indices = torch.from_numpy(np.linspace(0, curobo_target_ee_pose.shape[0] - 1, max_length).astype(np.int32)).to(curobo_target_ee_pose.device)
            curobo_target_ee_pose = curobo_target_ee_pose[selected_indices]
            print(f"Trajectory truncated to {max_length} steps from {curobo_target_ee_pose.shape[0]} steps")

        print(f"Trajectory Generated: interpolation | len {len(curobo_target_ee_pose)}")

        return curobo_target_ee_pose, False


def get_place_location(layout_json_path, target_object_name):

    layout = get_layout_from_scene_json_path(layout_json_path)
    mesh_dict = export_layout_to_mesh_dict_object_id(layout, target_object_name)
    mesh = mesh_dict[target_object_name]["mesh"]

    # get the center top of the mesh
    mesh_center = mesh.vertices.mean(axis=0)

    # get the top of the mesh
    mesh_top = mesh.vertices[:, 2].max()

    place_location = np.array([mesh_center[0], mesh_center[1], mesh_top])
    return place_location


def get_grasp_object_height(layout_json_path, target_object_name):

    layout = get_layout_from_scene_json_path(layout_json_path)
    mesh_dict = export_layout_to_mesh_dict_object_id(layout, target_object_name)
    mesh = mesh_dict[target_object_name]["mesh"]

    return float(mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min())

def is_reach(ee_goal, ee_frame_data, loc_threshold=0.03, rot_threshold=15.0, ignore_z=False):

    ee_goal_pos = ee_goal[:3]
    ee_goal_quat = ee_goal[3:]

    ee_frame_data_pos = ee_frame_data[:3]
    ee_frame_data_quat = ee_frame_data[3:]

    delta_pos = ee_goal_pos - ee_frame_data_pos 
    if ignore_z:
        delta_pos[2] = 0.0

    target_quat_np = ee_goal_quat.cpu().numpy()
    current_quat_np = ee_frame_data_quat.cpu().numpy()
    
    # Convert to scipy Rotation objects (scalar-first format)
    target_rot = R.from_quat(target_quat_np[[1, 2, 3, 0]])  # Convert to (x, y, z, w)
    current_rot = R.from_quat(current_quat_np[[1, 2, 3, 0]])  # Convert to (x, y, z, w)
    
    # Calculate relative rotation: target = current * delta_rot
    # So delta_rot = current.inv() * target
    delta_rot = target_rot * current_rot.inv()
    
    # Convert to Euler angles (roll, pitch, yaw) in radians
    delta_euler = delta_rot.as_rotvec(degrees=True)
    delta_euler = torch.tensor(delta_euler, dtype=torch.float, device=ee_goal.device)

    delta_pos_max = torch.max(torch.abs(delta_pos)) if not ignore_z else torch.max(torch.abs(delta_pos[:2]))
    delta_euler_norm = torch.norm(delta_euler)

    loc_success = delta_pos_max < loc_threshold
    rot_success = delta_euler_norm < rot_threshold

    # print(f"loc_success: {delta_pos_max}/{loc_threshold}, rot_success: {delta_euler_norm}/{rot_threshold}")

    return loc_success and rot_success

def detect_success_placement(
    target_object_state,
    pick_object_id,
    place_table_id,
    scene_save_dir,
    layout_name, 
    room_id
):
    """
    Detect if an object has been successfully placed by checking if its position
    is within the valid placement region on the placement table.
    
    Args:
        target_object_state: torch tensor (1, 7) - object state [pos, quat] relative to env root
        pick_object_id: ID of the object being placed
        scene_save_dir: Directory containing scene data
        layout_name: Name of the layout
        room_id: ID of the room
    
    Returns:
        bool: True if placement is successful, False otherwise
    """
    # Extract object position (only need x, y coordinates)
    object_pos = target_object_state[0, :3]  # [x, y, z]
    object_x, object_y = float(object_pos[0]), float(object_pos[1])
    
    try:
        # Create table-specific occupancy grid for the placement table
        table_occupancy_grid, table_bounds, table_mesh, floor_plan, target_room = create_table_occupancy_grid(
            scene_save_dir, layout_name, room_id, place_table_id
        )
        
        table_min_x, table_min_y, table_max_x, table_max_y = table_bounds
        
        # Find valid placement cells on table that are far enough from boundaries
        valid_place_cells = []
        table_grid_res = CollisionCheckingConfig.GRID_RES / 2  # Finer resolution used in table occupancy
        
        if table_occupancy_grid.size == 1:
            # Fallback case: single cell table - check if object is within table bounds
            table_center_x = (table_min_x + table_max_x) / 2
            table_center_y = (table_min_y + table_max_y) / 2
            
            # Check if object is reasonably close to table center
            distance_to_center = ((object_x - table_center_x)**2 + (object_y - table_center_y)**2)**0.5
            max_distance = min((table_max_x - table_min_x) / 2, (table_max_y - table_min_y) / 2)
            return distance_to_center <= max_distance
        else:
            # Check each table cell for valid placement
            for i in range(table_occupancy_grid.shape[0]):
                for j in range(table_occupancy_grid.shape[1]):
                    if table_occupancy_grid[i, j]:  # Cell is part of table
                        # Convert grid coordinates to world coordinates
                        cell_x = table_min_x + i * table_grid_res + table_grid_res / 2
                        cell_y = table_min_y + j * table_grid_res + table_grid_res / 2
                        
                        # Check distance to table boundaries
                        dist_to_table_edges = min(
                            cell_x - table_min_x,           # distance to left edge
                            table_max_x - cell_x,           # distance to right edge
                            cell_y - table_min_y,           # distance to bottom edge
                            table_max_y - cell_y            # distance to top edge
                        )
                        
                        if dist_to_table_edges >= CollisionCheckingConfig.MIN_DIST_TO_BOUNDARY:
                            valid_place_cells.append((cell_x, cell_y))
        
        if len(valid_place_cells) == 0:
            print(f"Warning: No valid place locations found on table {place_table_id}")
            return False
        
        # Check if object position is within any valid placement cell
        # Use the same logic as in the training code for consistency
        min_distance_to_valid_cell = float('inf')
        for cell_x, cell_y in valid_place_cells:
            distance_to_cell = ((object_x - cell_x)**2 + (object_y - cell_y)**2)**0.5
            min_distance_to_valid_cell = min(min_distance_to_valid_cell, distance_to_cell)
        
        # Use the same threshold as in the training code: min(CollisionCheckingConfig.MIN_DIST_TO_BOUNDARY, 0.1)
        success_threshold = min(CollisionCheckingConfig.MIN_DIST_TO_BOUNDARY, 0.1)
        is_success = min_distance_to_valid_cell <= success_threshold
        
        if is_success:
            print(f"Placement SUCCESS: object at ({object_x:.3f}, {object_y:.3f}), "
                  f"min_distance_to_valid_cell: {min_distance_to_valid_cell:.3f}, threshold: {success_threshold:.3f}")
        
        return is_success
        
    except Exception as e:
        print(f"Error in detect_success_placement: {e}")
        print(f"Falling back to simple table bounds check for {place_table_id}")
        
        # Fallback: simple check if object is within the table's general area
        # Load the layout to get table bounds
        try:
            layout_json_path = os.path.join(scene_save_dir, f"{layout_name}.json")
            with open(layout_json_path, "r") as f:
                layout_info = json.load(f)
            
            floor_plan: FloorPlan = dict_to_floor_plan(layout_info)
            target_room = next(room for room in floor_plan.rooms if room.id == room_id)
            table_object = next(obj for obj in target_room.objects if obj.id == place_table_id)
            
            # Simple bounding box check
            table_min_x = table_object.position.x
            table_min_y = table_object.position.y
            table_max_x = table_object.position.x + table_object.dimensions.width
            table_max_y = table_object.position.y + table_object.dimensions.length
            
            # Check if object is within table bounds with some tolerance
            tolerance = 0.1  # 10cm tolerance
            within_bounds = (table_min_x - tolerance <= object_x <= table_max_x + tolerance and
                           table_min_y - tolerance <= object_y <= table_max_y + tolerance)
            
            if within_bounds:
                print(f"Fallback placement success: object at ({object_x:.3f}, {object_y:.3f}) within table bounds")
            
            return within_bounds
            
        except Exception as fallback_e:
            print(f"Fallback also failed: {fallback_e}")
            return False

def mm_arm_abs_to_base_rel(actions, fixed_support_frame_data_pose):
    ee_target_pose = actions[:, 6:6+7]
    ee_target_pose_to_fixed_support_frame_pos, ee_target_pose_to_fixed_support_frame_quat = math_utils.subtract_frame_transforms(
        fixed_support_frame_data_pose[:, :3], fixed_support_frame_data_pose[:, 3:7],
        ee_target_pose[:, :3], ee_target_pose[:, 3:7],
    )
    pos_z_offset = -0.5
    ee_target_pose_to_fixed_support_frame_pos[:, 2] += pos_z_offset
    ee_target_pose_to_fixed_support_frame_pose = torch.cat([ee_target_pose_to_fixed_support_frame_pos, ee_target_pose_to_fixed_support_frame_quat], dim=-1)
    processed_actions = actions.clone()
    processed_actions[:, 6:6+7] = ee_target_pose_to_fixed_support_frame_pose
    return processed_actions

def mm_base_rel_to_arm_abs(actions, fixed_support_frame_data_pose):
    # Extract the relative EE target pose from actions
    ee_target_pose_relative = actions[:, 6:6+7]
    
    # Extract position and quaternion
    ee_target_pose_to_fixed_support_frame_pos = ee_target_pose_relative[:, :3].clone()
    ee_target_pose_to_fixed_support_frame_quat = ee_target_pose_relative[:, 3:7]
    
    # Reverse the z-offset (subtract the offset that was added in mm_arm_abs_to_base_rel)
    pos_z_offset = -0.5
    ee_target_pose_to_fixed_support_frame_pos[:, 2] -= pos_z_offset
    
    # Convert from relative (fixed_support_frame) to absolute (world) frame using combine_frame_transforms
    ee_target_pose_pos, ee_target_pose_quat = math_utils.combine_frame_transforms(
        fixed_support_frame_data_pose[:, :3], fixed_support_frame_data_pose[:, 3:7],
        ee_target_pose_to_fixed_support_frame_pos, ee_target_pose_to_fixed_support_frame_quat,
    )
    
    # Combine position and quaternion back into a single pose tensor
    ee_target_pose = torch.cat([ee_target_pose_pos, ee_target_pose_quat], dim=-1)
    
    # Replace the relative pose with the absolute pose in actions
    processed_actions = actions.clone()
    processed_actions[:, 6:6+7] = ee_target_pose
    return processed_actions


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    log_dir = os.path.abspath(os.path.join("./robomimic_data", args_cli.task+"-"+args_cli.post_fix))
    debug_dir = os.path.join(log_dir, "debug")

    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir, exist_ok=True)
    

    layout_id = "layout_fac9613b"
    pick_table_id = "room_0efe6071_table_eb113586"
    pick_object_id = "room_0efe6071_coke_29651336"
    # pick_object_id = "room_0efe6071_mug_13fa7a11"
    place_table_id = "room_0efe6071_desk_0aac9647"
    place_object_id = "room_0efe6071_plate_cdfd1c12"
    room_id = "room_0efe6071"

    # layout_id = "layout_3f2c14ff"
    # pick_table_id = "room_96fbc459_coffeetable_7c2106a1"
    # pick_object_id = "room_96fbc459_coke_2f3a6f2e"
    # place_table_id = "room_96fbc459_desk_277c90bf"
    # place_object_id = "room_96fbc459_plate_daa757d4"
    # room_id = "room_96fbc459"

    # layout_id = "layout_6b1bf1b5"
    # pick_table_id = "room_4e051f77_coffeetable_2c551044"
    # pick_object_id = "room_4e051f77_coke_d833ab23"
    # place_table_id = "room_4e051f77_desk_411d1a35"
    # place_object_id = "room_4e051f77_plate_883c693c"
    # room_id = "room_4e051f77"

    eval_spawn_pose_json_path = "/home/hongchix/main/server/robomimic_data/Isaac-Mobile-Manipulation-Obj-Scene-Franka-IK-Abs-v0-pose_aug_eval_00/spawn_pose.json"
    eval_spawn_layout_and_pose_dict = json.load(open(eval_spawn_pose_json_path, "r"))

    all_test_cases = []
    repeat_nums = 3

    for layout_dict_current in eval_spawn_layout_and_pose_dict:
        spawn_pos = torch.tensor(layout_dict_current["spawn_pose"]["spawn_pos"], device="cuda").float().reshape(-1, 2)
        spawn_angles = torch.tensor(layout_dict_current["spawn_pose"]["spawn_angles"], device="cuda").float().reshape(-1, 1)
        
        num_spawn_pos = spawn_pos.shape[0]
        for spawn_pose_i in range(num_spawn_pos):
            spawn_pos_i = spawn_pos[spawn_pose_i, :].reshape(1, 2)
            spawn_angles_i = spawn_angles[spawn_pose_i, :].reshape(1, 1)
            for _ in range(repeat_nums):
                all_test_cases.append({
                    "layout_id": layout_dict_current["layout_id"],
                    "rigid_object_transform_dict": layout_dict_current["rigid_object_transform_dict"],
                    "spawn_pos": spawn_pos_i,
                    "spawn_angles": spawn_angles_i,
                })

    # Visualize all loaded spawn poses if visualization is enabled
    # if args_cli.visualize_figs:
    #     try:
    #         scene_save_dir = os.path.join(SERVER_ROOT_DIR, f"results/{layout_id}")
    #         visualize_loaded_spawn_poses(
    #             eval_spawn_pos, eval_spawn_angles, 
    #             scene_save_dir, layout_id, room_id, debug_dir,
    #             layout_name=layout_id
    #         )
    #     except Exception as e:
    #         print(f"Warning: Failed to create loaded spawn poses visualization: {e}")

    robomimic_policy_path = args_cli.policy_path
    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(device="cuda", ckpt_path=robomimic_policy_path)

    scene_save_dir = os.path.join(SERVER_ROOT_DIR, f"results/{layout_id}")
    layout_json_path = os.path.join(scene_save_dir, f"{layout_id}.json")

    layout_data = json.load(open(layout_json_path, "r"))

    room_dict = {}
    for room in layout_data["rooms"]:
        room_x_min = room["position"]["x"]
        room_y_min = room["position"]["y"]

        room_x_max = room["position"]["x"] + room["dimensions"]["width"]
        room_y_max = room["position"]["y"] + room["dimensions"]["length"]

        room_dict[room["id"]] = {
            "x_min": room_x_min,
            "y_min": room_y_min,
            "x_max": room_x_max,
            "y_max": room_y_max,
            "height": room["dimensions"]["height"],
            "id": room["id"],
        }


    camera_pos, camera_lookat = get_default_camera_view()
    base_pos = get_default_base_pos()
    usd_collection_dir = os.path.join(scene_save_dir, layout_id+"_usd_collection")
    mass_dict_path = os.path.join(usd_collection_dir, "rigid_object_property_dict.json")
    mass_dict = json.load(open(mass_dict_path, "r"))
    all_object_rigids = list(mass_dict.keys())
    
    # create a yaml file to store the config
    config_dict = {
        "scene_save_dir": scene_save_dir,
        "usd_collection_dir": usd_collection_dir,
        "base_pos": base_pos,
        "camera_pos": camera_pos,
        "camera_lookat": camera_lookat,
        "mass_dict": mass_dict,
        "room_dict": room_dict,
    }
    env_init_config_yaml = os.path.join(log_dir, "params", "env_init.yaml")
    dump_yaml(env_init_config_yaml, config_dict)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, config_yaml=env_init_config_yaml)
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    env_cfg = load_pickle(os.path.join(log_dir, "params", "env.pkl"))
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).env

    # reset environment
    obs_dict, _ = env.reset()
    # reset interfaces
    iteration = 0
    T_grasp = 2000
    T_hold = 50
    T_init = 20
    T_drop = 100
    current_data_idx = 0
    success_cnt = 0

    # Initialize test results tracking for JSON output
    test_results = {
        "task": args_cli.task,
        "policy_path": args_cli.policy_path,
        "layout_id": layout_id,
        "room_id": room_id,
        "pick_object_id": pick_object_id,
        "place_table_id": place_table_id,
        "total_test_cases": len(all_test_cases),
        "num_envs_per_test": num_envs,
        "test_cases": [],
        "summary": {}
    }

    joint_name_base_x = "base_joint_mobile_side"
    joint_name_base_y = "base_joint_mobile_forward"
    joint_name_base_angle = "base_joint_mobile_yaw"

    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt):
        while True:

            if iteration % T_grasp == 0:

                test_case_current = all_test_cases[current_data_idx]
                layout_name = test_case_current["layout_id"]
                rigid_object_transform_dict = test_case_current["rigid_object_transform_dict"]
                spawn_pos = test_case_current["spawn_pos"].reshape(1, 2)
                spawn_angles = test_case_current["spawn_angles"].reshape(1, 1)

                object_transform_dict = {}

                for object_name in rigid_object_transform_dict:
                    object_transform_dict[object_name] = {}
                    object_position = rigid_object_transform_dict[object_name]["position"]
                    object_transform_dict[object_name]["position"] = np.array([object_position["x"], object_position["y"], object_position["z"]]).reshape(3)
                    object_rotation = rigid_object_transform_dict[object_name]["rotation"]
                    object_transform_dict[object_name]["rotation"] = np.array(R.from_euler("xyz", [object_rotation["x"], object_rotation["y"], object_rotation["z"]], degrees=True).as_quat(scalar_first=True)).reshape(4)


                iteration = 0
                env.reset()
                sim = env.sim

                # set objects
                # print(f"env.scene: ", env.scene.keys())
                envs_translate = []
                for env_i in range(num_envs):
                    env_root_prim_path = f"/World/envs/env_{env_i}"
                    env_root_prim = prim_utils.get_prim_at_path(env_root_prim_path)
                    envs_translate.append(env_root_prim.GetAttribute("xformOp:translate").Get())
                envs_pos = torch.tensor(envs_translate, device=sim.device).reshape(num_envs, 3).float()
                envs_quat = torch.tensor([1, 0, 0, 0], device=sim.device).reshape(1, 4).repeat(num_envs, 1).float()

                for object_name in all_object_rigids:
                    object_state = torch.zeros(num_envs, 13).to(sim.device)
                    object_state[..., :3] = torch.tensor(object_transform_dict[object_name]["position"], device=sim.device)
                    object_state[..., 3:7] = torch.tensor(object_transform_dict[object_name]["rotation"], device=sim.device)
                    object_state_pos, object_state_quat = math_utils.combine_frame_transforms(
                        envs_pos, envs_quat,
                        object_state[..., :3], object_state[..., 3:7]
                    )
                    object_state = torch.cat([object_state_pos, object_state_quat, object_state[..., 7:]], dim=-1)
                    env.scene[object_name].write_root_state_to_sim(object_state)
                    env.scene[object_name].reset()

                robot = env.scene["robot"]

                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()

                joint_idx_base_x = robot.data.joint_names.index(joint_name_base_x)
                joint_idx_base_y = robot.data.joint_names.index(joint_name_base_y)
                joint_idx_base_angle = robot.data.joint_names.index(joint_name_base_angle)

                joint_pos[:, joint_idx_base_x] = spawn_pos[:, 0]
                joint_pos[:, joint_idx_base_y] = spawn_pos[:, 1]
                joint_pos[:, joint_idx_base_angle] = spawn_angles[:, 0]

                robot.write_joint_state_to_sim(joint_pos, joint_vel)

                robot_base_pos = torch.tensor([0., 0., 0.01], device=sim.device).reshape(1, 3).repeat(num_envs, 1)
                robot_base_quat = torch.tensor([1, 0, 0, 0], device=sim.device).reshape(1, 4).repeat(num_envs, 1)

                base_w = torch.zeros(num_envs, 13).to(sim.device)
                base_w[..., :3] = robot_base_pos
                base_w[..., 3:7] = robot_base_quat

                base_w_pos, base_w_quat = math_utils.combine_frame_transforms(
                    envs_pos, envs_quat,
                    base_w[..., :3], base_w[..., 3:7]
                )
                base_w = torch.cat([base_w_pos, base_w_quat, base_w[..., 7:]], dim=-1)

                robot.write_root_link_state_to_sim(base_w)
                robot.reset()
                robot_base_w = env.scene["robot"].data.root_state_w[:, :7]

                env.scene["ee_frame"].reset()
                env.scene["fixed_support_frame"].reset()
                env.scene["support_frame"].reset()

                camera_list = []
                rollout_policy.start_episode()

                success_envs = [False] * num_envs



            actions_list = []

            body_names = robot.data.body_names
            body_link_pose = robot.data.body_link_state_w[..., :7]

            ee_body_name = "arm_right_hand"
            fixed_support_body_name = "base_fixed_support"
            support_body_name = "base_support"

            ee_body_idx = body_names.index(ee_body_name)
            fixed_support_body_idx = body_names.index(fixed_support_body_name)
            support_body_idx = body_names.index(support_body_name)

            ee_frame_data_pos = body_link_pose[:, ee_body_idx, :3].reshape(-1, 3).clone()
            ee_frame_data_quat = body_link_pose[:, ee_body_idx, 3:7].reshape(-1, 4).clone()
            ee_frame_data_pos, ee_frame_data_quat = math_utils.subtract_frame_transforms(
                envs_pos, envs_quat,
                ee_frame_data_pos, ee_frame_data_quat,
            )
            ee_frame_data_pose = torch.cat([ee_frame_data_pos, ee_frame_data_quat], dim=-1)

            fixed_support_frame_data_pos = body_link_pose[:, fixed_support_body_idx, :3].reshape(-1, 3).clone()
            fixed_support_frame_data_quat = body_link_pose[:, fixed_support_body_idx, 3:7].reshape(-1, 4).clone()
            fixed_support_frame_data_pos, fixed_support_frame_data_quat = math_utils.subtract_frame_transforms(
                envs_pos, envs_quat,
                fixed_support_frame_data_pos, fixed_support_frame_data_quat,
            )
            fixed_support_frame_data_pose = torch.cat([fixed_support_frame_data_pos, fixed_support_frame_data_quat], dim=-1)

            ee_to_fixed_support_frame_pos, ee_to_fixed_support_frame_quat = math_utils.subtract_frame_transforms(
                fixed_support_frame_data_pos, fixed_support_frame_data_quat,
                ee_frame_data_pos, ee_frame_data_quat
            )
            ee_to_fixed_support_frame_pose = torch.cat([ee_to_fixed_support_frame_pos, ee_to_fixed_support_frame_quat], dim=-1)

            support_frame_data_pos = body_link_pose[:, support_body_idx, :3].reshape(-1, 3).clone()
            support_frame_data_quat = body_link_pose[:, support_body_idx, 3:7].reshape(-1, 4).clone()
            support_frame_data_pos, support_frame_data_quat = math_utils.subtract_frame_transforms(
                envs_pos, envs_quat,
                support_frame_data_pos, support_frame_data_quat,
            )
            support_frame_data_pose = torch.cat([support_frame_data_pos, support_frame_data_quat], dim=-1)

            ee_to_support_frame_pos, ee_to_support_frame_quat = math_utils.subtract_frame_transforms(
                support_frame_data_pos, support_frame_data_quat,
                ee_frame_data_pos, ee_frame_data_quat
            )
            ee_to_support_frame_pose = torch.cat([ee_to_support_frame_pos, ee_to_support_frame_quat], dim=-1)



            if iteration == 0:
                ee_frame_data_pos_initial = ee_frame_data_pos.clone()
                ee_frame_data_quat_initial = ee_frame_data_quat.clone()
                fixed_support_frame_data_pos_initial = fixed_support_frame_data_pos.clone()
                fixed_support_frame_data_quat_initial = fixed_support_frame_data_quat.clone()
                support_frame_data_pos_initial = support_frame_data_pos.clone()
                support_frame_data_quat_initial = support_frame_data_quat.clone()

                ee_to_fixed_support_frame_pos_initial = ee_to_fixed_support_frame_pos.clone()
                ee_to_fixed_support_frame_quat_initial = ee_to_fixed_support_frame_quat.clone()
                ee_to_support_frame_pos_initial = ee_to_support_frame_pos.clone()
                ee_to_support_frame_quat_initial = ee_to_support_frame_quat.clone()


            robot_qpos = robot.data.joint_pos

            target_object_w_state = env.scene[pick_object_id].data.root_state_w[:, :7]
            target_object_w_state_pos_to_env_root, target_object_w_state_quat_to_env_root = math_utils.subtract_frame_transforms(
                envs_pos, envs_quat,
                target_object_w_state[:, :3], target_object_w_state[:, 3:7],
            )
            target_object_w_state_to_env_root = torch.cat([target_object_w_state_pos_to_env_root, target_object_w_state_quat_to_env_root], dim=-1)


            def get_obs_dict_for_policy(obs_dict):
                return {
                    k: v[0] for k, v in obs_dict["policy"].items()
                }

            
            for env_i in range(num_envs):

                if iteration % T_grasp < T_init:

                    ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat = math_utils.combine_frame_transforms(
                        fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1],
                        ee_to_fixed_support_frame_pos_initial[env_i:env_i+1], ee_to_fixed_support_frame_quat_initial[env_i:env_i+1]
                    )

                    actions = torch.cat([
                        get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], 
                            fixed_support_frame_data_pos_initial[env_i:env_i+1, :3].reshape(1, 3), 
                            fixed_support_frame_data_quat_initial[env_i:env_i+1, :4].reshape(1, 4)),
                        torch.cat([ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat], dim=-1),
                        torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                    ], dim=-1)

                else:
                    obs_dict = env.observation_manager.compute()
                    actions = rollout_policy(ob=get_obs_dict_for_policy(obs_dict))
                    if isinstance(actions, np.ndarray):
                        actions = torch.from_numpy(actions).to(sim.device)
                    actions = actions.reshape(1, -1).float()
                    actions = mm_base_rel_to_arm_abs(actions, fixed_support_frame_data_pose)

                actions_list.append(actions)

                

            actions = torch.cat(actions_list, dim=0)

            iteration_status = {}

            obs_dict = env.observation_manager.compute()


            for key, value in obs_dict["policy"].items():
                # print(f"key: {key}; value: {value.shape}")
                iteration_status[f"obs/{key}"] = value
            iteration_status["actions"] = actions
            
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            for key, value in obs_dict["policy"].items():
                iteration_status[f"next_obs/{key}"] = value

            iteration_status["rewards"] = rewards
            iteration_status["dones"] = dones


            def depth_to_rgb(depth):
                depth = depth[..., None].repeat(3, axis=2)
                return depth


            if iteration % T_grasp >= T_init:
                camera_frame_envs = []
                for env_i in range(min(num_envs, 8)):
                    camera_frame_envs.append(
                        np.concatenate([
                            iteration_status["obs/rgb_front"][env_i].cpu().numpy(),
                            depth_to_rgb(iteration_status["obs/depth_front"][env_i, :, :, 0].cpu().numpy()),
                            iteration_status["obs/rgb_back"][env_i].cpu().numpy(),
                            depth_to_rgb(iteration_status["obs/depth_back"][env_i, :, :, 0].cpu().numpy()),
                            iteration_status["obs/rgb_left"][env_i].cpu().numpy(),
                            depth_to_rgb(iteration_status["obs/depth_left"][env_i, :, :, 0].cpu().numpy()),
                            iteration_status["obs/rgb_right"][env_i].cpu().numpy(),
                            depth_to_rgb(iteration_status["obs/depth_right"][env_i, :, :, 0].cpu().numpy()),
                            iteration_status["obs/rgb_wrist"][env_i].cpu().numpy(),
                            depth_to_rgb(iteration_status["obs/depth_wrist"][env_i, :, :, 0].cpu().numpy()),
                        ], axis=1)
                    )
                camera_frame_envs = np.concatenate(camera_frame_envs, axis=0)
                camera_list.append(camera_frame_envs)

            if iteration % 100 == 0:
                print("iteration: ", iteration)

            if iteration % T_grasp == T_grasp - 1:

                # Track success for each environment in this test case
                test_case_results = {
                    "test_case_id": current_data_idx,
                    "spawn_pos": [float(spawn_pos[0, 0]), float(spawn_pos[0, 1])],
                    "spawn_angle": float(spawn_angles[0, 0]),
                    "environments": [],
                    "success_count": 0,
                    "total_envs": num_envs,
                    "success_rate": 0.0
                }

                for env_i in range(num_envs):
                    success_envs[env_i] = detect_success_placement(
                        target_object_w_state_to_env_root[env_i:env_i+1],
                        pick_object_id,
                        place_table_id,
                        scene_save_dir,
                        layout_name, 
                        room_id
                    )
                    
                    # Record individual environment result
                    env_result = {
                        "env_id": env_i,
                        "success": bool(success_envs[env_i]),
                        "object_final_position": {
                            "x": float(target_object_w_state_to_env_root[env_i, 0]),
                            "y": float(target_object_w_state_to_env_root[env_i, 1]),
                            "z": float(target_object_w_state_to_env_root[env_i, 2])
                        }
                    }
                    test_case_results["environments"].append(env_result)
                    
                    if success_envs[env_i]:
                        print(f"env {env_i} placement success")
                        success_cnt += 1
                        test_case_results["success_count"] += 1

                # Calculate success rate for this test case
                test_case_results["success_rate"] = test_case_results["success_count"] / test_case_results["total_envs"]
                test_results["test_cases"].append(test_case_results)

                cameras_debug_dir = os.path.join(debug_dir, "cameras")
                os.makedirs(cameras_debug_dir, exist_ok=True)
                video_path = os.path.join(cameras_debug_dir, f"{current_data_idx:0>3d}_cameras.mp4")
                imageio.mimwrite(video_path, camera_list, fps=30)
                print(f"saving {current_data_idx} data to video: {video_path}")
                
                torch.cuda.empty_cache()
                gc.collect()

                current_data_idx += 1

                print(f"success_cnt: {success_cnt}; total test clips: {current_data_idx}; current success rate: {success_cnt / current_data_idx}")
            
            iteration = (iteration + 1) % T_grasp

            if current_data_idx == len(all_test_cases):
                break

    # Calculate final summary statistics
    total_test_cases = current_data_idx
    total_environments = current_data_idx * num_envs
    overall_success_rate = success_cnt / total_environments if total_environments > 0 else 0.0
    
    # Calculate test case success rate (percentage of test cases with at least one successful environment)
    successful_test_cases = sum(1 for test_case in test_results["test_cases"] if test_case["success_count"] > 0)
    test_case_success_rate = successful_test_cases / total_test_cases if total_test_cases > 0 else 0.0
    
    # Calculate perfect test case success rate (percentage of test cases where all environments succeeded)
    perfect_test_cases = sum(1 for test_case in test_results["test_cases"] if test_case["success_count"] == test_case["total_envs"])
    perfect_test_case_rate = perfect_test_cases / total_test_cases if total_test_cases > 0 else 0.0

    # Fill in the summary
    import time
    test_results["summary"] = {
        "total_test_cases": total_test_cases,
        "total_environments": total_environments,
        "total_successful_environments": success_cnt,
        "overall_success_rate": overall_success_rate,
        "successful_test_cases": successful_test_cases,
        "test_case_success_rate": test_case_success_rate,
        "perfect_test_cases": perfect_test_cases,
        "perfect_test_case_rate": perfect_test_case_rate,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_settings": {
            "num_envs": num_envs,
            "T_grasp": T_grasp,
            "T_init": T_init,
            "T_drop": T_drop
        }
    }

    # Save results to JSON file
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"evaluation_results_{layout_id}_{room_id}_{timestamp}.json"
    results_path = os.path.join(log_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    print(f"Total test cases: {total_test_cases}")
    print(f"Total environments: {total_environments}")
    print(f"Successful environments: {success_cnt}")
    print(f"Overall success rate: {overall_success_rate:.2%}")
    print(f"Test cases with at least one success: {successful_test_cases}")
    print(f"Test case success rate: {test_case_success_rate:.2%}")
    print(f"Test cases with perfect success: {perfect_test_cases}")
    print(f"Perfect test case rate: {perfect_test_case_rate:.2%}")
    print(f"Results saved to: {results_path}")
    print("="*80)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
