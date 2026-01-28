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
import torch

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
# Add parent directory to Python path to import constants
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR, ROBOMIMIC_ROOT_DIR, M2T2_ROOT_DIR, PHYSICS_CRITIC_ENABLED, SEMANTIC_CRITIC_ENABLED

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


import importlib.util
utils_spec = importlib.util.spec_from_file_location("tex_utils", os.path.join(SERVER_ROOT_DIR, "tex_utils.py"))
tex_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(tex_utils)

# Import the specific functions from server utils
export_layout_to_mesh_dict_list_tree_search_with_object_id = tex_utils.export_layout_to_mesh_dict_list_tree_search_with_object_id
export_layout_to_mesh_dict_object_id = tex_utils.export_layout_to_mesh_dict_object_id
get_textured_object_mesh = tex_utils.get_textured_object_mesh

import trimesh
import trimesh.transformations as tf
from models import FloorPlan
import copy

from isaaclab.omron_franka_occupancy import occupancy_map, support_point, get_forward_side_from_support_point_and_yaw

# Import helper functions from object_augmentation.py for consistency
def _object_pose_to_matrix(obj) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from an object's position and Euler rotation (degrees).
    Rotation order matches apply_object_transform: Z @ Y @ X (Euler xyz).
    """
    rx = np.radians(obj.rotation.x)
    ry = np.radians(obj.rotation.y)
    rz = np.radians(obj.rotation.z)

    rot_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1],
    ])
    rot_y = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1],
    ])
    rot_z = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    rot = rot_z @ rot_y @ rot_x

    trans = np.eye(4)
    trans[:3, 3] = np.array([obj.position.x, obj.position.y, obj.position.z])

    return trans @ rot


def _matrix_to_pose(matrix: np.ndarray):
    """
    Convert a 4x4 transform into position (meters) and Euler rotation (degrees, xyz order).
    """
    from models import Point3D, Euler
    
    pos = matrix[:3, 3]
    rot_mat = matrix[:3, :3]
    # trimesh.transformations expects a 4x4 matrix for euler_from_matrix
    rot4 = np.eye(4)
    rot4[:3, :3] = rot_mat
    # 'sxyz' corresponds to static axes X->Y->Z; matches our construction
    ex, ey, ez = tf.euler_from_matrix(rot4, axes='sxyz')
    euler_deg = np.degrees([ex, ey, ez])
    return Point3D(float(pos[0]), float(pos[1]), float(pos[2])), Euler(float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2]))


def _check_reachability(layout_copy, room_copy, all_objects_post_order, reach_threshold: float) -> bool:
    """
    Check if objects in the room are reachable by a robot.
    
    Args:
        layout_copy: The floor plan layout
        room_copy: The room containing the objects
        all_objects_post_order: List of objects to check for reachability
        reach_threshold: Maximum reach distance for the robot
        
    Returns:
        True if at least one robot location can reach all non-floor objects, False otherwise
    """
    # Parameters (same as in sample_robot_location)
    robot_min_dist_to_room_edge = 0.5
    robot_min_dist_to_object = 0.15
    grid_res = 0.02
    num_sample_points = 10000
    
    # Get room rectangle bounds
    room_min_x = room_copy.position.x
    room_min_y = room_copy.position.y
    room_max_x = room_copy.position.x + room_copy.dimensions.width
    room_max_y = room_copy.position.y + room_copy.dimensions.length
    
    # Get all object meshes in the room for occupancy calculation
    object_meshes = []
    for obj in room_copy.objects:
        try:
            mesh_info = get_textured_object_mesh(layout_copy, room_copy, room_copy.id, obj.id)
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
    
    # Create occupancy grid using ray casting
    grid_x = np.arange(room_min_x, room_max_x, grid_res)
    grid_y = np.arange(room_min_y, room_max_y, grid_res)
    
    # Create ray origins at grid centers, elevated above the room
    grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x + grid_res/2, grid_y + grid_res/2, indexing='ij')
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
    edge_valid_mask = dist_to_edges >= robot_min_dist_to_room_edge
    edge_valid_points = sample_points[edge_valid_mask]
    
    if len(edge_valid_points) == 0:
        return False  # No valid robot positions
    
    # Convert to grid coordinates
    grid_coords = np.floor((edge_valid_points - [room_min_x, room_min_y]) / grid_res).astype(int)
    
    # Check if points are within grid bounds and not in occupied cells
    grid_valid_mask = (
        (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < len(grid_x)) &
        (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < len(grid_y)) &
        (~occupancy_grid[grid_coords[:, 0], grid_coords[:, 1]])
    )
    
    grid_valid_points = edge_valid_points[grid_valid_mask]
    
    if len(grid_valid_points) == 0:
        return False  # No valid robot positions after grid filtering
    
    # Check distance to occupied cells using vectorized operations
    occupied_indices = np.where(occupancy_grid)
    if len(occupied_indices[0]) > 0:
        occupied_positions = np.column_stack([
            room_min_x + occupied_indices[0] * grid_res + grid_res/2,
            room_min_y + occupied_indices[1] * grid_res + grid_res/2
        ])
        
        # Calculate distances from each valid point to all occupied cells
        distances = np.linalg.norm(
            grid_valid_points[:, np.newaxis, :] - occupied_positions[np.newaxis, :, :], 
            axis=2
        )
        
        # Find minimum distance to any occupied cell for each point
        min_distances = np.min(distances, axis=1)
        
        # Filter points that are far enough from occupied cells
        distance_valid_mask = min_distances >= robot_min_dist_to_object
        valid_robot_points = grid_valid_points[distance_valid_mask]
    else:
        # No occupied cells, all grid-valid points are valid
        valid_robot_points = grid_valid_points
    
    if len(valid_robot_points) == 0:
        return False  # No valid robot positions after distance filtering
    
    # Check reachability: for each object not on the floor, check if any robot location is within reach
    objects_to_check = [obj for obj in all_objects_post_order if obj.place_id != "floor"]
    
    if len(objects_to_check) == 0:
        return True  # No objects to check, consider reachable
    
    # Get object positions (2D)
    object_positions = np.array([[obj.position.x, obj.position.y] for obj in objects_to_check])
    
    # Calculate distances from each robot location to each object (10k x num_objects)
    # Shape: (n_robot_points, n_objects)
    distances_to_objects = np.linalg.norm(
        valid_robot_points[:, np.newaxis, :] - object_positions[np.newaxis, :, :], 
        axis=2
    )
    
    # Check if any robot location is within reach_threshold of all objects
    # For each robot location, check if it can reach all objects
    robot_can_reach_all = np.all(distances_to_objects <= reach_threshold, axis=1)
    
    # Return True if at least one robot location can reach all objects
    is_reachable = np.any(robot_can_reach_all)
    
    print(f"Reach test: {len(valid_robot_points)} valid robot locations, {len(objects_to_check)} objects to check, reachable: {is_reachable}")
    
    return is_reachable

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
    MIN_DIST_TO_BOUNDARY = 0.2   # Minimum distance from place location to table boundary
    MAX_DIST_TO_OBJECT = 0.8      # Maximum distance from robot to place location for feasibility


def create_unified_occupancy_grid(scene_save_dir, layout_name, room_id, only_floor=True, return_idx=False):
    """
    Create a unified occupancy grid that will be consistent across all functions.
    
    Args:
        scene_save_dir: Directory containing the scene data
        layout_name: Name of the layout
        room_id: ID of the room
        only_floor: If True, only include objects that are ultimately placed on the floor (default: True)
        return_idx: If True, return additional dict with occupancy_idx_grid and idx_to_object_name
    
    Returns:
        tuple: (occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, floor_plan, target_room)
               If return_idx=True, returns (occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, 
               floor_plan, target_room, idx_dict) where idx_dict contains 'occupancy_idx_grid' and 
               'idx_to_object_name'
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
    
    # Helper function to check if an object is ultimately placed on the floor
    def is_ultimately_on_floor(obj, visited=None):
        """Recursively check if an object is ultimately placed on the floor."""
        if visited is None:
            visited = set()
        
        # Prevent infinite loops
        if obj.id in visited:
            return False
        visited.add(obj.id)
        
        # Check if directly on floor
        if not hasattr(obj, 'place_id') or obj.place_id is None:
            return False
        if obj.place_id == "floor":
            return True
        
        # Recursively check the object it's placed on
        for parent_obj in target_room.objects:
            if parent_obj.id == obj.place_id:
                return is_ultimately_on_floor(parent_obj, visited)
        
        # place_id not found in room objects
        return False
    
    # Get all object meshes in the room for occupancy calculation
    object_meshes = []
    object_names = []  # Track object names for return_idx
    for obj in target_room.objects:
        # Filter based on only_floor parameter
        if only_floor and not is_ultimately_on_floor(obj):
            continue
        
        try:
            mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, obj.id)
            if mesh_info and mesh_info["mesh"] is not None:
                object_meshes.append(mesh_info["mesh"])
                object_names.append(obj.id)
        except Exception as e:
            print(f"Warning: Could not load mesh for object {obj.id}: {e}")
            continue
    
    # Combine all object meshes for ray casting
    # Track triangle to object mapping for return_idx
    tri_to_object_idx = []
    if object_meshes:
        combined_mesh = trimesh.util.concatenate(object_meshes)
        # Build mapping from triangle index to object index
        if return_idx:
            cumulative_faces = 0
            for obj_idx, mesh in enumerate(object_meshes):
                num_faces = len(mesh.faces)
                tri_to_object_idx.extend([obj_idx] * num_faces)
                cumulative_faces += num_faces
            tri_to_object_idx = np.array(tri_to_object_idx)
    else:
        # If no meshes available, create an empty mesh
        combined_mesh = trimesh.Trimesh()
        tri_to_object_idx = np.array([])
    
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
    occupancy_idx_grid = None
    if return_idx:
        # Initialize with -1 to indicate no object hit
        occupancy_idx_grid = np.full((len(grid_x), len(grid_y)), -1, dtype=np.int32)
    
    if len(object_meshes) > 0:
        try:
            locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(combined_mesh).intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                multiple_hits=False
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
                
                # If return_idx is True, also track which object was hit
                if return_idx and len(tri_to_object_idx) > 0:
                    valid_index_tri = index_tri[valid_mask]
                    # Map triangle indices to object indices
                    object_indices = tri_to_object_idx[valid_index_tri]
                    occupancy_idx_grid[valid_i, valid_j] = object_indices
        except Exception as e:
            print(f"Warning: Ray casting failed: {e}")
    
    if return_idx:
        # Create idx_to_object_name dictionary
        idx_to_object_name = {i: name for i, name in enumerate(object_names)}
        idx_dict = {
            'occupancy_idx_grid': occupancy_idx_grid,
            'idx_to_object_name': idx_to_object_name
        }
        return occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, floor_plan, target_room, idx_dict
    else:
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
    num_sample_points = int(10000 * num_envs)
    
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
    if True:
        try:
            # Create save directory if it doesn't exist
            robot_planning_debug_dir = os.path.join(debug_dir, "robot_planning")
            os.makedirs(robot_planning_debug_dir, exist_ok=True)
            
            # Generate filename with timestamp and identifiers
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            viz_filename = f"robot_planning_{layout_name.replace('/', '_')}_{room_id}_{target_object_name}_{timestamp}.png"
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
    num_sample_robot_points = 10000 * num_envs # For robot position sampling
    
    print(f"Sampling robot place locations for table {table_object_name} in room {room_id}")
    
    # Create unified occupancy grid for scene collision checking
    occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, floor_plan, target_room, idx_dict = create_unified_occupancy_grid(
        scene_save_dir, layout_name, room_id, return_idx=True
    )
    occupancy_idx_grid = idx_dict['occupancy_idx_grid']
    idx_to_object_name = idx_dict['idx_to_object_name']
    scene_occupancy_fn = create_unified_scene_occupancy_fn(occupancy_grid, grid_x, grid_y, room_bounds)
    
    # Find table object index
    table_object = next(obj for obj in target_room.objects if obj.id == table_object_name)
    assert table_object is not None, f"table_object {table_object_name} not found in floor_plan"
    
    # Find the table's index in idx_to_object_name
    table_idx = None
    for idx, obj_name in idx_to_object_name.items():
        if obj_name == table_object_name:
            table_idx = idx
            break
    
    assert table_idx is not None, f"table_object {table_object_name} not found in idx_to_object_name"
    
    # Create table-specific occupancy grid from occupancy_idx_grid
    # Only keep cells where the occupancy_idx_grid matches the table index
    table_occupancy_grid = (occupancy_idx_grid == table_idx)
    
    # Calculate table bounds from the occupancy grid
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    table_cells = np.argwhere(table_occupancy_grid)
    
    if len(table_cells) > 0:
        # Convert grid indices to world coordinates
        table_cell_i_min = table_cells[:, 0].min()
        table_cell_i_max = table_cells[:, 0].max()
        table_cell_j_min = table_cells[:, 1].min()
        table_cell_j_max = table_cells[:, 1].max()
        
        table_min_x = room_min_x + table_cell_i_min * CollisionCheckingConfig.GRID_RES
        table_max_x = room_min_x + (table_cell_i_max + 1) * CollisionCheckingConfig.GRID_RES
        table_min_y = room_min_y + table_cell_j_min * CollisionCheckingConfig.GRID_RES
        table_max_y = room_min_y + (table_cell_j_max + 1) * CollisionCheckingConfig.GRID_RES
        table_bounds = (table_min_x, table_min_y, table_max_x, table_max_y)
    else:
        # Fallback: use object dimensions
        table_min_x = table_object.position.x
        table_min_y = table_object.position.y
        table_max_x = table_object.position.x + table_object.dimensions.width
        table_max_y = table_object.position.y + table_object.dimensions.length
        table_bounds = (table_min_x, table_min_y, table_max_x, table_max_y)
        print(f"Warning: No table cells found in occupancy grid, using object dimensions as fallback", file=sys.stderr)
    
    # Get table mesh for height calculation
    try:
        table_mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, table_object_name)
        if table_mesh_info and table_mesh_info["mesh"] is not None:
            table_mesh = table_mesh_info["mesh"]
        else:
            table_mesh = None
    except Exception as e:
        print(f"Warning: Could not load mesh for table {table_object_name}: {e}")
        table_mesh = None
    
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
    table_grid_res = CollisionCheckingConfig.GRID_RES  # Using unified grid resolution
    
    # Get all table cells from the occupancy_idx_grid
    table_cells_in_grid = np.argwhere(table_occupancy_grid)
    
    if len(table_cells_in_grid) == 0:
        # Fallback case: no table cells found
        table_center_x = (table_min_x + table_max_x) / 2
        table_center_y = (table_min_y + table_max_y) / 2
        valid_place_cells = [(table_center_x, table_center_y)]
        print("Using table center as single place location (fallback)")
    else:
        # Check each table cell for valid placement
        for cell_idx in table_cells_in_grid:
            i, j = cell_idx[0], cell_idx[1]
            # Convert grid coordinates to world coordinates
            cell_x = room_min_x + i * table_grid_res + table_grid_res / 2
            cell_y = room_min_y + j * table_grid_res + table_grid_res / 2
            
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
    if True:
        try:
            # Create save directory if it doesn't exist
            robot_place_planning_debug_dir = os.path.join(debug_dir, "robot_place_planning")
            os.makedirs(robot_place_planning_debug_dir, exist_ok=True)
            
            # Generate filename with timestamp and identifiers
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            viz_filename = f"robot_place_planning_{layout_name.replace('/', '_')}_{room_id}_{table_object_name}_{timestamp}.png"
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
    
    if not place_found:
        return None, None, None
    else:
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
    scene_save_dir, layout_name, room_id, num_envs, debug_dir
):
    """
    sample collision-free robot spawn positions

    Uses unified collision checking configuration for consistency with trajectory planning.

    no need to consider the target object

    Args:
        scene_save_dir: Directory containing scene data
        layout_name: ID of the layout
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
        scene_save_dir, layout_name, room_id
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
    if True:
        try:
            # Create save directory if it doesn't exist
            robot_spawn_planning_debug_dir = os.path.join(debug_dir, "robot_spawn_planning")
            os.makedirs(robot_spawn_planning_debug_dir, exist_ok=True)
            
            # Generate filename with timestamp and identifiers
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            viz_filename = f"robot_spawn_planning_{layout_name.replace('/', '_')}_{room_id}_{timestamp}.png"
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
                layout_name=layout_name,
                room_id=room_id,
                save_path=viz_path
            )
        except Exception as e:
            print(f"Warning: Failed to create robot spawn location planning visualization: {e}")
    
    return spawn_pos, spawn_angles_tensor

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


def visualize_object_removal_decision(
    room_bounds, occupancy_grid, grid_x, grid_y, grid_res,
    start_pos, end_pos, all_objects, required_obj_ids, closest_object,
    layout_name="", room_id="",
    save_path=None
):
    """
    Visualize the object removal decision process during trajectory correction.
    
    Shows:
    - Room boundaries and occupancy grid
    - Trajectory line from start to end position
    - All objects in the scene
    - Required objects (highlighted, cannot be removed)
    - Closest object to trajectory line (highlighted, will be removed)
    
    Args:
        room_bounds: tuple of (min_x, min_y, max_x, max_y)
        occupancy_grid: 2D numpy array of boolean occupancy for scene objects
        grid_x, grid_y: arrays defining grid coordinates
        grid_res: grid resolution
        start_pos: start position [x, y] (numpy array)
        end_pos: end position [x, y] (numpy array)
        all_objects: list of all objects in the scene (with position attributes)
        required_obj_ids: set of object IDs that cannot be removed
        closest_object: the object closest to the trajectory line (will be removed)
        layout_name, room_id: strings for labeling
        save_path: path to save the PNG file
    """
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Import required matplotlib components
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    
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
            alpha=0.4,
            aspect='equal'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Occupancy (White=Free, Red=Occupied)', rotation=270, labelpad=20)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Free', 'Occupied'])
    
    # 3. Draw trajectory line from start to end
    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
           'b-', linewidth=4, alpha=0.7, label='Planned Trajectory Line', zorder=3)
    
    # 4. Draw start and end positions
    ax.scatter(*start_pos[:2], c='green', s=300, marker='o', 
              edgecolors='darkgreen', linewidth=3, 
              label='Start Position', zorder=5)
    ax.scatter(*end_pos[:2], c='red', s=300, marker='*', 
              edgecolors='darkred', linewidth=3,
              label='End Position', zorder=5)
    
    # 5. Draw all objects
    # Separate objects into different categories for visualization
    regular_objects = []
    required_objects = []
    
    for obj in all_objects:
        obj_pos = [obj.position.x, obj.position.y]
        
        if obj.id in required_obj_ids:
            required_objects.append(obj_pos)
        else:
            regular_objects.append(obj_pos)
    
    # Draw regular objects (can be removed)
    if regular_objects:
        regular_objects_np = np.array(regular_objects)
        ax.scatter(regular_objects_np[:, 0], regular_objects_np[:, 1], 
                  c='lightblue', s=100, marker='o', 
                  edgecolors='blue', linewidth=1.5, alpha=0.6,
                  label=f'Regular Objects ({len(regular_objects)})', zorder=4)
    
    # Draw required objects (cannot be removed) - highlighted in orange
    if required_objects:
        required_objects_np = np.array(required_objects)
        ax.scatter(required_objects_np[:, 0], required_objects_np[:, 1], 
                  c='orange', s=150, marker='D', 
                  edgecolors='darkorange', linewidth=2, alpha=0.8,
                  label=f'Required Objects (Cannot Remove) ({len(required_objects)})', zorder=6)
    
    # 6. Draw closest object (to be removed) - highlighted in bright red/magenta
    if closest_object is not None:
        closest_pos = [closest_object.position.x, closest_object.position.y]
        ax.scatter(*closest_pos, c='magenta', s=250, marker='X', 
                  edgecolors='darkred', linewidth=3, alpha=0.9,
                  label=f'Object to Remove: {closest_object.type} ({closest_object.id})', zorder=7)
        
        # Draw line from object to closest point on trajectory
        # Calculate closest point on line segment
        line_vec = end_pos - start_pos
        point_vec = np.array(closest_pos) - start_pos
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq > 0:
            t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
            closest_point_on_line = start_pos + t * line_vec
            
            # Draw perpendicular line to show distance
            ax.plot([closest_pos[0], closest_point_on_line[0]], 
                   [closest_pos[1], closest_point_on_line[1]], 
                   '--', color='magenta', linewidth=2, alpha=0.7, zorder=4)
            
            # Calculate and display distance
            distance = np.linalg.norm(np.array(closest_pos) - closest_point_on_line)
            mid_point = [(closest_pos[0] + closest_point_on_line[0]) / 2,
                        (closest_pos[1] + closest_point_on_line[1]) / 2]
            ax.text(mid_point[0], mid_point[1], f'd={distance:.2f}m', 
                   fontsize=10, color='darkred', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                   zorder=8)
    
    # 7. Add legend and labels
    ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left', fontsize=11)
    ax.set_xlabel('X Position (m)', fontsize=13)
    ax.set_ylabel('Y Position (m)', fontsize=13)
    
    # Title with information
    title = f'Object Removal Decision Visualization\n'
    title += f'Layout: {layout_name}, Room: {room_id}\n'
    if closest_object:
        title += f'Removing: {closest_object.type} (ID: {closest_object.id})'
    ax.set_title(title, fontsize=14, pad=20)
    
    # 8. Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 9. Add text annotations with statistics
    trajectory_length = np.linalg.norm(end_pos - start_pos)
    
    info_text = (f"Grid Resolution: {grid_res:.3f}m\n"
                f"Room Size: {room_max_x-room_min_x:.2f}×{room_max_y-room_min_y:.2f}m\n"
                f"Trajectory Length: {trajectory_length:.2f}m\n"
                f"Total Objects: {len(all_objects)}\n"
                f"Required Objects: {len(required_obj_ids)}\n"
                f"Removable Objects: {len(all_objects) - len(required_obj_ids)}")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
           verticalalignment='top', fontsize=11)
    
    # 10. Tight layout and save with extra padding for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Object removal visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plan_robot_traj(start_pos, start_quat, end_pos, end_quat,
    scene_save_dir, layout_name, room_id, debug_dir,
    max_move_distance_per_step = 0.02,
    max_rotate_degree_per_step = 2.0,
    min_move_distance_per_step = 0.01,
    min_rotate_degree_per_step = 1.0,
    max_length=75,
    return_visualization_data=False,
    return_plan_status=False
):

    """
    plan the robot traj from start_pos, start_quat to end_pos, end_quat
    pos and quat are support_point_pos and support_point_quat

    plan the traj with RRT
    you need to ensure no collisions between the robot and the environment

    Uses unified collision checking configuration for consistency with sampling functions.

    you need to conside max_move_distance_per_step and max_rotate_degree_per_step

    Args:
        start_pos: Starting position [x, y, z]
        start_quat: Starting quaternion [w, x, y, z] 
        end_pos: End position [x, y, z]
        end_quat: End quaternion [w, x, y, z]
        scene_save_dir: Directory containing scene data
        layout_name: Name of the layout
        room_id: ID of the room
        debug_dir: Directory for debug visualizations
        max_move_distance_per_step: Maximum movement per step
        max_rotate_degree_per_step: Maximum rotation per step (degrees)
        min_move_distance_per_step: Minimum movement per step
        min_rotate_degree_per_step: Minimum rotation per step (degrees)
        max_length: Maximum trajectory length
        return_visualization_data: Whether to return visualization data
        return_plan_status: Whether to return planning success status
        
    Returns:
        If return_visualization_data=True and return_plan_status=True:
            (trajectory_tensor, visualization_data, plan_successful)
        If return_visualization_data=True:
            (trajectory_tensor, visualization_data)
        If return_plan_status=True:
            (trajectory_tensor, plan_successful)
        Otherwise:
            trajectory_tensor
            
        Where:
        - trajectory_tensor: torch tensor of shape (num_steps, 7) [x, y, z, qw, qx, qy, qz]
        - visualization_data: dict with visualization information (if requested)
        - plan_successful: bool, True if RRT succeeded, False if fallback used (if requested)
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
    
    # RRT-Connect Algorithm
    print(f"Planning robot trajectory from {start_pos_np[:2]} to {end_pos_np[:2]} (RRT-Connect)")
    
    # Tree structure: [pos, quat, parent_index] - use lists for faster appending
    tree_positions = [start_pos_np]
    tree_quaternions = [start_quat_np] 
    tree_parents = [-1]
    
    # Accurate collision checks for start/goal (critical)
    if check_collision_unified(start_pos_np, start_quat_np):
        print("Warning: Start pose is in collision!")
        fallback_trajectory = torch.tensor([[*start_pos_np, *start_quat_np]], dtype=torch.float, device="cuda")
        if return_visualization_data and return_plan_status:
            return fallback_trajectory, None, False
        elif return_visualization_data:
            return fallback_trajectory, None
        elif return_plan_status:
            return fallback_trajectory, False
        else:
            return fallback_trajectory
    
    if check_collision_unified(end_pos_np, end_quat_np):
        print("Warning: Goal pose is in collision!")
        fallback_trajectory = torch.tensor([[*start_pos_np, *start_quat_np]], dtype=torch.float, device="cuda")
        if return_visualization_data and return_plan_status:
            return fallback_trajectory, None, False
        elif return_visualization_data:
            return fallback_trajectory, None
        elif return_plan_status:
            return fallback_trajectory, False
        else:
            return fallback_trajectory
    
    if is_goal_reached_fast(start_pos_np, start_quat_np, end_pos_np, end_quat_np):
        print("Already at goal!")
        fallback_trajectory = torch.tensor([[*start_pos_np, *start_quat_np]], dtype=torch.float, device="cuda")
        if return_visualization_data and return_plan_status:
            return fallback_trajectory, None, True  # This is success case - already at goal
        elif return_visualization_data:
            return fallback_trajectory, None
        elif return_plan_status:
            return fallback_trajectory, True  # This is success case - already at goal
        else:
            return fallback_trajectory
    
    goal_found = False
    goal_node_idx = -1
    plan_successful = False  # Track whether RRT planning succeeded without fallback
    
    # Pre-allocate arrays for random sampling (avoid repeated allocation)
    room_bounds_sampling = np.array([
        room_min_x + CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE,
        room_min_y + CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE,
        room_max_x - CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE,
        room_max_y - CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE
    ])
    
    # RRT-Connect: Initialize second tree from goal
    tree_goal_positions = [end_pos_np]
    tree_goal_quaternions = [end_quat_np]
    tree_goal_parents = [-1]
    
    # Helper function to find nearest node in a tree
    def find_nearest_in_tree(tree_pos, tree_quat, target_pos, target_quat):
        """Find nearest node in tree to target."""
        min_dist = float('inf')
        nearest_idx = 0
        
        if len(tree_pos) < 100:
            for i, (node_pos, node_quat) in enumerate(zip(tree_pos, tree_quat)):
                pos_dist, rot_dist = distance_pos_rot_fast(node_pos, node_quat, target_pos, target_quat)
                total_dist = pos_dist + 0.5 * rot_dist
                if total_dist < min_dist:
                    min_dist = total_dist
                    nearest_idx = i
        else:
            positions_array = np.array(tree_pos)
            pos_distances = np.linalg.norm(positions_array[:, :2] - target_pos[:2], axis=1)
            nearest_candidates = np.argsort(pos_distances)[:5]
            
            for i in nearest_candidates:
                pos_dist, rot_dist = distance_pos_rot_fast(tree_pos[i], tree_quat[i], target_pos, target_quat)
                total_dist = pos_dist + 0.5 * rot_dist
                if total_dist < min_dist:
                    min_dist = total_dist
                    nearest_idx = i
        
        return nearest_idx
    
    # Helper function to extend tree towards target
    def extend_tree(tree_pos, tree_quat, tree_parents, target_pos, target_quat):
        """Extend tree towards target. Returns (success, new_node_idx, new_pos, new_quat)."""
        nearest_idx = find_nearest_in_tree(tree_pos, tree_quat, target_pos, target_quat)
        nearest_pos = tree_pos[nearest_idx]
        nearest_quat = tree_quat[nearest_idx]
        
        # Steer towards target
        new_pos, new_quat = steer_fast(nearest_pos, nearest_quat, target_pos, target_quat)
        
        # Check collision
        pos_progress = np.linalg.norm(new_pos[:2] - nearest_pos[:2])
        if pos_progress > step_size_pos * 0.05:
            collision = check_collision_unified(new_pos, new_quat)
            
            if not collision:
                tree_pos.append(new_pos)
                tree_quat.append(new_quat)
                tree_parents.append(nearest_idx)
                new_node_idx = len(tree_pos) - 1
                return True, new_node_idx, new_pos, new_quat
        
        return False, -1, None, None
    
    # Variables to track connection
    connect_start_idx = -1
    connect_goal_idx = -1
    
    for iteration in range(max_iterations):
        if iteration % 2000 == 0:
            print(f"RRT-Connect iteration {iteration}/{max_iterations}, start_tree: {len(tree_positions)}, goal_tree: {len(tree_goal_positions)}")
        
        # Sample random configuration
        rand_pos = np.array([
            np.random.uniform(room_bounds_sampling[0], room_bounds_sampling[2]),
            np.random.uniform(room_bounds_sampling[1], room_bounds_sampling[3]),
            start_pos_np[2]
        ])
        rand_yaw = np.random.uniform(-np.pi, np.pi)
        rand_quat = np.array([np.cos(rand_yaw/2), 0, 0, np.sin(rand_yaw/2)])
        
        # Alternate which tree extends first (but don't swap the tree variables)
        if iteration % 2 == 0:
            # Extend start tree towards random sample
            success, new_idx, new_pos, new_quat = extend_tree(
                tree_positions, tree_quaternions, tree_parents, rand_pos, rand_quat
            )
            
            if success:
                # Try to connect goal tree to the new node in start tree
                target_pos, target_quat = new_pos, new_quat
                
                for _ in range(10):  # Try up to 10 extensions
                    success2, new_idx2, new_pos2, new_quat2 = extend_tree(
                        tree_goal_positions, tree_goal_quaternions, tree_goal_parents,
                        target_pos, target_quat
                    )
                    
                    if not success2:
                        break
                    
                    # Check if trees are connected
                    pos_dist, rot_dist = distance_pos_rot_fast(new_pos2, new_quat2, target_pos, target_quat)
                    
                    if pos_dist < goal_threshold_pos and rot_dist < goal_threshold_rot:
                        # Trees connected!
                        connect_start_idx = new_idx
                        connect_goal_idx = new_idx2
                        goal_found = True
                        plan_successful = True
                        print(f"Trees connected at iteration {iteration}!")
                        break
        else:
            # Extend goal tree towards random sample
            success, new_idx, new_pos, new_quat = extend_tree(
                tree_goal_positions, tree_goal_quaternions, tree_goal_parents, rand_pos, rand_quat
            )
            
            if success:
                # Try to connect start tree to the new node in goal tree
                target_pos, target_quat = new_pos, new_quat
                
                for _ in range(10):  # Try up to 10 extensions
                    success2, new_idx2, new_pos2, new_quat2 = extend_tree(
                        tree_positions, tree_quaternions, tree_parents,
                        target_pos, target_quat
                    )
                    
                    if not success2:
                        break
                    
                    # Check if trees are connected
                    pos_dist, rot_dist = distance_pos_rot_fast(new_pos2, new_quat2, target_pos, target_quat)
                    
                    if pos_dist < goal_threshold_pos and rot_dist < goal_threshold_rot:
                        # Trees connected!
                        connect_start_idx = new_idx2
                        connect_goal_idx = new_idx
                        goal_found = True
                        plan_successful = True
                        print(f"Trees connected at iteration {iteration}!")
                        break
        
        if goal_found:
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
        # Reconstruct path from RRT-Connect (merge two trees)
        # Start tree: tree_positions (from start to connection point)
        # Goal tree: tree_goal_positions (from goal to connection point)
        
        # Reconstruct path from start tree to connection point
        path_from_start = []
        current_idx = connect_start_idx
        while current_idx != -1:
            path_from_start.append((tree_positions[current_idx], tree_quaternions[current_idx]))
            current_idx = tree_parents[current_idx]
        path_from_start.reverse()  # Reverse to get start -> connection
        
        # Reconstruct path from goal tree to connection point
        path_from_goal = []
        current_idx = connect_goal_idx
        while current_idx != -1:
            path_from_goal.append((tree_goal_positions[current_idx], tree_goal_quaternions[current_idx]))
            current_idx = tree_goal_parents[current_idx]
        path_from_goal.reverse()  # Reverse to get goal -> connection, then we'll reverse again
        
        # Build complete trajectory: start -> connection -> goal
        trajectory = []
        
        # Add path from start to connection
        for pos, quat in path_from_start:
            trajectory.append([*pos, *quat])
        
        # Add path from connection to goal (reversed, skipping first if duplicate)
        for i in range(len(path_from_goal) - 1, -1, -1):
            pos, quat = path_from_goal[i]
            
            # Skip if this is a duplicate of the connection point
            if len(trajectory) > 0:
                last_pos = np.array(trajectory[-1][:3])
                if np.linalg.norm(last_pos - pos) < 0.001:
                    continue
            
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
            plan_successful = False  # RRT path had collisions, using fallback
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
        # Convert tree data for visualization (both trees for RRT-Connect)
        tree_for_viz = [(tree_positions[i], tree_quaternions[i], tree_parents[i]) 
                        for i in range(len(tree_positions))] if goal_found else None
        tree_goal_for_viz = [(tree_goal_positions[i], tree_goal_quaternions[i], tree_goal_parents[i]) 
                            for i in range(len(tree_goal_positions))] if goal_found else None
        
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
            'tree_goal_nodes': tree_goal_for_viz,  # Add goal tree for visualization
            'layout_name': layout_name,
            'room_id': room_id
        }
    
    # Create individual visualization (simplified for speed) - only if not returning visualization data
    if True and not return_visualization_data:
        try:
            trajectory_planning_debug_dir = os.path.join(debug_dir, "trajectory_planning")
            os.makedirs(trajectory_planning_debug_dir, exist_ok=True)
            
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            viz_filename = f"trajectory_planning_{layout_name.replace('/', '_')}_{room_id}_{timestamp}.png"
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
    
    if return_visualization_data and return_plan_status:
        return trajectory_tensor, visualization_data, plan_successful
    elif return_visualization_data:
        return trajectory_tensor, visualization_data
    elif return_plan_status:
        return trajectory_tensor, plan_successful
    else:
        return trajectory_tensor

def visualize_pick_object_pose_sampling(
    room_bounds, table_bounds, successful_samples, pick_object, pick_table,
    layout_name="", room_id="", pick_object_name="", pick_table_name="",
    save_path=None
):
    """
    Visualize pick object pose sampling results in 2D.
    
    Args:
        room_bounds: tuple of (min_x, min_y, max_x, max_y) for the room
        table_bounds: tuple of (min_x, min_y, max_x, max_y) for the table
        successful_samples: list of successful sample dictionaries
        pick_object: the pick object with position and dimensions
        pick_table: the pick table object with position
        layout_name, room_id, pick_object_name, pick_table_name: strings for labeling
        save_path: path to save the PNG file
    """
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    table_min_x, table_min_y, table_max_x, table_max_y = table_bounds
    
    # Import required matplotlib components
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    
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
    
    # 2. Draw table boundaries
    table_rect = patches.Rectangle(
        (table_min_x, table_min_y), 
        table_max_x - table_min_x, 
        table_max_y - table_min_y,
        linewidth=2, edgecolor='brown', facecolor='tan', alpha=0.4,
        label='Table Surface'
    )
    ax.add_patch(table_rect)
    
    # 3. Draw table center
    table_center_x = pick_table.position.x
    table_center_y = pick_table.position.y
    ax.scatter(table_center_x, table_center_y, c='brown', s=150, marker='s', 
              edgecolors='darkred', linewidth=2,
              label=f'Table Center ({pick_table_name})')
    
    # 4. Draw original object position
    ax.scatter(pick_object.position.x, pick_object.position.y, 
              c='red', s=120, marker='x', linewidth=3,
              label=f'Original Position ({pick_object_name})')
    
    if len(successful_samples) > 0:
        # 5. Draw successful sample positions
        sample_positions = np.array([[s['position'][0], s['position'][1]] for s in successful_samples])
        
        # Use different colors for different samples
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_positions)))
        
        for i, (pos, color) in enumerate(zip(sample_positions, colors)):
            ax.scatter(pos[0], pos[1], c=[color], s=80, marker='o', 
                      edgecolors='darkgreen', linewidth=2, alpha=0.8,
                      label=f'Sample {i}' if i < 5 else '')  # Only label first 5 to avoid clutter
        
        # 6. Draw clearance zones for arm picking space (show first few to avoid clutter)
        pick_clearance_width = 2.0 * pick_object.dimensions.width
        pick_clearance_length = 2.0 * pick_object.dimensions.length
        
        max_clearance_zones = min(5, len(sample_positions))  # Show max 5 clearance zones
        for i in range(max_clearance_zones):
            pos = sample_positions[i]
            color = colors[i]
            
            # Draw clearance zone rectangle
            clearance_rect = patches.Rectangle(
                (pos[0] - pick_clearance_width/2, pos[1] - pick_clearance_length/2), 
                pick_clearance_width, pick_clearance_length,
                linewidth=1.5, edgecolor=color, facecolor='none', alpha=0.6,
                linestyle='--'
            )
            ax.add_patch(clearance_rect)
            
            # Draw object bounding box at sampled position
            obj_rect = patches.Rectangle(
                (pos[0] - pick_object.dimensions.width/2, pos[1] - pick_object.dimensions.length/2),
                pick_object.dimensions.width, pick_object.dimensions.length,
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.3
            )
            ax.add_patch(obj_rect)
        
        # 7. Add connecting lines from original position to samples
        for i, pos in enumerate(sample_positions):
            ax.plot([pick_object.position.x, pos[0]], [pick_object.position.y, pos[1]], 
                   '--', color='gray', alpha=0.5, linewidth=1)
            
            # Add distance annotation for first few samples
            if i < 3:
                distance = np.linalg.norm([pos[0] - pick_object.position.x, pos[1] - pick_object.position.y])
                mid_x = (pick_object.position.x + pos[0]) / 2
                mid_y = (pick_object.position.y + pos[1]) / 2
                ax.annotate(f'{distance:.2f}m', (mid_x, mid_y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7, color='gray',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # 8. Add legend entries
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]
    
    # Add custom legend entries
    if len(successful_samples) > 0:
        legend_elements.extend([
            Patch(facecolor='none', edgecolor='gray', linestyle='--', alpha=0.6, label='Arm Picking Clearance'),
            Patch(facecolor='gray', alpha=0.3, label='Object Footprint')
        ])
        legend_labels.extend(['Arm Picking Clearance', 'Object Footprint'])
    
    # 9. Add legend and labels
    ax.legend(handles=legend_elements, labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Pick Object Pose Sampling Results\n'
                f'Layout: {layout_name}, Room: {room_id}\n'
                f'Object: {pick_object_name}, Table: {pick_table_name}', 
                fontsize=14, pad=20)
    
    # 10. Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 11. Add text annotations with statistics
    if len(successful_samples) > 0:
        # Calculate some statistics
        sample_positions = np.array([[s['position'][0], s['position'][1]] for s in successful_samples])
        distances_from_original = np.linalg.norm(
            sample_positions - np.array([pick_object.position.x, pick_object.position.y]), 
            axis=1
        )
        
        avg_distance = np.mean(distances_from_original)
        max_distance = np.max(distances_from_original)
        min_distance = np.min(distances_from_original)
        
        info_text = (f"Successful Samples: {len(successful_samples)}\n"
                    f"Object Size: {pick_object.dimensions.width:.2f}×{pick_object.dimensions.length:.2f}×{pick_object.dimensions.height:.2f}m\n"
                    f"Clearance Zone: {2.0 * pick_object.dimensions.width:.2f}×{2.0 * pick_object.dimensions.length:.2f}m\n"
                    f"Distance from Original:\n"
                    f"  Average: {avg_distance:.2f}m\n"
                    f"  Range: {min_distance:.2f}m - {max_distance:.2f}m")
    else:
        info_text = (f"Successful Samples: 0\n"
                    f"Object Size: {pick_object.dimensions.width:.2f}×{pick_object.dimensions.length:.2f}×{pick_object.dimensions.height:.2f}m\n"
                    f"No valid poses found")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
           verticalalignment='top', fontsize=10)
    
    # 12. Tight layout and save
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Pick object pose sampling visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def sample_pick_object_pose(
    scene_save_dir, 
    layout_name, 
    room_id, 
    pick_object_id,
    pick_table_id,
    num_samples,
    save_dir,
    debug_dir,
):
    """
    Random sample poses for the pick object on the pick table using efficient batch processing.
    
    Algorithm:
    1. Sample poses in batches of 20
    2. Filter each batch for support and collision constraints
    3. Test remaining poses for reachability 
    4. Select first reachable pose and test with IsaacSim physics
    5. If physics test passes, save sample; otherwise try next batch
    
    This approach minimizes expensive IsaacSim calls by doing cheaper tests first.
    
    Args:
        scene_save_dir: Directory containing scene layout JSON files
        layout_name: Name of layout file (without .json extension) 
        room_id: ID of the room containing the objects
        pick_object_id: ID of the object to sample poses for
        pick_table_id: ID of the table the object should be placed on
        num_samples: Number of successful samples to generate (max 100)
        save_dir: Directory to save generated layout JSON files and visualization
        debug_dir: Directory to save debug visualizations
    
    Returns:
        List of successful sample dictionaries with metadata
    
    Features:
    - Efficient batch processing (20 poses per batch)
    - Arm picking space validation (2x object dimensions clearance)
    - Physics stability testing with IsaacSim
    - Robot reachability validation
    - Comprehensive 2D visualization saved to save_dir
    """
    import json
    from dataclasses import asdict
    from utils import export_layout_to_json, generate_unique_id
    from isaacsim.isaac_mcp.server import (
        create_single_room_layout_scene_from_room,
        simulate_the_scene
    )
    
    # Load the layout
    layout_json_path = os.path.join(scene_save_dir, f"{layout_name}.json")
    with open(layout_json_path, "r") as f:
        layout_info = json.load(f)
    
    floor_plan: FloorPlan = dict_to_floor_plan(layout_info)
    target_room = next(room for room in floor_plan.rooms if room.id == room_id)
    assert target_room is not None, f"target_room {room_id} not found in floor_plan"
    
    # Find pick object and pick table
    pick_object = next(obj for obj in target_room.objects if obj.id == pick_object_id)
    pick_table = next(obj for obj in target_room.objects if obj.id == pick_table_id)
    assert pick_object is not None, f"pick_object {pick_object_id} not found in room"
    assert pick_table is not None, f"pick_table {pick_table_id} not found in room"
    
    print(f"Sampling {num_samples} poses for pick object {pick_object_id} on table {pick_table_id}")
    
    # Get pick table mesh for surface sampling
    try:
        table_mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, pick_table_id)
        if table_mesh_info and table_mesh_info["mesh"] is not None:
            table_mesh = table_mesh_info["mesh"]
        else:
            print(f"Warning: Could not load mesh for table {pick_table_id}")
            return []
    except Exception as e:
        print(f"Warning: Could not load mesh for table {pick_table_id}: {e}")
        return []
    
    # Get pick object mesh for collision testing
    try:
        pick_obj_mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, pick_object_id)
        if pick_obj_mesh_info and pick_obj_mesh_info["mesh"] is not None:
            pick_obj_mesh = pick_obj_mesh_info["mesh"]
        else:
            print(f"Warning: Could not load mesh for pick object {pick_object_id}")
            return []
    except Exception as e:
        print(f"Warning: Could not load mesh for pick object {pick_object_id}: {e}")
        return []
    
    # Create support surface from upward-facing faces of the table
    face_normals = table_mesh.face_normals.reshape(-1, 3)
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)
    up_axis = np.array([0.0, 0.0, 1.0]).reshape(1, 3)
    support_faces_mask = (face_normals @ up_axis.T) > 0.5
    support_faces_idxs = np.where(support_faces_mask)[0]
    
    if support_faces_idxs.size == 0:
        print(f"Warning: No upward-facing surfaces found on table {pick_table_id}")
        return []
    
    support_mesh = table_mesh.submesh([support_faces_idxs], append=True)
    
    # Sample candidate positions on table surface
    sample_count = min(1000, num_samples * 20)  # Sample more candidates than needed
    samples, sample_face_idxs = trimesh.sample.sample_surface(support_mesh, sample_count)
    sample_face_normals = support_mesh.face_normals[sample_face_idxs]
    candidate_positions = samples + sample_face_normals * 0.01  # Slightly above surface
    
    # Filter candidates to ensure they're within table bounds with some margin
    table_vertices_2d = table_mesh.vertices[:, :2]
    table_min_x = np.min(table_vertices_2d[:, 0])
    table_min_y = np.min(table_vertices_2d[:, 1])
    table_max_x = np.max(table_vertices_2d[:, 0])
    table_max_y = np.max(table_vertices_2d[:, 1])
    table_bounds = (table_min_x, table_min_y, table_max_x, table_max_y)
    
    # Add margin from table edges
    margin = CollisionCheckingConfig.MIN_DIST_TO_BOUNDARY
    valid_candidates_mask = (
        (candidate_positions[:, 0] >= table_min_x + margin) &
        (candidate_positions[:, 0] <= table_max_x - margin) &
        (candidate_positions[:, 1] >= table_min_y + margin) &
        (candidate_positions[:, 1] <= table_max_y - margin)
    )
    
    valid_candidates = candidate_positions[valid_candidates_mask]
    
    if len(valid_candidates) == 0:
        print(f"Warning: No valid candidate positions found on table {pick_table_id}")
        return []
    
    print(f"Found {len(valid_candidates)} valid candidate positions on table surface")
    
    # Get other objects on the table for collision testing
    other_table_objects = []
    for obj in target_room.objects:
        if obj.id != pick_object_id and obj.place_id == pick_table_id:
            try:
                mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, obj.id)
                if mesh_info and mesh_info["mesh"] is not None:
                    other_table_objects.append(mesh_info["mesh"])
            except Exception as e:
                print(f"Warning: Could not load mesh for object {obj.id}: {e}")
                continue
    
    # Current pose matrix of pick object
    T_curr = _object_pose_to_matrix(pick_object)
    
    successful_samples = []
    batch_size = 20  # Sample 20 poses per batch
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting {num_samples} iterations of batch sampling (batch size: {batch_size})")
    print(f"Each iteration: sample {batch_size} poses → reachability test → IsaacSim physics test")
    batch_size = 20  # Sample 20 poses per batch
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting {num_samples} iterations of batch sampling (batch size: {batch_size})")
    print(f"Each iteration: sample {batch_size} poses → reachability test → IsaacSim physics test")
    
    # Run exactly num_samples iterations (each with IsaacSim test)
    for iteration in range(num_samples):
        print(f"\n=== Iteration {iteration + 1}/{num_samples} ===")
        
        # Step 1: Sample 20 possible poses for this iteration
        print(f"Step 1: Sampling {batch_size} candidate poses...")
        
        # Randomly sample from valid candidates
        if len(valid_candidates) < batch_size:
            # If not enough candidates, use all and repeat some
            current_batch_candidates = []
            for i in range(batch_size):
                idx = i % len(valid_candidates)
                current_batch_candidates.append(valid_candidates[idx])
            current_batch_candidates = np.array(current_batch_candidates)
        else:
            # Randomly sample batch_size candidates
            batch_indices = np.random.choice(len(valid_candidates), size=batch_size, replace=False)
            current_batch_candidates = valid_candidates[batch_indices]
        
        # Test all poses in the batch for support and collision
        batch_valid_poses = []
        
        for i, placement_location in enumerate(current_batch_candidates):
            # Random rotation around Z-axis (keeping object upright)
            rx = 0
            ry = 0
            rz = np.deg2rad(np.random.uniform(-180.0, 180.0))
            
            # Build candidate transform
            T_candidate = np.eye(4)
            T_candidate[:3, :3] = tf.euler_matrix(rx, ry, rz, axes='sxyz')[:3, :3]
            T_candidate[:3, 3] = placement_location
            
            # Transform the pick object mesh to candidate pose
            transformed_mesh = pick_obj_mesh.copy()
            transformed_mesh.apply_transform(T_candidate @ np.linalg.inv(T_curr))
            
            # Test 1: Support test - all vertices must have support from table
            try:
                transformed_vertices = transformed_mesh.vertices
                ray_origins = transformed_vertices
                ray_directions = np.tile(np.array([0.0, 0.0, -1.0]), (len(transformed_vertices), 1))
                
                _, index_ray, _ = trimesh.ray.ray_pyembree.RayMeshIntersector(table_mesh).intersects_location(
                    ray_origins=ray_origins,
                    ray_directions=ray_directions,
                    multiple_hits=False
                )
                
                # All vertices should have support
                if index_ray.shape[0] < transformed_vertices.shape[0]:
                    continue
                    
                # Check no vertices penetrate table from above
                ray_directions_up = np.tile(np.array([0.0, 0.0, 1.0]), (len(transformed_vertices), 1))
                _, index_ray_up, _ = trimesh.ray.ray_pyembree.RayMeshIntersector(table_mesh).intersects_location(
                    ray_origins=ray_origins,
                    ray_directions=ray_directions_up,
                    multiple_hits=False
                )
                
                if index_ray_up.shape[0] > 0:  # Some vertices penetrate from above
                    continue
                    
            except Exception as e:
                print(f"Warning: Support test failed: {e}")
                continue
            
            # Test 2: Collision test with other objects on table - ensure arm picking space
            collision_detected = False
            if other_table_objects:
                # Define arm picking clearance zone around object center
                pick_clearance_width = 2.0 * pick_object.dimensions.width
                pick_clearance_length = 2.0 * pick_object.dimensions.length
                
                # Get object center position from candidate transform
                object_center = placement_location[:2]  # [x, y] only
                
                # Create clearance zone bounds around object center
                clearance_min_x = object_center[0] - pick_clearance_width / 2
                clearance_max_x = object_center[0] + pick_clearance_width / 2
                clearance_min_y = object_center[1] - pick_clearance_length / 2
                clearance_max_y = object_center[1] + pick_clearance_length / 2
                
                # Check if any other object intersects with the clearance zone
                for other_mesh in other_table_objects:
                    try:
                        # Get bounding box of other object
                        other_bounds = other_mesh.bounds
                        other_min_x, other_min_y = other_bounds[0][:2]
                        other_max_x, other_max_y = other_bounds[1][:2]
                        
                        # Check for 2D bounding box overlap
                        overlap_x = not (other_max_x <= clearance_min_x or other_min_x >= clearance_max_x)
                        overlap_y = not (other_max_y <= clearance_min_y or other_min_y >= clearance_max_y)
                        
                        if overlap_x and overlap_y:
                            # Additional precise check using ray casting
                            clearance_sample_points = []
                            num_samples_per_dim = 10
                            for i in range(num_samples_per_dim):
                                for j in range(num_samples_per_dim):
                                    sample_x = clearance_min_x + (i / (num_samples_per_dim - 1)) * pick_clearance_width
                                    sample_y = clearance_min_y + (j / (num_samples_per_dim - 1)) * pick_clearance_length
                                    clearance_sample_points.append([sample_x, sample_y, placement_location[2] + 0.1])
                            
                            clearance_sample_points = np.array(clearance_sample_points)
                            ray_origins = clearance_sample_points
                            ray_directions = np.tile([0, 0, -1], (len(clearance_sample_points), 1))
                            
                            locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(other_mesh).intersects_location(
                                ray_origins=ray_origins,
                                ray_directions=ray_directions,
                                multiple_hits=False
                            )
                            
                            if len(index_ray) > 0:
                                collision_detected = True
                                break
                    except Exception as e:
                        print(f"Warning: Clearance zone collision test failed: {e}")
                        continue
            
            if collision_detected:
                continue
            
            # This pose passed support and collision tests - add to batch
            batch_valid_poses.append({
                'T_candidate': T_candidate,
                'placement_location': placement_location,
                'new_pos': None,  # Will be calculated later
                'new_rot': None   # Will be calculated later
            })
        
        print(f"Step 1 complete: {len(batch_valid_poses)} poses passed support/collision tests out of {batch_size}")
        
        if len(batch_valid_poses) == 0:
            print(f"Iteration {iteration + 1}: No valid poses found, skipping IsaacSim test")
            continue
        
        # Step 2: Test reachability and choose the first one that passes
        print(f"Step 2: Testing reachability for {len(batch_valid_poses)} valid poses...")
        
        selected_pose = None
        for pose_idx, pose_data in enumerate(batch_valid_poses):
            try:
                # Create modified layout with new object pose
                layout_copy = copy.deepcopy(floor_plan)
                room_copy = next(r for r in layout_copy.rooms if r.id == room_id)
                pick_obj_copy = next(obj for obj in room_copy.objects if obj.id == pick_object_id)
                
                # Update object pose
                new_pos, new_rot = _matrix_to_pose(pose_data['T_candidate'])
                pick_obj_copy.position = new_pos
                pick_obj_copy.rotation = new_rot
                
                # Store calculated pose data
                pose_data['new_pos'] = new_pos
                pose_data['new_rot'] = new_rot
                pose_data['layout_copy'] = layout_copy
                pose_data['room_copy'] = room_copy
                
                # Test reachability - only check the pick object
                pick_obj_to_check = [obj for obj in room_copy.objects if obj.id == pick_object_id]
                is_reachable = _check_reachability(layout_copy, room_copy, pick_obj_to_check, reach_threshold=0.4)
                
                if is_reachable:
                    selected_pose = pose_data
                    print(f"Step 2 complete: Found first reachable pose (pose {pose_idx + 1}/{len(batch_valid_poses)})")
                    break
                    
            except Exception as e:
                print(f"Warning: Reachability test failed for pose {pose_idx + 1}: {e}")
                continue
        
        if selected_pose is None:
            print(f"Iteration {iteration + 1}: No reachable poses found, skipping IsaacSim test")
            continue
        
        # Step 3: Test the selected pose with IsaacSim physics
        print(f"Step 3: Testing selected pose with IsaacSim physics simulation...")
        
        try:

            layout_copy_id = generate_unique_id(f"{layout_name}_pick_sample_{iteration}")

            if PHYSICS_CRITIC_ENABLED:
                # only maintain the objects including the table and objects on top 
                room_copy_selected_pose = selected_pose['room_copy']
            
                # Test physics stability using IsaacSim
                # Create and simulate the room
                room_dict_save_path = os.path.join(scene_save_dir, f"{layout_copy_id}_{room_id}.json")
                with open(room_dict_save_path, "w") as f:
                    json.dump(asdict(room_copy_selected_pose), f)
                
                result_create = create_single_room_layout_scene_from_room(scene_save_dir, room_dict_save_path)
                if not isinstance(result_create, dict) or result_create.get("status") != "success":
                    os.remove(room_dict_save_path)
                    print(f"Iteration {iteration + 1}: Physics test failed - IsaacSim scene creation failed")
                    continue
                
                result_sim = simulate_the_scene()
                if not isinstance(result_sim, dict) or result_sim.get("status") != "success":
                    os.remove(room_dict_save_path)
                    print(f"Iteration {iteration + 1}: Physics test failed - IsaacSim simulation failed")
                    continue
                    
                next_step = result_sim.get("next_step", "") or ""
                is_stable = "stable" in next_step.lower() and "unstable" not in next_step.lower()
                
                # Clean up temporary file
                os.remove(room_dict_save_path)
                
                if not is_stable:
                    print(f"Iteration {iteration + 1}: Physics test failed - pose is unstable")
                    continue

            # Step 4: Physics test passed - save the successful sample
            print(f"✅ Iteration {iteration + 1}: Physics test passed! Saving successful sample...")
            
            sample_filename = f"{len(successful_samples):02d}.json"
            sample_path = os.path.join(save_dir, sample_filename)
            
            with open(sample_path, "w") as f:
                json.dump(asdict(selected_pose['layout_copy']), f, indent=2)
            
            successful_samples.append({
                'sample_id': len(successful_samples),
                'layout_id': layout_copy_id,
                'position': [float(selected_pose['new_pos'].x), float(selected_pose['new_pos'].y), float(selected_pose['new_pos'].z)],
                'rotation': [float(selected_pose['new_rot'].x), float(selected_pose['new_rot'].y), float(selected_pose['new_rot'].z)],
                'file_path': sample_path
            })
            
            print(f"Successfully saved sample {len(successful_samples)}: {sample_filename}")
            
        except Exception as e:
            print(f"Iteration {iteration + 1}: IsaacSim physics test failed: {e}")
            continue
        
    # Create comprehensive visualization and save to save_dir (always create if samples exist)
    if len(successful_samples) > 0:
        try:
            # Get room bounds
            room_min_x = target_room.position.x
            room_min_y = target_room.position.y
            room_max_x = target_room.position.x + target_room.dimensions.width
            room_max_y = target_room.position.y + target_room.dimensions.length
            room_bounds = (room_min_x, room_min_y, room_max_x, room_max_y)
            
            # Create comprehensive visualization and save to save_dir
            viz_filename = f"pick_object_pose_sampling_{pick_object_id}_{pick_table_id}.png"
            viz_path = os.path.join(save_dir, viz_filename)
            
            visualize_pick_object_pose_sampling(
                room_bounds=room_bounds,
                table_bounds=table_bounds,
                successful_samples=successful_samples,
                pick_object=pick_object,
                pick_table=pick_table,
                layout_name=layout_name,
                room_id=room_id,
                pick_object_name=pick_object_id,
                pick_table_name=pick_table_id,
                save_path=viz_path
            )
            
        except Exception as e:
            print(f"Warning: Failed to create pose sampling visualization: {e}")
    
    # Create additional debug visualization if debug_dir is provided
    if debug_dir and len(successful_samples) > 0:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create a copy in debug_dir as well for debugging purposes
            debug_viz_path = os.path.join(debug_dir, f"debug_pick_object_sampling_{pick_object_id}_{pick_table_id}.png")
            
            # Get room bounds
            room_min_x = target_room.position.x
            room_min_y = target_room.position.y
            room_max_x = target_room.position.x + target_room.dimensions.width
            room_max_y = target_room.position.y + target_room.dimensions.length
            room_bounds = (room_min_x, room_min_y, room_max_x, room_max_y)
            
            visualize_pick_object_pose_sampling(
                room_bounds=room_bounds,
                table_bounds=table_bounds,
                successful_samples=successful_samples,
                pick_object=pick_object,
                pick_table=pick_table,
                layout_name=layout_name,
                room_id=room_id,
                pick_object_name=pick_object_id,
                pick_table_name=pick_table_id,
                save_path=debug_viz_path
            )
            
        except Exception as e:
            print(f"Warning: Failed to create debug visualization: {e}")
    
    print(f"\n=== Sample Generation Complete ===")
    print(f"Iterations run: {num_samples}")
    print(f"Successful samples: {len(successful_samples)}")
    print(f"Success rate: {100*len(successful_samples)/max(1,num_samples):.1f}% ({len(successful_samples)}/{num_samples} iterations)")
    print(f"Each iteration tested up to {batch_size} poses for support/collision, then reachability, then 1 IsaacSim test")
    
    return successful_samples

def sample_pick_object_pose_with_mobile_franka_occupancy(
    scene_save_dir, 
    layout_name, 
    room_id, 
    pick_object_id,
    pick_table_id,
    num_samples,
    save_dir,
    debug_dir,
):
    """
    Combine sample_pick_object_pose and sample_robot_location to efficiently sample
    valid pick object poses with corresponding reachable robot base positions.
    
    Algorithm:
    1. Sample pick object poses on table (position + rotation)
    2. For each pose, test support and collision constraints
    3. For valid poses, sample robot positions that can reach the object
    4. Test robot occupancy collision
    5. Filter by reachability (distance < 0.8m)
    6. Return successful (pick_pose, robot_pose) pairs
    
    Args:
        scene_save_dir: Directory containing scene layout JSON files
        layout_name: Name of layout file (without .json extension)
        room_id: ID of the room containing the objects
        pick_object_id: ID of the object to sample poses for
        pick_table_id: ID of the table the object should be placed on
        num_samples: Number of successful samples to generate
        save_dir: Not used (kept for API compatibility)
        debug_dir: Directory to save debug visualizations
    
    Returns:
        List of successful sample dictionaries with:
        - sample_id: Sample identifier
        - pick_position: [x, y, z] of pick object
        - pick_rotation: [rx, ry, rz] of pick object (degrees)
        - position: [x, y, z] of pick object (alias for visualization)
        - rotation: [rx, ry, rz] of pick object (alias for visualization)
        - robot_position: [x, y, z] of robot base
        - robot_quaternion: [w, x, y, z] of robot base
        - robot_distance_to_object: Distance from robot to object (meters)
    """
    import json
    
    # Load the layout
    layout_json_path = os.path.join(scene_save_dir, f"{layout_name}.json")
    with open(layout_json_path, "r") as f:
        layout_info = json.load(f)
    
    floor_plan: FloorPlan = dict_to_floor_plan(layout_info)
    target_room = next(room for room in floor_plan.rooms if room.id == room_id)
    assert target_room is not None, f"target_room {room_id} not found in floor_plan"
    
    # Find pick object and pick table
    pick_object = next(obj for obj in target_room.objects if obj.id == pick_object_id)
    pick_table = next(obj for obj in target_room.objects if obj.id == pick_table_id)
    assert pick_object is not None, f"pick_object {pick_object_id} not found in room"
    assert pick_table is not None, f"pick_table {pick_table_id} not found in room"
    
    print(f"Sampling {num_samples} pick object poses with robot positions for {pick_object_id}")
    
    # Create unified occupancy grid for robot collision checking
    occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, _, _ = create_unified_occupancy_grid(
        scene_save_dir, layout_name, room_id
    )
    scene_occupancy_fn = create_unified_scene_occupancy_fn(occupancy_grid, grid_x, grid_y, room_bounds)
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Get pick table mesh for surface sampling
    try:
        table_mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, pick_table_id)
        if table_mesh_info and table_mesh_info["mesh"] is not None:
            table_mesh = table_mesh_info["mesh"]
        else:
            print(f"Warning: Could not load mesh for table {pick_table_id}")
            return []
    except Exception as e:
        print(f"Warning: Could not load mesh for table {pick_table_id}: {e}")
        return []
    
    # Get pick object mesh for collision testing
    try:
        pick_obj_mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, pick_object_id)
        if pick_obj_mesh_info and pick_obj_mesh_info["mesh"] is not None:
            pick_obj_mesh = pick_obj_mesh_info["mesh"]
        else:
            print(f"Warning: Could not load mesh for pick object {pick_object_id}")
            return []
    except Exception as e:
        print(f"Warning: Could not load mesh for pick object {pick_object_id}: {e}")
        return []
    
    # Create support surface from upward-facing faces of the table
    face_normals = table_mesh.face_normals.reshape(-1, 3)
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)
    up_axis = np.array([0.0, 0.0, 1.0]).reshape(1, 3)
    support_faces_mask = (face_normals @ up_axis.T) > 0.5
    support_faces_idxs = np.where(support_faces_mask)[0]
    
    if support_faces_idxs.size == 0:
        print(f"Warning: No upward-facing surfaces found on table {pick_table_id}")
        return []
    
    support_mesh = table_mesh.submesh([support_faces_idxs], append=True)
    
    # Get table bounds
    table_vertices_2d = table_mesh.vertices[:, :2]
    table_min_x = np.min(table_vertices_2d[:, 0])
    table_min_y = np.min(table_vertices_2d[:, 1])
    table_max_x = np.max(table_vertices_2d[:, 0])
    table_max_y = np.max(table_vertices_2d[:, 1])
    table_bounds = (table_min_x, table_min_y, table_max_x, table_max_y)
    
    # Get other objects on the table for collision testing
    other_table_objects = []
    for obj in target_room.objects:
        if obj.id != pick_object_id and obj.place_id == pick_table_id:
            try:
                mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, obj.id)
                if mesh_info and mesh_info["mesh"] is not None:
                    other_table_objects.append(mesh_info["mesh"])
            except Exception as e:
                print(f"Warning: Could not load mesh for object {obj.id}: {e}")
                continue
    
    # Current pose matrix of pick object
    T_curr = _object_pose_to_matrix(pick_object)
    
    # Calculate table height for robot z position
    if table_mesh and len(table_mesh.vertices) > 0:
        table_height = np.max(table_mesh.vertices[:, 2])
    else:
        table_height = pick_table.position.z + pick_table.dimensions.height
    
    robot_height_offset = 0.20
    robot_z = max(table_height - robot_height_offset, 0)
    
    # Sample robot positions in room (pre-compute for efficiency)
    num_robot_sample_points = 5000
    sample_points = np.random.uniform(
        low=[room_min_x, room_min_y], 
        high=[room_max_x, room_max_y], 
        size=(num_robot_sample_points, 2)
    )
    
    # Filter robot positions by room edges and occupancy
    dist_to_edges = np.minimum.reduce([
        sample_points[:, 0] - room_min_x,
        room_max_x - sample_points[:, 0],
        sample_points[:, 1] - room_min_y,
        room_max_y - sample_points[:, 1]
    ])
    
    edge_valid_mask = dist_to_edges >= CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE
    edge_valid_points = sample_points[edge_valid_mask]
    
    valid_robot_points = []
    if len(edge_valid_points) > 0:
        grid_coords = np.floor((edge_valid_points - [room_min_x, room_min_y]) / CollisionCheckingConfig.GRID_RES).astype(int)
        
        grid_valid_mask = (
            (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < len(grid_x)) &
            (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < len(grid_y)) &
            (~occupancy_grid[grid_coords[:, 0], grid_coords[:, 1]])
        )
        
        grid_valid_points = edge_valid_points[grid_valid_mask]
        
        if len(grid_valid_points) > 0:
            # Check distance to occupied cells
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
                valid_robot_points = grid_valid_points
    
    if len(valid_robot_points) == 0:
        print("Warning: No valid robot positions found in room")
        return []
    
    print(f"Found {len(valid_robot_points)} valid robot positions in room")
    
    # Sample candidate pick object positions on table surface
    sample_count = min(2000, num_samples * 40)
    samples, sample_face_idxs = trimesh.sample.sample_surface(support_mesh, sample_count)
    sample_face_normals = support_mesh.face_normals[sample_face_idxs]
    candidate_positions = samples + sample_face_normals * 0.01
    
    # Filter candidates within table bounds with margin
    margin = CollisionCheckingConfig.MIN_DIST_TO_BOUNDARY
    valid_candidates_mask = (
        (candidate_positions[:, 0] >= table_min_x + margin) &
        (candidate_positions[:, 0] <= table_max_x - margin) &
        (candidate_positions[:, 1] >= table_min_y + margin) &
        (candidate_positions[:, 1] <= table_max_y - margin)
    )
    
    valid_candidates = candidate_positions[valid_candidates_mask]
    
    if len(valid_candidates) == 0:
        print(f"Warning: No valid candidate positions found on table {pick_table_id}")
        return []
    
    print(f"Found {len(valid_candidates)} valid candidate positions on table surface")
    
    successful_samples = []
    
    batch_size = 20
    max_reachability_distance = 0.8  # Distance threshold from pose_aug_with_mm_v0.py
    
    print(f"Starting sampling with batches of {batch_size} poses per iteration...")
    
    iteration = 0
    while len(successful_samples) < num_samples:
        iteration += 1
        print(f"\n=== Iteration {iteration} (samples: {len(successful_samples)}/{num_samples}) ===")
        
        # Sample batch of candidate poses
        if len(valid_candidates) < batch_size:
            batch_candidates = []
            for i in range(batch_size):
                idx = i % len(valid_candidates)
                batch_candidates.append(valid_candidates[idx])
            batch_candidates = np.array(batch_candidates)
        else:
            batch_indices = np.random.choice(len(valid_candidates), size=batch_size, replace=False)
            batch_candidates = valid_candidates[batch_indices]
        
        # Test each pose in batch for support and collision
        batch_valid_poses = []
        
        for placement_location in batch_candidates:
            # Random rotation around Z-axis
            rz = np.deg2rad(np.random.uniform(-180.0, 180.0))
            T_candidate = np.eye(4)
            T_candidate[:3, :3] = tf.euler_matrix(0, 0, rz, axes='sxyz')[:3, :3]
            T_candidate[:3, 3] = placement_location
            
            # Transform pick object mesh to candidate pose
            transformed_mesh = pick_obj_mesh.copy()
            transformed_mesh.apply_transform(T_candidate @ np.linalg.inv(T_curr))
            
            # Support test
            try:
                transformed_vertices = transformed_mesh.vertices
                ray_origins = transformed_vertices
                ray_directions = np.tile(np.array([0.0, 0.0, -1.0]), (len(transformed_vertices), 1))
                
                _, index_ray, _ = trimesh.ray.ray_pyembree.RayMeshIntersector(table_mesh).intersects_location(
                    ray_origins=ray_origins,
                    ray_directions=ray_directions,
                    multiple_hits=False
                )
                
                if index_ray.shape[0] < transformed_vertices.shape[0]:
                    continue
                
                ray_directions_up = np.tile(np.array([0.0, 0.0, 1.0]), (len(transformed_vertices), 1))
                _, index_ray_up, _ = trimesh.ray.ray_pyembree.RayMeshIntersector(table_mesh).intersects_location(
                    ray_origins=ray_origins,
                    ray_directions=ray_directions_up,
                    multiple_hits=False
                )
                
                if index_ray_up.shape[0] > 0:
                    continue
            except Exception as e:
                continue
            
            # Collision test with arm clearance
            collision_detected = False
            if other_table_objects:
                pick_clearance_width = 2.0 * pick_object.dimensions.width
                pick_clearance_length = 2.0 * pick_object.dimensions.length
                
                object_center = placement_location[:2]
                clearance_min_x = object_center[0] - pick_clearance_width / 2
                clearance_max_x = object_center[0] + pick_clearance_width / 2
                clearance_min_y = object_center[1] - pick_clearance_length / 2
                clearance_max_y = object_center[1] + pick_clearance_length / 2
                
                for other_mesh in other_table_objects:
                    try:
                        other_bounds = other_mesh.bounds
                        other_min_x, other_min_y = other_bounds[0][:2]
                        other_max_x, other_max_y = other_bounds[1][:2]
                        
                        overlap_x = not (other_max_x <= clearance_min_x or other_min_x >= clearance_max_x)
                        overlap_y = not (other_max_y <= clearance_min_y or other_min_y >= clearance_max_y)
                        
                        if overlap_x and overlap_y:
                            collision_detected = True
                            break
                    except Exception as e:
                        continue
            
            if collision_detected:
                continue
            
            # Pose passed support and collision tests
            batch_valid_poses.append({
                'T_candidate': T_candidate,
                'placement_location': placement_location
            })
        
        print(f"  {len(batch_valid_poses)} poses passed support/collision tests")
        
        if len(batch_valid_poses) == 0:
            continue
        
        # For each valid pose, find reachable robot position
        for pose_data in batch_valid_poses:
            T_candidate = pose_data['T_candidate']
            placement_location = pose_data['placement_location']
            
            # Calculate new object position
            new_pos, new_rot = _matrix_to_pose(T_candidate)
            pick_object_xy = np.array([new_pos.x, new_pos.y])
            
            # Find robot positions within reachability distance
            distances_to_object = np.linalg.norm(valid_robot_points - pick_object_xy, axis=1)
            reachable_mask = distances_to_object <= max_reachability_distance
            reachable_robot_points = valid_robot_points[reachable_mask]
            reachable_distances = distances_to_object[reachable_mask]
            
            if len(reachable_robot_points) == 0:
                continue
            
            # Sort by distance (closest first)
            sorted_indices = np.argsort(reachable_distances)
            
            # Try robot positions until we find one without collision
            robot_found = False
            for idx in sorted_indices:
                robot_pos_2d = reachable_robot_points[idx]
                
                # Calculate robot orientation toward object
                direction_to_object = pick_object_xy - robot_pos_2d
                yaw = np.arctan2(direction_to_object[1], direction_to_object[0])
                
                # Create quaternion
                robot_quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
                robot_pos_3d = np.array([robot_pos_2d[0], robot_pos_2d[1], 0])
                
                # Check robot collision
                collision = check_unified_robot_collision(robot_pos_3d, robot_quat, scene_occupancy_fn, room_bounds)
                
                if not collision:
                    # Found valid (pose, robot) pair!
                    robot_distance = reachable_distances[idx]
                    
                    # Optional IsaacSim physics validation (currently disabled)
                    if True:
                        from dataclasses import asdict
                        from utils import generate_unique_id
                        from isaacsim.isaac_mcp.server import (
                            create_single_room_layout_scene_from_room,
                            simulate_the_scene
                        )
                        
                        # Create layout copy with updated pick object pose
                        layout_copy = copy.deepcopy(floor_plan)
                        room_copy = next(r for r in layout_copy.rooms if r.id == room_id)
                        pick_obj_copy = next(obj for obj in room_copy.objects if obj.id == pick_object_id)
                        
                        pick_obj_copy.position = new_pos
                        pick_obj_copy.rotation = new_rot
                        
                        layout_copy_id = generate_unique_id(f"{layout_name}_pick_sample_{len(successful_samples)}")
                        
                        # Test physics stability using IsaacSim
                        room_dict_save_path = os.path.join(scene_save_dir, f"{layout_copy_id}_{room_id}.json")
                        with open(room_dict_save_path, "w") as f:
                            json.dump(asdict(room_copy), f)
                        
                        result_create = create_single_room_layout_scene_from_room(scene_save_dir, room_dict_save_path)
                        if not isinstance(result_create, dict) or result_create.get("status") != "success":
                            os.remove(room_dict_save_path)
                            print(f"  Physics test failed - IsaacSim scene creation failed")
                            continue
                        
                        result_sim = simulate_the_scene()
                        if not isinstance(result_sim, dict) or result_sim.get("status") != "success":
                            os.remove(room_dict_save_path)
                            print(f"  Physics test failed - IsaacSim simulation failed")
                            continue
                        
                        next_step = result_sim.get("next_step", "") or ""
                        is_stable = "stable" in next_step.lower() and "unstable" not in next_step.lower()
                        
                        # Clean up temporary file
                        os.remove(room_dict_save_path)
                        
                        if not is_stable:
                            print(f"  Physics test failed - pose is unstable")
                            continue
                    
                    successful_samples.append({
                        'sample_id': len(successful_samples),
                        'pick_position': [float(new_pos.x), float(new_pos.y), float(new_pos.z)],
                        'pick_rotation': [float(new_rot.x), float(new_rot.y), float(new_rot.z)],
                        'position': [float(new_pos.x), float(new_pos.y), float(new_pos.z)],  # Alias for visualization compatibility
                        'rotation': [float(new_rot.x), float(new_rot.y), float(new_rot.z)],  # Alias for visualization compatibility
                        'robot_position': [float(robot_pos_2d[0]), float(robot_pos_2d[1]), float(robot_z)],
                        'robot_quaternion': [float(robot_quat[0]), float(robot_quat[1]), float(robot_quat[2]), float(robot_quat[3])],
                        'robot_distance_to_object': float(robot_distance)
                    })
                    
                    print(f"  ✅ Sample {len(successful_samples)}: robot dist={robot_distance:.2f}m")
                    robot_found = True
                    break
            
            if robot_found and len(successful_samples) >= num_samples:
                break
        
        if len(successful_samples) >= num_samples:
            break
    
    # Create visualization
    if len(successful_samples) > 0 and debug_dir:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            
            # Visualize pick object poses and robot positions
            viz_filename = f"pick_object_with_robot_{pick_object_id}_{pick_table_id}.png"
            viz_path = os.path.join(debug_dir, viz_filename)
            
            room_bounds_viz = (
                target_room.position.x,
                target_room.position.y,
                target_room.position.x + target_room.dimensions.width,
                target_room.position.y + target_room.dimensions.length
            )
            
            visualize_pick_object_pose_sampling(
                room_bounds=room_bounds_viz,
                table_bounds=table_bounds,
                successful_samples=successful_samples,
                pick_object=pick_object,
                pick_table=pick_table,
                layout_name=layout_name,
                room_id=room_id,
                pick_object_name=pick_object_id,
                pick_table_name=pick_table_id,
                save_path=viz_path
            )
        except Exception as e:
            print(f"Warning: Failed to create visualization: {e}")
    
    print(f"\n=== Sampling Complete ===")
    print(f"Successful samples: {len(successful_samples)}/{num_samples}")
    print(f"Average robot distance: {np.mean([s['robot_distance_to_object'] for s in successful_samples]):.2f}m")
    
    return successful_samples

def sample_place_object_pose_with_mobile_franka_occupancy(
    scene_save_dir, 
    layout_name, 
    room_id, 
    place_object_id,
    place_table_id,
    num_samples,
    save_dir,
    debug_dir,
):
    """
    Sample valid place locations with robot positions for placing objects.
    
    Two modes:
    1. If place_object_id == place_table_id: Place directly on table
       - Use sample_robot_place_location to find robot pose and place location on table
       
    2. If place_object_id != place_table_id: Place on top of another object
       - Sample place object pose on table with physics validation
       - Place location is center top of the place object
       - Sample robot positions that can reach the place location
       - Filter by reachability (distance < 0.8m)
    
    Args:
        scene_save_dir: Directory containing scene layout JSON files
        layout_name: Name of layout file (without .json extension)
        room_id: ID of the room containing the objects
        place_object_id: ID of object to place on (can be table or another object)
        place_table_id: ID of the table (used when place_object_id is the table itself)
        num_samples: Number of successful samples to generate
        save_dir: Not used (kept for API compatibility)
        debug_dir: Directory to save debug visualizations
    
    Returns:
        List of successful sample dictionaries:
        - If place_object_id == place_table_id:
            - sample_id, robot_position, robot_quaternion, place_location, robot_distance_to_place
        - If place_object_id != place_table_id:
            - sample_id, robot_position, robot_quaternion, place_location, robot_distance_to_place,
              place_object_position, place_object_rotation
    """
    import json
    from scipy.spatial.transform import Rotation as R
    
    # Load the layout
    layout_json_path = os.path.join(scene_save_dir, f"{layout_name}.json")
    with open(layout_json_path, "r") as f:
        layout_info = json.load(f)
    
    floor_plan: FloorPlan = dict_to_floor_plan(layout_info)
    target_room = next(room for room in floor_plan.rooms if room.id == room_id)
    assert target_room is not None, f"target_room {room_id} not found in floor_plan"
    
    # Case 1: Placing directly on table (place_object_id == place_table_id)
    if place_object_id == place_table_id:
        print(f"Placing directly on table {place_table_id}")
        
        # Use sample_robot_place_location
        robot_base_pos, robot_base_quat, place_locations = sample_robot_place_location(
            scene_save_dir, layout_name, room_id, place_table_id, num_samples, debug_dir
        )
        
        # Convert to numpy for processing
        robot_positions_np = robot_base_pos.cpu().numpy() if hasattr(robot_base_pos, 'cpu') else np.array(robot_base_pos)
        robot_quats_np = robot_base_quat.cpu().numpy() if hasattr(robot_base_quat, 'cpu') else np.array(robot_base_quat)
        place_locations_np = place_locations.cpu().numpy() if hasattr(place_locations, 'cpu') else np.array(place_locations)
        
        # Filter by reachability distance (< 0.8m)
        successful_samples = []
        max_reachability_distance = 0.8
        
        for i in range(len(robot_positions_np)):
            robot_pos_xy = robot_positions_np[i][:2]
            place_loc_xy = place_locations_np[i][:2]
            
            distance = np.linalg.norm(robot_pos_xy - place_loc_xy)
            
            if distance <= max_reachability_distance:
                successful_samples.append({
                    'sample_id': len(successful_samples),
                    'robot_position': robot_positions_np[i].tolist(),
                    'robot_quaternion': robot_quats_np[i].tolist(),
                    'place_location': place_locations_np[i].tolist(),
                    'robot_distance_to_place': float(distance)
                })
                print(f"  ✅ Sample {len(successful_samples)}: robot dist={distance:.2f}m")
        
        print(f"\n=== Sampling Complete (Table Placement) ===")
        print(f"Successful samples: {len(successful_samples)}/{num_samples}")
        if successful_samples:
            print(f"Average robot distance: {np.mean([s['robot_distance_to_place'] for s in successful_samples]):.2f}m")
        
        return successful_samples
    
    # Case 2: Placing on top of another object
    print(f"Placing on top of object {place_object_id}")
    
    place_object = next(obj for obj in target_room.objects if obj.id == place_object_id)
    assert place_object is not None, f"place_object {place_object_id} not found in room"
    
    # Create unified occupancy grid for robot collision checking
    occupancy_grid, grid_x, grid_y, room_bounds, combined_mesh, _, _ = create_unified_occupancy_grid(
        scene_save_dir, layout_name, room_id
    )
    scene_occupancy_fn = create_unified_scene_occupancy_fn(occupancy_grid, grid_x, grid_y, room_bounds)
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Get place object mesh
    try:
        place_obj_mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, place_object_id)
        if place_obj_mesh_info and place_obj_mesh_info["mesh"] is not None:
            place_obj_mesh = place_obj_mesh_info["mesh"]
        else:
            print(f"Warning: Could not load mesh for place object {place_object_id}")
            return []
    except Exception as e:
        print(f"Warning: Could not load mesh for place object {place_object_id}: {e}")
        return []
    
    # Get table mesh (the surface the place object sits on)
    table_for_place_object = place_object.place_id
    if not table_for_place_object:
        print(f"Warning: place_object {place_object_id} has no place_id (table)")
        return []
    
    try:
        table_mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, table_for_place_object)
        if table_mesh_info and table_mesh_info["mesh"] is not None:
            table_mesh = table_mesh_info["mesh"]
        else:
            print(f"Warning: Could not load mesh for table {table_for_place_object}")
            return []
    except Exception as e:
        print(f"Warning: Could not load mesh for table {table_for_place_object}: {e}")
        return []
    
    # Calculate table height for robot z position
    if table_mesh and len(table_mesh.vertices) > 0:
        table_height = np.max(table_mesh.vertices[:, 2])
    else:
        table_obj = next(obj for obj in target_room.objects if obj.id == table_for_place_object)
        table_height = table_obj.position.z + table_obj.dimensions.height
    
    robot_height_offset = 0.20
    robot_z = max(table_height - robot_height_offset, 0)
    
    # Get other objects on the table for collision testing
    other_table_objects = []
    for obj in target_room.objects:
        if obj.id != place_object_id and obj.place_id == table_for_place_object:
            try:
                mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, obj.id)
                if mesh_info and mesh_info["mesh"] is not None:
                    other_table_objects.append(mesh_info["mesh"])
            except Exception as e:
                print(f"Warning: Could not load mesh for object {obj.id}: {e}")
                continue
    
    # Sample valid robot positions in room (pre-compute)
    num_robot_sample_points = 5000
    sample_points = np.random.uniform(
        low=[room_min_x, room_min_y], 
        high=[room_max_x, room_max_y], 
        size=(num_robot_sample_points, 2)
    )
    
    # Filter robot positions
    dist_to_edges = np.minimum.reduce([
        sample_points[:, 0] - room_min_x,
        room_max_x - sample_points[:, 0],
        sample_points[:, 1] - room_min_y,
        room_max_y - sample_points[:, 1]
    ])
    
    edge_valid_mask = dist_to_edges >= CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE
    edge_valid_points = sample_points[edge_valid_mask]
    
    valid_robot_points = []
    if len(edge_valid_points) > 0:
        grid_coords = np.floor((edge_valid_points - [room_min_x, room_min_y]) / CollisionCheckingConfig.GRID_RES).astype(int)
        
        grid_valid_mask = (
            (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < len(grid_x)) &
            (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < len(grid_y)) &
            (~occupancy_grid[grid_coords[:, 0], grid_coords[:, 1]])
        )
        
        grid_valid_points = edge_valid_points[grid_valid_mask]
        
        if len(grid_valid_points) > 0:
            occupied_indices = np.where(occupancy_grid)
            if len(occupied_indices[0]) > 0:
                occupied_positions = np.column_stack([
                    room_min_x + occupied_indices[0] * CollisionCheckingConfig.GRID_RES + CollisionCheckingConfig.GRID_RES/2,
                    room_min_y + occupied_indices[1] * CollisionCheckingConfig.GRID_RES + CollisionCheckingConfig.GRID_RES/2
                ])
                
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
                valid_robot_points = grid_valid_points
    
    if len(valid_robot_points) == 0:
        print("Warning: No valid robot positions found in room")
        return []
    
    print(f"Found {len(valid_robot_points)} valid robot positions in room")
    
    # Sample candidate place object positions on table (using current object as template)
    # Create support surface from upward-facing faces
    face_normals = table_mesh.face_normals.reshape(-1, 3)
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)
    up_axis = np.array([0.0, 0.0, 1.0]).reshape(1, 3)
    support_faces_mask = (face_normals @ up_axis.T) > 0.5
    support_faces_idxs = np.where(support_faces_mask)[0]
    
    if support_faces_idxs.size == 0:
        print(f"Warning: No upward-facing surfaces found on table {table_for_place_object}")
        return []
    
    support_mesh = table_mesh.submesh([support_faces_idxs], append=True)
    
    # Sample positions
    sample_count = min(2000, num_samples * 40)
    samples, sample_face_idxs = trimesh.sample.sample_surface(support_mesh, sample_count)
    sample_face_normals = support_mesh.face_normals[sample_face_idxs]
    candidate_positions = samples + sample_face_normals * 0.01
    
    # Get table bounds
    table_vertices_2d = table_mesh.vertices[:, :2]
    table_min_x = np.min(table_vertices_2d[:, 0])
    table_min_y = np.min(table_vertices_2d[:, 1])
    table_max_x = np.max(table_vertices_2d[:, 0])
    table_max_y = np.max(table_vertices_2d[:, 1])
    
    # Filter candidates within table bounds
    margin = CollisionCheckingConfig.MIN_DIST_TO_BOUNDARY
    valid_candidates_mask = (
        (candidate_positions[:, 0] >= table_min_x + margin) &
        (candidate_positions[:, 0] <= table_max_x - margin) &
        (candidate_positions[:, 1] >= table_min_y + margin) &
        (candidate_positions[:, 1] <= table_max_y - margin)
    )
    
    valid_candidates = candidate_positions[valid_candidates_mask]
    
    if len(valid_candidates) == 0:
        print(f"Warning: No valid candidate positions found on table")
        return []
    
    print(f"Found {len(valid_candidates)} valid candidate positions on table surface")
    
    # Current pose matrix of place object
    T_curr = _object_pose_to_matrix(place_object)
    
    successful_samples = []
    batch_size = 20
    max_reachability_distance = 0.8
    
    print(f"Starting sampling with batches of {batch_size} poses per iteration...")
    
    iteration = 0
    while len(successful_samples) < num_samples:
        iteration += 1
        print(f"\n=== Iteration {iteration} (samples: {len(successful_samples)}/{num_samples}) ===")
        
        # Sample batch of candidate poses
        if len(valid_candidates) < batch_size:
            batch_candidates = []
            for i in range(batch_size):
                idx = i % len(valid_candidates)
                batch_candidates.append(valid_candidates[idx])
            batch_candidates = np.array(batch_candidates)
        else:
            batch_indices = np.random.choice(len(valid_candidates), size=batch_size, replace=False)
            batch_candidates = valid_candidates[batch_indices]
        
        # Test each pose in batch for support and collision
        batch_valid_poses = []
        
        for placement_location in batch_candidates:
            # Random rotation around Z-axis
            rz = np.deg2rad(np.random.uniform(-180.0, 180.0))
            T_candidate = np.eye(4)
            T_candidate[:3, :3] = tf.euler_matrix(0, 0, rz, axes='sxyz')[:3, :3]
            T_candidate[:3, 3] = placement_location
            
            # Transform place object mesh
            transformed_mesh = place_obj_mesh.copy()
            transformed_mesh.apply_transform(T_candidate @ np.linalg.inv(T_curr))
            
            # Support test
            try:
                transformed_vertices = transformed_mesh.vertices
                ray_origins = transformed_vertices
                ray_directions = np.tile(np.array([0.0, 0.0, -1.0]), (len(transformed_vertices), 1))
                
                _, index_ray, _ = trimesh.ray.ray_pyembree.RayMeshIntersector(table_mesh).intersects_location(
                    ray_origins=ray_origins,
                    ray_directions=ray_directions,
                    multiple_hits=False
                )
                
                if index_ray.shape[0] < transformed_vertices.shape[0]:
                    continue
                
                ray_directions_up = np.tile(np.array([0.0, 0.0, 1.0]), (len(transformed_vertices), 1))
                _, index_ray_up, _ = trimesh.ray.ray_pyembree.RayMeshIntersector(table_mesh).intersects_location(
                    ray_origins=ray_origins,
                    ray_directions=ray_directions_up,
                    multiple_hits=False
                )
                
                if index_ray_up.shape[0] > 0:
                    continue
            except Exception as e:
                continue
            
            # Collision test
            collision_detected = False
            if other_table_objects:
                for other_mesh in other_table_objects:
                    try:
                        collision_manager = trimesh.collision.CollisionManager()
                        collision_manager.add_object('other', other_mesh)
                        
                        if collision_manager.in_collision_single(transformed_mesh):
                            collision_detected = True
                            break
                    except Exception as e:
                        continue
            
            if collision_detected:
                continue
            
            # Pose passed support and collision tests
            batch_valid_poses.append({
                'T_candidate': T_candidate,
                'placement_location': placement_location
            })
        
        print(f"  {len(batch_valid_poses)} poses passed support/collision tests")
        
        if len(batch_valid_poses) == 0:
            continue
        
        # For each valid pose, find reachable robot position
        for pose_data in batch_valid_poses:
            T_candidate = pose_data['T_candidate']
            
            # Calculate new object position
            new_pos, new_rot = _matrix_to_pose(T_candidate)
            
            # Place location is center top of the place object
            # Get top center of transformed mesh
            transformed_mesh = place_obj_mesh.copy()
            transformed_mesh.apply_transform(T_candidate @ np.linalg.inv(T_curr))
            
            top_z = np.max(transformed_mesh.vertices[:, 2])
            place_location = np.array([new_pos.x, new_pos.y, top_z])
            place_location_xy = place_location[:2]
            
            # Physics validation with IsaacSim (optional)
            if True:
                from dataclasses import asdict
                from utils import generate_unique_id
                from isaacsim.isaac_mcp.server import (
                    create_single_room_layout_scene_from_room,
                    simulate_the_scene
                )
                
                # Create layout copy with updated place object pose
                layout_copy = copy.deepcopy(floor_plan)
                room_copy = next(r for r in layout_copy.rooms if r.id == room_id)
                place_obj_copy = next(obj for obj in room_copy.objects if obj.id == place_object_id)
                
                place_obj_copy.position = new_pos
                place_obj_copy.rotation = new_rot
                
                layout_copy_id = generate_unique_id(f"{layout_name}_place_sample_{len(successful_samples)}")
                
                # Test physics stability
                room_dict_save_path = os.path.join(scene_save_dir, f"{layout_copy_id}_{room_id}.json")
                with open(room_dict_save_path, "w") as f:
                    json.dump(asdict(room_copy), f)
                
                result_create = create_single_room_layout_scene_from_room(scene_save_dir, room_dict_save_path)
                if not isinstance(result_create, dict) or result_create.get("status") != "success":
                    os.remove(room_dict_save_path)
                    print(f"  Physics test failed - IsaacSim scene creation failed")
                    continue
                
                result_sim = simulate_the_scene()
                if not isinstance(result_sim, dict) or result_sim.get("status") != "success":
                    os.remove(room_dict_save_path)
                    print(f"  Physics test failed - IsaacSim simulation failed")
                    continue
                
                next_step = result_sim.get("next_step", "") or ""
                is_stable = "stable" in next_step.lower() and "unstable" not in next_step.lower()
                
                os.remove(room_dict_save_path)
                
                if not is_stable:
                    print(f"  Physics test failed - pose is unstable")
                    continue
            
            # Find robot positions within reachability distance
            distances_to_place = np.linalg.norm(valid_robot_points - place_location_xy, axis=1)
            reachable_mask = distances_to_place <= max_reachability_distance
            reachable_robot_points = valid_robot_points[reachable_mask]
            reachable_distances = distances_to_place[reachable_mask]
            
            if len(reachable_robot_points) == 0:
                continue
            
            # Sort by distance
            sorted_indices = np.argsort(reachable_distances)
            
            # Try robot positions
            robot_found = False
            for idx in sorted_indices:
                robot_pos_2d = reachable_robot_points[idx]
                
                # Calculate robot orientation toward place location
                direction_to_place = place_location_xy - robot_pos_2d
                yaw = np.arctan2(direction_to_place[1], direction_to_place[0])
                
                # Create quaternion
                robot_quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
                robot_pos_3d = np.array([robot_pos_2d[0], robot_pos_2d[1], 0])
                
                # Check robot collision
                collision = check_unified_robot_collision(robot_pos_3d, robot_quat, scene_occupancy_fn, room_bounds)
                
                if not collision:
                    # Found valid pair!
                    robot_distance = reachable_distances[idx]
                    
                    successful_samples.append({
                        'sample_id': len(successful_samples),
                        'robot_position': [float(robot_pos_2d[0]), float(robot_pos_2d[1]), float(robot_z)],
                        'robot_quaternion': [float(robot_quat[0]), float(robot_quat[1]), float(robot_quat[2]), float(robot_quat[3])],
                        'place_location': place_location.tolist(),
                        'robot_distance_to_place': float(robot_distance),
                        'place_object_position': [float(new_pos.x), float(new_pos.y), float(new_pos.z)],
                        'place_object_rotation': [float(new_rot.x), float(new_rot.y), float(new_rot.z)]
                    })
                    
                    print(f"  ✅ Sample {len(successful_samples)}: robot dist={robot_distance:.2f}m")
                    robot_found = True
                    break
            
            if robot_found and len(successful_samples) >= num_samples:
                break
        
        if len(successful_samples) >= num_samples:
            break
    
    print(f"\n=== Sampling Complete (Object Placement) ===")
    print(f"Successful samples: {len(successful_samples)}/{num_samples}")
    if successful_samples:
        print(f"Average robot distance: {np.mean([s['robot_distance_to_place'] for s in successful_samples]):.2f}m")
    
    return successful_samples