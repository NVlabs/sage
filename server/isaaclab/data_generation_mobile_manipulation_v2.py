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

num_envs = 2

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=num_envs, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Mobile-Manipulation-Obj-Scene-Franka-IK-Abs-v0", help="Name of the task.")
parser.add_argument("--num_demos", type=int, default=1280, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
# Add parent directory to Python path to import constants
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR, ROBOMIMIC_ROOT_DIR, M2T2_ROOT_DIR

sys.path.insert(0, SERVER_ROOT_DIR)
sys.path.insert(0, M2T2_ROOT_DIR)
from utils import (
    dict_to_floor_plan, 
    get_layout_from_scene_save_dir,
    get_layout_from_scene_json_path
)
from tex_utils import (
    export_layout_to_mesh_dict_list_tree_search_with_object_id,
    export_layout_to_mesh_dict_object_id,
    get_textured_object_mesh
)
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
    arm_action = torch.cat([target_ee_pos - ee_frame_pos, d_rotvec.reshape(-1, 3)], dim=-1)
    # print(f"arm_action: {arm_action.shape} {arm_action}")
    return arm_action


def visualize_robot_planning_data(
    room_bounds, occupancy_grid, grid_x, grid_y, grid_res,
    robot_base_positions, robot_base_quats, camera_lookats,
    target_object, table_object, valid_points=None,
    layout_name="", room_id="", target_object_name="", table_object_name="",
    save_path=None
):
    """
    Visualize robot planning data including occupancy grid, robot positions, orientations, and camera lookat points.
    
    Args:
        room_bounds: tuple of (min_x, min_y, max_x, max_y)
        occupancy_grid: 2D numpy array of boolean occupancy
        grid_x, grid_y: arrays defining grid coordinates
        grid_res: grid resolution
        robot_base_positions: tensor/array of robot positions (N, 3)
        robot_base_quats: tensor/array of robot orientations (N, 1) - yaw angles
        camera_lookats: tensor/array of camera lookat positions (N, 3)
        target_object: object with position attributes
        table_object: object with position attributes
        valid_points: optional array of valid robot positions for debugging
        layout_name, room_id, target_object_name, table_object_name: strings for labeling
        save_path: path to save the PNG file
    """
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
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
    
    # 6. Draw robot positions and orientations
    for i, (pos, yaw, cam_lookat) in enumerate(zip(robot_positions_np, robot_quats_np, camera_lookats_np)):
        x, y = pos[0], pos[1]
        
        # Robot base position
        color = plt.cm.viridis(i / max(1, len(robot_positions_np) - 1))
        ax.scatter(x, y, c=[color], s=100, marker='o', 
                  edgecolors='black', linewidth=1,
                  label=f'Robot {i}' if i < 5 else '')  # Only label first 5 to avoid clutter
        
        # Robot orientation arrow
        arrow_length = 0.3
        dx = arrow_length * np.cos(yaw[0] if len(yaw.shape) > 0 else yaw)
        dy = arrow_length * np.sin(yaw[0] if len(yaw.shape) > 0 else yaw)
        
        ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.05,
                fc=color, ec='black', linewidth=1, alpha=0.8)
        
        # Camera lookat position
        cam_x, cam_y = cam_lookat[0], cam_lookat[1]
        ax.scatter(cam_x, cam_y, c=[color], s=60, marker='^', 
                  edgecolors='black', linewidth=1, alpha=0.7)
        
        # Line from robot to camera lookat
        ax.plot([x, cam_x], [y, cam_y], '--', color=color, alpha=0.5, linewidth=1)
    
    # 7. Add legend and labels
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Robot Planning Visualization\n'
                f'Layout: {layout_name}, Room: {room_id}\n'
                f'Target: {target_object_name}, Table: {table_object_name}', 
                fontsize=14, pad=20)
    
    # 8. Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 9. Add text annotations
    info_text = (f"Grid Resolution: {grid_res:.3f}m\n"
                f"Room Size: {room_max_x-room_min_x:.2f}×{room_max_y-room_min_y:.2f}m\n"
                f"Num Robots: {len(robot_positions_np)}")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
           verticalalignment='top', fontsize=10)
    
    # 10. Tight layout and save
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Robot planning visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def sample_robot_location(
    scene_save_dir, layout_name, room_id,
    target_object_name, table_object_name, num_envs
):

    """
    the sampling is solving a math problem:
    the room is a rectangle, and we treat the whole room as a 2d rectangle space, divided by grid_res x grid_res small occupancy rectangles.
    each object take over the occupancy rectangles that it covers.

    some variables that we can change:
    robot_min_dist_to_room_edge = 0.5: the minimum distance from the robot base to the room edge.
    robot_min_dist_to_object = 0.1: the minimum distance from the robot base to the object occupancy rectangles.
    robot_height_offset = 0.2: the height offset from the table top to the robot base.
    grid_res = 0.05: the resolution of the grid.

    the way to sample robot location is:
    1. sample 10k points in the room rectangle, remove the points that: 
        i. inside the object occupancy rectangles 
        ii. has a distance less than robot_min_dist_to_object to the object occupancy rectangles.
        iii. has a distance less than robot_min_dist_to_room_edge to the room rectangle edge.
    2. among the remaining points, find the point that minimizes the distance to the target_object, 
    choose the point as the robot pos x and y, robot z is max(the height of the table top - robot_height_offset, 0) (use the table object height);
    3. the robot 3d rotation quaternion is a z-axis rotation from the robot pos towards the target_object_name.

    extra for debugging: save a 2d vis image of the room rectangle, object occupancy rectangles, all valid points, and the final robot pos and quat.
    """

    # Parameters
    robot_min_dist_to_room_edge = 0.5
    robot_min_dist_to_object = 0.30
    robot_height_offset = 0.2
    grid_res = 0.05
    num_sample_points = 100000
    
    layout_json_path = os.path.join(scene_save_dir, f"{layout_name}.json")
    with open(layout_json_path, "r") as f:
        layout_info = json.load(f)
    
    floor_plan: FloorPlan = dict_to_floor_plan(layout_info)
    target_room = next(room for room in floor_plan.rooms if room.id == room_id)
    assert target_room is not None, f"target_room {room_id} not found in floor_plan"

    target_object = next(obj for obj in target_room.objects if obj.id == target_object_name)
    table_object = next(obj for obj in target_room.objects if obj.id == table_object_name)
    
    assert target_object is not None, f"target_object {target_object_name} not found in floor_plan"
    assert table_object is not None, f"table_object {table_object_name} not found in floor_plan"

    # Get room rectangle bounds
    room_min_x = target_room.position.x
    room_min_y = target_room.position.y
    room_max_x = target_room.position.x + target_room.dimensions.width
    room_max_y = target_room.position.y + target_room.dimensions.length
    
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
        valid_points = np.array([])
    else:
        # Convert to grid coordinates
        grid_coords = np.floor((edge_valid_points - [room_min_x, room_min_y]) / grid_res).astype(int)
        
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
            search_radius = int(np.ceil(robot_min_dist_to_object / grid_res))
            
            # Find all occupied cell positions
            occupied_indices = np.where(occupancy_grid)
            if len(occupied_indices[0]) > 0:
                occupied_positions = np.column_stack([
                    room_min_x + occupied_indices[0] * grid_res + grid_res/2,
                    room_min_y + occupied_indices[1] * grid_res + grid_res/2
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
                    distance_valid_mask = min_distances >= robot_min_dist_to_object
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
        
        # Choose optimal point with minimum distance to target
        optimal_idx = np.argmin(target_distances)
        optimal_point = valid_points[optimal_idx]
        
        if num_envs == 1:
            robot_positions = np.array([optimal_point])
        else:
            # For remaining environments, find closest points to optimal point
            distances_to_optimal = np.linalg.norm(valid_points - optimal_point, axis=1)
            
            # Sort indices by distance to optimal point
            sorted_indices = np.argsort(distances_to_optimal)
            
            # Select num_envs closest points (including the optimal point itself)
            selected_indices = sorted_indices[:num_envs]
            robot_positions = valid_points[selected_indices]
            
            # If we don't have enough valid points, repeat the closest ones
            if len(robot_positions) < num_envs:
                print(f"Warning: Only {len(robot_positions)} valid points available, repeating closest points")
                # Pad with repeated positions from what we have
                while len(robot_positions) < num_envs:
                    additional_needed = num_envs - len(robot_positions)
                    repeat_count = min(additional_needed, len(valid_points))
                    robot_positions = np.concatenate([
                        robot_positions, 
                        valid_points[sorted_indices[:repeat_count]]
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
    # try:
    #     # Create save directory if it doesn't exist
    #     debug_dir = os.path.join(os.path.dirname(scene_save_dir), "debug", "robot_planning")
    #     os.makedirs(debug_dir, exist_ok=True)
        
    #     # Generate filename with timestamp and identifiers
    #     import time
    #     timestamp = time.strftime("%Y%m%d_%H%M%S")
    #     viz_filename = f"robot_planning_{layout_name}_{room_id}_{target_object_name}_{timestamp}.png"
    #     viz_path = os.path.join(debug_dir, viz_filename)
        
    #     # Call visualization function
    #     visualize_robot_planning_data(
    #         room_bounds=(room_min_x, room_min_y, room_max_x, room_max_y),
    #         occupancy_grid=occupancy_grid,
    #         grid_x=grid_x,
    #         grid_y=grid_y,
    #         grid_res=grid_res,
    #         robot_base_positions=robot_base_pos,
    #         robot_base_quats=robot_base_quat,
    #         camera_lookats=camera_lookat,
    #         target_object=target_object,
    #         table_object=table_object,
    #         valid_points=valid_points if len(valid_points) > 0 else None,
    #         layout_name=layout_name,
    #         room_id=room_id,
    #         target_object_name=target_object_name,
    #         table_object_name=table_object_name,
    #         save_path=viz_path
    #     )
    # except Exception as e:
    #     print(f"Warning: Failed to create robot planning visualization: {e}")

    return robot_base_pos, robot_base_quat, camera_lookat

def get_grasp_transforms(layout_json_path, target_object_name, base_pos):
    
    layout = get_layout_from_scene_json_path(layout_json_path)
    mesh_dict_list = export_layout_to_mesh_dict_list_tree_search_with_object_id(layout, target_object_name)
    print(f"mesh_dict_list: {mesh_dict_list.keys()}")
    meta_data, vis_data = generate_m2t2_data(mesh_dict_list, target_object_name, base_pos)
    model, cfg = load_m2t2()
    total_trials = 0
    while True:
        grasp_transforms = infer_m2t2(meta_data, vis_data, model, cfg)
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


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    log_dir = os.path.abspath(os.path.join("./logs/robomimic", args_cli.task+"-test"))
    os.makedirs(os.path.join(log_dir, "debug"), exist_ok=True)

    layout_id = "layout_fac9613b"
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
    transform_dict_path = os.path.join(usd_collection_dir, "rigid_object_transform_dict.json")
    mass_dict = json.load(open(mass_dict_path, "r"))
    all_object_rigids = list(mass_dict.keys())
    object_transform_dict = json.load(open(transform_dict_path, "r"))

    for object_name in object_transform_dict:
        object_position = object_transform_dict[object_name]["position"]
        object_transform_dict[object_name]["position"] = np.array([object_position["x"], object_position["y"], object_position["z"]]).reshape(3)
        object_rotation = object_transform_dict[object_name]["rotation"]
        object_transform_dict[object_name]["rotation"] = np.array(R.from_euler("xyz", [object_rotation["x"], object_rotation["y"], object_rotation["z"]], degrees=True).as_quat(scalar_first=True)).reshape(4)

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
    T_grasp = 1000
    T_hold = 50
    T_init = 20
    first_round = True
    current_data_idx = 0

    pick_table_id = "room_0efe6071_table_eb113586"
    pick_object_id = "room_0efe6071_coke_29651336"
    # pick_object_id = "room_0efe6071_mug_13fa7a11"
    place_table_id = "room_0efe6071_desk_0aac9647"
    place_object_id = "room_0efe6071_plate_cdfd1c12"
    room_id = "room_0efe6071"


    print("start creating motion planner")
    motion_planner = MotionPlanner(
        env,
        collision_checker=True,
        reference_prim_path="/World/envs/env_0/Robot",
        ignore_substring=[
            "/World/envs/env_0/Robot",
            "/World/GroundPlane",
            "/World/collisions",
            "/World/light",
            "/curobo",
        ]+[f"/World/envs/env_{env_i}" for env_i in range(1, num_envs)],
        collision_avoidance_distance=0.001
        # collision_avoidance_distance=grasp_object_height
    )
    print("end creating motion planner")


    robot_base_pick_pos, robot_base_pick_quat, _ = sample_robot_location(
        scene_save_dir, layout_id, room_id,
        pick_object_id, pick_table_id, num_envs
    )
    robot_base_place_pos, robot_base_place_quat, _ = sample_robot_location(
        scene_save_dir, layout_id, room_id,
        place_object_id, place_table_id, num_envs
    )

    print(f"robot_base_pos: {robot_base_pick_pos}")
    print(f"robot_base_pick_quat: {robot_base_pick_quat}")

    place_location_w = get_place_location(os.path.join(scene_save_dir, f"{layout_id}.json"), place_object_id)
    place_location_w = torch.tensor(place_location_w, device="cuda").float()

    pick_object_height = get_grasp_object_height(os.path.join(scene_save_dir, f"{layout_id}.json"), pick_object_id)

    
    grasp_up_offset = 0.35
    place_up_offset = 0.8
    place_down_offset = 0.5
    grasp_down_offset = 0.02

    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt):
        while True:

            if iteration % T_grasp == 0:

                iteration = 0
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


                place_location_w_pos_envs, place_location_w_quat_envs = math_utils.combine_frame_transforms(
                    envs_pos, envs_quat,
                    place_location_w.reshape(1, 3).repeat(num_envs, 1), torch.tensor([1, 0, 0, 0], device=env.sim.device).reshape(1, 4).repeat(num_envs, 1)
                )
                place_location_w_envs = torch.cat([place_location_w_pos_envs, place_location_w_quat_envs], dim=-1)

                is_reach_pick_goal_envs = []
                is_reach_grasp_goal_envs = []
                is_reach_grasp_goal_down_envs = []
                is_reach_grasp_goal_up_envs = []
                is_reach_place_goal_envs = []
                is_reach_place_goal_up_envs = []

                start_grasp_T_envs = {}

                reach_pick_base_traj_envs = {}
                reach_pick_base_T_envs = {}
                reach_pick_up_traj_envs = {}
                reach_pick_up_T_envs = {}
                reach_pick_down_traj_envs = {}
                reach_pick_down_T_envs = {}
                reach_grasp_up_traj_envs = {}
                reach_grasp_up_T_envs = {}
                reach_place_base_traj_envs = {}
                reach_place_base_T_envs = {}
                reach_place_T_envs = {}
                reach_place_up_traj_envs = {}
                reach_place_goal_T_envs = {}
                place_ee_goals_up_envs = {}

                ee_frame_data_pos_after_grasp_relative_base_support_envs = {}
                ee_frame_data_quat_after_grasp_relative_base_support_envs = {}
                support_frame_data_pos_envs = {}

                for env_i in range(num_envs):
                    is_reach_pick_goal_envs.append(False)
                    is_reach_grasp_goal_envs.append(False)
                    is_reach_grasp_goal_down_envs.append(False)
                    is_reach_grasp_goal_up_envs.append(False)
                    is_reach_place_goal_envs.append(False)
                    is_reach_place_goal_up_envs.append(False)

                pick_ee_goals = sample_grasp(os.path.join(scene_save_dir, f"{layout_id}.json"), pick_object_id, robot_base_pick_pos[0, :3], num_envs, env.sim.device)
                
                pick_ee_goals_translate, pick_ee_goals_quat = math_utils.combine_frame_transforms(
                    envs_pos, envs_quat,
                    pick_ee_goals[:, :3], pick_ee_goals[:, 3:7]
                )
                pick_ee_goals = torch.cat([pick_ee_goals_translate, pick_ee_goals_quat], dim=1)
                print(f"pick_ee_goals: {pick_ee_goals.shape} {pick_ee_goals}")

                pick_ee_goals_up = pick_ee_goals + torch.tensor([0., 0., grasp_up_offset, 0., 0., 0., 0.], device=env.sim.device).reshape(1, 7)
                pick_ee_goals_up_after = pick_ee_goals + torch.tensor([0., 0., place_up_offset, 0., 0., 0., 0.], device=env.sim.device).reshape(1, 7)
                pick_ee_goals_down = pick_ee_goals + torch.tensor([0., 0., -grasp_down_offset, 0., 0., 0., 0.], device=env.sim.device).reshape(1, 7)

                target_object_com_state = env.scene[pick_object_id].data.root_com_state_w
                target_object_initial_z = float(target_object_com_state[:, 2].mean())
                print(f"target_object_initial_z: {target_object_initial_z}")

                robot_base_pick_pose = torch.cat([robot_base_pick_pos, robot_base_pick_quat], dim=-1)
                robot_base_place_pose = torch.cat([robot_base_place_pos, robot_base_place_quat], dim=-1)

                if first_round:

                    ee_frame_data_pos_initial = env.scene["ee_frame"].data.target_pos_source.reshape(-1, 3).clone()
                    ee_frame_data_quat_initial = env.scene["ee_frame"].data.target_quat_source.reshape(-1, 4).clone()

                    fixed_support_frame_data_pos_initial = env.scene["fixed_support_frame"].data.target_pos_source.reshape(-1, 3).clone()
                    fixed_support_frame_data_quat_initial = env.scene["fixed_support_frame"].data.target_quat_source.reshape(-1, 4).clone()

                    ee_frame_to_fixed_support_frame_pos_initial, ee_frame_to_fixed_support_frame_quat_initial = math_utils.subtract_frame_transforms(
                        fixed_support_frame_data_pos_initial, fixed_support_frame_data_quat_initial,
                        ee_frame_data_pos_initial, ee_frame_data_quat_initial
                    )

                    first_round = False

                

                

                camera_list = []


            # joint_idx_base_x = robot.data.joint_names.index(joint_name_base_x)
            # joint_idx_base_y = robot.data.joint_names.index(joint_name_base_y)
            # joint_idx_base_angle = robot.data.joint_names.index(joint_name_base_angle)

            # joint_pos = robot.data.joint_pos

            # joint_pos_base_x = joint_pos[:, joint_idx_base_x]
            # joint_pos_base_y = joint_pos[:, joint_idx_base_y]
            # joint_pos_base_angle = joint_pos[:, joint_idx_base_angle]

            # link_idx_base_support = robot.data.body_names.index(link_name_base_support)
            # base_support_pos = robot.data.body_state_w[:, link_idx_base_support, :3]
            # base_support_quat = robot.data.body_state_w[:, link_idx_base_support, 3:7]

            # support_pos_and_angle = torch.cat([base_support_pos[:, 0].reshape(-1, 1), base_support_pos[:, 1].reshape(-1, 1), joint_pos_base_angle.reshape(-1, 1)], dim=-1)
            # robot_base_pick = torch.cat([robot_base_pick_pos[:, :2], robot_base_pick_angle.reshape(-1, 1)], dim=-1)
            # robot_base_place = torch.cat([robot_base_place_pos[:, :2], robot_base_place_angle.reshape(-1, 1)], dim=-1)

            max_velocity = 2.0
            min_velocity = 0.1

            max_velocity_grasp = 2.0 * 0.5
            min_velocity_grasp = 0.1

            actions_list = []

            ee_frame_data_pos = env.scene["ee_frame"].data.target_pos_source.reshape(-1, 3)
            ee_frame_data_quat = env.scene["ee_frame"].data.target_quat_source.reshape(-1, 4)
            ee_frame_data_pose = torch.cat([ee_frame_data_pos, ee_frame_data_quat], dim=-1)

            fixed_support_frame_data_pos = env.scene["fixed_support_frame"].data.target_pos_source.reshape(-1, 3).clone()
            fixed_support_frame_data_quat = env.scene["fixed_support_frame"].data.target_quat_source.reshape(-1, 4).clone()
            fixed_support_frame_data_pose = torch.cat([fixed_support_frame_data_pos, fixed_support_frame_data_quat], dim=-1)

            ee_to_fixed_support_frame_pos, ee_to_fixed_support_frame_quat = math_utils.subtract_frame_transforms(
                fixed_support_frame_data_pos, fixed_support_frame_data_quat,
                ee_frame_data_pos, ee_frame_data_quat
            )
            ee_to_fixed_support_frame_pose = torch.cat([ee_to_fixed_support_frame_pos, ee_to_fixed_support_frame_quat], dim=-1)

            support_frame_data_pos = env.scene["support_frame"].data.target_pos_source.reshape(-1, 3).clone()
            support_frame_data_quat = env.scene["support_frame"].data.target_quat_source.reshape(-1, 4).clone()
            support_frame_data_pose = torch.cat([support_frame_data_pos, support_frame_data_quat], dim=-1)

            ee_to_support_frame_pos, ee_to_support_frame_quat = math_utils.subtract_frame_transforms(
                support_frame_data_pos, support_frame_data_quat,
                ee_frame_data_pos, ee_frame_data_quat
            )
            ee_to_support_frame_pose = torch.cat([ee_to_support_frame_pos, ee_to_support_frame_quat], dim=-1)


            robot_qpos = robot.data.joint_pos

            target_object_com_state = env.scene[pick_object_id].data.root_com_state_w
            for env_i in range(num_envs):
                target_object_current_z = float(target_object_com_state[env_i, 2])

                if iteration % T_grasp == 0:

                    reach_pick_base_traj_envs[env_i], _ = curobo_plan_traj(
                        None, None,
                        fixed_support_frame_data_pos[env_i:env_i+1],
                        robot_base_pick_pose[env_i:env_i+1, :3],
                        robot_base_pick_pose[env_i:env_i+1, 3:7],
                        interpolate=True,
                        max_length=150,
                        max_interpolation_step_distance=0.01,
                    )
                    reach_pick_base_T_envs[env_i] = iteration

                if not is_reach_pick_goal_envs[env_i]:
                    # print("iteration: ", iteration, "reaching pick goal envs...")
                    # is_reach_pick_goal_envs[env_i] = torch.all(torch.abs(robot_base_pick[env_i] - support_pos_and_angle[env_i]) < 0.01)
                    is_reach_pick_goal_envs[env_i] = is_reach(robot_base_pick_pose[env_i], fixed_support_frame_data_pose[env_i], ignore_z=True, loc_threshold=0.03, rot_threshold=5.0)

                    if is_reach_pick_goal_envs[env_i]:
                        reach_pick_up_traj, success = curobo_plan_traj(
                            motion_planner,
                            robot_qpos[env_i:env_i+1, :],
                            ee_frame_data_pos[env_i:env_i+1, :3],
                            pick_ee_goals_up[env_i:env_i+1, :3],
                            pick_ee_goals_up[env_i:env_i+1, 3:7],
                            interpolate=True,
                        )
                        reach_pick_up_traj_envs[env_i] = reach_pick_up_traj
                        reach_pick_up_T_envs[env_i] = iteration
                
                elif not is_reach_grasp_goal_envs[env_i]:
                    # is_reach_grasp_goal_envs[env_i] = torch.all(torch.abs(ee_frame_data_pos[env_i] - pick_ee_goals_up[env_i, :3]) < 0.03)
                    is_reach_grasp_goal_envs[env_i] = is_reach(pick_ee_goals_up[env_i, :], ee_frame_data_pose[env_i], loc_threshold=0.03, rot_threshold=30.0)

                    if is_reach_grasp_goal_envs[env_i]:
                        reach_pick_down_traj, success = curobo_plan_traj(
                            motion_planner,
                            robot_qpos[env_i:env_i+1, :],
                            ee_frame_data_pos[env_i:env_i+1, :3],
                            pick_ee_goals_down[env_i:env_i+1, :3],
                            pick_ee_goals_down[env_i:env_i+1, 3:7],
                            interpolate=True,
                        )
                        reach_pick_down_traj_envs[env_i] = reach_pick_down_traj
                        reach_pick_down_T_envs[env_i] = iteration
                
                elif not is_reach_grasp_goal_down_envs[env_i]:
                    # is_reach_grasp_goal_down_envs[env_i] = torch.all(torch.abs(ee_frame_data_pos[env_i] - pick_ee_goals_down[env_i, :3]) < 0.03)
                    is_reach_grasp_goal_down_envs[env_i] = is_reach(pick_ee_goals_down[env_i, :], ee_frame_data_pose[env_i], loc_threshold=0.03, rot_threshold=30.0)

                    if is_reach_grasp_goal_down_envs[env_i]:
                        reach_grasp_up_traj, success = curobo_plan_traj(
                            motion_planner,
                            robot_qpos[env_i:env_i+1, :],
                            ee_frame_data_pos[env_i:env_i+1, :3],
                            pick_ee_goals_up_after[env_i:env_i+1, :3],
                            pick_ee_goals_up_after[env_i:env_i+1, 3:7],
                            interpolate=True,
                        )
                        reach_grasp_up_traj_envs[env_i] = reach_grasp_up_traj
                        reach_grasp_up_T_envs[env_i] = iteration

                elif not is_reach_grasp_goal_up_envs[env_i]:
                    # is_reach_grasp_goal_up_envs[env_i] = torch.all(torch.abs(ee_frame_data_pos[env_i] - pick_ee_goals_up_after[env_i, :3]) < 0.08)
                    is_reach_grasp_goal_up_envs[env_i] = is_reach(pick_ee_goals_up_after[env_i, :], ee_frame_data_pose[env_i], loc_threshold=0.08, rot_threshold=10.0)

                    if is_reach_grasp_goal_up_envs[env_i]:

                        ee_frame_data_pos_after_grasp_relative_base_support_envs[env_i] = ee_to_support_frame_pos[env_i].clone()
                        ee_frame_data_quat_after_grasp_relative_base_support_envs[env_i] = ee_to_support_frame_quat[env_i].clone()

                        support_frame_data_pos_envs[env_i] = support_frame_data_pos[env_i].clone()


                        reach_place_base_traj_envs[env_i], _ = curobo_plan_traj(
                            None, None,
                            fixed_support_frame_data_pos[env_i:env_i+1],
                            robot_base_place_pose[env_i:env_i+1, :3],
                            robot_base_place_pose[env_i:env_i+1, 3:7],
                            interpolate=True,
                            max_length=150,
                            max_interpolation_step_distance=0.01,
                        )
                        reach_place_base_T_envs[env_i] = iteration


                elif not is_reach_place_goal_envs[env_i]:
                    # is_reach_place_goal_envs[env_i] = torch.all(torch.abs(robot_base_place[env_i] - support_pos_and_angle[env_i]) < 0.01)
                    is_reach_place_goal_envs[env_i] = is_reach(robot_base_place_pose[env_i], fixed_support_frame_data_pose[env_i], ignore_z=True, loc_threshold=0.03, rot_threshold=5.0)

                    if is_reach_place_goal_envs[env_i]:

                        robot_base_w = env.scene["robot"].data.root_state_w[env_i:env_i+1, :7]
                        target_object_w = env.scene[pick_object_id].data.root_state_w[env_i:env_i+1, :7]
                        target_object_r_pos, target_object_r_quat = math_utils.subtract_frame_transforms(
                            robot_base_w[:, :3], robot_base_w[:, 3:7],
                            target_object_w[:, :3], target_object_w[:, 3:7]
                        )

                        target_to_ee_pos = (ee_frame_data_pos[env_i:env_i+1].reshape(3) - target_object_r_pos.reshape(3))
                        target_to_ee_pos[2] = 0
                        # ee_goals_place[env_i, :2] += target_to_ee_pos.reshape(3)[:2]

                        place_ee_goals_up_env_i = place_location_w_envs[env_i:env_i+1, :3] + torch.tensor([0., 0., place_down_offset+pick_object_height], device=env.sim.device).reshape(1, 3)
                        place_ee_goals_up_env_i[:, :2] += target_to_ee_pos.reshape(3)[:2].reshape(1, 2)

                        print("target_to_ee_pos: ", target_to_ee_pos)
                        
                        reach_place_up_traj, success = curobo_plan_traj(
                            motion_planner,
                            robot_qpos[env_i:env_i+1, :],
                            ee_frame_data_pos[env_i:env_i+1, :3],
                            place_ee_goals_up_env_i,
                            ee_frame_data_quat[env_i:env_i+1],
                            interpolate=True,
                        )
                        reach_place_up_traj_envs[env_i] = reach_place_up_traj
                        reach_place_T_envs[env_i] = iteration

                        place_ee_goals_up_envs[env_i] = place_ee_goals_up_env_i.reshape(-1)

                elif not is_reach_place_goal_up_envs[env_i]:
                    is_reach_place_goal_up_envs[env_i] = torch.all(torch.abs(ee_frame_data_pos[env_i] - place_ee_goals_up_envs[env_i][:3]) < 0.10)
                    # is_reach_place_goal_up_envs[env_i] = is_reach(place_ee_goals_up_envs[env_i], ee_frame_data_pose[env_i], loc_threshold=0.10, rot_threshold=10.0)

                    reach_place_goal_T_envs[env_i] = iteration


                if not is_reach_pick_goal_envs[env_i]:
                    # velocity_x = torch.clip((robot_base_pick_pos[env_i, 0] - base_support_pos_env_i[0]), -max_velocity, max_velocity)
                    # velocity_y = torch.clip((robot_base_pick_pos[env_i, 1] - base_support_pos_env_i[1]), -max_velocity, max_velocity)
                    # velocity_x = torch.where(torch.abs(velocity_x) < min_velocity, torch.sign(velocity_x) * min_velocity, velocity_x)
                    # velocity_y = torch.where(torch.abs(velocity_y) < min_velocity, torch.sign(velocity_y) * min_velocity, velocity_y)
                    # velocity_x = velocity_x.reshape(1, 1)
                    # velocity_y = velocity_y.reshape(1, 1)
                    # velocity_angle = torch.clip((robot_base_pick_angle[env_i] - joint_pos_base_angle[env_i]), -max_velocity, max_velocity)
                    # velocity_angle = torch.where(torch.abs(velocity_angle) < min_velocity, torch.sign(velocity_angle) * min_velocity, velocity_angle)
                    # velocity_angle = velocity_angle.reshape(1, 1)
                    # height_target = height_target_pick[env_i].reshape(1, 1)

                    # ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat = math_utils.combine_frame_transforms(
                    #     base_support_pos_env_i.reshape(1, 3), base_support_quat_env_i.reshape(1, 4),
                    #     ee_frame_data_pos_initial_relative_base_support_env_i.reshape(1, 3), ee_frame_data_quat_initial_relative_base_support_env_i.reshape(1, 4)
                    # )

                    actions = torch.cat([
                        get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], 
                            reach_pick_base_traj_envs[env_i][min(max(iteration - reach_pick_base_T_envs[env_i], 0), len(reach_pick_base_traj_envs[env_i]) - 1), :3].reshape(1, 3), 
                            reach_pick_base_traj_envs[env_i][min(max(iteration - reach_pick_base_T_envs[env_i], 0), len(reach_pick_base_traj_envs[env_i]) - 1), 3:7].reshape(1, 4)),
                        get_action_relative(ee_to_fixed_support_frame_pos[env_i:env_i+1], ee_to_fixed_support_frame_quat[env_i:env_i+1], ee_frame_to_fixed_support_frame_pos_initial[env_i:env_i+1], ee_frame_to_fixed_support_frame_quat_initial[env_i:env_i+1]),
                        torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                    ], dim=-1)

                elif not is_reach_grasp_goal_envs[env_i]:
                    # velocity_x = torch.clip((robot_base_pick_pos[env_i, 0] - base_support_pos_env_i[0]), -max_velocity, max_velocity)
                    # velocity_y = torch.clip((robot_base_pick_pos[env_i, 1] - base_support_pos_env_i[1]), -max_velocity, max_velocity)
                    # velocity_x = velocity_x.reshape(1, 1)
                    # velocity_y = velocity_y.reshape(1, 1)
                    # velocity_angle = torch.clip((robot_base_pick_angle[env_i] - joint_pos_base_angle[env_i]), -max_velocity, max_velocity)
                    # velocity_angle = velocity_angle.reshape(1, 1)
                    # height_target = height_target_pick[env_i].reshape(1, 1)

                    pick_ee_goals_up_iter = reach_pick_up_traj_envs[env_i][min(max(iteration - reach_pick_up_T_envs[env_i], 0), len(reach_pick_up_traj_envs[env_i]) - 1)].reshape(1, 7)
                    pick_ee_goals_up_iter_to_fixed_support_frame_pos, pick_ee_goals_up_iter_to_fixed_support_frame_quat = math_utils.subtract_frame_transforms(
                        fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1],
                        pick_ee_goals_up_iter[:, :3], pick_ee_goals_up_iter[:, 3:7]
                    )

                    actions = torch.cat([
                        get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], robot_base_pick_pos[env_i:env_i+1], robot_base_pick_quat[env_i:env_i+1]),
                        get_action_relative(ee_to_fixed_support_frame_pos[env_i:env_i+1], ee_to_fixed_support_frame_quat[env_i:env_i+1], pick_ee_goals_up_iter_to_fixed_support_frame_pos, pick_ee_goals_up_iter_to_fixed_support_frame_quat),
                        torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                    ], dim=-1)
                    # plan the traj to pick object

                elif not is_reach_grasp_goal_down_envs[env_i]:

                    # velocity_x = torch.clip((robot_base_pick_pos[env_i, 0] - base_support_pos_env_i[0]), -max_velocity, max_velocity)
                    # velocity_y = torch.clip((robot_base_pick_pos[env_i, 1] - base_support_pos_env_i[1]), -max_velocity, max_velocity)
                    # velocity_x = velocity_x.reshape(1, 1)
                    # velocity_y = velocity_y.reshape(1, 1)
                    # velocity_angle = torch.clip((robot_base_pick_angle[env_i] - joint_pos_base_angle[env_i]), -max_velocity, max_velocity)
                    # velocity_angle = velocity_angle.reshape(1, 1)
                    # height_target = height_target_pick[env_i].reshape(1, 1)

                    pick_ee_goals_down_iter = reach_pick_down_traj_envs[env_i][min(max(iteration - reach_pick_down_T_envs[env_i], 0), len(reach_pick_down_traj_envs[env_i]) - 1)].reshape(1, 7)
                    pick_ee_goals_down_iter_to_fixed_support_frame_pos, pick_ee_goals_down_iter_to_fixed_support_frame_quat = math_utils.subtract_frame_transforms(
                        fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1],
                        pick_ee_goals_down_iter[:, :3], pick_ee_goals_down_iter[:, 3:7]
                    )

                    actions = torch.cat([
                        get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], robot_base_pick_pos[env_i:env_i+1], robot_base_pick_quat[env_i:env_i+1]),
                        get_action_relative(ee_to_fixed_support_frame_pos[env_i:env_i+1], ee_to_fixed_support_frame_quat[env_i:env_i+1], pick_ee_goals_down_iter_to_fixed_support_frame_pos, pick_ee_goals_down_iter_to_fixed_support_frame_quat),
                        torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                    ], dim=-1)
                    # plan the traj to pick object

                elif is_reach_grasp_goal_down_envs[env_i] and start_grasp_T_envs.get(env_i, 0) < T_hold:
                    start_grasp_T_envs[env_i] = start_grasp_T_envs.get(env_i, 0) + 1

                    # velocity_x = torch.clip(robot_base_pick_pos[env_i, 0] - base_support_pos_env_i[0], -max_velocity, max_velocity)
                    # velocity_y = torch.clip(robot_base_pick_pos[env_i, 1] - base_support_pos_env_i[1], -max_velocity, max_velocity)
                    # velocity_x = velocity_x.reshape(1, 1)
                    # velocity_y = velocity_y.reshape(1, 1)
                    # velocity_angle = torch.clip(robot_base_pick_angle[env_i] - joint_pos_base_angle[env_i], -max_velocity, max_velocity)
                    # velocity_angle = velocity_angle.reshape(1, 1)
                    # height_target = height_target_pick[env_i].reshape(1, 1)

                    # actions = torch.cat(
                    #     [
                    #     velocity_y, velocity_x, velocity_angle, height_target,
                    #     # torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0,], device=env.sim.device).reshape(1, -1).repeat(num_envs, 1),
                    #     torch.tensor([-1.0 if start_grasp_T_envs[env_i] > T_hold / 2 else 1.0,], device=env.sim.device).reshape(1, 1),
                    #     get_arm_action_relative(ee_frame_data_pos[env_i:env_i+1], ee_frame_data_quat[env_i:env_i+1], pick_ee_goals_down[env_i:env_i+1, :3], pick_ee_goals_down[env_i:env_i+1, 3:7])],
                    #     dim=-1
                    # )
                    # plan the traj to pick object

                    pick_ee_goals_down_to_fixed_support_frame_pos, pick_ee_goals_down_to_fixed_support_frame_quat = math_utils.subtract_frame_transforms(
                        fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1],
                        pick_ee_goals_down[env_i:env_i+1, :3], pick_ee_goals_down[env_i:env_i+1, 3:7]
                    )

                    actions = torch.cat([
                        get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], robot_base_pick_pos[env_i:env_i+1], robot_base_pick_quat[env_i:env_i+1]),
                        get_action_relative(ee_to_fixed_support_frame_pos[env_i:env_i+1], ee_to_fixed_support_frame_quat[env_i:env_i+1], pick_ee_goals_down_to_fixed_support_frame_pos, pick_ee_goals_down_to_fixed_support_frame_quat),
                        torch.tensor([-1.0 if start_grasp_T_envs[env_i] > T_hold / 2 else 1.0,], device=env.sim.device).reshape(1, 1),
                    ], dim=-1)
                
                elif not is_reach_grasp_goal_up_envs[env_i]:
                    # velocity_x = torch.clip(robot_base_pick_pos[env_i, 0] - base_support_pos_env_i[0], -max_velocity, max_velocity)
                    # velocity_y = torch.clip(robot_base_pick_pos[env_i, 1] - base_support_pos_env_i[1], -max_velocity, max_velocity)
                    # velocity_x = velocity_x.reshape(1, 1)
                    # velocity_y = velocity_y.reshape(1, 1)
                    # velocity_angle = torch.clip(robot_base_pick_angle[env_i] - joint_pos_base_angle[env_i], -max_velocity, max_velocity)
                    # velocity_angle = velocity_angle.reshape(1, 1)
                    # height_target = height_target_pick[env_i].reshape(1, 1)

                    grasp_up_goals_iter = reach_grasp_up_traj_envs[env_i][min(max(iteration - reach_grasp_up_T_envs[env_i], 0), len(reach_grasp_up_traj_envs[env_i]) - 1)].reshape(1, 7)
                    grasp_up_goals_iter_to_fixed_support_frame_pos, grasp_up_goals_iter_to_fixed_support_frame_quat = math_utils.subtract_frame_transforms(
                        fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1],
                        grasp_up_goals_iter[:, :3], grasp_up_goals_iter[:, 3:7]
                    )

                    # actions = torch.cat(
                    #     [
                    #     velocity_y, velocity_x, velocity_angle, height_target,
                    #     # torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0,], device=env.sim.device).reshape(1, -1).repeat(num_envs, 1),
                    #     torch.tensor([-1.0,], device=env.sim.device).reshape(1, 1),
                    #     get_arm_action_relative(ee_frame_data_pos[env_i:env_i+1], ee_frame_data_quat[env_i:env_i+1], grasp_up_goals_iter[:, :3], grasp_up_goals_iter[:, 3:7])],
                    #     dim=-1
                    # )
                    # plan the traj to pick object

                    actions = torch.cat([
                        get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], robot_base_pick_pos[env_i:env_i+1], robot_base_pick_quat[env_i:env_i+1]),
                        get_action_relative(ee_to_fixed_support_frame_pos[env_i:env_i+1], ee_to_fixed_support_frame_quat[env_i:env_i+1], grasp_up_goals_iter_to_fixed_support_frame_pos, grasp_up_goals_iter_to_fixed_support_frame_quat),
                        torch.tensor([-1.0], device=env.sim.device).reshape(1, 1),
                    ], dim=-1)

                elif not is_reach_place_goal_envs[env_i]:


                    # velocity_x = torch.clip((robot_base_place_pos[env_i, 0] - base_support_pos_env_i[0]), -max_velocity_grasp, max_velocity_grasp)
                    # velocity_y = torch.clip((robot_base_place_pos[env_i, 1] - base_support_pos_env_i[1]), -max_velocity_grasp, max_velocity_grasp)
                    # velocity_x = torch.where(torch.abs(velocity_x) < min_velocity_grasp, torch.sign(velocity_x) * min_velocity_grasp, velocity_x)
                    # velocity_y = torch.where(torch.abs(velocity_y) < min_velocity_grasp, torch.sign(velocity_y) * min_velocity_grasp, velocity_y)
                    # velocity_x = velocity_x.reshape(1, 1)
                    # velocity_y = velocity_y.reshape(1, 1)
                    # velocity_angle = torch.clip((robot_base_place_angle[env_i] - joint_pos_base_angle[env_i]), -max_velocity_grasp, max_velocity_grasp)
                    # velocity_angle = torch.where(torch.abs(velocity_angle) < min_velocity_grasp, torch.sign(velocity_angle) * min_velocity_grasp, velocity_angle)
                    # velocity_angle = velocity_angle.reshape(1, 1)
                    # # height_target = height_target_place[env_i].reshape(1, 1)
                    # height_target = torch.clamp(height_target_place[env_i], min=height_target_limit).reshape(1, 1)

                    # ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat = math_utils.combine_frame_transforms(
                    #     base_support_pos_env_i.reshape(1, 3), base_support_quat_env_i.reshape(1, 4),
                    #     ee_frame_data_pos_after_grasp_relative_base_support_envs[env_i].reshape(1, 3), ee_frame_data_quat_after_grasp_relative_base_support_envs[env_i].reshape(1, 4)
                    # )

                    # actions = torch.cat(
                    #     [
                    #     velocity_y, velocity_x, velocity_angle, height_target,
                    #     # torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0,], device=env.sim.device).reshape(1, -1).repeat(num_envs, 1),
                    #     torch.tensor([-1.0,], device=env.sim.device).reshape(1, 1),
                    #     get_arm_action_relative(ee_frame_data_pos[env_i:env_i+1], ee_frame_data_quat[env_i:env_i+1], ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat)],
                    #     dim=-1
                    # )

                    # print("ee_frame_data_pos_after_grasp_relative_base_support_envs[env_i]: ", ee_frame_data_pos_after_grasp_relative_base_support_envs[env_i])
                    # print("ee_frame_data_quat_after_grasp_relative_base_support_envs[env_i]: ", ee_frame_data_quat_after_grasp_relative_base_support_envs[env_i])

                    support_frame_data_pos_env_i = support_frame_data_pos[env_i:env_i+1].clone()
                    support_frame_data_pos_env_i[:, 2] = support_frame_data_pos_envs[env_i][2]

                    ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat = math_utils.combine_frame_transforms(
                        support_frame_data_pos_env_i, support_frame_data_quat[env_i:env_i+1],
                        ee_frame_data_pos_after_grasp_relative_base_support_envs[env_i].reshape(1, 3), ee_frame_data_quat_after_grasp_relative_base_support_envs[env_i].reshape(1, 4)
                    )

                    actions = torch.cat([
                        get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], 
                        reach_place_base_traj_envs[env_i][min(max(iteration - reach_place_base_T_envs[env_i], 0), len(reach_place_base_traj_envs[env_i]) - 1), :3].reshape(1, 3), 
                        reach_place_base_traj_envs[env_i][min(max(iteration - reach_place_base_T_envs[env_i], 0), len(reach_place_base_traj_envs[env_i]) - 1), 3:7].reshape(1, 4)),
                        get_action_relative(ee_frame_data_pos[env_i:env_i+1], ee_frame_data_quat[env_i:env_i+1], 
                        ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat),
                        torch.tensor([-1.0], device=env.sim.device).reshape(1, 1),
                    ], dim=-1)

                elif not is_reach_place_goal_up_envs[env_i]:
                    # velocity_x = torch.clip((robot_base_place_pos[env_i, 0] - base_support_pos_env_i[0]), -max_velocity_grasp, max_velocity_grasp)
                    # velocity_y = torch.clip((robot_base_place_pos[env_i, 1] - base_support_pos_env_i[1]), -max_velocity_grasp, max_velocity_grasp)
                    # velocity_x = torch.where(torch.abs(velocity_x) < min_velocity_grasp, torch.sign(velocity_x) * min_velocity_grasp, velocity_x)
                    # velocity_y = torch.where(torch.abs(velocity_y) < min_velocity_grasp, torch.sign(velocity_y) * min_velocity_grasp, velocity_y)
                    # velocity_x = velocity_x.reshape(1, 1)
                    # velocity_y = velocity_y.reshape(1, 1)
                    # velocity_angle = torch.clip((robot_base_place_angle[env_i] - joint_pos_base_angle[env_i]), -max_velocity_grasp, max_velocity_grasp)
                    # velocity_angle = torch.where(torch.abs(velocity_angle) < min_velocity_grasp, torch.sign(velocity_angle) * min_velocity_grasp, velocity_angle)
                    # velocity_angle = velocity_angle.reshape(1, 1)
                    # # height_target = height_target_place[env_i].reshape(1, 1)
                    # height_target = torch.clamp(height_target_place[env_i], min=height_target_limit).reshape(1, 1)

                    # ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat = math_utils.combine_frame_transforms(
                    #     base_support_pos_env_i.reshape(1, 3), base_support_quat_env_i.reshape(1, 4),
                    #     ee_frame_data_pos_after_grasp_relative_base_support_envs[env_i].reshape(1, 3), ee_frame_data_quat_after_grasp_relative_base_support_envs[env_i].reshape(1, 4)
                    # )

                    place_up_goals_iter = reach_place_up_traj_envs[env_i][min(max(iteration - reach_place_T_envs[env_i], 0), len(reach_place_up_traj_envs[env_i]) - 1)].reshape(1, 7)
                    place_up_goals_iter_to_fixed_support_frame_pos, place_up_goals_iter_to_fixed_support_frame_quat = math_utils.subtract_frame_transforms(
                        fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1],
                        place_up_goals_iter[:, :3], place_up_goals_iter[:, 3:7]
                    )

                    # actions = torch.cat(
                    #     [
                    #     velocity_y, velocity_x, velocity_angle, height_target,
                    #     # torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0,], device=env.sim.device).reshape(1, -1).repeat(num_envs, 1),
                    #     torch.tensor([-1.0,], device=env.sim.device).reshape(1, 1),
                    #     get_arm_action_relative(ee_frame_data_pos[env_i:env_i+1], ee_frame_data_quat[env_i:env_i+1], place_up_goals_iter[:, :3], place_up_goals_iter[:, 3:7])],
                    #     dim=-1
                    # )
                    actions = torch.cat([
                        get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], robot_base_place_pos[env_i:env_i+1], robot_base_place_quat[env_i:env_i+1]),
                        get_arm_action_relative(ee_frame_data_pos[env_i:env_i+1], ee_frame_data_quat[env_i:env_i+1], place_up_goals_iter[:, :3], place_up_goals_iter[:, 3:7]),
                        torch.tensor([-1.0], device=env.sim.device).reshape(1, 1),
                    ], dim=-1)

                else:
                    # velocity_x = torch.clip((robot_base_place_pos[env_i, 0] - base_support_pos_env_i[0]), -max_velocity_grasp, max_velocity_grasp)
                    # velocity_y = torch.clip((robot_base_place_pos[env_i, 1] - base_support_pos_env_i[1]), -max_velocity_grasp, max_velocity_grasp)
                    # velocity_x = torch.where(torch.abs(velocity_x) < min_velocity_grasp, torch.sign(velocity_x) * min_velocity_grasp, velocity_x)
                    # velocity_y = torch.where(torch.abs(velocity_y) < min_velocity_grasp, torch.sign(velocity_y) * min_velocity_grasp, velocity_y)
                    # velocity_x = velocity_x.reshape(1, 1)
                    # velocity_y = velocity_y.reshape(1, 1)
                    # velocity_angle = torch.clip((robot_base_place_angle[env_i] - joint_pos_base_angle[env_i]), -max_velocity_grasp, max_velocity_grasp)
                    # velocity_angle = torch.where(torch.abs(velocity_angle) < min_velocity_grasp, torch.sign(velocity_angle) * min_velocity_grasp, velocity_angle)
                    # velocity_angle = velocity_angle.reshape(1, 1)
                    # # height_target = height_target_place[env_i].reshape(1, 1)
                    # height_target = torch.clamp(torch.tensor([height_target_limit-place_down_offset*0.8+0.01], device=env.sim.device).reshape(1, 1), min=0.0).reshape(1, 1)

                    # ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat = math_utils.combine_frame_transforms(
                    #     base_support_pos_env_i.reshape(1, 3), base_support_quat_env_i.reshape(1, 4),
                    #     ee_frame_data_pos_after_grasp_relative_base_support_envs[env_i].reshape(1, 3), ee_frame_data_quat_after_grasp_relative_base_support_envs[env_i].reshape(1, 4)
                    # )

                    place_up_goals_iter = reach_place_up_traj_envs[env_i][min(max(iteration - reach_place_T_envs[env_i], 0), len(reach_place_up_traj_envs[env_i]) - 1)].reshape(1, 7)
                    place_up_goals_iter = place_up_goals_iter.clone()
                    place_up_goals_iter[:, 2] += -place_down_offset*0.8+0.1

                    # actions = torch.cat(
                    #     [
                    #     velocity_y, velocity_x, velocity_angle, height_target,
                    #     # torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0,], device=env.sim.device).reshape(1, -1).repeat(num_envs, 1),
                    #     torch.tensor([1.0 if iteration % T_grasp - reach_place_goal_T_envs[env_i] > 20 else -1.0,], device=env.sim.device).reshape(1, 1),
                    #     get_arm_action_relative(ee_frame_data_pos[env_i:env_i+1], ee_frame_data_quat[env_i:env_i+1], place_up_goals_iter[:, :3], place_up_goals_iter[:, 3:7])],
                    #     dim=-1
                    # )
                    actions = torch.cat([
                        get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], robot_base_place_pos[env_i:env_i+1], robot_base_place_quat[env_i:env_i+1]),
                        get_arm_action_relative(ee_frame_data_pos[env_i:env_i+1], ee_frame_data_quat[env_i:env_i+1], place_up_goals_iter[:, :3], place_up_goals_iter[:, 3:7]),
                        torch.tensor([1.0 if iteration % T_grasp - reach_place_goal_T_envs[env_i] > 20 else -1.0,], device=env.sim.device).reshape(1, 1),
                    ], dim=-1)


                actions_list.append(actions)

            actions = torch.cat(actions_list, dim=0)

            iteration_status = {}

            obs_dict = env.observation_manager.compute()


            for key, value in obs_dict["policy"].items():
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

            if current_data_idx < 5:
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

            iteration = (iteration + 1) % T_grasp

            if iteration % T_grasp == T_grasp - 1:

                if current_data_idx < 5:
                    os.makedirs(os.path.join(scene_save_dir, "debug", "cameras"), exist_ok=True)
                    imageio.mimwrite(os.path.join(scene_save_dir, "debug", "cameras", f"{current_data_idx:0>3d}_cameras.mp4"), camera_list, fps=30)
                    print(f"saving {current_data_idx} data to video: {os.path.join(scene_save_dir, 'debug', 'cameras', f'{current_data_idx:0>3d}_cameras.mp4')}")
                
                current_data_idx += 1


    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
