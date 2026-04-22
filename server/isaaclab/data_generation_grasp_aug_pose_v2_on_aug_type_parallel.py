# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
from omni.isaac.lab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Single-Obj-Scene-Franka-IK-Abs-v0", help="Name of the task.")
parser.add_argument("--num_demos", type=int, default=1280, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
parser.add_argument("--type_aug_name", type=str, required=True, help="Name of the augmentation.")
parser.add_argument("--pose_aug_name", type=str, required=True, help="Name of the augmentation.")
parser.add_argument("--type_candidate_id", type=str, required=True, help="Name of the augmentation.")
parser.add_argument("--layout_id", type=str, required=True, help="Name of the layout.")
parser.add_argument("--room_id", type=str, required=True, help="Name of the room.")
parser.add_argument("--target_object_name", type=str, required=True, help="Name of the target object.")
parser.add_argument("--place_object_name", type=str, required=True, help="Name of the place object.")
parser.add_argument("--table_object_name", type=str, required=True, help="Name of the table object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from doctest import FAIL_FAST
import gc
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
# Add parent directory to Python path to import constants
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR, ROBOMIMIC_ROOT_DIR, M2T2_ROOT_DIR
print("SERVER_ROOT_DIR: ", SERVER_ROOT_DIR)
sys.path.insert(0, SERVER_ROOT_DIR)
sys.path.insert(0, M2T2_ROOT_DIR)

# Import utils functions using importlib to avoid conflicts with cv2.utils
import importlib.util
utils_spec = importlib.util.spec_from_file_location("server_utils", os.path.join(SERVER_ROOT_DIR, "utils.py"))
server_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(server_utils)

# Import the specific functions from server utils
dict_to_floor_plan = server_utils.dict_to_floor_plan
get_layout_from_scene_save_dir = server_utils.get_layout_from_scene_save_dir
get_layout_from_scene_json_path = server_utils.get_layout_from_scene_json_path

from tex_utils import (
    export_layout_to_mesh_dict_list_tree_search_with_object_id,
    export_layout_to_mesh_dict_object_id,
    get_textured_object_mesh
)
from m2t2_utils.data import generate_m2t2_data
from m2t2_utils.infer import load_m2t2, infer_m2t2
import trimesh
from models import FloorPlan
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml, load_yaml, load_pickle

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.data_collector import RobomimicDataCollector
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

##
# Pre-defined configs
##

import omni.isaac.lab.utils.math as math_utils
import omni.isaac.core.utils.prims as prim_utils
from isaaclab.curobo_tools.curobo_planner import MotionPlanner


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
    rotate_90 = R.from_euler('z', 90, degrees=True).as_matrix()
    grasp_transforms_rotation = grasp_transforms_rotation @ rotate_90

    grasp_transforms_updated = np.concatenate([grasp_transforms_rotation, grasp_transforms_translation], axis=2)
    
    grasp_transforms[:, :3, :] = grasp_transforms_updated

    return grasp_transforms


def get_grasp_transforms_with_additional_transform(layout_json_path, target_object_name, base_pos, additional_transform_dict):
    
    layout = get_layout_from_scene_json_path(layout_json_path)
    mesh_dict_list = export_layout_to_mesh_dict_list_tree_search_with_object_id(layout, target_object_name)

    for object_name in mesh_dict_list.keys():
        assert object_name in additional_transform_dict.keys(), f"object_name {object_name} not in additional_transform_dict"
        object_transform = additional_transform_dict[object_name]
        self_transform = mesh_dict_list[object_name]["transform"]
        mesh_dict_list[object_name]["mesh"].apply_transform(object_transform @ np.linalg.inv(self_transform))

    meta_data, vis_data = generate_m2t2_data(mesh_dict_list, target_object_name, base_pos)
    model, cfg = load_m2t2()
    while True:
        grasp_transforms = infer_m2t2(meta_data, vis_data, model, cfg)
        if grasp_transforms.shape[0] > 0:
            break

    grasp_transforms_rotation = grasp_transforms[:, :3, :3]
    grasp_transforms_translation = grasp_transforms[:, :3, 3].reshape(-1, 3, 1)

    # create a rotation matrix which rotates along the z axis by 90 degrees
    rotate_90 = R.from_euler('z', 90, degrees=True).as_matrix()
    grasp_transforms_rotation = grasp_transforms_rotation @ rotate_90

    grasp_transforms_updated = np.concatenate([grasp_transforms_rotation, grasp_transforms_translation], axis=2)
    
    grasp_transforms[:, :3, :] = grasp_transforms_updated

    return grasp_transforms


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


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)

def process_action(original_ee_goals, gripper_switch):
    # print(f"original_ee_goals: {original_ee_goals.shape}; {original_ee_goals}")
    original_ee_goals_pos = original_ee_goals[:3]
    original_ee_goals_quat = original_ee_goals[3:]

    abs_pose_pos = original_ee_goals_pos.reshape(-1, 3)
    abs_pose_quat = original_ee_goals_quat.reshape(-1, 4)


    gripper_vel = torch.tensor([gripper_switch], dtype=torch.float, device="cuda").reshape(-1, 1)

    actions = torch.cat([abs_pose_pos, abs_pose_quat, gripper_vel], dim=-1)

    return actions

def is_ee_reach_goal(ee_goal, ee_frame_data, env_i):

    ee_goal_pos = ee_goal[:3]
    ee_goal_quat = ee_goal[3:]

    ee_frame_data_pos = ee_frame_data.target_pos_source[env_i, :].reshape(3)
    ee_frame_data_quat = ee_frame_data.target_quat_source[env_i, :].reshape(4)

    delta_pos = ee_goal_pos - ee_frame_data_pos 


    target_quat_np = ee_goal_quat.cpu().numpy()
    current_quat_np = ee_frame_data_quat.cpu().numpy()
    
    # Convert to scipy Rotation objects (scalar-first format)
    target_rot = R.from_quat(target_quat_np[[1, 2, 3, 0]])  # Convert to (x, y, z, w)
    current_rot = R.from_quat(current_quat_np[[1, 2, 3, 0]])  # Convert to (x, y, z, w)
    
    # Calculate relative rotation: target = current * delta_rot
    # So delta_rot = current.inv() * target
    delta_rot = target_rot * current_rot.inv()
    
    # Convert to Euler angles (roll, pitch, yaw) in radians
    delta_euler = delta_rot.as_rotvec()
    delta_euler = torch.tensor(delta_euler, dtype=torch.float, device=ee_goal.device)

    loc_success = delta_pos.abs().max() < 0.03
    rot_success = delta_euler.abs().max() < 0.01

    # print(f"loc_success: {loc_success}; rot_success: {rot_success}")

    return loc_success

def get_default_camera_view():
    return [0., 0., 0.], [1., 0., 0.]

def get_default_base_pos():
    return [0., 0., 0.]

def sample_robot_location(
    scene_save_dir, layout_name, room_id, aug_name,
    target_object_name, place_object_name, table_object_name, num_envs
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
    2. among the remaining points, find the point that minimizes the maximum distance between robot-target_object and robot-place_object, 
    choose the point as the robot pos x and y, robot z is max(the height of the table top - robot_height_offset, 0) (use the table object height);
    3. the robot 3d rotation quaternion is a z-axis rotation from the robot pos towards the mid point of target_object_name and place_object_name.

    extra for debugging: save a 2d vis image of the room rectangle, object occupancy rectangles, all valid points, and the final robot pos and quat.
    """

    # Parameters
    robot_min_dist_to_room_edge = 0.5
    robot_min_dist_to_object = 0.15
    robot_height_offset = 0.2
    grid_res = 0.02
    num_sample_points = 100000
    
    layout_json_path = os.path.join(scene_save_dir, aug_name, f"{layout_name}.json")
    with open(layout_json_path, "r") as f:
        layout_info = json.load(f)
    
    floor_plan: FloorPlan = dict_to_floor_plan(layout_info)
    target_room = next(room for room in floor_plan.rooms if room.id == room_id)
    assert target_room is not None, f"target_room {room_id} not found in floor_plan"

    target_object = next(obj for obj in target_room.objects if obj.id == target_object_name)
    place_object = next(obj for obj in target_room.objects if obj.id == place_object_name)
    table_object = next(obj for obj in target_room.objects if obj.id == table_object_name)
    
    assert target_object is not None, f"target_object {target_object_name} not found in floor_plan"
    assert place_object is not None, f"place_object {place_object_name} not found in floor_plan"
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
                
                # Calculate distances from each valid point to all occupied cells
                # Shape: (n_valid_points, n_occupied_cells)
                distances = np.linalg.norm(
                    grid_valid_points[:, np.newaxis, :] - occupied_positions[np.newaxis, :, :], 
                    axis=2
                )
                
                # Find minimum distance to any occupied cell for each point
                min_distances = np.min(distances, axis=1)
                
                # Filter points that are far enough from occupied cells
                distance_valid_mask = min_distances >= robot_min_dist_to_object
                valid_points = grid_valid_points[distance_valid_mask]
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
        # Find optimal point based on min-max distance to target and place objects
        target_pos = np.array([target_object.position.x, target_object.position.y])
        place_pos = np.array([place_object.position.x, place_object.position.y])
        
        # Calculate distances for each valid point
        target_distances = np.linalg.norm(valid_points - target_pos, axis=1)
        place_distances = np.linalg.norm(valid_points - place_pos, axis=1)
        max_distances = np.maximum(target_distances, place_distances)
        
        # Choose optimal point with minimum of the maximum distances
        optimal_idx = np.argmin(max_distances)
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
    place_pos_2d = np.array([place_object.position.x, place_object.position.y])
    midpoint = (target_pos_2d + place_pos_2d) / 2
    
    robot_base_positions = []
    robot_base_quats = []
    camera_lookats = []
    
    for i in range(num_envs):
        robot_x, robot_y = robot_positions[i]
        
        # Calculate robot orientation towards midpoint of target and place objects
        direction_to_midpoint = midpoint - np.array([robot_x, robot_y])
        yaw = np.arctan2(direction_to_midpoint[1], direction_to_midpoint[0])
        
        # Convert to torch tensors on CUDA
        robot_base_pos = torch.tensor([robot_x, robot_y, robot_z], dtype=torch.float, device="cuda")
        robot_base_quat = torch.tensor(
            R.from_euler('z', yaw).as_quat(scalar_first=True), 
            dtype=torch.float, device="cuda"
        )
        
        camera_lookat = torch.tensor([
            (target_object.position.x + place_object.position.x) / 2, 
            (target_object.position.y + place_object.position.y) / 2, 
            table_height
        ], dtype=torch.float, device="cuda")
        
        robot_base_positions.append(robot_base_pos)
        robot_base_quats.append(robot_base_quat)
        camera_lookats.append(camera_lookat)
    
    # Stack into tensors
    robot_base_pos = torch.stack(robot_base_positions)
    robot_base_quat = torch.stack(robot_base_quats)
    camera_lookat = torch.stack(camera_lookats)

    return robot_base_pos, robot_base_quat, camera_lookat

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

def sample_robot_location_on_type_aug(
    scene_save_dir, layout_name, room_id, type_aug_name, type_candidate_id, pose_aug_name,
    target_object_name, place_object_name, table_object_name, num_envs
):
    """
    Modified version of sample_robot_location that works with type augmentation directory structure.
    Layout JSON files are now located at: scene_save_dir/type_aug_name/type_candidate_id/pose_aug_name/layout_name.json
    """

    # Parameters
    robot_min_dist_to_room_edge = 0.5
    robot_min_dist_to_object = 0.15
    robot_height_offset = 0.2
    grid_res = 0.02
    num_sample_points = 100000
    
    # Updated path for type augmentation structure
    layout_json_path = os.path.join(scene_save_dir, type_aug_name, type_candidate_id, pose_aug_name, f"{layout_name}.json")
    with open(layout_json_path, "r") as f:
        layout_info = json.load(f)
    
    floor_plan: FloorPlan = dict_to_floor_plan(layout_info)
    target_room = next(room for room in floor_plan.rooms if room.id == room_id)
    assert target_room is not None, f"target_room {room_id} not found in floor_plan"

    target_object = next(obj for obj in target_room.objects if obj.id == target_object_name)
    place_object = next(obj for obj in target_room.objects if obj.id == place_object_name)
    table_object = next(obj for obj in target_room.objects if obj.id == table_object_name)
    
    assert target_object is not None, f"target_object {target_object_name} not found in floor_plan"
    assert place_object is not None, f"place_object {place_object_name} not found in floor_plan"
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
                
                # Calculate distances from each valid point to all occupied cells
                # Shape: (n_valid_points, n_occupied_cells)
                distances = np.linalg.norm(
                    grid_valid_points[:, np.newaxis, :] - occupied_positions[np.newaxis, :, :], 
                    axis=2
                )
                
                # Find minimum distance to any occupied cell for each point
                min_distances = np.min(distances, axis=1)
                
                # Filter points that are far enough from occupied cells
                distance_valid_mask = min_distances >= robot_min_dist_to_object
                valid_points = grid_valid_points[distance_valid_mask]
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
        # Find optimal point based on min-max distance to target and place objects
        target_pos = np.array([target_object.position.x, target_object.position.y])
        place_pos = np.array([place_object.position.x, place_object.position.y])
        
        # Calculate distances for each valid point
        target_distances = np.linalg.norm(valid_points - target_pos, axis=1)
        place_distances = np.linalg.norm(valid_points - place_pos, axis=1)
        max_distances = np.maximum(target_distances, place_distances)
        
        # Choose optimal point with minimum of the maximum distances
        optimal_idx = np.argmin(max_distances)
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
    place_pos_2d = np.array([place_object.position.x, place_object.position.y])
    midpoint = (target_pos_2d + place_pos_2d) / 2
    
    robot_base_positions = []
    robot_base_quats = []
    camera_lookats = []
    
    for i in range(num_envs):
        robot_x, robot_y = robot_positions[i]
        
        # Calculate robot orientation towards midpoint of target and place objects
        direction_to_midpoint = midpoint - np.array([robot_x, robot_y])
        yaw = np.arctan2(direction_to_midpoint[1], direction_to_midpoint[0])
        
        # Convert to torch tensors on CUDA
        robot_base_pos = torch.tensor([robot_x, robot_y, robot_z], dtype=torch.float, device="cuda")
        robot_base_quat = torch.tensor(
            R.from_euler('z', yaw).as_quat(scalar_first=True), 
            dtype=torch.float, device="cuda"
        )
        
        camera_lookat = torch.tensor([
            (target_object.position.x + place_object.position.x) / 2, 
            (target_object.position.y + place_object.position.y) / 2, 
            table_height
        ], dtype=torch.float, device="cuda")
        
        robot_base_positions.append(robot_base_pos)
        robot_base_quats.append(robot_base_quat)
        camera_lookats.append(camera_lookat)
    
    # Stack into tensors
    robot_base_pos = torch.stack(robot_base_positions)
    robot_base_quat = torch.stack(robot_base_quats)
    camera_lookat = torch.stack(camera_lookats)

    # Save debug visualization
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Draw room rectangle
        room_rect = patches.Rectangle(
            (room_min_x, room_min_y), 
            room_max_x - room_min_x, 
            room_max_y - room_min_y,
            linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3
        )
        ax.add_patch(room_rect)
        
        # Draw occupancy grid
        for i in range(len(grid_x)):
            for j in range(len(grid_y)):
                if occupancy_grid[i, j]:
                    x = room_min_x + i * grid_res
                    y = room_min_y + j * grid_res
                    occupied_rect = patches.Rectangle(
                        (x, y), grid_res, grid_res,
                        facecolor='red', alpha=0.5
                    )
                    ax.add_patch(occupied_rect)
        
        # Draw valid points
        if len(valid_points) > 0:
            ax.scatter(valid_points[:, 0], valid_points[:, 1], 
                      c='green', alpha=0.5, s=1, label='Valid positions')
        
        # Draw object positions
        ax.scatter(target_object.position.x, target_object.position.y, 
                  c='blue', s=100, marker='s', label='Target object')
        ax.scatter(place_object.position.x, place_object.position.y, 
                  c='orange', s=100, marker='s', label='Place object')
        ax.scatter(table_object.position.x, table_object.position.y, 
                  c='brown', s=100, marker='s', label='Table object')
        
        # Draw robot positions for all environments
        for i in range(num_envs):
            robot_x, robot_y = robot_positions[i]
            
            # Calculate orientation for this environment
            direction_to_midpoint = midpoint - np.array([robot_x, robot_y])
            yaw = np.arctan2(direction_to_midpoint[1], direction_to_midpoint[0])
            
            # Draw robot position
            color = 'red' if i == 0 else 'darkred'  # Highlight first environment
            alpha = 1.0 if i == 0 else 0.7
            size = 200 if i == 0 else 100
            ax.scatter(robot_x, robot_y, c=color, s=size, marker='*', alpha=alpha,
                      label='Robot position' if i == 0 else None)
            
            # Draw orientation arrow
            arrow_length = 0.3 if i == 0 else 0.2
            arrow_dx = arrow_length * np.cos(yaw)
            arrow_dy = arrow_length * np.sin(yaw)
            ax.arrow(robot_x, robot_y, arrow_dx, arrow_dy, 
                    head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=alpha)
            
            # Add environment index label
            ax.annotate(f'E{i}', (robot_x + 0.1, robot_y + 0.1), fontsize=8, color=color)
        
        ax.set_xlim(room_min_x - 0.5, room_max_x + 0.5)
        ax.set_ylim(room_min_y - 0.5, room_max_y + 0.5)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'Robot Location Sampling - Room {room_id} ({num_envs} envs)')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, alpha=0.3)
        
        debug_path = os.path.join(scene_save_dir, f"robot_sampling_debug_{room_id}_{type_candidate_id}.png")
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Debug visualization saved to: {debug_path}")
        
    except Exception as e:
        print(f"Warning: Could not save debug visualization: {e}")

    return robot_base_pos, robot_base_quat, camera_lookat

def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    log_dir = os.path.abspath(os.path.join("./robomimic_data/", args_cli.task+args_cli.type_aug_name+args_cli.pose_aug_name+args_cli.type_candidate_id))
    os.makedirs(os.path.join(log_dir, "debug"), exist_ok=True)

    # scene parameters
    layout_id = args_cli.layout_id
    scene_save_dir = os.path.join(SERVER_ROOT_DIR, f"results/{layout_id}")
    
    room_id = args_cli.room_id
    type_aug_name = args_cli.type_aug_name
    pose_aug_name = args_cli.pose_aug_name
    type_candidate_id = args_cli.type_candidate_id
    
    # Original object names (from the base layout)
    original_target_object_name = args_cli.target_object_name
    original_place_object_name = args_cli.place_object_name
    original_table_object_name = args_cli.table_object_name
    
    num_envs = args_cli.num_envs

    # Load the type candidate data to get ID mappings
    type_candidate_path = os.path.join(scene_save_dir, type_aug_name, f"{type_candidate_id}_type_candidate.json")
    with open(type_candidate_path, "r") as f:
        type_candidate_data = json.load(f)
    
    old_id_to_new_id_map = type_candidate_data["old_id_to_new_id_map"]
    
    # Map original object IDs to new object IDs using the type augmentation mapping
    target_object_name = old_id_to_new_id_map.get(original_target_object_name, original_target_object_name)
    place_object_name = old_id_to_new_id_map.get(original_place_object_name, original_place_object_name)
    table_object_name = old_id_to_new_id_map.get(original_table_object_name, original_table_object_name)
    
    print(f"Object ID mapping:")
    print(f"  Target: {original_target_object_name} -> {target_object_name}")
    print(f"  Place: {original_place_object_name} -> {place_object_name}")
    print(f"  Table: {original_table_object_name} -> {table_object_name}")

    # Load augmentation metadata from the new location
    aug_pose_json_name = f"all_augmented_layouts_info_{pose_aug_name}.json"
    aug_pose_json_path = os.path.join(scene_save_dir, type_aug_name, type_candidate_id, aug_pose_json_name)
    
    with open(aug_pose_json_path, "r") as f:
        all_augmented_layouts_info = json.load(f)
    
    mass_dict = all_augmented_layouts_info["mass_dict"]
    print(f"mass_dict: {mass_dict}")
    usd_collection_dir = all_augmented_layouts_info["usd_collection_dir"]
    object_transform_layouts_dict = all_augmented_layouts_info["object_transform_dict"]
    all_object_rigids = list(object_transform_layouts_dict[
        next(iter(object_transform_layouts_dict.keys()))
    ].keys())

    print(f"All training layouts number: {len(object_transform_layouts_dict)}")

    object_transform_layouts_dict_train = {}

    layout_aug_ids = sorted(list(object_transform_layouts_dict.keys()))
    train_layout_aug_ids = layout_aug_ids

    for layout_aug_id in train_layout_aug_ids:
        object_transform_layouts_dict_train[layout_aug_id] = object_transform_layouts_dict[layout_aug_id]

    print(f"Train layouts number: {len(object_transform_layouts_dict_train)}")


    camera_pos, camera_lookat = get_default_camera_view()
    base_pos = get_default_base_pos()

    # create a yaml file to store the config
    config_dict = {
        "scene_save_dir": scene_save_dir,
        "usd_collection_dir": usd_collection_dir,
        "base_pos": base_pos,
        "camera_pos": camera_pos,
        "camera_lookat": camera_lookat,
        "mass_dict": mass_dict,
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

    collected_data_episodes = 0
    num_demos = 1280

    T_grasp = 400
    T_init = 20
    T_open_gap = 50
    T_close_gap = 15

    grasp_up_offset = 0.1
    grasp_down_offset = 0.01
    max_interpolation_step_distance = 0.01  # Maximum distance per interpolation step
    camera_height = 0.9
    camera_to_base = 0.4
    current_data_idx = 0

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


    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
        flush_freq=1,
        env_config={
            "cfg": os.path.join(log_dir, "params", "env.yaml"),
            "env_init_config_yaml": os.path.join(log_dir, "params", "env_init.yaml")
        },
    )
    collector_interface.reset()
    

    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt):
        while True:

            if iteration % T_grasp == 0:

                print(f"collected_data_episodes: {collected_data_episodes}; num_demos: {num_demos}; current_data_idx: {current_data_idx}; total_layouts: {len(object_transform_layouts_dict_train)}")

                if collected_data_episodes >= num_demos:
                    break

                if current_data_idx >= len(object_transform_layouts_dict_train):
                    break

                # scene initialization
                iteration = 0
                sim = env.sim

                # scene index sampling
                layout_name = sorted(list(object_transform_layouts_dict_train.keys()))[current_data_idx % len(object_transform_layouts_dict_train)]
                print(f"current_data_idx: {current_data_idx}; layout_name: {layout_name}")
                object_transform_dict = object_transform_layouts_dict_train[layout_name]
                
                # Layout JSON files are now in the pose augmentation directory
                layout_json_path = os.path.join(scene_save_dir, type_aug_name, type_candidate_id, pose_aug_name, f"{layout_name}.json")

                # compute place location
                place_location_w = get_place_location(layout_json_path, place_object_name)
                place_location_w = torch.tensor(place_location_w, device="cuda").float()

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

                # sample robot and camera locations - use the updated function with new directory structure
                robot_base_pos, robot_base_quat, camera_lookat = sample_robot_location_on_type_aug(
                    scene_save_dir, layout_name, room_id, type_aug_name, type_candidate_id, pose_aug_name,
                    target_object_name, place_object_name, table_object_name, num_envs
                )
                
                # set robot
                robot = env.scene["robot"]

                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)

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

                # place location to each env coordinate
                place_location_w_pos, place_location_w_quat = math_utils.combine_frame_transforms(
                    envs_pos, envs_quat,
                    place_location_w.reshape(1, 3).repeat(num_envs, 1), torch.tensor([1, 0, 0, 0], device=env.sim.device).reshape(1, 4).repeat(num_envs, 1)
                )
                place_location_w = torch.cat([place_location_w_pos, place_location_w_quat], dim=-1)

                # place location to each robot coordinate
                place_location_r_pos, place_location_r_quat = math_utils.subtract_frame_transforms(
                    robot_base_w[:, :3], robot_base_w[:, 3:7],
                    place_location_w[:, :3], place_location_w[:, 3:7]
                )
                place_location_r = torch.cat([place_location_r_pos, place_location_r_quat], dim=-1)

                target_object_height = get_grasp_object_height(layout_json_path, target_object_name)
                place_location_r[:, 2] = place_location_r[:, 2] + target_object_height + 0.15
                expected_success_height = target_object_height * 0.6

                camera_lookat, _ = math_utils.combine_frame_transforms(
                    envs_pos, envs_quat,
                    camera_lookat.reshape(-1, 3), torch.tensor([1, 0, 0, 0], device=env.sim.device).reshape(1, 4).repeat(num_envs, 1)
                )

                camera_left = env.scene["camera_left"]

                camera_pos_left, _ = math_utils.combine_frame_transforms(
                    robot_base_w[:, :3], robot_base_w[:, 3:7],
                    torch.tensor([0, camera_to_base, camera_height], device=env.sim.device).reshape(1, 3).repeat(num_envs, 1), torch.tensor([1, 0, 0, 0], device=env.sim.device).reshape(1, 4).repeat(num_envs, 1)
                )

                camera_left.set_world_poses_from_view(
                    camera_pos_left.reshape(-1, 3),
                    camera_lookat.reshape(-1, 3)
                )
                camera_left.reset()

                camera_right = env.scene["camera_right"]

                camera_pos_right, _ = math_utils.combine_frame_transforms(
                    robot_base_w[:, :3], robot_base_w[:, 3:7],
                    torch.tensor([0, -camera_to_base, camera_height], device=env.sim.device).reshape(1, 3).repeat(num_envs, 1), torch.tensor([1, 0, 0, 0], device=env.sim.device).reshape(1, 4).repeat(num_envs, 1)
                )

                camera_right.set_world_poses_from_view(
                    camera_pos_right.reshape(-1, 3),
                    camera_lookat.reshape(-1, 3)
                )
                camera_right.reset()
                
                camera_list = []

                data_buffer = []

            if iteration % T_grasp == T_init + 1:

                target_object_com_state = env.scene[target_object_name].data.root_com_state_w
                target_object_initial_z = float(target_object_com_state[:, 2].mean())

                # Sample grasp poses for all environments
                ee_goals = sample_grasp(layout_json_path, target_object_name, robot_base_pos[0, :3], num_envs, env.sim.device)

                # ee_goals to each env coordinate
                ee_goals_translate, ee_goals_quat = math_utils.combine_frame_transforms(
                    envs_pos, envs_quat,
                    ee_goals[:, :3], ee_goals[:, 3:7]
                )
                ee_goals = torch.cat([ee_goals_translate, ee_goals_quat], dim=1)

                # ee_goals to each robot coordinate
                ee_goals_translate, ee_goals_quat = math_utils.subtract_frame_transforms(
                    robot_base_w[:, :3], robot_base_w[:, 3:7],
                    ee_goals[:, :3], ee_goals[:, 3:7]
                )
                ee_goals = torch.cat([ee_goals_translate, ee_goals_quat], dim=1)

                ee_goals_grasp_up = ee_goals.clone()
                ee_goals_grasp_up[:, 2] = ee_goals_grasp_up[:, 2] + grasp_up_offset

                ee_goals_grasp_down = ee_goals.clone()
                ee_goals_grasp_down[:, 2] = ee_goals_grasp_down[:, 2] - grasp_down_offset

                ee_goals_lift_up = ee_goals.clone()
                ee_goals_lift_up[:, 2] = place_location_r[:, 2]

                ee_goals_place = place_location_r.clone()

                robot_qpos = robot.data.joint_pos
                ee_frame_data = env.scene["ee_frame"].data
                current_ee_pose = torch.cat([
                    ee_frame_data.target_pos_source.reshape(-1, 3), 
                    ee_frame_data.target_quat_source.reshape(-1, 4)], dim=-1)

                reach_up_traj_envs = {}
                reach_down_traj_envs = {}
                lift_up_traj_envs = {}
                place_traj_envs = {}

                is_reach_grasp_up_goal_envs = []
                is_reach_grasp_down_goal_envs = []
                is_reach_lift_up_goal_envs = []
                is_reach_place_goal_envs = []
                is_start_lift_envs = []

                close_iters_envs = []
                T_reach_up_envs = {}
                T_start_lift_envs = {}
                T_start_place_envs = {}
                T_reach_place_envs = {}

                for env_i in range(num_envs):

                    print(f"planning reach up traj for env {env_i}")

                    reach_up_traj, success = curobo_plan_traj(
                        motion_planner,
                        robot_qpos[env_i:env_i+1, :],
                        current_ee_pose[env_i:env_i+1, :3],
                        ee_goals_grasp_up[env_i:env_i+1, :3],
                        ee_goals_grasp_up[env_i:env_i+1, 3:7],
                    )
                    reach_up_traj_envs[env_i] = reach_up_traj
                    is_reach_grasp_up_goal_envs.append(False)
                    is_reach_grasp_down_goal_envs.append(False)
                    is_reach_lift_up_goal_envs.append(False)
                    is_reach_place_goal_envs.append(False)
                    is_start_lift_envs.append(False)
                    close_iters_envs.append(T_close_gap)

            ee_frame_data = env.scene["ee_frame"].data
            # print(f"ee_frame_data: {ee_frame_data.target_pos_source.shape}; {ee_frame_data.target_pos_source}")
            target_object_com_state = env.scene[target_object_name].data.root_com_state_w
            if iteration % T_grasp > T_init:
                actions = []
                robot_qpos = robot.data.joint_pos
                ee_frame_data = env.scene["ee_frame"].data
                current_ee_pose = torch.cat([
                    ee_frame_data.target_pos_source.reshape(-1, 3), 
                    ee_frame_data.target_quat_source.reshape(-1, 4)], 
                    dim=-1
                )

                for env_i in range(num_envs):
                    is_reach_grasp_up_goal = is_reach_grasp_up_goal_envs[env_i]
                    is_reach_grasp_down_goal = is_reach_grasp_down_goal_envs[env_i]
                    is_reach_lift_up_goal = is_reach_lift_up_goal_envs[env_i]
                    is_reach_place_goal = is_reach_place_goal_envs[env_i]
                    is_start_lift = is_start_lift_envs[env_i]
                 
                    target_object_current_z = float(target_object_com_state[env_i, 2])

                    if not is_reach_grasp_up_goal:
                        is_reach_grasp_up_goal = is_ee_reach_goal(ee_goals_grasp_up[env_i, :].reshape(-1), ee_frame_data, env_i)
                        is_reach_grasp_up_goal_envs[env_i] = is_reach_grasp_up_goal
                        if is_reach_grasp_up_goal:
                            print(f"planning reach down traj for env {env_i}")
                            reach_down_traj_envs[env_i], _ = curobo_plan_traj(
                                motion_planner,
                                robot_qpos[env_i:env_i+1, :],
                                current_ee_pose[env_i:env_i+1, :3],
                                ee_goals_grasp_down[env_i:env_i+1, :3],
                                ee_goals_grasp_down[env_i:env_i+1, 3:7],
                                interpolate=True
                            )
                            T_reach_up_envs[env_i] = iteration

                    elif not is_reach_grasp_down_goal:
                        is_reach_grasp_down_goal = is_ee_reach_goal(ee_goals_grasp_down[env_i, :].reshape(-1), ee_frame_data, env_i)
                        is_reach_grasp_down_goal_envs[env_i] = is_reach_grasp_down_goal
                        # if is_reach_grasp_down_goal:
                        #     print(f"planning reach up traj for env {env_i}")
                        #     reach_up_traj_envs[env_i], _ = curobo_plan_traj(
                        #         motion_planner,
                        #         robot_qpos[env_i:env_i+1, :],
                        #         current_ee_pose[env_i:env_i+1, :3],
                        #         ee_goals_lift_up[env_i:env_i+1, :3],
                        #         ee_goals_lift_up[env_i:env_i+1, 3:7],
                        #     )

                    elif not is_reach_lift_up_goal:
                        is_reach_lift_up_goal = is_ee_reach_goal(ee_goals_lift_up[env_i, :].reshape(-1), ee_frame_data, env_i) and target_object_current_z - target_object_initial_z > expected_success_height
                        is_reach_lift_up_goal_envs[env_i] = is_reach_lift_up_goal

                        if is_reach_lift_up_goal:
                            ee_frame_data_pos = ee_frame_data.target_pos_source[env_i, :].reshape(3)
                            ee_frame_data_quat = ee_frame_data.target_quat_source[env_i, :].reshape(4)
                            
                            robot_base_w = env.scene["robot"].data.root_state_w[env_i:env_i+1, :7]
                            target_object_w = env.scene[target_object_name].data.root_state_w[env_i:env_i+1, :7]
                            target_object_r_pos, target_object_r_quat = math_utils.subtract_frame_transforms(
                                robot_base_w[:, :3], robot_base_w[:, 3:7],
                                target_object_w[:, :3], target_object_w[:, 3:7]
                            )

                            target_to_ee_pos = (ee_frame_data_pos.reshape(3) - target_object_r_pos.reshape(3))
                            target_to_ee_pos[2] = 0
                            ee_goals_place[env_i, :2] += target_to_ee_pos.reshape(3)[:2]

                            print(f"planning place traj for env {env_i}")
                            place_traj_envs[env_i], _ = curobo_plan_traj(
                                motion_planner,
                                robot_qpos[env_i:env_i+1, :],
                                current_ee_pose[env_i:env_i+1, :3],
                                ee_goals_place[env_i:env_i+1, :3],
                                current_ee_pose[env_i:env_i+1, 3:7],
                            )
                            place_traj_envs[env_i][:, 2] = torch.clip(place_traj_envs[env_i][:, 2], min=ee_goals_place[env_i, 2])
                            place_traj_envs[env_i][:, 3:7] = current_ee_pose[env_i:env_i+1, 3:7]
                            place_traj_envs[env_i] = place_traj_envs[env_i].reshape(-1, 1, 7).repeat(1, 2, 1).reshape(-1, 7)

                            T_start_place_envs[env_i] = iteration
                    
                    elif not is_reach_place_goal:
                        ee_reach_place_goal = is_ee_reach_goal(ee_goals_place[env_i, :].reshape(-1), ee_frame_data, env_i)

                        place_xy_w = torch.tensor([place_location_w[env_i, 0], place_location_w[env_i, 1]], device=env.sim.device).float()
                        target_pos_xy_w = env.scene[target_object_name].data.root_state_w[env_i, :2].reshape(2)
                        place_pos_xy_w = env.scene[place_object_name].data.root_state_w[env_i, :2].reshape(2)

                        is_place_stay = torch.norm(place_pos_xy_w - place_xy_w) < 0.08
                        is_target_reach = torch.norm(target_pos_xy_w - place_xy_w) < 0.08
                        is_reach_place_goal = ee_reach_place_goal and is_place_stay and is_target_reach

                        is_reach_place_goal_envs[env_i] = is_reach_place_goal
                        if is_reach_place_goal:
                            print(f"is_reach_place_goal for env {env_i}: {is_reach_place_goal}")
                            T_reach_place_envs[env_i] = iteration



                    if not is_reach_grasp_up_goal:
                        ee_goal = reach_up_traj_envs[env_i][min(max(iteration - T_init, 0), len(reach_up_traj_envs[env_i]) - 1)]
                        gripper_switch = 1
                    elif not is_reach_grasp_down_goal:
                        T_reach_up = T_reach_up_envs[env_i]
                        ee_goal = reach_down_traj_envs[env_i][min(max(iteration - T_reach_up, 0), len(reach_down_traj_envs[env_i]) - 1)]
                        gripper_switch = 1
                    elif not is_reach_lift_up_goal:
                        gripper_switch = -1
                        close_iters = close_iters_envs[env_i]
                        if close_iters > 0:
                            close_iters -= 1
                            close_iters_envs[env_i] = close_iters
                            ee_goal = reach_down_traj_envs[env_i][-1]
                        elif not is_start_lift and close_iters == 0:
                            is_start_lift = True
                            is_start_lift_envs[env_i] = True
                            T_start_lift = iteration
                            T_start_lift_envs[env_i] = T_start_lift
                            print(f"planning lift up traj for env {env_i}")
                            lift_up_traj_envs[env_i], _ = curobo_plan_traj(
                                motion_planner,
                                robot_qpos[env_i:env_i+1, :],
                                current_ee_pose[env_i:env_i+1, :3],
                                ee_goals_lift_up[env_i:env_i+1, :3],
                                ee_goals_lift_up[env_i:env_i+1, 3:7],
                                interpolate=True
                            )
                            ee_goal = lift_up_traj_envs[env_i][min(max(iteration - T_start_lift, 0), len(lift_up_traj_envs[env_i]) - 1)]
                        else:
                            T_start_lift = T_start_lift_envs[env_i]
                            ee_goal = lift_up_traj_envs[env_i][min(max(iteration - T_start_lift, 0), len(lift_up_traj_envs[env_i]) - 1)]
                    else:
                        T_start_place = T_start_place_envs[env_i]
                        ee_goal = place_traj_envs[env_i][min(max(iteration - T_start_place, 0), len(place_traj_envs[env_i]) - 1)]
                        is_reach_place_goal = is_reach_place_goal_envs[env_i]
                        if is_reach_place_goal:
                            gripper_switch = 1
                        else:    
                            gripper_switch = -1

                    # print(f"action for env {env_i}: {ee_goal.shape}, {gripper_switch}")

                    actions_env_i = process_action(ee_goal, gripper_switch)
                    actions.append(actions_env_i)
                actions = torch.cat(actions, dim=0)
            else:
                actions = torch.cat([ee_frame_data.target_pos_source.reshape(-1, 3), ee_frame_data.target_quat_source.reshape(-1, 4), torch.ones(num_envs, 1).to(env.sim.device)], dim=-1)

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

            if iteration % T_grasp > T_init:
                data_buffer.append(iteration_status)

            if iteration % T_grasp > T_init and current_data_idx < 5:
                camera_frame_envs = []
                for env_i in range(min(num_envs, 8)):
                    camera_frame_envs.append(
                        np.concatenate([
                            iteration_status["obs/rgb_left"][env_i].cpu().numpy(),
                            depth_to_rgb(iteration_status["obs/depth_left"][env_i, :, :, 0].cpu().numpy()),
                            iteration_status["obs/rgb_right"][env_i].cpu().numpy(),
                            depth_to_rgb(iteration_status["obs/depth_right"][env_i, :, :, 0].cpu().numpy()),
                            iteration_status["obs/rgb_wrist"][env_i].cpu().numpy(),
                            depth_to_rgb(iteration_status["obs/depth_wrist"][env_i, :, :, 0].cpu().numpy())], axis=1)
                    )
                camera_frame_envs = np.concatenate(camera_frame_envs, axis=0)
                camera_list.append(camera_frame_envs)

            if iteration % T_grasp == T_grasp - 1:
                if current_data_idx < 5:
                    imageio.mimwrite(os.path.join(log_dir, "debug", f"{current_data_idx:0>3d}_cameras.mp4"), camera_list, fps=30)
                    print(f"saving {current_data_idx} data to video")
                current_data_idx += 1
                for env_i in range(num_envs):
                    is_reach_place_goal = is_reach_place_goal_envs[env_i]
                    if is_reach_place_goal:
                        for data_i, iteration_i in enumerate(range(T_init+1, min(T_reach_place_envs[env_i] + T_open_gap, T_grasp))):
                            for key, value in data_buffer[data_i].items():
                                collector_interface.add(key, value[env_i:env_i+1])
                        collector_interface.flush()
                        collected_data_episodes += 1
                
                data_buffer = []
                torch.cuda.empty_cache()
                gc.collect()


            iteration = (iteration + 1) % T_grasp

    collector_interface.close()

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
