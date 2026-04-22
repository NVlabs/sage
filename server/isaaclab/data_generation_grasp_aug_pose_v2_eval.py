# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import torch

import argparse
from doctest import FAIL_FAST
import sys
import os

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Single-Obj-Scene-Franka-IK-Abs-v0", help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--num_demos", type=int, default=100, help="Number of episodes to store in the dataset.")
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
from isaaclab.curobo_tools.curobo_planner import MotionPlanner

from robomimic.algo import RolloutPolicy
import robomimic.utils.file_utils as FileUtils


def get_grasp_transforms(scene_save_dir, target_object_name, base_pos):
    
    layout = get_layout_from_scene_save_dir(scene_save_dir)
    mesh_dict_list = export_layout_to_mesh_dict_list_tree_search_with_object_id(layout, target_object_name)
    meta_data, vis_data = generate_m2t2_data(mesh_dict_list, target_object_name, base_pos)
    model, cfg = load_m2t2()
    grasp_transforms = infer_m2t2(meta_data, vis_data, model, cfg)

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

def get_grasp_object_height(scene_save_dir, target_object_name, grasp_loc):

    layout = get_layout_from_scene_save_dir(scene_save_dir)
    mesh_dict = export_layout_to_mesh_dict_object_id(layout, target_object_name)
    mesh = mesh_dict[target_object_name]["mesh"]

    return float(grasp_loc[..., 2].reshape(1) - mesh.vertices[:, 2].min())


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

def process_action(original_ee_goals, gripper_switch, ee_frame_data):
    # print(f"original_ee_goals: {original_ee_goals.shape}; {original_ee_goals}")
    original_ee_goals_pos = original_ee_goals[:3]
    original_ee_goals_quat = original_ee_goals[3:]

    abs_pose_pos = original_ee_goals_pos.reshape(-1, 3)
    abs_pose_quat = original_ee_goals_quat.reshape(-1, 4)


    gripper_vel = torch.tensor([gripper_switch], dtype=torch.float, device="cuda").reshape(-1, 1)

    actions = torch.cat([abs_pose_pos, abs_pose_quat, gripper_vel], dim=-1)

    return actions

def is_ee_reach_goal(ee_goal, ee_frame_data):

    ee_goal_pos = ee_goal[:3]
    ee_goal_quat = ee_goal[3:]

    ee_frame_data_pos = ee_frame_data.target_pos_source.reshape(3)
    ee_frame_data_quat = ee_frame_data.target_quat_source.reshape(4)

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

    loc_success = delta_pos.abs().max() < 0.02
    rot_success = delta_euler.abs().max() < 0.01

    # print(f"loc_success: {loc_success}; rot_success: {rot_success}")

    return loc_success

def get_default_camera_view():
    return [0., 0., 0.], [1., 0., 0.]

def get_default_base_pos():
    return [0., 0., 0.]

def sample_robot_location(
    scene_save_dir, layout_name, room_id, aug_name,
    target_object_name, place_object_name, table_object_name
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
    num_sample_points = 10000
    
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
    
    if len(valid_points) == 0:
        print("Warning: No valid robot positions found, using room center")
        robot_x = (room_min_x + room_max_x) / 2
        robot_y = (room_min_y + room_max_y) / 2
    else:
        # Find optimal point based on min-max distance to target and place objects
        target_pos = np.array([target_object.position.x, target_object.position.y])
        place_pos = np.array([place_object.position.x, place_object.position.y])
        
        # Calculate distances for each valid point
        target_distances = np.linalg.norm(valid_points - target_pos, axis=1)
        place_distances = np.linalg.norm(valid_points - place_pos, axis=1)
        max_distances = np.maximum(target_distances, place_distances)
        
        # Choose point with minimum of the maximum distances
        optimal_idx = np.argmin(max_distances)
        robot_x, robot_y = valid_points[optimal_idx]
    
    # Calculate robot z position based on table height
    table_mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, table_object_name)
    if table_mesh_info and table_mesh_info["mesh"] is not None:
        table_mesh = table_mesh_info["mesh"]
        table_height = np.max(table_mesh.vertices[:, 2])
    else:
        # Fallback to object position + dimensions
        table_height = table_object.position.z + table_object.dimensions.height
    
    robot_z = max(table_height - robot_height_offset, 0)
    
    # Calculate robot orientation towards midpoint of target and place objects
    target_pos_2d = np.array([target_object.position.x, target_object.position.y])
    place_pos_2d = np.array([place_object.position.x, place_object.position.y])
    midpoint = (target_pos_2d + place_pos_2d) / 2
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
        
        # Draw final robot position and orientation
        ax.scatter(robot_x, robot_y, c='red', s=200, marker='*', label='Robot position')
        
        # Draw orientation arrow
        arrow_length = 0.3
        arrow_dx = arrow_length * np.cos(yaw)
        arrow_dy = arrow_length * np.sin(yaw)
        ax.arrow(robot_x, robot_y, arrow_dx, arrow_dy, 
                head_width=0.05, head_length=0.05, fc='red', ec='red')
        
        ax.set_xlim(room_min_x - 0.5, room_max_x + 0.5)
        ax.set_ylim(room_min_y - 0.5, room_max_y + 0.5)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'Robot Location Sampling - Room {room_id}')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, alpha=0.3)
        
        debug_path = os.path.join(scene_save_dir, f"robot_sampling_debug_{room_id}.png")
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Debug visualization saved to: {debug_path}")
        
    except Exception as e:
        print(f"Warning: Could not save debug visualization: {e}")

    return robot_base_pos, robot_base_quat, camera_lookat

def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    log_dir = os.path.abspath(os.path.join("./robomimic_data", args_cli.task+"aug_pose_v2_reach_group_00"))
    # aug 100
    robomimic_policy_path = "augment/ckpts/dp_25_2000.pth"
    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(device="cuda", ckpt_path=robomimic_policy_path)
    
    eval_dir = os.path.join(log_dir, "eval_dp_25")
    os.makedirs(eval_dir, exist_ok=True)

    layout_id = "layout_625b9812"
    scene_save_dir = os.path.join(SERVER_ROOT_DIR, f"results/{layout_id}")
    
    room_id = "room_0736c934"
    target_object_name = "room_0736c934_ceramic_mug_with_handle_dcaede53"
    place_object_name = "room_0736c934_ceramic_bowl_empty_opening_e3270589"
    table_object_name = "room_0736c934_wooden_rectangular_coffee_table_ad3b7c58"

    aug_name = "aug_pose_v2_reach_group_00"
    aug_pose_json_name = f"all_augmented_layouts_info_{aug_name}.json"
    usd_collection_dir = f"usd_collection_{aug_name}"

    with open(os.path.join(scene_save_dir, aug_pose_json_name), "r") as f:
        all_augmented_layouts_info = json.load(f)
    
    mass_dict = all_augmented_layouts_info["mass_dict"]
    usd_collection_dir = all_augmented_layouts_info["usd_collection_dir"]
    object_transform_layouts_dict = all_augmented_layouts_info["object_transform_dict"]
    
    all_object_rigids = list(object_transform_layouts_dict[
        next(iter(object_transform_layouts_dict.keys()))
    ].keys())

    print(f"All training layouts number: {len(object_transform_layouts_dict)}")

    object_transform_layouts_dict_eval = {}

    layout_aug_ids = sorted(list(object_transform_layouts_dict.keys()))
    eval_layout_aug_ids = layout_aug_ids

    for layout_aug_id in eval_layout_aug_ids:
        object_transform_layouts_dict_eval[layout_aug_id] = object_transform_layouts_dict[layout_aug_id]

    print(f"Eval layouts number: {len(object_transform_layouts_dict_eval)}")

    print("starting evaluating the trained policy in eval set")

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

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).env

    # reset environment
    obs_dict, _ = env.reset()


    success_count = 0
    T_init = 20
    camera_height = 0.9
    camera_to_base = 0.4

    total_layout_count = len(object_transform_layouts_dict_eval.keys())
    sorted_layout_names = sorted(list(object_transform_layouts_dict_eval.keys()))
    print(f"Total layout count: {total_layout_count}")

    success_dict = {}

    # Repeat each layout name 5 times: [a, b, c] -> [a, a, a, a, a, b, b, b, b, b, c, c, c, c, c]
    attempts_per_layout = 5
    attempts_per_layout_dict = {layout_name: 0 for layout_name in sorted_layout_names}
    sorted_layout_names = [layout_name for layout_name in sorted_layout_names for _ in range(attempts_per_layout)]



    # simulate environment -- run everything in inference mode
    for layout_name in sorted_layout_names:
        print(f"Processing layout: {layout_name}")
        attempts_per_layout_dict[layout_name] += 1
        T_grasp = 600
        iteration = 0
        is_reach = False

        while True:

            if iteration == 0:

                # scene initialization
                iteration = 0
                sim = env.sim

                # scene index sampling
                object_transform_dict = object_transform_layouts_dict_eval[layout_name]

                # compute place location and related variables
                layout_json_path = os.path.join(scene_save_dir, aug_name, f"{layout_name}.json")
                place_location_w = get_place_location(layout_json_path, place_object_name)
                place_location_w = torch.tensor(place_location_w, device="cuda").float()
                print(f"place_location_w: {place_location_w.shape}; {place_location_w}")

                # set objects
                for object_name in all_object_rigids:
                    object_state = torch.zeros(1, 13).to(sim.device)
                    object_state[..., :3] = torch.tensor(object_transform_dict[object_name]["position"], device=sim.device)
                    object_state[..., 3:7] = torch.tensor(object_transform_dict[object_name]["rotation"], device=sim.device)
                    env.scene[object_name].write_root_state_to_sim(object_state)
                    env.scene[object_name].reset()

                # sample robot and camera locations
                robot_base_pos, robot_base_quat, camera_lookat = sample_robot_location(
                    scene_save_dir, layout_name, room_id, aug_name,
                    target_object_name, place_object_name, table_object_name
                )
                
                # set robot
                robot = env.scene["robot"]

                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)

                base_w = torch.zeros(1, 13).to(sim.device)
                base_w[..., :3] = robot_base_pos.reshape(1, 3)
                base_w[..., 3:7] = robot_base_quat.reshape(1, 4)

                robot.write_root_link_state_to_sim(base_w)
                robot.reset()
                robot_base_w = env.scene["robot"].data.root_state_w[:, :7]


                camera_left = env.scene["camera_left"]
                camera_pos_left, _ = math_utils.combine_frame_transforms(
                    robot_base_w[:, :3], robot_base_w[:, 3:7],
                    torch.tensor([0, camera_to_base, camera_height], device=env.sim.device).reshape(1, 3), torch.tensor([1, 0, 0, 0], device=env.sim.device).reshape(1, 4)
                )

                camera_left.set_world_poses_from_view(
                    camera_pos_left.reshape(1, 3),
                    camera_lookat.reshape(1, 3)
                )
                camera_left.reset()

                camera_right = env.scene["camera_right"]

                camera_pos_right, _ = math_utils.combine_frame_transforms(
                    robot_base_w[:, :3], robot_base_w[:, 3:7],
                    torch.tensor([0, -camera_to_base, camera_height], device=env.sim.device).reshape(1, 3), torch.tensor([1, 0, 0, 0], device=env.sim.device).reshape(1, 4)
                )

                camera_right.set_world_poses_from_view(
                    camera_pos_right.reshape(1, 3),
                    camera_lookat.reshape(1, 3)
                )
                camera_right.reset()

                camera_list = []

                rollout_policy.start_episode()
            def get_obs_dict_for_policy(obs_dict):
                return {
                    k: v[0] for k, v in obs_dict["policy"].items()
                }

            
            ee_frame_data = env.scene["ee_frame"].data
            iteration_status = {}
            obs_dict = env.observation_manager.compute()

            if iteration % T_grasp > T_init:
                actions = rollout_policy(ob=get_obs_dict_for_policy(obs_dict))
                if isinstance(actions, np.ndarray):
                    actions = torch.from_numpy(actions).to(sim.device)
                actions = actions.unsqueeze(0)
            else:
                ee_frame_data = env.scene["ee_frame"].data
                actions = torch.cat([ee_frame_data.target_pos_source.reshape(1, 3), ee_frame_data.target_quat_source.reshape(1, 4), torch.ones(1, 1).to(env.unwrapped.sim.device)], dim=-1)



            for key, value in obs_dict["policy"].items():
                iteration_status[f"obs/{key}"] = value
            iteration_status["actions"] = actions
            # perform action on environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated
            if env.sim.is_stopped():
                break

            for key, value in obs_dict["policy"].items():
                iteration_status[f"next_obs/{key}"] = value
            iteration_status["rewards"] = rewards
            iteration_status["dones"] = dones

            def depth_to_rgb(depth):
                depth = depth[..., None].repeat(3, axis=2)
                return depth

            if iteration % T_grasp > T_init:
                camera_list.append(
                    np.concatenate([
                        iteration_status["obs/rgb_left"][0].cpu().numpy(),
                        depth_to_rgb(iteration_status["obs/depth_left"][0, :, :, 0].cpu().numpy()),
                        iteration_status["obs/rgb_right"][0].cpu().numpy(),
                        depth_to_rgb(iteration_status["obs/depth_right"][0, :, :, 0].cpu().numpy()),
                        iteration_status["obs/rgb_wrist"][0].cpu().numpy(),
                        depth_to_rgb(iteration_status["obs/depth_wrist"][0, :, :, 0].cpu().numpy())], axis=1)
                )


            iteration_status = {}
            iteration = (iteration + 1) % T_grasp

            place_xy_w = torch.tensor([place_location_w[0], place_location_w[1]], device=env.sim.device).float()
            target_pos_xy_w = env.scene[target_object_name].data.root_state_w[:, :2].reshape(2)
            place_pos_xy_w = env.scene[place_object_name].data.root_state_w[:, :2].reshape(2)

            is_place_stay = torch.norm(place_pos_xy_w - place_xy_w) < 0.08
            is_target_reach = torch.norm(target_pos_xy_w - place_xy_w) < 0.08

            if not is_reach and is_place_stay and is_target_reach:
                print("success: reach place up")
                is_reach = True
                success_dict[layout_name] = success_dict.get(layout_name, 0) + 1
                # T_grasp = iteration + 30

            if iteration % T_grasp == 0:
                imageio.mimwrite(os.path.join(eval_dir, f"{layout_name}_{attempts_per_layout_dict[layout_name]}.mp4"), camera_list, fps=30)
                break
    
    print(f"Success dict: {success_dict}")
    
    # Calculate statistics
    unique_layouts = sorted(list(object_transform_layouts_dict_eval.keys()))
    total_attempts = len(sorted_layout_names)
    total_successes = sum(success_dict.values())
    
    print("\n=== EVALUATION STATISTICS ===")
    print(f"Total layouts evaluated: {len(unique_layouts)}")
    print(f"Attempts per layout: {attempts_per_layout}")
    print(f"Total attempts: {total_attempts}")
    print(f"Total successes: {total_successes}")
    print(f"Overall success rate: {total_successes / total_attempts:.3f} ({total_successes}/{total_attempts})")
    
    print("\n=== PER-LAYOUT SUCCESS RATES ===")
    layout_success_rates = []
    for layout_name in unique_layouts:
        successes = success_dict.get(layout_name, 0)
        success_rate = successes / attempts_per_layout
        layout_success_rates.append(success_rate)
        print(f"{layout_name}: {success_rate:.3f} ({successes}/{attempts_per_layout})")
    
    print("\n=== SUMMARY STATISTICS ===")
    if layout_success_rates:
        avg_success_rate = sum(layout_success_rates) / len(layout_success_rates)
        max_success_rate = max(layout_success_rates)
        min_success_rate = min(layout_success_rates)
        
        print(f"Average success rate per layout: {avg_success_rate:.3f}")
        print(f"Best performing layout success rate: {max_success_rate:.3f}")
        print(f"Worst performing layout success rate: {min_success_rate:.3f}")
        
        # Count layouts by success rate
        perfect_layouts = sum(1 for rate in layout_success_rates if rate == 1.0)
        good_layouts = sum(1 for rate in layout_success_rates if 0.6 <= rate < 1.0)
        poor_layouts = sum(1 for rate in layout_success_rates if 0.0 < rate < 0.6)
        failed_layouts = sum(1 for rate in layout_success_rates if rate == 0.0)
        
        # Layouts with at least one success
        layouts_with_success = sum(1 for rate in layout_success_rates if rate > 0.0)
        layouts_with_success_rate = layouts_with_success / len(layout_success_rates)
        
        print(f"Perfect layouts (100% success): {perfect_layouts}")
        print(f"Good layouts (60-99% success): {good_layouts}")
        print(f"Poor layouts (1-59% success): {poor_layouts}")
        print(f"Failed layouts (0% success): {failed_layouts}")
        print(f"Layouts with at least one success: {layouts_with_success}/{len(layout_success_rates)} ({layouts_with_success_rate:.3f})")
    else:
        print("No success rate data available")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
