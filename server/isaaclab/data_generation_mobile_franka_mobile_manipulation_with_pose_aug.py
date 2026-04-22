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
total_recording_num = 20

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=num_envs, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Mobile-Manipulation-Obj-Scene-Franka-IK-Abs-v3", help="Name of the task.")
parser.add_argument("--num_demos", type=int, default=128, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
parser.add_argument("--visualize_figs", action="store_true", help="Generate visualization figures for debugging (default: False)")
parser.add_argument("--post_fix", type=str, default="", help="Postfix for the log directory")
parser.add_argument("--pose_aug_name", type=str, default="", help="Pose augmentation name")
parser.add_argument("--layout_id", type=str, default="", help="Layout ID")

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


import importlib.util
utils_spec = importlib.util.spec_from_file_location("server_objects_mm_utils", os.path.join(SERVER_ROOT_DIR, "objects", "object_mobile_manipulation_utils.py"))
server_objects_mm_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(server_objects_mm_utils)

# Import the specific functions from server utils
CollisionCheckingConfig = server_objects_mm_utils.CollisionCheckingConfig
create_unified_occupancy_grid = server_objects_mm_utils.create_unified_occupancy_grid
create_unified_scene_occupancy_fn = server_objects_mm_utils.create_unified_scene_occupancy_fn
check_unified_robot_collision = server_objects_mm_utils.check_unified_robot_collision
visualize_robot_place_planning_data = server_objects_mm_utils.visualize_robot_place_planning_data
visualize_robot_planning_data = server_objects_mm_utils.visualize_robot_planning_data
sample_robot_location = server_objects_mm_utils.sample_robot_location
create_table_occupancy_grid = server_objects_mm_utils.create_table_occupancy_grid
sample_robot_place_location = server_objects_mm_utils.sample_robot_place_location
visualize_robot_spawn_planning_data = server_objects_mm_utils.visualize_robot_spawn_planning_data
sample_robot_spawn = server_objects_mm_utils.sample_robot_spawn
visualize_multi_env_trajectory_planning = server_objects_mm_utils.visualize_multi_env_trajectory_planning
visualize_trajectory_planning = server_objects_mm_utils.visualize_trajectory_planning
plan_robot_traj = server_objects_mm_utils.plan_robot_traj


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

def mm_arm_abs_to_base_rel(actions, fixed_support_frame_data_pose):
    # print("actions: ", actions.shape)
    ee_target_pose = actions[:, 6:6+7]
    # print("ee_target_pose: ", ee_target_pose.shape)
    ee_target_pose_to_fixed_support_frame_pos, ee_target_pose_to_fixed_support_frame_quat = math_utils.subtract_frame_transforms(
        fixed_support_frame_data_pose[:, :3], fixed_support_frame_data_pose[:, 3:7],
        ee_target_pose[:, :3], ee_target_pose[:, 3:7],
    )
    pos_z_offset = -0.5
    ee_target_pose_to_fixed_support_frame_pos[:, 2] += pos_z_offset
    ee_target_pose_to_fixed_support_frame_pose = torch.cat([ee_target_pose_to_fixed_support_frame_pos, ee_target_pose_to_fixed_support_frame_quat], dim=-1)
    processed_actions = actions.clone()
    processed_actions[:, 6:6+7] = ee_target_pose_to_fixed_support_frame_pose
    # print("ee_target_pose_to_fixed_support_frame_pose: ", processed_actions[:, 6:6+7])
    return processed_actions

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
    current_ee_quat=None,  # Current end-effector quaternion for SLERP interpolation
):
    """
    robot_qpos: (1, num_joints)
    current_ee_pos: (1, 3)
    target_ee_pos: (1, 3)
    target_ee_quat: (1, 4)
    current_ee_quat: (1, 4) - Current end-effector quaternion for SLERP interpolation (optional, required when interpolate=True)

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
        
        # Validate that current_ee_quat is provided when interpolate=True
        if current_ee_quat is None:
            raise ValueError("current_ee_quat must be provided when interpolate=True for proper quaternion interpolation")

        # Calculate total distance and number of steps needed
        total_distance = torch.norm(target_ee_pos - current_ee_pos).item()
        num_steps = max(1, int(torch.ceil(torch.tensor(total_distance / max_interpolation_step_distance)).item()))
        
        # Helper function for torch SLERP (Spherical Linear Interpolation)
        def slerp_torch(q1, q2, t):
            """
            Spherical Linear Interpolation for quaternions in PyTorch.
            
            Args:
                q1: Start quaternion [w, x, y, z] (scalar-first format)
                q2: End quaternion [w, x, y, z] (scalar-first format)
                t: Interpolation parameter (0.0 to 1.0)
            
            Returns:
                Interpolated quaternion [w, x, y, z]
            """
            # Ensure quaternions are normalized
            q1 = q1 / torch.norm(q1)
            q2 = q2 / torch.norm(q2)
            
            # Compute dot product
            dot = torch.sum(q1 * q2)
            
            # If dot product is negative, negate one quaternion to take shorter path
            if dot < 0.0:
                q2 = -q2
                dot = -dot
            
            # If quaternions are very close, use linear interpolation to avoid numerical issues
            if dot > 0.995:
                result = q1 + t * (q2 - q1)
                return result / torch.norm(result)
            
            # Calculate angle between quaternions
            theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
            sin_theta_0 = torch.sin(theta_0)
            theta = theta_0 * t
            sin_theta = torch.sin(theta)
            
            # Calculate interpolation weights
            s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0
            
            return s0 * q1 + s1 * q2
        
        # Create interpolated trajectory with both position and quaternion interpolation
        interpolated_positions = []
        interpolated_quaternions = []
        
        current_ee_quat_normalized = current_ee_quat / torch.norm(current_ee_quat)
        target_ee_quat_normalized = target_ee_quat / torch.norm(target_ee_quat)
        
        for i in range(num_steps + 1):
            alpha = i / num_steps
            
            # Interpolate position linearly
            interp_pos = current_ee_pos + alpha * (target_ee_pos - current_ee_pos)
            interpolated_positions.append(interp_pos.reshape(-1))
            
            # Interpolate quaternion using SLERP
            interp_quat = slerp_torch(
                current_ee_quat_normalized.reshape(-1), 
                target_ee_quat_normalized.reshape(-1), 
                alpha
            )
            interpolated_quaternions.append(interp_quat.reshape(-1))
        
        # Stack positions and quaternions
        interpolated_positions = torch.stack(interpolated_positions)
        interpolated_quaternions = torch.stack(interpolated_quaternions)
        
        curobo_target_ee_pose = torch.cat([
            interpolated_positions,
            interpolated_quaternions,
        ], dim=1).float()

        if curobo_target_ee_pose.shape[0] > max_length:
            selected_indices = torch.from_numpy(np.linspace(0, curobo_target_ee_pose.shape[0] - 1, max_length).astype(np.int32)).to(curobo_target_ee_pose.device)
            curobo_target_ee_pose = curobo_target_ee_pose[selected_indices]
            print(f"Trajectory truncated to {max_length} steps from {curobo_target_ee_pose.shape[0]} steps")

        print(f"Trajectory Generated: position + quaternion interpolation | len {len(curobo_target_ee_pose)}")

        return curobo_target_ee_pose, False

def interpolate_pose(
    start_pos,
    start_quat,
    end_pos,
    end_quat,
    steps,
    max_interpolation_step_distance=0.001,
):
    # linear interpolation

    # Calculate total distance and number of steps needed
    total_distance = torch.norm(end_pos - start_pos).item()
    num_steps = max(1, int(torch.ceil(torch.tensor(total_distance / max_interpolation_step_distance)).item()))
    
    # Helper function for torch SLERP (Spherical Linear Interpolation)
    def slerp_torch(q1, q2, t):
        """
        Spherical Linear Interpolation for quaternions in PyTorch.
        
        Args:
            q1: Start quaternion [w, x, y, z] (scalar-first format)
            q2: End quaternion [w, x, y, z] (scalar-first format)
            t: Interpolation parameter (0.0 to 1.0)
        
        Returns:
            Interpolated quaternion [w, x, y, z]
        """
        # Ensure quaternions are normalized
        q1 = q1 / torch.norm(q1)
        q2 = q2 / torch.norm(q2)
        
        # Compute dot product
        dot = torch.sum(q1 * q2)
        
        # If dot product is negative, negate one quaternion to take shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are very close, use linear interpolation to avoid numerical issues
        if dot > 0.995:
            result = q1 + t * (q2 - q1)
            return result / torch.norm(result)
        
        # Calculate angle between quaternions
        theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
        sin_theta_0 = torch.sin(theta_0)
        theta = theta_0 * t
        sin_theta = torch.sin(theta)
        
        # Calculate interpolation weights
        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
    # Create interpolated trajectory with both position and quaternion interpolation
    interpolated_positions = []
    interpolated_quaternions = []
    
    current_ee_quat_normalized = start_quat / torch.norm(start_quat)
    target_ee_quat_normalized = end_quat / torch.norm(end_quat)
    
    for i in range(num_steps + 1):
        alpha = i / num_steps
        
        # Interpolate position linearly
        interp_pos = start_pos + alpha * (end_pos - start_pos)
        interpolated_positions.append(interp_pos.reshape(-1))
        
        # Interpolate quaternion using SLERP
        interp_quat = slerp_torch(
            current_ee_quat_normalized.reshape(-1), 
            target_ee_quat_normalized.reshape(-1), 
            alpha
        )
        interpolated_quaternions.append(interp_quat.reshape(-1))
    
    # Stack positions and quaternions
    interpolated_positions = torch.stack(interpolated_positions)
    interpolated_quaternions = torch.stack(interpolated_quaternions)
    
    curobo_target_ee_pose = torch.cat([
        interpolated_positions,
        interpolated_quaternions,
    ], dim=1).float()

    if curobo_target_ee_pose.shape[0] > steps:
        selected_indices = torch.from_numpy(np.linspace(0, curobo_target_ee_pose.shape[0] - 1, steps).astype(np.int32)).to(curobo_target_ee_pose.device)
        curobo_target_ee_pose = curobo_target_ee_pose[selected_indices]
        print(f"Trajectory truncated to {steps} steps from {curobo_target_ee_pose.shape[0]} steps")

    print(f"Trajectory Generated: position + quaternion interpolation | len {len(curobo_target_ee_pose)}")

    return curobo_target_ee_pose[:, :3], curobo_target_ee_pose[:, 3:]
    

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

def get_state_binary_tensor(list_of_states):
    num_envs = len(list_of_states[0])
    num_states = len(list_of_states)

    states = torch.zeros(num_envs, num_states, device="cuda").float()

    for i in range(num_states):
        assert len(list_of_states[i]) == num_envs, f"list_of_states[{i}] has {len(list_of_states[i])} states, but num_envs is {num_envs}"
        for j in range(num_envs):
            if list_of_states[i][j]:
                states[j, i] = 1.0
            else:
                states[j, i] = -1.0

    return states

def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    log_dir = os.path.abspath(os.path.join("./robomimic_data", args_cli.task+"-"+args_cli.post_fix))
    debug_dir = os.path.join(log_dir, "debug")

    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir, exist_ok=True)

    layout_id = args_cli.layout_id

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

    pose_aug_dir = args_cli.pose_aug_name
    meta_json_path = os.path.join(SERVER_ROOT_DIR, f"results/{layout_id}/{pose_aug_dir}/meta.json")
    meta_layouts = json.load(open(meta_json_path, "r"))["layouts"]
    print(f"loading {len(meta_layouts)} layouts from {meta_json_path}")
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
    
    # collection process
    iteration = 0
    current_data_idx = 0
    collected_data_episodes = 0

    # some motion planning parameters
    T_grasp = 2000
    T_hold = 80
    T_init = 20
    T_grasp_init = 35
    T_drop = 100
    

    # pick and place offsets
    grasp_up_offset = 0.1
    place_up_offset = 0.8
    place_down_offset = 0.5
    grasp_down_offset = 0.03

    # joint names
    joint_name_base_x = "base_joint_mobile_side"
    joint_name_base_y = "base_joint_mobile_forward"
    joint_name_base_angle = "base_joint_mobile_yaw"

    # task decomposition policy parse results
    robot_task_dict_list = layout_data["policy_analysis"]["updated_task_decomposition"]
    
    # Extract room_id from the first task with an object
    room_id = None
    for task in robot_task_dict_list:
        if "target_object_id" in task and task["target_object_id"]:
            # Extract room_id from object_id format: room_XXX_objecttype_YYY
            object_id_parts = task["target_object_id"].split("_")
            if len(object_id_parts) >= 2:
                room_id = "_".join(object_id_parts[:2])  # room_XXX
                break
    
    if room_id is None:
        raise ValueError("Could not extract room_id from robot_task_dict_list")
    
    print(f"Extracted room_id: {room_id}")

    # create motion planner
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

    # create collector interface
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
        while not collector_interface.is_stopped():

            if iteration % T_grasp == 0:

                layout_dict_current = meta_layouts[current_data_idx % len(meta_layouts)]
                layout_name = layout_dict_current["layout_id"]
                layout_json_path = os.path.join(scene_save_dir, f"{layout_name}.json")
                rigid_object_transform_dict = layout_dict_current["rigid_object_transform_dict"]

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
                
                # Sample new spawn positions for this episode
                spawn_pos, spawn_angles = sample_robot_spawn(
                    scene_save_dir, layout_name, room_id, num_envs, debug_dir
                )

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
                    if object_name in object_transform_dict:
                        object_position = object_transform_dict[object_name]["position"]
                        object_rotation = object_transform_dict[object_name]["rotation"]
                    else:
                        object_position = np.random.uniform(1000, 2000, (num_envs, 3))
                        object_rotation = np.zeros((num_envs, 4))
                        object_rotation[:, 0] = 1
                        print(f"object {object_name} not found in object_transform_dict, randomly generated position and rotation")

                    object_state = torch.zeros(num_envs, 13).to(sim.device)
                    object_state[..., :3] = torch.tensor(object_position, device=sim.device)
                    object_state[..., 3:7] = torch.tensor(object_rotation, device=sim.device)
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
                data_buffer = []

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

            # Parse robot task dict list to identify pick-place sequences
            if iteration == 0:
                # Parse task decomposition into actionable sequences
                task_sequences = []
                current_pick_task = None
                
                for task in robot_task_dict_list:
                    action = task["action"]
                    
                    if action == "navigate":
                        # Store navigation target for upcoming pick/place
                        current_nav_target = task["location_object_id"]
                        
                    elif action == "pick":
                        # Create pick sequence
                        pick_sequence = {
                            "type": "pick",
                            "pick_object_id": task["target_object_id"],
                            "pick_table_id": task["location_object_id"],
                            "nav_target": current_nav_target if 'current_nav_target' in locals() else task["location_object_id"]
                        }
                        task_sequences.append(pick_sequence)
                        current_pick_task = pick_sequence
                        
                    elif action == "place":
                        # Create place sequence
                        if current_pick_task is not None:
                            place_sequence = {
                                "type": "place",
                                "place_table_id": task["location_object_id"],
                                "pick_object_id": current_pick_task["pick_object_id"],
                                "nav_target": current_nav_target if 'current_nav_target' in locals() else task["location_object_id"]
                            }
                            task_sequences.append(place_sequence)
                            current_pick_task = None
                
                print(f"Task sequences parsed: {len(task_sequences)} tasks")
                for i, seq in enumerate(task_sequences):
                    print(f"  Task {i}: {seq}")
                
                # Initialize per-environment task state
                current_task_idx_envs = {env_i: 0 for env_i in range(num_envs)}
                task_phase_envs = {env_i: "init" for env_i in range(num_envs)}

                # traj plan failure record
                traj_plan_failure_record_envs = set([])
                
                # State tracking for each environment
                is_reach_pick_goal_envs = []
                is_reach_grasp_goal_envs = []
                is_reach_grasp_goal_down_envs = []
                is_reach_grasp_goal_up_envs = []
                is_reach_place_goal_envs = []
                is_reach_place_goal_up_envs = []
                is_place_success_envs = []
                
                start_grasp_T_envs = {}
                
                # Trajectory storage
                reach_pick_base_traj_envs = {}
                reach_pick_base_traj_pass_envs = {}
                reach_pick_base_T_envs = {}
                reach_pick_up_traj_envs = {}
                reach_pick_up_T_envs = {}
                reach_pick_down_traj_envs = {}
                reach_pick_down_T_envs = {}
                reach_grasp_up_traj_envs = {}
                reach_grasp_up_T_envs = {}
                reach_place_base_traj_envs = {}
                reach_place_base_traj_pass_envs = {}
                reach_place_base_T_envs = {}
                reach_place_T_envs = {}
                reach_place_up_traj_envs = {}
                reach_place_goal_T_envs = {}
                place_ee_goals_up_envs = {}
                place_success_T_envs = {}
                start_place_T_envs = {}
                
                ee_frame_data_pos_after_grasp_relative_base_support_envs = {}
                ee_frame_data_quat_after_grasp_relative_base_support_envs = {}
                support_frame_data_pos_envs = {}
                
                # Per-task data storage for each environment
                current_pick_object_id_envs = {}
                current_pick_table_id_envs = {}
                current_place_table_id_envs = {}
                pick_ee_goals_envs = {}
                pick_ee_goals_up_envs = {}
                pick_ee_goals_up_after_envs = {}
                pick_ee_goals_down_envs = {}
                robot_base_pick_pose_envs = {}
                robot_base_place_pose_envs = {}
                place_location_w_envs_dict = {}
                target_object_initial_z_envs = {}
                
                for env_i in range(num_envs):
                    is_reach_pick_goal_envs.append(False)
                    is_reach_grasp_goal_envs.append(False)
                    is_reach_grasp_goal_down_envs.append(False)
                    is_reach_grasp_goal_up_envs.append(False)
                    is_reach_place_goal_envs.append(False)
                    is_reach_place_goal_up_envs.append(False)
                    reach_pick_base_traj_pass_envs[env_i] = 0
                    reach_place_base_traj_pass_envs[env_i] = 0
                    is_place_success_envs.append(False)
            
            # Helper function to reset state for new task phase
            def reset_task_state_for_env(env_i):
                """Reset state variables for a new task phase"""
                is_reach_pick_goal_envs[env_i] = False
                is_reach_grasp_goal_envs[env_i] = False
                is_reach_grasp_goal_down_envs[env_i] = False
                is_reach_grasp_goal_up_envs[env_i] = False
                is_reach_place_goal_envs[env_i] = False
                is_reach_place_goal_up_envs[env_i] = False
                is_place_success_envs[env_i] = False
                
                # Trajectory storage
                if env_i in start_grasp_T_envs:
                    del start_grasp_T_envs[env_i]
                if env_i in reach_pick_base_traj_envs:
                    del reach_pick_base_traj_envs[env_i]
                if env_i in reach_pick_base_traj_pass_envs:
                    del reach_pick_base_traj_pass_envs[env_i]
                if env_i in reach_pick_base_T_envs:
                    del reach_pick_base_T_envs[env_i]
                if env_i in reach_pick_up_traj_envs:
                    del reach_pick_up_traj_envs[env_i]
                if env_i in reach_pick_up_T_envs:
                    del reach_pick_up_T_envs[env_i]
                if env_i in reach_pick_down_traj_envs:
                    del reach_pick_down_traj_envs[env_i]
                if env_i in reach_pick_down_T_envs:
                    del reach_pick_down_T_envs[env_i]
                if env_i in reach_grasp_up_traj_envs:
                    del reach_grasp_up_traj_envs[env_i]
                if env_i in reach_grasp_up_T_envs:
                    del reach_grasp_up_T_envs[env_i]
                if env_i in reach_place_base_traj_envs:
                    del reach_place_base_traj_envs[env_i]
                if env_i in reach_place_base_traj_pass_envs:
                    del reach_place_base_traj_pass_envs[env_i]
                if env_i in reach_place_base_T_envs:
                    del reach_place_base_T_envs[env_i]
                if env_i in reach_place_T_envs:
                    del reach_place_T_envs[env_i]
                if env_i in reach_place_up_traj_envs:
                    del reach_place_up_traj_envs[env_i]
                if env_i in reach_place_goal_T_envs:
                    del reach_place_goal_T_envs[env_i]
                if env_i in place_ee_goals_up_envs:
                    del place_ee_goals_up_envs[env_i]
                if env_i in place_success_T_envs:
                    del place_success_T_envs[env_i]
                if env_i in start_place_T_envs:
                    del start_place_T_envs[env_i]
                
                if env_i in ee_frame_data_pos_after_grasp_relative_base_support_envs:
                    del ee_frame_data_pos_after_grasp_relative_base_support_envs[env_i]
                if env_i in ee_frame_data_quat_after_grasp_relative_base_support_envs:
                    del ee_frame_data_quat_after_grasp_relative_base_support_envs[env_i]
                if env_i in support_frame_data_pos_envs:
                    del support_frame_data_pos_envs[env_i]
                
                # Per-task data storage for each environment
                if env_i in current_pick_object_id_envs:
                    del current_pick_object_id_envs[env_i]
                if env_i in current_pick_table_id_envs:
                    del current_pick_table_id_envs[env_i]
                if env_i in current_place_table_id_envs:
                    del current_place_table_id_envs[env_i]
                if env_i in pick_ee_goals_envs:
                    del pick_ee_goals_envs[env_i]
                if env_i in pick_ee_goals_up_envs:
                    del pick_ee_goals_up_envs[env_i]
                if env_i in pick_ee_goals_up_after_envs:
                    del pick_ee_goals_up_after_envs[env_i]
                if env_i in pick_ee_goals_down_envs:
                    del pick_ee_goals_down_envs[env_i]
                if env_i in robot_base_pick_pose_envs:
                    del robot_base_pick_pose_envs[env_i]
                if env_i in robot_base_place_pose_envs:
                    del robot_base_place_pose_envs[env_i]
                if env_i in place_location_w_envs_dict:
                    del place_location_w_envs_dict[env_i]
                if env_i in target_object_initial_z_envs:
                    del target_object_initial_z_envs[env_i]
                
                reach_pick_base_traj_pass_envs[env_i] = 0
                reach_place_base_traj_pass_envs[env_i] = 0
            
            # Helper function to initialize pick task for an environment
            def initialize_pick_task(env_i, pick_object_id, pick_table_id):
                """Initialize a new pick task for an environment with trajectory planning validation"""
                print(f"[Env {env_i}] Initializing pick task: object={pick_object_id}, table={pick_table_id}")
                reset_task_state_for_env(env_i)
                
                # Store current task info
                current_pick_object_id_envs[env_i] = pick_object_id
                current_pick_table_id_envs[env_i] = pick_table_id
                
                # Try up to 3 times to sample a robot location with valid trajectory
                max_attempts = 3
                planning_success = False
                
                for attempt in range(max_attempts):
                    # Sample robot location for picking
                    robot_base_pick_pos, robot_base_pick_quat, _ = sample_robot_location(
                        scene_save_dir, layout_name, room_id,
                        pick_object_id, pick_table_id, 1, debug_dir
                    )
                    
                    # Validate trajectory planning from current position to sampled position
                    _, plan_success = plan_robot_traj(
                        fixed_support_frame_data_pos[env_i], fixed_support_frame_data_quat[env_i],
                        robot_base_pick_pos[0], robot_base_pick_quat[0],
                        scene_save_dir, layout_name, room_id, debug_dir,
                        return_visualization_data=False,
                        return_plan_status=True
                    )
                    
                    if plan_success:
                        planning_success = True
                        print(f"[Env {env_i}] Pick location sampling successful on attempt {attempt + 1}")
                        break
                    else:
                        print(f"[Env {env_i}] Pick trajectory planning failed on attempt {attempt + 1}, resampling...")
                
                if not planning_success:
                    print(f"[Env {env_i}] WARNING: Failed to find valid pick location after {max_attempts} attempts. Using last sample.")
                
                # Store the sampled (and validated) robot pose
                robot_base_pick_pose_envs[env_i] = torch.cat([robot_base_pick_pos[0], robot_base_pick_quat[0]], dim=-1)
                
                # Sample grasp poses
                pick_ee_goals = sample_grasp(layout_json_path, pick_object_id, robot_base_pick_pos[0, :3], 1, env.sim.device)
                pick_ee_goals_envs[env_i] = pick_ee_goals[0]
                pick_ee_goals_up_envs[env_i] = pick_ee_goals[0] + torch.tensor([0., 0., grasp_up_offset, 0., 0., 0., 0.], device=env.sim.device)
                pick_ee_goals_up_after_envs[env_i] = pick_ee_goals[0] + torch.tensor([0., 0., place_up_offset, 0., 0., 0., 0.], device=env.sim.device)
                pick_ee_goals_down_envs[env_i] = pick_ee_goals[0] + torch.tensor([0., 0., -grasp_down_offset, 0., 0., 0., 0.], device=env.sim.device)
                
                # Get initial object height
                target_object_com_state = env.scene[pick_object_id].data.root_com_state_w
                target_object_initial_z_envs[env_i] = float(target_object_com_state[env_i, 2])
                
                task_phase_envs[env_i] = "pick_navigate"
                
                return planning_success
            
            # Helper function to initialize place task for an environment
            def initialize_place_task(env_i, place_table_id, pick_object_id):
                """Initialize a new place task for an environment with trajectory planning validation"""
                print(f"[Env {env_i}] Initializing place task: table={place_table_id}, object={pick_object_id}")
                
                # Store current task info
                current_place_table_id_envs[env_i] = place_table_id
                
                # Try up to 3 times to sample a robot location with valid trajectory
                max_attempts = 3
                planning_success = False
                
                for attempt in range(max_attempts):
                    # Sample robot location for placing
                    robot_base_place_pos, robot_base_place_quat, place_location_w = sample_robot_place_location(
                        scene_save_dir, layout_name, room_id,
                        place_table_id, 1, debug_dir
                    )
                    
                    # Validate trajectory planning from current position to sampled position
                    _, plan_success = plan_robot_traj(
                        fixed_support_frame_data_pos[env_i], fixed_support_frame_data_quat[env_i],
                        robot_base_place_pos[0], robot_base_place_quat[0],
                        scene_save_dir, layout_name, room_id, debug_dir,
                        return_visualization_data=False,
                        return_plan_status=True
                    )
                    
                    if plan_success:
                        planning_success = True
                        print(f"[Env {env_i}] Place location sampling successful on attempt {attempt + 1}")
                        break
                    else:
                        print(f"[Env {env_i}] Place trajectory planning failed on attempt {attempt + 1}, resampling...")
                
                if not planning_success:
                    print(f"[Env {env_i}] WARNING: Failed to find valid place location after {max_attempts} attempts. Using last sample.")
                
                # Store the sampled (and validated) robot pose and place location
                robot_base_place_pose_envs[env_i] = torch.cat([robot_base_place_pos[0], robot_base_place_quat[0]], dim=-1)
                place_location_w_envs_dict[env_i] = torch.cat([place_location_w[0], torch.tensor([1, 0, 0, 0], device=env.sim.device)], dim=-1)
                
                # Keep pick-related state for the held object
                task_phase_envs[env_i] = "place_navigate"
                
                return planning_success
            
            # Main state machine loop for each environment
            for env_i in range(num_envs):
                
                # Check if we need to initialize a new task
                if task_phase_envs[env_i] == "init":
                    if current_task_idx_envs[env_i] < len(task_sequences):
                        current_task = task_sequences[current_task_idx_envs[env_i]]
                        print(f"[Env {env_i}] Initializing new task: {current_task}")
                        
                        if current_task["type"] == "pick":
                            initialize_pick_task(env_i, current_task["pick_object_id"], current_task["pick_table_id"])
                        elif current_task["type"] == "place":
                            initialize_place_task(env_i, current_task["place_table_id"], current_task["pick_object_id"])
                    else:
                        # All tasks completed for this environment
                        task_phase_envs[env_i] = "complete"
                        continue
                
                # Skip if all tasks are complete
                if task_phase_envs[env_i] == "complete":
                    continue
                
                # Get current task info
                current_task_idx = current_task_idx_envs[env_i]
                if current_task_idx >= len(task_sequences):
                    continue
                    
                current_task = task_sequences[current_task_idx]
                current_pick_object_id = current_pick_object_id_envs.get(env_i, None)
                
                # PICK PHASE STATE MACHINE
                if current_task["type"] == "pick":
                    
                    # Phase 1: Navigate to pick location
                    if task_phase_envs[env_i] == "pick_navigate":
                        if iteration % T_grasp >= T_init and env_i in robot_base_pick_pose_envs and env_i not in reach_pick_base_traj_envs:
                            # Plan navigation trajectory
                            reach_pick_base_traj_envs[env_i], plan_success = plan_robot_traj(
                                fixed_support_frame_data_pos[env_i], fixed_support_frame_data_quat[env_i],
                                robot_base_pick_pose_envs[env_i][:3], robot_base_pick_pose_envs[env_i][3:7],
                                scene_save_dir, layout_name, room_id, debug_dir,
                                return_visualization_data=False,
                                return_plan_status=True
                            )
                            
                            if not plan_success:
                                print(f"[Env {env_i}] Pick navigation planning failed")
                                traj_plan_failure_record_envs.add(env_i)
                                reach_pick_base_traj_envs[env_i] = reach_pick_base_traj_envs[env_i][0:1]
                            
                            reach_pick_base_T_envs[env_i] = iteration
                        
                        # Check if navigation is complete
                        if iteration % T_grasp >= T_init and env_i in reach_pick_base_traj_envs:
                            if not is_reach_pick_goal_envs[env_i]:
                                is_reach_pick_goal_envs[env_i] = reach_pick_base_traj_pass_envs[env_i] >= len(reach_pick_base_traj_envs[env_i]) - 1
                                
                                if is_reach_pick_goal_envs[env_i]:
                                    # Transition to move up phase
                                    task_phase_envs[env_i] = "pick_move_up"
                                    print(f"[Env {env_i}] Transitioning to pick_move_up phase")
                                else:
                                    # Continue navigation
                                    while reach_pick_base_traj_pass_envs[env_i] < len(reach_pick_base_traj_envs[env_i]) - 1 and is_reach(
                                        reach_pick_base_traj_envs[env_i][reach_pick_base_traj_pass_envs[env_i]], 
                                        fixed_support_frame_data_pose[env_i], ignore_z=True, 
                                        loc_threshold=0.05, rot_threshold=5.0
                                    ):
                                        reach_pick_base_traj_pass_envs[env_i] += 1
                    
                    # Phase 2: Move EE up to pre-grasp
                    elif task_phase_envs[env_i] == "pick_move_up":
                        if is_reach_pick_goal_envs[env_i] and env_i not in reach_pick_up_T_envs:
                            # Plan upward trajectory
                            reach_pick_up_traj, success = curobo_plan_traj(
                                motion_planner,
                                robot_qpos[env_i:env_i+1, :],
                                ee_frame_data_pos[env_i:env_i+1, :3],
                                pick_ee_goals_up_envs[env_i].reshape(1, -1)[:, :3],
                                pick_ee_goals_up_envs[env_i].reshape(1, -1)[:, 3:7],
                                max_interpolation_step_distance=0.001,
                                interpolate=True,
                                current_ee_quat=ee_frame_data_quat[env_i:env_i+1],
                            )
                            reach_pick_up_traj_envs[env_i] = reach_pick_up_traj
                            reach_pick_up_T_envs[env_i] = iteration
                        
                        # Check if move up is complete
                        if env_i in reach_pick_up_T_envs:
                            if not is_reach_grasp_goal_envs[env_i]:
                                is_reach_grasp_goal_envs[env_i] = is_reach(
                                    pick_ee_goals_up_envs[env_i], ee_frame_data_pose[env_i], 
                                    loc_threshold=0.03, rot_threshold=30.0
                                )
                                
                                if is_reach_grasp_goal_envs[env_i]:
                                    task_phase_envs[env_i] = "pick_move_down"
                                    print(f"[Env {env_i}] Transitioning to pick_move_down phase")
                    
                    # Phase 3: Move EE down to grasp
                    elif task_phase_envs[env_i] == "pick_move_down":
                        if is_reach_grasp_goal_envs[env_i] and env_i not in reach_pick_down_T_envs:
                            # Plan downward trajectory
                            reach_pick_down_traj, success = curobo_plan_traj(
                                motion_planner,
                                robot_qpos[env_i:env_i+1, :],
                                ee_frame_data_pos[env_i:env_i+1, :3],
                                pick_ee_goals_down_envs[env_i].reshape(1, -1)[:, :3],
                                pick_ee_goals_down_envs[env_i].reshape(1, -1)[:, 3:7],
                                max_interpolation_step_distance=0.001,
                                interpolate=True,
                                current_ee_quat=ee_frame_data_quat[env_i:env_i+1],
                            )
                            reach_pick_down_traj_envs[env_i] = reach_pick_down_traj
                            reach_pick_down_T_envs[env_i] = iteration
                        
                        # Check if move down is complete
                        if env_i in reach_pick_down_T_envs:
                            if not is_reach_grasp_goal_down_envs[env_i]:
                                is_reach_grasp_goal_down_envs[env_i] = is_reach(
                                    pick_ee_goals_down_envs[env_i], ee_frame_data_pose[env_i], 
                                    loc_threshold=0.05+grasp_down_offset, rot_threshold=30.0
                                )
                                
                                if is_reach_grasp_goal_down_envs[env_i]:
                                    task_phase_envs[env_i] = "pick_grasp_hold"
                                    start_grasp_T_envs[env_i] = 0
                                    print(f"[Env {env_i}] Transitioning to pick_grasp_hold phase")
                    
                    # Phase 4: Close gripper and hold
                    elif task_phase_envs[env_i] == "pick_grasp_hold":
                        if is_reach_grasp_goal_down_envs[env_i]:
                            start_grasp_T_envs[env_i] = start_grasp_T_envs.get(env_i, 0) + 1
                            
                            if start_grasp_T_envs[env_i] >= T_hold:
                                task_phase_envs[env_i] = "pick_lift_up"
                                print(f"[Env {env_i}] Transitioning to pick_lift_up phase")
                    
                    # Phase 5: Lift up with object
                    elif task_phase_envs[env_i] == "pick_lift_up":
                        if env_i not in reach_grasp_up_T_envs:
                            # Update pick_ee_goals_up_after_envs

                            pick_ee_goals_up_after_envs_to_support_pos, pick_ee_goals_up_after_envs_to_support_quat = math_utils.subtract_frame_transforms(
                                support_frame_data_pos[env_i].reshape(1, 3), support_frame_data_quat[env_i].reshape(1, 4),
                                pick_ee_goals_up_after_envs[env_i].reshape(1, -1)[:, :3], pick_ee_goals_up_after_envs[env_i].reshape(1, -1)[:, 3:7]
                            )

                            pick_ee_goals_up_after_envs_to_support_pos[:, 0] = 0.3
                            pick_ee_goals_up_after_envs_pos, pick_ee_goals_up_after_envs_quat = math_utils.combine_frame_transforms(
                                support_frame_data_pos[env_i].reshape(1, 3), support_frame_data_quat[env_i].reshape(1, 4),
                                pick_ee_goals_up_after_envs_to_support_pos, pick_ee_goals_up_after_envs_to_support_quat,
                            )
                            pick_ee_goals_up_after_envs[env_i] = torch.cat([pick_ee_goals_up_after_envs_pos, pick_ee_goals_up_after_envs_quat], dim=-1).reshape(-1)


                            # Plan lift up trajectory
                            reach_grasp_up_traj, success = curobo_plan_traj(
                                motion_planner,
                                robot_qpos[env_i:env_i+1, :],
                                ee_frame_data_pos[env_i:env_i+1, :3],
                                pick_ee_goals_up_after_envs[env_i].reshape(1, -1)[:, :3],
                                pick_ee_goals_up_after_envs[env_i].reshape(1, -1)[:, 3:7],
                                max_interpolation_step_distance=0.001,
                                interpolate=True,
                                current_ee_quat=ee_frame_data_quat[env_i:env_i+1],
                            )
                            reach_grasp_up_traj_envs[env_i] = reach_grasp_up_traj
                            reach_grasp_up_T_envs[env_i] = iteration
                        
                        # Check if lift up is complete
                        if env_i in reach_grasp_up_T_envs:
                            if not is_reach_grasp_goal_up_envs[env_i]:
                                is_reach_grasp_goal_up_envs[env_i] = is_reach(
                                    pick_ee_goals_up_after_envs[env_i], ee_frame_data_pose[env_i], 
                                    loc_threshold=0.03, rot_threshold=10.0
                                )
                                
                                if is_reach_grasp_goal_up_envs[env_i]:
                                    #
                                    pick_ee_goals_up_after_env_i_pos = pick_ee_goals_up_after_envs[env_i].reshape(1, -1)[:, :3]
                                    pick_ee_goals_up_after_env_i_quat = pick_ee_goals_up_after_envs[env_i].reshape(1, -1)[:, 3:7]

                                    pick_ee_goals_up_after_env_i_pos_to_support_pos, pick_ee_goals_up_after_env_i_pos_to_support_quat = math_utils.subtract_frame_transforms(
                                        support_frame_data_pos[env_i].reshape(1, 3), support_frame_data_quat[env_i].reshape(1, 4),
                                        pick_ee_goals_up_after_env_i_pos, pick_ee_goals_up_after_env_i_quat
                                    )


                                    ee_to_support_frame_pos_traj, ee_to_support_frame_quat_traj = interpolate_pose(
                                        ee_to_support_frame_pos[env_i],
                                        ee_to_support_frame_quat[env_i],
                                        pick_ee_goals_up_after_env_i_pos_to_support_pos.clone().reshape(3),
                                        pick_ee_goals_up_after_env_i_pos_to_support_quat.clone().reshape(4),
                                        steps=100,
                                    )

                                    

                                    # Save EE pose relative to base support for place phase
                                    ee_frame_data_pos_after_grasp_relative_base_support_envs[env_i] = ee_to_support_frame_pos_traj.clone()
                                    ee_frame_data_quat_after_grasp_relative_base_support_envs[env_i] = ee_to_support_frame_quat_traj.clone()
                                    support_frame_data_pos_envs[env_i] = support_frame_data_pos[env_i].clone()
                                    start_place_T_envs[env_i] = iteration
                                    
                                    # Pick task complete, initialize next task
                                    current_task_idx_envs[env_i] += 1
                                    task_phase_envs[env_i] = "init"
                                    print(f"[Env {env_i}] Pick task complete, initializing next task {current_task_idx_envs[env_i]}")

                                    ee_frame_data_pos_initial[env_i] = ee_frame_data_pos[env_i].clone()
                                    ee_frame_data_quat_initial[env_i] = ee_frame_data_quat[env_i].clone()
                                    fixed_support_frame_data_pos_initial[env_i] = fixed_support_frame_data_pos[env_i].clone()
                                    fixed_support_frame_data_quat_initial[env_i] = fixed_support_frame_data_quat[env_i].clone()
                                    support_frame_data_pos_initial[env_i] = support_frame_data_pos[env_i].clone()
                                    support_frame_data_quat_initial[env_i] = support_frame_data_quat[env_i].clone()

                                    ee_to_fixed_support_frame_pos_initial[env_i] = ee_to_fixed_support_frame_pos[env_i].clone()
                                    ee_to_fixed_support_frame_quat_initial[env_i] = ee_to_fixed_support_frame_quat[env_i].clone()
                                    ee_to_support_frame_pos_initial[env_i] = ee_to_support_frame_pos[env_i].clone()
                                    ee_to_support_frame_quat_initial[env_i] = ee_to_support_frame_quat[env_i].clone()
                
                # PLACE PHASE STATE MACHINE
                elif current_task["type"] == "place":
                    
                    # Phase 1: Navigate to place location
                    if task_phase_envs[env_i] == "place_navigate":
                        if env_i not in reach_place_base_T_envs:
                            # Plan navigation trajectory
                            reach_place_base_traj_envs[env_i], plan_success = plan_robot_traj(
                                fixed_support_frame_data_pos[env_i], fixed_support_frame_data_quat[env_i],
                                robot_base_place_pose_envs[env_i][:3], robot_base_place_pose_envs[env_i][3:7],
                                scene_save_dir, layout_name, room_id, debug_dir,
                                return_visualization_data=False,
                                return_plan_status=True
                            )

                            robot_base_place_pose_envs[env_i] = reach_place_base_traj_envs[env_i][-1, :]
                            
                            if not plan_success:
                                print(f"[Env {env_i}] Place navigation planning failed")
                                traj_plan_failure_record_envs.add(env_i)
                                reach_place_base_traj_envs[env_i] = reach_place_base_traj_envs[env_i][0:1]
                            
                            reach_place_base_T_envs[env_i] = iteration
                        
                        # Check if navigation is complete
                        if env_i in reach_place_base_traj_envs:
                            if not is_reach_place_goal_envs[env_i]:
                                is_reach_place_goal_envs[env_i] = reach_place_base_traj_pass_envs[env_i] >= len(reach_place_base_traj_envs[env_i]) - 1
                                
                                if is_reach_place_goal_envs[env_i]:
                                    task_phase_envs[env_i] = "place_move_to_position"
                                    print(f"[Env {env_i}] Transitioning to place_move_to_position phase")
                                else:
                                    # Continue navigation
                                    while reach_place_base_traj_pass_envs[env_i] < len(reach_place_base_traj_envs[env_i]) - 1 and is_reach(
                                        reach_place_base_traj_envs[env_i][reach_place_base_traj_pass_envs[env_i]], 
                                        fixed_support_frame_data_pose[env_i], ignore_z=True, 
                                        loc_threshold=0.05, rot_threshold=5.0
                                    ):
                                        reach_place_base_traj_pass_envs[env_i] += 1
                    
                    # Phase 2: Move EE to place position
                    elif task_phase_envs[env_i] == "place_move_to_position":
                        if is_reach_place_goal_envs[env_i] and env_i not in reach_place_T_envs:
                            # Calculate place position
                            robot_base_w = env.scene["robot"].data.root_state_w[env_i:env_i+1, :7]
                            target_object_w = env.scene[current_pick_object_id].data.root_state_w[env_i:env_i+1, :7]
                            target_object_r_pos, target_object_r_quat = math_utils.subtract_frame_transforms(
                                robot_base_w[:, :3], robot_base_w[:, 3:7],
                                target_object_w[:, :3], target_object_w[:, 3:7]
                            )
                            
                            target_to_ee_pos = (ee_frame_data_pos[env_i:env_i+1].reshape(3) - target_object_r_pos.reshape(3))
                            target_to_ee_pos[2] = 0
                            
                            pick_object_height = get_grasp_object_height(layout_json_path, current_pick_object_id)
                            place_ee_goals_up_env_i = place_location_w_envs_dict[env_i][:3] + torch.tensor([0., 0., place_down_offset+pick_object_height], device=env.sim.device)
                            place_ee_goals_up_env_i[:2] += target_to_ee_pos.reshape(3)[:2]
                            
                            reach_place_up_traj, success = curobo_plan_traj(
                                motion_planner,
                                robot_qpos[env_i:env_i+1, :],
                                ee_frame_data_pos[env_i:env_i+1, :3],
                                place_ee_goals_up_env_i.reshape(1, -1),
                                ee_frame_data_quat[env_i:env_i+1],
                                interpolate=True,
                                current_ee_quat=ee_frame_data_quat[env_i:env_i+1],
                            )
                            reach_place_up_traj_envs[env_i] = reach_place_up_traj
                            reach_place_T_envs[env_i] = iteration
                            place_ee_goals_up_envs[env_i] = place_ee_goals_up_env_i
                        
                        # Check if move to position is complete
                        if env_i in reach_place_T_envs:
                            if not is_reach_place_goal_up_envs[env_i]:
                                is_reach_place_goal_up_envs[env_i] = torch.all(
                                    torch.abs(ee_frame_data_pos[env_i] - place_ee_goals_up_envs[env_i][:3]) < 0.10
                                )
                                
                                if is_reach_place_goal_up_envs[env_i]:
                                    reach_place_goal_T_envs[env_i] = iteration
                                    task_phase_envs[env_i] = "place_release"
                                    print(f"[Env {env_i}] Transitioning to place_release phase")
                    
                    # Phase 3: Release object
                    elif task_phase_envs[env_i] == "place_release":
                        if is_reach_place_goal_up_envs[env_i] and not is_place_success_envs[env_i]:
                            # Check if object is in place
                            target_object_w_state = env.scene[current_pick_object_id].data.root_state_w[:, :7]
                            target_object_w_state_pos_to_env_root, target_object_w_state_quat_to_env_root = math_utils.subtract_frame_transforms(
                                envs_pos, envs_quat,
                                target_object_w_state[:, :3], target_object_w_state[:, 3:7],
                            )
                            target_object_w_state_env_i_pos = target_object_w_state_pos_to_env_root[env_i]
                            place_location_w_env_i_pos = place_location_w_envs_dict[env_i][:3]
                            
                            if torch.all(torch.abs(target_object_w_state_env_i_pos[:2] - place_location_w_env_i_pos[:2]) < min(CollisionCheckingConfig.MIN_DIST_TO_BOUNDARY, 0.1)) and \
                                env_i not in traj_plan_failure_record_envs:
                                is_place_success_envs[env_i] = True
                                place_success_T_envs[env_i] = iteration
                                
                                # Place task complete, initialize next task
                                current_task_idx_envs[env_i] += 1
                                task_phase_envs[env_i] = "init"
                                print(f"[Env {env_i}] Place task complete, initializing next task {current_task_idx_envs[env_i]}")

                                ee_frame_data_pos_initial[env_i] = ee_frame_data_pos[env_i].clone()
                                ee_frame_data_quat_initial[env_i] = ee_frame_data_quat[env_i].clone()
                                fixed_support_frame_data_pos_initial[env_i] = fixed_support_frame_data_pos[env_i].clone()
                                fixed_support_frame_data_quat_initial[env_i] = fixed_support_frame_data_quat[env_i].clone()
                                support_frame_data_pos_initial[env_i] = support_frame_data_pos[env_i].clone()
                                support_frame_data_quat_initial[env_i] = support_frame_data_quat[env_i].clone()

                                ee_to_fixed_support_frame_pos_initial[env_i] = ee_to_fixed_support_frame_pos[env_i].clone()
                                ee_to_fixed_support_frame_quat_initial[env_i] = ee_to_fixed_support_frame_quat[env_i].clone()
                                ee_to_support_frame_pos_initial[env_i] = ee_to_support_frame_pos[env_i].clone()
                                ee_to_support_frame_quat_initial[env_i] = ee_to_support_frame_quat[env_i].clone()
                                
            
            # Generate actions based on current phase
            actions_list = []
            
            for env_i in range(num_envs):
                current_phase = task_phase_envs[env_i]
                # print(f"[Env {env_i}] Current phase: {current_phase}")
                
                if current_phase == "complete":
                    # Task complete, hold current position
                    actions = torch.cat([
                        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,], device=env.sim.device).reshape(1, -1),
                        ee_frame_data_pose[env_i:env_i+1].reshape(1, -1),
                        torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                    ], dim=-1)
                
                elif iteration % T_grasp < T_init or current_phase == "init":
                    # Initialization phase - maintain initial pose
                    ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat = math_utils.combine_frame_transforms(
                        fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1],
                        ee_to_fixed_support_frame_pos_initial[env_i:env_i+1], ee_to_fixed_support_frame_quat_initial[env_i:env_i+1]
                    )
                    
                    actions = torch.cat([
                        get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], 
                            fixed_support_frame_data_pos_initial[env_i:env_i+1, :3].reshape(1, 3), 
                            fixed_support_frame_data_quat_initial[env_i:env_i+1, :4].reshape(1, 4)),
                        torch.cat([ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat], dim=-1),
                        torch.tensor([-1.0,], device=env.sim.device).reshape(1, 1),
                    ], dim=-1)
                
                elif current_phase == "pick_navigate":
                    # Navigate to pick location
                    if env_i in reach_pick_base_traj_envs and env_i in reach_pick_base_T_envs:
                        ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat = math_utils.combine_frame_transforms(
                            fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1],
                            ee_to_fixed_support_frame_pos_initial[env_i:env_i+1], ee_to_fixed_support_frame_quat_initial[env_i:env_i+1]
                        )
                        
                        actions = torch.cat([
                            get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], 
                                reach_pick_base_traj_envs[env_i][reach_pick_base_traj_pass_envs[env_i], :3].reshape(1, 3), 
                                reach_pick_base_traj_envs[env_i][reach_pick_base_traj_pass_envs[env_i], 3:7].reshape(1, 4)),
                            torch.cat([ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat], dim=-1),
                            torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                    else:
                        # Fallback: hold position
                        actions = torch.cat([
                            torch.zeros(1, 6, device=env.sim.device),
                            ee_frame_data_pose[env_i:env_i+1],
                            torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                
                elif current_phase == "pick_move_up":
                    # Move EE up to pre-grasp position
                    if env_i in reach_pick_up_traj_envs and env_i in reach_pick_up_T_envs and env_i in robot_base_pick_pose_envs:
                        pick_ee_goals_up_iter = reach_pick_up_traj_envs[env_i][min(max(iteration - reach_pick_up_T_envs[env_i] - T_grasp_init, 0), len(reach_pick_up_traj_envs[env_i]) - 1)].reshape(1, 7)
                        
                        actions = torch.cat([
                            get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], 
                                robot_base_pick_pose_envs[env_i][:3].reshape(1, 3), robot_base_pick_pose_envs[env_i][3:7].reshape(1, 4)),
                            torch.cat([pick_ee_goals_up_iter[:, :3], pick_ee_goals_up_iter[:, 3:7]], dim=-1),
                            torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                    else:
                        actions = torch.cat([
                            torch.zeros(1, 6, device=env.sim.device),
                            ee_frame_data_pose[env_i:env_i+1],
                            torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                
                elif current_phase == "pick_move_down":
                    # Move EE down to grasp
                    if env_i in reach_pick_down_traj_envs and env_i in reach_pick_down_T_envs and env_i in robot_base_pick_pose_envs:
                        pick_ee_goals_down_iter = reach_pick_down_traj_envs[env_i][min(max(iteration - reach_pick_down_T_envs[env_i], 0), len(reach_pick_down_traj_envs[env_i]) - 1)].reshape(1, 7)
                        
                        actions = torch.cat([
                            get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], 
                                robot_base_pick_pose_envs[env_i][:3].reshape(1, 3), robot_base_pick_pose_envs[env_i][3:7].reshape(1, 4)),
                            torch.cat([pick_ee_goals_down_iter[:, :3], pick_ee_goals_down_iter[:, 3:7]], dim=-1),
                            torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                    else:
                        actions = torch.cat([
                            torch.zeros(1, 6, device=env.sim.device),
                            ee_frame_data_pose[env_i:env_i+1],
                            torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                
                elif current_phase == "pick_grasp_hold":
                    # Close gripper and hold
                    if env_i in pick_ee_goals_down_envs and env_i in robot_base_pick_pose_envs:
                        gripper_action = -1.0 if start_grasp_T_envs.get(env_i, 0) > T_hold * 0.7 else 1.0
                        
                        actions = torch.cat([
                            get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], 
                                robot_base_pick_pose_envs[env_i][:3].reshape(1, 3), robot_base_pick_pose_envs[env_i][3:7].reshape(1, 4)),
                            torch.cat([pick_ee_goals_down_envs[env_i][:3].reshape(1, -1), pick_ee_goals_down_envs[env_i][3:7].reshape(1, -1)], dim=-1),
                            torch.tensor([gripper_action,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                    else:
                        actions = torch.cat([
                            torch.zeros(1, 6, device=env.sim.device),
                            ee_frame_data_pose[env_i:env_i+1],
                            torch.tensor([-1.0,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                
                elif current_phase == "pick_lift_up":
                    # Lift up with object
                    if env_i in reach_grasp_up_traj_envs and env_i in reach_grasp_up_T_envs and env_i in robot_base_pick_pose_envs:
                        grasp_up_goals_iter = reach_grasp_up_traj_envs[env_i][min(max(iteration - reach_grasp_up_T_envs[env_i], 0), len(reach_grasp_up_traj_envs[env_i]) - 1)].reshape(1, 7)
                        
                        actions = torch.cat([
                            get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], 
                                robot_base_pick_pose_envs[env_i][:3].reshape(1, 3), robot_base_pick_pose_envs[env_i][3:7].reshape(1, 4)),
                            torch.cat([grasp_up_goals_iter[:, :3], grasp_up_goals_iter[:, 3:7]], dim=-1),
                            torch.tensor([-1.0], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                    else:
                        actions = torch.cat([
                            torch.zeros(1, 6, device=env.sim.device),
                            ee_frame_data_pose[env_i:env_i+1],
                            torch.tensor([-1.0,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                
                elif current_phase == "place_navigate":
                    # Navigate to place location while maintaining EE relative to base
                    if env_i in reach_place_base_traj_envs and env_i in support_frame_data_pos_envs:
                        support_frame_data_pos_env_i = support_frame_data_pos[env_i:env_i+1].clone()
                        support_frame_data_pos_env_i[:, 2] = support_frame_data_pos_envs[env_i][2]
                        
                        ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat = math_utils.combine_frame_transforms(
                            support_frame_data_pos_env_i, support_frame_data_quat[env_i:env_i+1],
                            ee_frame_data_pos_after_grasp_relative_base_support_envs[env_i][min(max(start_place_T_envs[env_i] - iteration, 0), len(ee_frame_data_pos_after_grasp_relative_base_support_envs[env_i]) - 1), :].reshape(1, 3), 
                            ee_frame_data_quat_after_grasp_relative_base_support_envs[env_i][min(max(start_place_T_envs[env_i] - iteration, 0), len(ee_frame_data_quat_after_grasp_relative_base_support_envs[env_i]) - 1), :].reshape(1, 4)
                        )
                        
                        actions = torch.cat([
                            get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], 
                                reach_place_base_traj_envs[env_i][reach_place_base_traj_pass_envs[env_i], :3].reshape(1, 3), 
                                reach_place_base_traj_envs[env_i][reach_place_base_traj_pass_envs[env_i], 3:7].reshape(1, 4)),
                            torch.cat([ee_frame_data_pos_target_pos, ee_frame_data_pos_target_quat], dim=-1),
                            torch.tensor([-1.0], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                    else:
                        actions = torch.cat([
                            torch.zeros(1, 6, device=env.sim.device),
                            ee_frame_data_pose[env_i:env_i+1],
                            torch.tensor([-1.0,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                
                elif current_phase == "place_move_to_position":
                    # Move EE to place position
                    if env_i in reach_place_up_traj_envs and env_i in reach_place_T_envs and env_i in robot_base_place_pose_envs:
                        place_up_goals_iter = reach_place_up_traj_envs[env_i][min(max(iteration - reach_place_T_envs[env_i], 0), len(reach_place_up_traj_envs[env_i]) - 1)].reshape(1, 7)
                        
                        actions = torch.cat([
                            get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], 
                                robot_base_place_pose_envs[env_i][:3].reshape(1, 3), robot_base_place_pose_envs[env_i][3:7].reshape(1, 4)),
                            torch.cat([place_up_goals_iter[:, :3], place_up_goals_iter[:, 3:7]], dim=-1),
                            torch.tensor([-1.0], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                    else:
                        actions = torch.cat([
                            torch.zeros(1, 6, device=env.sim.device),
                            ee_frame_data_pose[env_i:env_i+1],
                            torch.tensor([-1.0,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                
                elif current_phase == "place_release":
                    # Release object
                    if env_i in reach_place_up_traj_envs and env_i in reach_place_T_envs and env_i in reach_place_goal_T_envs and env_i in robot_base_place_pose_envs:
                        place_up_goals_iter = reach_place_up_traj_envs[env_i][min(max(iteration - reach_place_T_envs[env_i], 0), len(reach_place_up_traj_envs[env_i]) - 1)].reshape(1, 7)
                        gripper_action = 1.0 if iteration % T_grasp - reach_place_goal_T_envs[env_i] > 20 else -1.0
                        
                        actions = torch.cat([
                            get_action_relative(fixed_support_frame_data_pos[env_i:env_i+1], fixed_support_frame_data_quat[env_i:env_i+1], 
                                robot_base_place_pose_envs[env_i][:3].reshape(1, 3), robot_base_place_pose_envs[env_i][3:7].reshape(1, 4)),
                            torch.cat([place_up_goals_iter[:, :3], place_up_goals_iter[:, 3:7]], dim=-1),
                            torch.tensor([gripper_action,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                    else:
                        actions = torch.cat([
                            torch.zeros(1, 6, device=env.sim.device),
                            ee_frame_data_pose[env_i:env_i+1],
                            torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                        ], dim=-1)
                
                else:
                    # Default: hold current position
                    actions = torch.cat([
                        torch.zeros(1, 6, device=env.sim.device),
                        ee_frame_data_pose[env_i:env_i+1],
                        torch.tensor([1.0,], device=env.sim.device).reshape(1, 1),
                    ], dim=-1)
                
                actions_list.append(actions)
            
            # Combine actions from all environments
            actions = torch.cat(actions_list, dim=0)
            # print(f"Actions: {actions.shape}, actions: {[f'{float(elem):0.4f}' for elem in actions.cpu().numpy().reshape(-1).tolist()]}")
            
            # Compute observations and store iteration status
            iteration_status = {}
            obs_dict_pre_step = env.observation_manager.compute()
            
            for key, value in obs_dict_pre_step["policy"].items():
                iteration_status[f"obs/{key}"] = value
            
            iteration_status["actions"] = mm_arm_abs_to_base_rel(actions, fixed_support_frame_data_pose)
            
            # Create state tensor for current task progress
            # States are organized in temporal order following task_sequences
            # For k rounds of pick-place, we have 4*k states in time sequence:
            # - Pick task 0: reach_pick_goal, grasp_goal_up
            # - Place task 0: reach_place_goal, place_complete
            # - Pick task 1: reach_pick_goal, grasp_goal_up
            # - Place task 1: reach_place_goal, place_complete
            # - ...
            # States are True if: (1) currently on task and flag is set, OR (2) task is already completed
            num_rounds = (len(task_sequences) + 1) // 2
            
            # Create state lists in temporal order
            state_lists = []
            for task_idx, task in enumerate(task_sequences):
                if task["type"] == "pick":
                    # Pick task has 2 states: reach_pick_goal, reach_grasp_goal_up
                    pick_goal_state = []
                    grasp_up_state = []
                    for env_i in range(num_envs):
                        is_current_task = (current_task_idx_envs[env_i] == task_idx)
                        is_past_task = (current_task_idx_envs[env_i] > task_idx)
                        # State is True if: currently on task and flag is set, OR task already completed
                        pick_goal_state.append((is_current_task and is_reach_pick_goal_envs[env_i]) or is_past_task)
                        grasp_up_state.append((is_current_task and is_reach_grasp_goal_up_envs[env_i]) or is_past_task)
                    state_lists.append(pick_goal_state)
                    state_lists.append(grasp_up_state)
                    
                elif task["type"] == "place":
                    # Place task has 2 states: reach_place_goal, place_complete
                    place_goal_state = []
                    place_complete_state = []
                    for env_i in range(num_envs):
                        is_current_task = (current_task_idx_envs[env_i] == task_idx)
                        is_past_task = (current_task_idx_envs[env_i] > task_idx)
                        # State is True if: currently on task and flag is set, OR task already completed
                        place_goal_state.append((is_current_task and is_reach_place_goal_envs[env_i]) or is_past_task)
                        place_complete_state.append(is_past_task)
                    state_lists.append(place_goal_state)
                    state_lists.append(place_complete_state)
            
            
            states_tensor = get_state_binary_tensor(state_lists)
            iteration_status["states"] = states_tensor

            # print(f"States tensor: {states_tensor.shape}, states_tensor: {states_tensor}")
            
            # Step the environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated
            
            for key, value in obs_dict["policy"].items():
                iteration_status[f"next_obs/{key}"] = value
            
            iteration_status["rewards"] = rewards
            iteration_status["dones"] = dones
            
            # Store data if past initialization phase
            if iteration % T_grasp > T_init:
                data_buffer.append(iteration_status)
            
            # Capture camera frames for debugging
            def depth_to_rgb(depth):
                depth = depth[..., None].repeat(3, axis=2)
                return depth
            
            if current_data_idx < total_recording_num and iteration % T_grasp >= T_init:
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
            
            # Increment iteration
            iteration = (iteration + 1) % T_grasp
            if iteration % 100 == 0:
                print(f"iteration: {iteration}")
            
            # Check for early stop - all environments completed all tasks
            all_envs_complete = all(current_task_idx_envs[env_i] >= len(task_sequences) for env_i in range(num_envs))
            
            if all_envs_complete:
                # Find the latest completion time
                latest_completion_T = 0
                for env_i in range(num_envs):
                    if env_i in place_success_T_envs:
                        latest_completion_T = max(latest_completion_T, place_success_T_envs[env_i])
                
                if iteration - latest_completion_T > T_drop:
                    print(f"All environments completed all tasks, early stop at iteration: {iteration}")
                    iteration = T_grasp - 1
            
            # End of episode - collect data and save
            if iteration % T_grasp == T_grasp - 1:
                
                # Save camera videos for debugging
                if current_data_idx < total_recording_num:
                    cameras_debug_dir = os.path.join(debug_dir, "cameras")
                    os.makedirs(cameras_debug_dir, exist_ok=True)
                    video_path = os.path.join(cameras_debug_dir, f"{current_data_idx:0>3d}_cameras.mp4")
                    imageio.mimwrite(video_path, camera_list, fps=30)
                    print(f"Saved {current_data_idx} data to video: {video_path}")
                
                # Collect successful trajectories
                for env_i in range(num_envs):
                    # Check if this environment completed all tasks successfully
                    env_completed_all_tasks = current_task_idx_envs[env_i] >= len(task_sequences)
                    
                    if env_completed_all_tasks and collected_data_episodes < args_cli.num_demos:
                        # Find the completion time (last place success)
                        completion_T = place_success_T_envs.get(env_i, T_grasp - T_drop)
                        
                        # Collect data from T_init+1 to completion + T_drop
                        for data_i, iteration_i in enumerate(range(T_init+1, min(completion_T + T_drop, T_grasp - 1))):
                            if data_i <= len(data_buffer) - 1:
                                for key, value in data_buffer[data_i].items():
                                    collector_interface.add(key, value[env_i:env_i+1])
                        
                        collector_interface.flush()
                        collected_data_episodes += 1
                        print(f"Collected {collected_data_episodes} data episodes from env {env_i}")
                
                # Clear buffers and increment counter
                data_buffer = []
                camera_list = []
                torch.cuda.empty_cache()
                gc.collect()
                
                current_data_idx += 1
                print(f"Motion planning success rate: {collected_data_episodes / (current_data_idx*num_envs)}")
            
            if collector_interface.is_stopped():
                break


    collector_interface.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
