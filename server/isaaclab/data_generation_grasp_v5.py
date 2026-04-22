# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import torch

import argparse
from doctest import FAIL_FAST

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
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR, ROBOMIMIC_ROOT_DIR, M2T2_ROOT_DIR

sys.path.insert(0, SERVER_ROOT_DIR)
sys.path.insert(0, M2T2_ROOT_DIR)
from utils import get_layout_from_scene_save_dir
from tex_utils import (
    export_layout_to_mesh_dict_list_tree_search_with_object_id,
    export_layout_to_mesh_dict_object_id
)
from m2t2_utils.data import generate_m2t2_data
from m2t2_utils.infer import load_m2t2, infer_m2t2

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

def get_place_location(scene_save_dir, target_object_name):

    layout = get_layout_from_scene_save_dir(scene_save_dir)
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

    # ee_frame_data_pos = ee_frame_data.target_pos_source.reshape(3)
    # ee_frame_data_quat = ee_frame_data.target_quat_source.reshape(4)

    # # print("original_ee_goals_pos:", original_ee_goals_pos)
    # # print("ee_frame_data_pos:", ee_frame_data_pos)

    # # Calculate the delta pose between the original ee goals and the ee frame data
    # # represent it as delta_pose = (dx, dy, dz, droll, dpitch, dyaw)
    
    # # Position delta
    # delta_pos = original_ee_goals_pos - ee_frame_data_pos 
    # delta_pos = delta_pos.clip(-1, 1)
    # # delta_pos = delta_pos / delta_pos.abs().max()
    
    # # Orientation delta - convert quaternions to rotation matrices and compute relative rotation
    # # Convert to numpy for scipy operations, assuming scalar-first quaternion format (w, x, y, z)
    # target_quat_np = original_ee_goals_quat.cpu().numpy()
    # current_quat_np = ee_frame_data_quat.cpu().numpy()
    
    # # Convert to scipy Rotation objects (scalar-first format)
    # target_rot = R.from_quat(target_quat_np[[1, 2, 3, 0]])  # Convert to (x, y, z, w)
    # current_rot = R.from_quat(current_quat_np[[1, 2, 3, 0]])  # Convert to (x, y, z, w)
    
    # # Calculate relative rotation: target = current * delta_rot
    # # So delta_rot = current.inv() * target
    # delta_rot = target_rot * current_rot.inv()
    
    # # Convert to Euler angles (roll, pitch, yaw) in radians
    # delta_euler = delta_rot.as_rotvec()
    # delta_euler = torch.tensor(delta_euler, dtype=torch.float, device=original_ee_goals.device)
    # if delta_euler.abs().max() > 1.0:
    #     delta_euler = delta_euler / delta_euler.abs().max()
    
    # # Combine position and orientation deltas
    # delta_pose = torch.cat([
    #     delta_pos,
    #     delta_euler,
    #     # torch.zeros_like(torch.tensor(delta_euler, dtype=torch.float, device=original_ee_goals.device))
    # ])

    # delta_pose = delta_pose.reshape(-1, 6)

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


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    log_dir = os.path.abspath(os.path.join("./logs/robomimic", args_cli.task+"-100-v2"))
    os.makedirs(os.path.join(log_dir, "debug"), exist_ok=True)

    # base_pos = [7.59, 6.76, 0.75]
    base_pos = [7.59, 6.76, 0.9]
    # rubiks cube
    # scene_save_dir = os.path.join(SERVER_ROOT_DIR, "results/layout_a2f73707")
    # target_object_name = "room_744fcab1_plastic_rubiks_cube_c7dee701"

    # mug
    scene_save_dir = os.path.join(SERVER_ROOT_DIR, "results/layout_4762e97d")
    target_object_name = "room_744fcab1_ceramic_black_mug_604a4d52"

    place_object_name = "room_744fcab1_ceramic_fruit_bowl_247ab0a6"

    all_object_names = [
        os.path.splitext(fname)[0] for fname in os.listdir(os.path.join(scene_save_dir, "usd_collection"))
    ]
    all_object_names = sorted(list(set(all_object_names)))

    all_object_rigids = [fname for fname in all_object_names if not( fname.startswith("floor_") or fname.startswith("wall_") \
                or fname.startswith('window_') or fname.startswith('door_'))]
    
    print(f"all_object_rigids: {all_object_rigids}")


    grasp_transforms = get_grasp_transforms(scene_save_dir, target_object_name, base_pos)
    grasp_center = grasp_transforms[..., :3, 3].reshape(-1, 3).mean(axis=0).reshape(3)

    grasp_object_height = get_grasp_object_height(scene_save_dir, target_object_name, grasp_center)
    print(f"grasp_object_height: {grasp_object_height}")

    place_location_w = get_place_location(scene_save_dir, place_object_name)
    print(f"place_location: {place_location_w}")
    place_location_w = place_location_w.reshape(3)
    place_location = place_location_w - np.array(base_pos).reshape(3)
    place_location = torch.tensor(place_location, device="cuda").float()
    place_location[2] = place_location[2] + grasp_object_height * 1.5

    grasp_center = torch.tensor(grasp_center, device="cuda").float()
    place_location_w = torch.tensor(place_location_w, device="cuda").float()

    camera_lookat = (grasp_center + place_location_w) / 2.0

    # Calculate camera_pos based on the following constraints:
    # 1. camera_pos - camera_lookat should be perpendicular to (grasp_center - place_location_w)
    # 2. |camera_pos - camera_lookat| >= 0.5
    # 3. cos sim: (camera_lookat - camera_pos) * (camera_lookat - base_pos) >= 0
    
    # Calculate the direction vector from grasp to place
    grasp_to_place = place_location_w - grasp_center
    
    # Get a vector perpendicular to grasp_to_place
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=grasp_center.device)
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=grasp_center.device)
    
    # Check if grasp_to_place is nearly parallel to z_axis
    grasp_to_place_norm = torch.norm(grasp_to_place)
    if grasp_to_place_norm > 1e-6:
        grasp_to_place_normalized = grasp_to_place / grasp_to_place_norm
        if torch.abs(torch.dot(grasp_to_place_normalized, z_axis)) > 0.9:
            # Use x_axis as reference
            perpendicular = torch.cross(grasp_to_place, x_axis)
        else:
            # Use z_axis as reference
            perpendicular = torch.cross(grasp_to_place, z_axis)
    else:
        # If grasp_to_place is zero, use a default perpendicular direction
        perpendicular = z_axis.clone()
    
    # Normalize the perpendicular vector
    perpendicular_norm = torch.norm(perpendicular)
    if perpendicular_norm > 1e-6:
        perpendicular = perpendicular / perpendicular_norm
    else:
        # Fallback if perpendicular is somehow zero
        perpendicular = z_axis.clone()
    
    # Scale to have minimum distance of 2.0
    min_distance = 1.2
    camera_offset = perpendicular * min_distance
    
    # Convert base_pos to tensor for calculations
    base_pos_tensor = torch.tensor(base_pos, device=grasp_center.device).float()
    
    # Try both directions for the camera position
    # Option 1: camera_pos = camera_lookat + camera_offset
    camera_pos_1 = camera_lookat + camera_offset
    view_direction_1 = camera_lookat - camera_pos_1  # Direction from camera to lookat
    lookat_to_base = camera_lookat - base_pos_tensor  # Vector from base_pos to lookat
    cos_sim_1 = torch.dot(view_direction_1, lookat_to_base)
    
    # Option 2: camera_pos = camera_lookat - camera_offset
    camera_pos_2 = camera_lookat - camera_offset
    view_direction_2 = camera_lookat - camera_pos_2  # Direction from camera to lookat
    cos_sim_2 = torch.dot(view_direction_2, lookat_to_base)
    
    # Choose the option that satisfies the cosine similarity constraint (>= 0)
    if cos_sim_1 >= 0:
        camera_pos = camera_pos_1
    elif cos_sim_2 >= 0:
        camera_pos = camera_pos_2
    else:
        # If neither satisfies the constraint, pick the one with higher cosine similarity
        if cos_sim_1 >= cos_sim_2:
            camera_pos = camera_pos_1
        else:
            camera_pos = camera_pos_2

    camera_pos[2] = camera_pos[2] + 1.0

    # create a yaml file to store the config
    config_dict = {
        "base_pos": base_pos,
        "target_object_name": target_object_name,
        "scene_save_dir": scene_save_dir,
        "camera_pos": camera_pos.tolist(),
        "camera_lookat": camera_lookat.tolist(),
    }
    env_init_config_yaml = os.path.join(log_dir, "params", "env_init.yaml")
    dump_yaml(env_init_config_yaml, config_dict)

    # assert (
    #     args_cli.task == "Isaac-Lift-Cube-Franka-IK-Rel-v0"
    # ), "Only 'Isaac-Lift-Cube-Franka-IK-Rel-v0' is supported currently."
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, config_yaml=env_init_config_yaml)

    # modify configuration such that the environment runs indefinitely
    # until goal is reached
    env_cfg.terminations.time_out = None
    # set the resampling time range to large number to avoid resampling
    # env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # add termination condition for reaching the goal otherwise the environment won't reset
    # env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # specify directory for logging experiments
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    env_cfg = load_pickle(os.path.join(log_dir, "params", "env.pkl"))

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).env

    # # create controller
    # if args_cli.teleop_device.lower() == "keyboard":
    #     teleop_interface = Se3Keyboard(pos_sensitivity=0.04, rot_sensitivity=0.08)
    # elif args_cli.teleop_device.lower() == "spacemouse":
    #     teleop_interface = Se3SpaceMouse(pos_sensitivity=0.05, rot_sensitivity=0.005)
    # else:
    #     raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # # add teleoperation key for env reset
    # teleop_interface.add_callback("L", env.reset)
    # # print helper
    # print(teleop_interface)




    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
        flush_freq=args_cli.num_envs,
        env_config={
            "cfg": os.path.join(log_dir, "params", "env.yaml"),
            "env_init_config_yaml": os.path.join(log_dir, "params", "env_init.yaml")
        },
    )

    # reset environment
    obs_dict, _ = env.reset()

    # reset interfaces
    # teleop_interface.reset()
    collector_interface.reset()

    iteration = 0

    ee_goals_translate = grasp_transforms[:, :3, 3].reshape(-1, 3) - np.array(base_pos).reshape(1, 3)
    ee_goals_quat = np.array([
        R.from_matrix(grasp_transforms_i[:3, :3]).as_quat(scalar_first=True) for grasp_transforms_i in grasp_transforms
    ]).reshape(-1, 4)
    ee_goals = np.concatenate([ee_goals_translate, ee_goals_quat], axis=1)
    ee_goals = torch.tensor(ee_goals, device=env.sim.device)
    print(f"ee_goals: {ee_goals.shape}")

    current_goal_idx = 0

    T_grasp = 350
    T_success_limit = 200
    T_close_gap = 10
    T_init = 10
    T_open_gap = 50

    grasp_up_offset = 0.1
    grasp_down_offset = 0.01
    lift_height_end = 0.3
    expected_success_height = lift_height_end * 0.6

    camera_0 = env.scene["camera_0"]

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
            "/World/camera_0",
            "/curobo",
        ],
        collision_avoidance_distance=0.01
        # collision_avoidance_distance=grasp_object_height
    )
    print("end creating motion planner")

    motion_planner.add_obstacle(
        plan_grasp=False,
        target_object_name=target_object_name
    )



    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt):
        while not collector_interface.is_stopped():
            # # get keyboard command
            # delta_pose, gripper_command = teleop_interface.advance()
            # # convert to torch
            # delta_pose = torch.tensor(delta_pose, dtype=torch.float, device="cuda").repeat(args_cli.num_envs, 1)
            # # compute actions based on environment
            # actions = pre_process_actions(delta_pose, gripper_command)

            # print(f"iteration: {iteration}")

            if iteration % T_grasp == 0:
                if current_goal_idx == 0:
                    current_pose_cnt = 0
                    all_ee_goals = []
                    while True:
                        grasp_transforms = get_grasp_transforms(scene_save_dir, target_object_name, base_pos)
                        ee_goals_translate = grasp_transforms[:, :3, 3].reshape(-1, 3) - np.array(base_pos).reshape(1, 3)
                        ee_goals_quat = np.array([
                            R.from_matrix(grasp_transforms_i[:3, :3]).as_quat(scalar_first=True) for grasp_transforms_i in grasp_transforms
                        ]).reshape(-1, 4)
                        ee_goals = np.concatenate([ee_goals_translate, ee_goals_quat], axis=1)
                        ee_goals = torch.tensor(ee_goals, device=env.sim.device).float()
                        print(f"ee_goals: {ee_goals.shape}")
                        current_pose_cnt += ee_goals.shape[0]
                        all_ee_goals.append(ee_goals)
                        if current_pose_cnt >= 100:
                            break
                    all_ee_goals = torch.cat(all_ee_goals, dim=0)
                    print(f"all_ee_goals: {all_ee_goals.shape}")
                    ee_goals = all_ee_goals

                iteration = 0
                robot = env.scene["robot"]
                sim = env.sim

                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                # print(f"joint_pos: {joint_pos.shape}; {joint_pos}")
                # print(f"joint_vel: {joint_vel.shape}; {joint_vel}")
                robot.write_joint_state_to_sim(joint_pos, joint_vel)

                # write_root_link_state_to_sim
                base_w = torch.zeros(1, 13).to(sim.device)
                random_pos = np.zeros(3)
                # random_pos = np.random.uniform(-0.1, 0.1, 3)
                robot_base = torch.tensor([
                    base_pos[0]+random_pos[0], 
                    base_pos[1]+random_pos[1], 
                    base_pos[2]+random_pos[2], 
                ]).to(sim.device)
                base_w[..., :3] = robot_base
                base_w[..., 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(sim.device)

                robot.write_root_link_state_to_sim(base_w)
                robot.reset()

                ee_goals_copy = ee_goals[current_goal_idx].clone()
                ee_goals_copy[0] -= float(random_pos[0])
                ee_goals_copy[1] -= float(random_pos[1])
                ee_goals_copy[2] -= float(random_pos[2])

                grasp_ee_goal_up = ee_goals_copy.clone()
                grasp_ee_goal_up[2] = grasp_ee_goal_up[2] + grasp_up_offset

                grasp_ee_goal_down = ee_goals_copy.clone()
                grasp_ee_goal_down[2] = grasp_ee_goal_down[2] - grasp_down_offset


                lift_ee_goal = ee_goals_copy.clone()
                # lift_ee_goal[2] = lift_ee_goal[2] + lift_height_end
                lift_ee_goal[2] = place_location[2]


                for object_name in all_object_rigids:
                    object_state = torch.zeros(1, 13).to(sim.device)
                    object_state[..., :3] = torch.zeros(1, 3).to(sim.device)
                    object_state[..., 3:4] = torch.ones(1, 1).to(sim.device)
                    object_state[..., 7:] = torch.zeros(1, 6).to(sim.device)
                    env.scene[object_name].write_root_state_to_sim(object_state)
                    env.scene[object_name].reset()

                camera_0.reset()

                buffer = []

                is_reach_grasp_up_goal = False
                is_reach_grasp_down_goal = False
                is_lift_up = False
                is_place_up = False



                # print("start adding obstacle")

                

                # print("end adding obstacle")

                target_ee_pose = grasp_ee_goal_up.unsqueeze(0)
                ee_frame_data = env.scene["ee_frame"].data
                current_ee_pose = torch.cat([ee_frame_data.target_pos_source.reshape(1, 3), ee_frame_data.target_quat_source.reshape(1, 4)], dim=-1)

                robot_qpos = robot.data.joint_pos

                # print(f"robot_qpos: {robot_qpos.shape}; {robot_qpos}")
                # print(f"target_ee_pose: {target_ee_pose.shape}; {target_ee_pose}")
                # print(f"current_ee_pose: {current_ee_pose.shape}; {current_ee_pose}")
                # print(current_ee_pose[..., :3] - target_ee_pose[..., :3])

                # target_ee_pose[..., 3:7] = math_utils.shortest_angles(target_ee_pose[..., 3:7], current_ee_pose[..., 3:7])

                # debug:
                camera_rgb_image_list = []
                camera_depth_image_list = []

                while True:
                
                    ee_goals_copy = ee_goals[current_goal_idx].clone()
                    ee_goals_copy[0] -= float(random_pos[0])
                    ee_goals_copy[1] -= float(random_pos[1])
                    ee_goals_copy[2] -= float(random_pos[2])

                    grasp_ee_goal_up = ee_goals_copy.clone()
                    grasp_ee_goal_up[2] = grasp_ee_goal_up[2] + grasp_up_offset

                    target_ee_pose = grasp_ee_goal_up.unsqueeze(0)
                
                    # print(f"target_ee_pose: {target_ee_pose.shape}; {target_ee_pose}")

                    ee_pose, _ = motion_planner.plan_motion(
                        robot_qpos,
                        target_ee_pose[:, :3],
                        target_ee_pose[:, 3:7],
                    )

                    if ee_pose is not None:

                        curobo_target_positions = ee_pose.ee_position
                        curobo_target_quaternion = ee_pose.ee_quaternion

                        curobo_target_ee_pos = torch.cat([
                            curobo_target_positions, curobo_target_quaternion,
                        ], dim=1).float()
                        
                        # print(f"curobo_target_ee_pos: {curobo_target_ee_pos.shape}; {curobo_target_ee_pos[0]}, {curobo_target_ee_pos[-1]}")

                        # calculate err between curobo_target_ee_pos[-1] and target_ee_pose (pos and angle in degree)
                        final_curobo_pose = curobo_target_ee_pos[-1]  # Final pose from trajectory
                        target_pose = target_ee_pose[0]  # Target pose
                        
                        # Position error (Euclidean distance)
                        pos_error = torch.norm(final_curobo_pose[:3] - target_pose[:3]).item()
                        
                        # Orientation error calculation
                        final_quat = final_curobo_pose[3:7]  # [w, x, y, z] format
                        target_quat = target_pose[3:7]       # [w, x, y, z] format
                        
                        # Calculate angular difference between quaternions
                        # Using the formula: angle = 2 * arccos(|q1 · q2|) where q1 and q2 are normalized quaternions
                        dot_product = torch.abs(torch.dot(final_quat, target_quat))
                        dot_product = torch.clamp(dot_product, 0.0, 1.0)  # Clamp to avoid numerical issues
                        angular_error_rad = 2 * torch.acos(dot_product)
                        angular_error_deg = torch.rad2deg(angular_error_rad).item()
                        
                        print(f"Motion planning success {current_goal_idx}: error - Position: {pos_error:.4f}m, Orientation: {angular_error_deg:.2f}°")

                        break

                    else:
                        print(f"Motion planning failed {current_goal_idx}; ee_pose is None")
                        curobo_target_ee_pos = None

                        if current_goal_idx == len(ee_goals) - 1:
                            break

                        current_goal_idx = (current_goal_idx + 1) % len(ee_goals)


            
            ee_frame_data = env.scene["ee_frame"].data
            # print(f"ee_frame_data pos: {ee_frame_data.target_pos_source.shape}; {ee_frame_data.target_pos_source}")
            # print(f"ee_frame_data quat: {ee_frame_data.target_quat_source.shape}; {ee_frame_data.target_quat_source}")

            com_state = env.scene[target_object_name].data.root_com_state_w
            target_object_current_z = float(com_state[:, 2])

            # print(f"root_state_w: {root_state_w.shape}; {root_state_w}")

            if not is_reach_grasp_up_goal:
                is_reach_grasp_up_goal = is_ee_reach_goal(grasp_ee_goal_up, ee_frame_data)

            elif not is_reach_grasp_down_goal:
                is_reach_grasp_down_goal = is_ee_reach_goal(grasp_ee_goal_down, ee_frame_data)
                if is_reach_grasp_down_goal:
                    close_iters = T_close_gap
                
            elif not is_lift_up:
                is_lift_up = is_ee_reach_goal(lift_ee_goal, ee_frame_data) and target_object_current_z - target_object_initial_z > expected_success_height

                if is_lift_up:
                    # plan the traj
                    ee_frame_data_pos = ee_frame_data.target_pos_source.reshape(3)
                    ee_frame_data_quat = ee_frame_data.target_quat_source.reshape(4)
                    robot_qpos = robot.data.joint_pos
                    ee_pose, _ = motion_planner.plan_motion(
                        robot_qpos,
                        place_location.reshape(1, 3),
                        lift_ee_goal[3:7].reshape(1, 4),
                        max_attempts=100
                    )

                    start_place_iteration = iteration
                    # root_state_w = env.scene[target_object_name].data.root_state_w

                    # rel_pos, rel_quat = math_utils.subtract_frame_transforms(
                    #     robot._data.root_state_w[:, :3],
                    #     robot._data.root_state_w[:, 3:7],
                    #     root_state_w[:, :3],
                    #     root_state_w[:, 3:7]
                    # )
                    # rel_pose = torch.cat([rel_pos, rel_quat], dim=1)

                    

                    if ee_pose is not None:
                        print("successfully plan the traj")

                        curobo_target_positions = ee_pose.ee_position
                        curobo_target_quaternion = ee_pose.ee_quaternion

                        curobo_target_ee_pos = torch.cat([
                            curobo_target_positions, curobo_target_quaternion,
                        ], dim=1).float()
                    
                    else:
                        print("failed to plan the traj")                        
                        curobo_target_ee_pos = torch.cat([
                            place_location.reshape(1, 3),
                            lift_ee_goal[3:7].reshape(1, 4),
                        ], dim=1).float()

                        iteration = T_grasp - 1

            elif not is_place_up:
                place_ee_goal = torch.cat([
                    place_location.reshape(1, 3),
                    lift_ee_goal[3:7].reshape(1, 4),
                ], dim=1).float()[0]
                is_place_up = is_ee_reach_goal(place_ee_goal, ee_frame_data)

                if is_place_up:
                    print("reach place up")
                    open_iters = T_open_gap
                        
            
            if not is_reach_grasp_up_goal:
                ee_goal = curobo_target_ee_pos[max(iteration - T_init, 0)] if curobo_target_ee_pos is not None and iteration - T_init < len(curobo_target_ee_pos) else grasp_ee_goal_up
                gripper_switch = 1
            elif not is_reach_grasp_down_goal:
                ee_goal = grasp_ee_goal_down
                gripper_switch = 1
            elif not is_lift_up:
                gripper_switch = -1
                if close_iters > 0:
                    close_iters -= 1
                    ee_goal = grasp_ee_goal_down
                else:
                    ee_goal = lift_ee_goal
            else:
                # ee_goal = torch.cat([place_location, lift_ee_goal[3:7]], dim=0)
                ee_goal = curobo_target_ee_pos[min(iteration - start_place_iteration, len(curobo_target_ee_pos) - 1)] 
                if not is_place_up:
                    gripper_switch = -1
                else:
                    gripper_switch = 1
                    if open_iters > 0:
                        open_iters -= 1
                    else:
                        iteration = T_grasp - 1

            # TODO: Get actions from motion planner
            # actions = torch.zeros((args_cli.num_envs, 7), dtype=torch.float, device="cuda")
            if iteration % T_grasp > T_init:
                actions = process_action(ee_goal, gripper_switch, ee_frame_data)
            else:
                actions = torch.cat([ee_frame_data.target_pos_source.reshape(1, 3), ee_frame_data.target_quat_source.reshape(1, 4), torch.ones(1, 1).to(env.sim.device)], dim=-1)
            # print(f"actions: {actions.shape}; {actions}")

            iteration_status = {}

            obs_dict = env.observation_manager.compute()

            # TODO: Deal with the case when reset is triggered by teleoperation device.
            #   The observations need to be recollected.
            # store signals before stepping
            # -- obs
            for key, value in obs_dict["policy"].items():
                # print(f"key: {key}; value: {value.shape};")
                # if "joint_qpos" in key:
                #     print("iteration: ", iteration % T_grasp, "obs", key, value.shape, value)
                #     print("actions: ", actions)
                    # print(f"ee_frame_data: {ee_frame_data.target_pos_source.shape}; {ee_frame_data.target_pos_source}; {ee_frame_data.target_quat_source.shape}; {ee_frame_data.target_quat_source};")
                # collector_interface.add(f"obs/{key}", value)
                iteration_status[f"obs/{key}"] = value
            # assert False
            # -- actions
            # print("actions", actions.shape, actions)
            # collector_interface.add("actions", actions)
            iteration_status["actions"] = actions

            # perform action on environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated
            # check that simulation is stopped or not
            if env.sim.is_stopped():
                break

            # robomimic only cares about policy observations
            # store signals from the environment
            # -- next_obs
            for key, value in obs_dict["policy"].items():
                # if "joint" in key and iteration % T_grasp < 2:
                #     print("iteration: ", iteration % T_grasp, "next obs", key, value.shape, value)
                # collector_interface.add(f"next_obs/{key}", value)
                iteration_status[f"next_obs/{key}"] = value
            # -- rewards
            # print("rewards", rewards)
            # collector_interface.add("rewards", rewards)
            iteration_status["rewards"] = rewards
            # -- dones
            # print("dones 0", dones.shape, dones)
            # collector_interface.add("dones", dones)
            iteration_status["dones"] = dones

            if iteration % T_grasp > T_init:
                buffer.append(iteration_status)

                # # debug
                # camera_rgb_image_list.append(iteration_status["obs/rgb"][0].cpu().numpy())
                # camera_depth_image_list.append(iteration_status["obs/depth"][0, :, :, 0].cpu().numpy())

            if iteration % T_grasp == 0:
                com_state = env.scene[target_object_name].data.root_com_state_w
                target_object_initial_z = float(com_state[:, 2])

            # if success:
            #     success_iteration = iteration
            #     iteration = T_grasp - 1

            if iteration % T_grasp == T_grasp - 1:
                # print(f"pose {current_goal_idx}: success = {success}; target_object_z_offset: {target_object_z_offset:.3f}; expected_success_height: {expected_success_height:.3f}")
                
                
                print(f"idx: {current_goal_idx}; cur_z: {target_object_current_z:.3f}; init_z: {target_object_initial_z:.3f}; exp_z: {expected_success_height:.3f}; is_place_up: {is_place_up}")

                current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
                if is_place_up:
                    # write all the status in buffer to data collector
                    print(f"iteration: {iteration}; writing {len(buffer)} status to data collector")
                    for status in buffer:
                        for key, value in status.items():
                            # print(f"key: {key}; value: {value.shape};")
                            collector_interface.add(key, value)
                    collector_interface.flush()

                    # debug:
                    # save video with imageio
                    # imageio.mimwrite(os.path.join(log_dir, "debug", f"{current_goal_idx:0>3d}_video.mp4"), camera_rgb_image_list, fps=10)
                    # imageio.mimwrite(os.path.join(log_dir, "debug", f"{current_goal_idx:0>3d}_depth_video.mp4"), camera_depth_image_list, fps=10)

                buffer = []




            iteration_status = {}
            iteration = (iteration + 1) % T_grasp

            # check if enough data is collected
            if collector_interface.is_stopped():
                break

    # close the simulator
    collector_interface.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
