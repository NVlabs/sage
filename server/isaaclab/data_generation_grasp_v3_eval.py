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
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import open3d as o3d

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR, ROBOMIMIC_ROOT_DIR, M2T2_ROOT_DIR

sys.path.insert(0, SERVER_ROOT_DIR)
sys.path.insert(0, M2T2_ROOT_DIR)
from utils import get_layout_from_scene_save_dir
from tex_utils import export_layout_to_mesh_dict_list_tree_search_with_object_id
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

from robomimic.algo import RolloutPolicy
import robomimic.utils.file_utils as FileUtils

from constants import SERVER_ROOT_DIR, ROBOMIMIC_ROOT_DIR



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

    ee_frame_data_pos = ee_frame_data.target_pos_source.reshape(3)
    ee_frame_data_quat = ee_frame_data.target_quat_source.reshape(4)

    # print("original_ee_goals_pos:", original_ee_goals_pos)
    # print("ee_frame_data_pos:", ee_frame_data_pos)

    # Calculate the delta pose between the original ee goals and the ee frame data
    # represent it as delta_pose = (dx, dy, dz, droll, dpitch, dyaw)
    
    # Position delta
    delta_pos = original_ee_goals_pos - ee_frame_data_pos 
    delta_pos = delta_pos.clip(-1, 1)
    # delta_pos = delta_pos / delta_pos.abs().max()
    
    # Orientation delta - convert quaternions to rotation matrices and compute relative rotation
    # Convert to numpy for scipy operations, assuming scalar-first quaternion format (w, x, y, z)
    target_quat_np = original_ee_goals_quat.cpu().numpy()
    current_quat_np = ee_frame_data_quat.cpu().numpy()
    
    # Convert to scipy Rotation objects (scalar-first format)
    target_rot = R.from_quat(target_quat_np[[1, 2, 3, 0]])  # Convert to (x, y, z, w)
    current_rot = R.from_quat(current_quat_np[[1, 2, 3, 0]])  # Convert to (x, y, z, w)
    
    # Calculate relative rotation: target = current * delta_rot
    # So delta_rot = current.inv() * target
    delta_rot = target_rot * current_rot.inv()
    
    # Convert to Euler angles (roll, pitch, yaw) in radians
    delta_euler = delta_rot.as_rotvec()
    delta_euler = torch.tensor(delta_euler, dtype=torch.float, device=original_ee_goals.device)
    if delta_euler.abs().max() > 1.0:
        delta_euler = delta_euler / delta_euler.abs().max()
    
    # Combine position and orientation deltas
    delta_pose = torch.cat([
        delta_pos,
        delta_euler,
        # torch.zeros_like(torch.tensor(delta_euler, dtype=torch.float, device=original_ee_goals.device))
    ])

    delta_pose = delta_pose.reshape(-1, 6)
    gripper_vel = torch.tensor([gripper_switch], dtype=torch.float, device="cuda").reshape(-1, 1)

    actions = torch.cat([delta_pose, gripper_vel], dim=-1)

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

    loc_success = delta_pos.abs().max() < 0.01
    rot_success = delta_euler.abs().max() < 0.01

    # print(f"loc_success: {loc_success}; rot_success: {rot_success}")

    return loc_success


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    log_dir = os.path.abspath(os.path.join("./logs/robomimic", args_cli.task+"-100-v2"))
    robomimic_policy_path = os.path.join(ROBOMIMIC_ROOT_DIR, "robomimic/../diffusion_policy_trained_models/dp_depth_mug/20250805141549/last.pth")

    base_pos = [7.59, 6.76, 0.9]
    # base_pos = [7.59, 6.76, 0.75]
    # rubiks cube
    # scene_save_dir = os.path.join(SERVER_ROOT_DIR, "results/layout_a2f73707")
    # target_object_name = "room_744fcab1_plastic_rubiks_cube_c7dee701"

    # mug
    scene_save_dir = os.path.join(SERVER_ROOT_DIR, "results/layout_4762e97d")
    target_object_name = "room_744fcab1_ceramic_black_mug_604a4d52"

    env_init_config_yaml = os.path.join(log_dir, "params", "env_init.yaml")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, config_yaml=env_init_config_yaml)

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).env

    # reset environment
    obs_dict, _ = env.reset()

    iteration = 0

    T_grasp = 350
    T_init = 10

    camera_0 = env.scene["camera_0"]

    all_object_names = [
        os.path.splitext(fname)[0] for fname in os.listdir(os.path.join(scene_save_dir, "usd_collection"))
    ]
    all_object_names = sorted(list(set(all_object_names)))


    all_object_rigids = [fname for fname in all_object_names if not( fname.startswith("floor_") or fname.startswith("wall_") \
            or fname.startswith('window_') or fname.startswith('door_'))]
    print(f"all_object_rigids: {all_object_rigids}")

    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(device="cuda", ckpt_path=robomimic_policy_path)

    # simulate environment -- run everything in inference mode
    while True:

        if iteration % T_grasp == 0:
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
            # random_pos = np.random.uniform(-0.05, 0.05, 3)
            robot_base = torch.tensor([
                base_pos[0]+random_pos[0], 
                base_pos[1]+random_pos[1], 
                base_pos[2]+random_pos[2], 
            ]).to(sim.device)
            base_w[..., :3] = robot_base
            base_w[..., 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(sim.device)

            robot.write_root_link_state_to_sim(base_w)
            robot.reset()

            # zero_object_state = torch.zeros_like(init_object_state)
            # zero_object_state[..., 3] = torch.ones_like(zero_object_state[..., 3])
            # rigid_object_collection.write_object_state_to_sim(zero_object_state)
            # # rigid_object_collection.write_object_state_to_sim(init_object_state)
            # rigid_object_collection.reset()

            for object_name in all_object_rigids:
                object_state = torch.zeros(1, 13).to(sim.device)
                object_state[..., :3] = torch.zeros(1, 3).to(sim.device)
                object_state[..., 3:4] = torch.ones(1, 1).to(sim.device)
                object_state[..., 7:] = torch.zeros(1, 6).to(sim.device)
                env.scene[object_name].write_root_state_to_sim(object_state)
                env.scene[object_name].reset()
            
            camera_0.reset()

            rollout_policy.start_episode()

        def get_obs_dict_for_policy(obs_dict):
            return {
                k: v[0] for k, v in obs_dict["policy"].items()
            }

        obs_dict = env.observation_manager.compute()
        
        if iteration % T_grasp > T_init:
            actions = rollout_policy(ob=get_obs_dict_for_policy(obs_dict))
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(sim.device)
            actions = actions.unsqueeze(0)
        else:
            ee_frame_data = env.scene["ee_frame"].data
            actions = torch.cat([ee_frame_data.target_pos_source.reshape(1, 3), ee_frame_data.target_quat_source.reshape(1, 4), torch.ones(1, 1).to(env.unwrapped.sim.device)], dim=-1)

        # perform action on environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        dones = terminated | truncated
        # check that simulation is stopped or not
        if env.sim.is_stopped():
            break

        iteration += 1


        # debug
        # os.makedirs(os.path.join(log_dir, "debug"), exist_ok=True)

        # # rgb_image = camera_0.data.output["rgb"][0].cpu().numpy().astype(np.uint8)
        # # depth_image = camera_0.data.output["distance_to_image_plane"][0].cpu().numpy()

        # # depth_image = depth_image[..., 0].clip(0, 10) / 10.0 * 255.0
        # # depth_image = depth_image.astype(np.uint8)


        # # Image.fromarray(rgb_image).save(os.path.join(log_dir, "debug", f"{iteration:0>3d}_rgb_image.png"))
        # # Image.fromarray(depth_image).save(os.path.join(log_dir, "debug", f"{iteration:0>3d}_depth_image.png"))

        # points = iteration_status["obs/points"].cpu().numpy().reshape(-1, 6)
        # # save to ply file with open3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(points[:, 3:])
        # o3d.io.write_point_cloud(os.path.join(log_dir, "debug", f"{iteration:0>3d}_points.ply"), pcd)


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
