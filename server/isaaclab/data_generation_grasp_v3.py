# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

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
    log_dir = os.path.abspath(os.path.join("./logs/robomimic", args_cli.task+"-100-v0"))

    base_pos = [7.59, 6.76, 0.9]
    # # rubiks cube
    scene_save_dir = os.path.join(SERVER_ROOT_DIR, "results/layout_a2f73707")
    target_object_name = "room_744fcab1_plastic_rubiks_cube_c7dee701"

    # mug
    # scene_save_dir = os.path.join(SERVER_ROOT_DIR, "results/layout_4762e97d")
    # target_object_name = "room_744fcab1_ceramic_black_mug_604a4d52"

    all_object_names = [
        os.path.splitext(fname)[0] for fname in os.listdir(os.path.join(scene_save_dir, "usd_collection"))
    ]
    all_object_names = sorted(list(set(all_object_names)))

    all_object_rigids = [fname for fname in all_object_names if not( fname.startswith("floor_") or fname.startswith("wall_") \
                or fname.startswith('window_') or fname.startswith('door_'))]
    
    print(f"all_object_rigids: {all_object_rigids}")

    

    grasp_transforms = get_grasp_transforms(scene_save_dir, target_object_name, base_pos)

    camera_lookat = grasp_transforms[..., :3, 3].reshape(-1, 3).mean(axis=0).reshape(3)

    # create a yaml file to store the config
    config_dict = {
        "base_pos": base_pos,
        "target_object_name": target_object_name,
        "scene_save_dir": scene_save_dir,
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
    env = gym.make(args_cli.task, cfg=env_cfg)

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
    ee_goals = torch.tensor(ee_goals, device=env.unwrapped.sim.device)
    print(f"ee_goals: {ee_goals.shape}")

    current_goal_idx = 0

    T_grasp = 250
    T_close_gap = 10
    T_init = 10

    grasp_up_offset = 0.1
    grasp_down_offset = 0.01
    lift_height_end = 0.1
    expected_success_height = lift_height_end * 0.6

    camera_0 = env.unwrapped.scene["camera_0"]


    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
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
                    grasp_transforms = get_grasp_transforms(scene_save_dir, target_object_name, base_pos)
                    ee_goals_translate = grasp_transforms[:, :3, 3].reshape(-1, 3) - np.array(base_pos).reshape(1, 3)
                    ee_goals_quat = np.array([
                        R.from_matrix(grasp_transforms_i[:3, :3]).as_quat(scalar_first=True) for grasp_transforms_i in grasp_transforms
                    ]).reshape(-1, 4)
                    ee_goals = np.concatenate([ee_goals_translate, ee_goals_quat], axis=1)
                    ee_goals = torch.tensor(ee_goals, device=env.unwrapped.sim.device)
                    print(f"ee_goals: {ee_goals.shape}")

                iteration = 0
                robot = env.unwrapped.scene["robot"]
                sim = env.unwrapped.sim

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
                lift_ee_goal[2] = lift_ee_goal[2] + lift_height_end

                for object_name in all_object_rigids:
                    object_state = torch.zeros(1, 13).to(sim.device)
                    object_state[..., :3] = torch.zeros(1, 3).to(sim.device)
                    object_state[..., 3:4] = torch.ones(1, 1).to(sim.device)
                    object_state[..., 7:] = torch.zeros(1, 6).to(sim.device)
                    env.unwrapped.scene[object_name].write_root_state_to_sim(object_state)
                    env.unwrapped.scene[object_name].reset()

                # zero_object_state = torch.zeros_like(init_object_state)
                # zero_object_state[..., 3] = torch.ones_like(zero_object_state[..., 3])
                # rigid_object_collection.write_object_state_to_sim(zero_object_state)
                # # rigid_object_collection.write_object_state_to_sim(init_object_state)
                # rigid_object_collection.reset()

                camera_0.reset()

                buffer = []

                is_reach_grasp_up_goal = False
                is_reach_grasp_down_goal = False

            
            ee_frame_data = env.unwrapped.scene["ee_frame"].data
            # print(f"ee_frame_data pos: {ee_frame_data.target_pos_source.shape}; {ee_frame_data.target_pos_source}")
            # print(f"ee_frame_data quat: {ee_frame_data.target_quat_source.shape}; {ee_frame_data.target_quat_source}")

            if not is_reach_grasp_up_goal:
                is_reach_grasp_up_goal = is_ee_reach_goal(grasp_ee_goal_up, ee_frame_data)

            if not is_reach_grasp_down_goal:
                is_reach_grasp_down_goal = is_ee_reach_goal(grasp_ee_goal_down, ee_frame_data)
                if is_reach_grasp_up_goal:
                    close_iters = T_close_gap

            
            if not is_reach_grasp_up_goal:
                ee_goal = grasp_ee_goal_up
                gripper_switch = 1
            elif not is_reach_grasp_down_goal:
                ee_goal = grasp_ee_goal_down
                gripper_switch = 1
            else:
                gripper_switch = -1
                if close_iters > 0:
                    close_iters -= 1
                    ee_goal = grasp_ee_goal_down
                else:
                    ee_goal = lift_ee_goal

            if iteration % T_grasp > 1:
                com_state = env.unwrapped.scene[target_object_name].data.root_com_state_w
                target_object_current_z = float(com_state[:, 2])
                success = target_object_current_z - target_object_initial_z > expected_success_height
            else:
                success = False

            # TODO: Get actions from motion planner
            # actions = torch.zeros((args_cli.num_envs, 7), dtype=torch.float, device="cuda")
            if iteration % T_grasp > T_init:
                actions = process_action(ee_goal, gripper_switch, ee_frame_data)
            else:
                actions = torch.cat([ee_frame_data.target_pos_source.reshape(1, 3), ee_frame_data.target_quat_source.reshape(1, 4), torch.ones(1, 1).to(env.unwrapped.sim.device)], dim=-1)
            # print(f"actions: {actions.shape}; {actions}")

            iteration_status = {}

            obs_dict = env.unwrapped.observation_manager.compute()

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
            if env.unwrapped.sim.is_stopped():
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

            if iteration % T_grasp == 0:
                com_state = env.unwrapped.scene[target_object_name].data.root_com_state_w
                target_object_initial_z = float(com_state[:, 2])

            if success:
                success_iteration = iteration
                iteration = T_grasp - 1

            if iteration % T_grasp == T_grasp - 1:
                # print(f"pose {current_goal_idx}: success = {success}; target_object_z_offset: {target_object_z_offset:.3f}; expected_success_height: {expected_success_height:.3f}")
                
                
                print(f"idx: {current_goal_idx}; success: {success}; cur_z: {target_object_current_z:.3f}; init_z: {target_object_initial_z:.3f}; exp_z: {expected_success_height:.3f}")

                current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
                if success:
                    # write all the status in buffer to data collector
                    print(f"iteration: {success_iteration}; writing {len(buffer)} status to data collector")
                    for status in buffer:
                        for key, value in status.items():
                            # print(f"key: {key}; value: {value.shape};")
                            collector_interface.add(key, value)
                    collector_interface.flush()

                buffer = []

            # -- is success label
            # collector_interface.add("success", env.unwrapped.termination_manager.get_term("object_reached_goal"))

            # # flush data from collector for successful environments
            # reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            # collector_interface.flush(reset_env_ids)

            # print("env scene: ", env.unwrapped.scene)
            # print("env scene robot: ", env.unwrapped.scene["robot"])
            # print("env scene rigid_object_collection object_com_state_w: ", 
            # env.unwrapped.scene["rigid_object_collection"].data.object_com_state_w)


            # debug
            # os.makedirs(os.path.join(log_dir, "debug"), exist_ok=True)

            # # rgb_image = camera_0.data.output["rgb"][0].cpu().numpy().astype(np.uint8)
            # # depth_image = camera_0.data.output["distance_to_image_plane"][0].cpu().numpy()

            # # depth_image = depth_image[..., 0].clip(0, 10) / 10.0 * 255.0
            # # depth_image = depth_image.astype(np.uint8)


            # Image.fromarray(iteration_status["obs/rgb"][0].cpu().numpy()).save(os.path.join(log_dir, "debug", f"{iteration:0>3d}_rgb.png"))
            # Image.fromarray(iteration_status["obs/depth"][0, :, :, 0].cpu().numpy()).save(os.path.join(log_dir, "debug", f"{iteration:0>3d}_depth.png"))
            # # Image.fromarray(depth_image).save(os.path.join(log_dir, "debug", f"{iteration:0>3d}_depth_image.png"))

            # points = iteration_status["obs/points"].cpu().numpy().reshape(-1, 6)
            # # save to ply file with open3d
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            # pcd.colors = o3d.utility.Vector3dVector(points[:, 3:])
            # o3d.io.write_point_cloud(os.path.join(log_dir, "debug", f"{iteration:0>3d}_points.ply"), pcd)

            iteration_status = {}
            iteration += 1

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
