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
from isaaclab.curobo_tools.curobo_planner import MotionPlanner


def get_action_relative(ee_frame_pos, ee_frame_quat, target_ee_pos, target_ee_quat):
    d_rotvec = math_utils.axis_angle_from_quat(math_utils.quat_unique(math_utils.quat_mul(target_ee_quat, math_utils.quat_inv(ee_frame_quat))))
    arm_action = torch.cat([target_ee_pos - ee_frame_pos, d_rotvec.reshape(-1, 3)], dim=-1)
    return arm_action

def get_default_camera_view():
    return [0., 0., 0.], [1., 0., 0.]

def get_default_base_pos():
    return [0., 0., 0.]

def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    log_dir = os.path.abspath(os.path.join("./logs/robomimic", args_cli.task+"-test"))
    os.makedirs(os.path.join(log_dir, "debug"), exist_ok=True)

    layout_id = "layout_fac9613b"
    scene_save_dir = os.path.join(SERVER_ROOT_DIR, f"results/{layout_id}")


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
    T_grasp = 200

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

                ee_frame_data = env.scene["ee_frame"].data
                ee_frame_data_pos_target = ee_frame_data.target_pos_source.reshape(-1, 3).clone().repeat(num_envs, 1)
                ee_frame_data_pos_target[:, 2] += 0.2
                ee_frame_data_quat_target = ee_frame_data.target_quat_source.reshape(-1, 4).clone().repeat(num_envs, 1)

                fixed_support_frame_data = env.scene["fixed_support_frame"].data
                fixed_support_frame_data_pos = fixed_support_frame_data.target_pos_source.reshape(-1, 3).clone().repeat(num_envs, 1)
                fixed_support_frame_data_quat = fixed_support_frame_data.target_quat_source.reshape(-1, 4).clone().repeat(num_envs, 1)

                fixed_support_frame_data_quat_initial = fixed_support_frame_data_quat.clone()

                ee_frame_to_fixed_support_frame_pos_target, ee_frame_to_fixed_support_frame_quat_target = math_utils.subtract_frame_transforms(
                    fixed_support_frame_data_pos, fixed_support_frame_data_quat,
                    ee_frame_data_pos_target, ee_frame_data_quat_target
                )

            ee_frame_data = env.scene["ee_frame"].data
            ee_frame_data_pos = ee_frame_data.target_pos_source.reshape(-1, 3).clone().repeat(num_envs, 1)
            ee_frame_data_quat = ee_frame_data.target_quat_source.reshape(-1, 4).clone().repeat(num_envs, 1)

            fixed_support_frame_data = env.scene["fixed_support_frame"].data
            fixed_support_frame_data_pos = fixed_support_frame_data.target_pos_source.reshape(-1, 3).clone().repeat(num_envs, 1)
            fixed_support_frame_data_quat = fixed_support_frame_data.target_quat_source.reshape(-1, 4).clone().repeat(num_envs, 1)

            ee_frame_to_fixed_support_frame_pos, ee_frame_to_fixed_support_frame_quat = math_utils.subtract_frame_transforms(
                fixed_support_frame_data_pos, fixed_support_frame_data_quat,
                ee_frame_data_pos, ee_frame_data_quat
            )

            fixed_support_frame_data_pos_target = fixed_support_frame_data_pos.clone()
            fixed_support_frame_data_pos_target[:, :2] += 0.5
            fixed_support_frame_data_quat_target = fixed_support_frame_data_quat_initial.clone()



            # actions = torch.tensor([0., 0., 0., 0., 0., 0.], device=env.sim.device).reshape(1, 6).repeat(num_envs, 1)
            print("iteration: ", iteration)
            actions = torch.cat([
                get_action_relative(
                    fixed_support_frame_data_pos, fixed_support_frame_data_quat, 
                    fixed_support_frame_data_pos_target, fixed_support_frame_data_quat_target
                ),
                get_action_relative(
                    ee_frame_to_fixed_support_frame_pos, ee_frame_to_fixed_support_frame_quat, 
                    ee_frame_to_fixed_support_frame_pos_target, ee_frame_to_fixed_support_frame_quat_target
                ),
                torch.tensor([1.0], device=env.sim.device).reshape(1, -1).repeat(num_envs, 1),
            ], dim=-1)

            print(f"actions: {actions.shape} {actions}")

            joint_pos = robot.data.joint_pos.cpu().numpy().reshape(-1).tolist()
            body_link_state_w = robot.data.body_link_state_w[:, :7]
            # joint_pos_str = [f"{j:.4f}" for j in joint_pos]
            # actions_str = [f"{a:.4f}" for a in actions.cpu().numpy().reshape(-1).tolist()]
            # print(f"actions: {actions_str}")
            # print(f"joint_pos: {joint_pos_str}")
            # print(f"body_names: {len(robot.data.body_names)} {robot.data.body_names}")
            # # print(f"joint_names: {robot.data.joint_names}")
            # print(f"ee_frame_data_pos: {ee_frame_data_pos}")
            # print(f"ee_frame_data_quat: {ee_frame_data_quat}")
            # print(f"body_link_state_w: {body_link_state_w.shape} {body_link_state_w}")

            iteration_status = {}

            obs_dict = env.observation_manager.compute()
            
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            iteration = (iteration + 1) % T_grasp


    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
