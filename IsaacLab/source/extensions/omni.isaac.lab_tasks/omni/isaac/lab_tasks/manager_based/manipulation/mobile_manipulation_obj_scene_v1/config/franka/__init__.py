# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg


##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Mobile-Manipulation-Obj-Scene-Franka-IK-Abs-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{ik_abs_env_cfg.RobotMobileManipulationObjSceneEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##
gym.register(
    id="Isaac-Mobile-Manipulation-Obj-Scene-Franka-IK-Rel-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{ik_rel_env_cfg.RobotMobileManipulationObjSceneEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)
