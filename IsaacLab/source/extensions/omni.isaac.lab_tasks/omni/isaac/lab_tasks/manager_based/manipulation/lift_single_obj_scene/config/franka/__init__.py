# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

# gym.register(
#     id="Isaac-Lift-Single-Obj-Scene-Franka-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaSingleObjSceneLiftEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Lift-Cube-Franka-Play-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#     },
#     disable_env_checker=True,
# )

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Lift-Single-Obj-Scene-Franka-IK-Abs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{ik_abs_env_cfg.FrankaSingleObjSceneLiftEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Single-Obj-Scene-Franka-IK-Abs-v0-vis",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{ik_abs_env_cfg.FrankaSingleObjSceneLiftEnvCfg_Vis}:config.yaml",
    },
    disable_env_checker=True,
)

# gym.register(
#     id="Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaTeddyBearLiftEnvCfg",
#     },
#     disable_env_checker=True,
# )

gym.register(
    id="Isaac-Lift-Single-Obj-Scene-Franka-IK-Abs-v0-vis-teaser",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{ik_abs_env_cfg.FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser}:config.yaml",
    },
    disable_env_checker=True,
)


##
# Inverse Kinematics - Relative Pose Control
##
gym.register(
    id="Isaac-Lift-Single-Obj-Scene-Franka-IK-Rel-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{ik_rel_env_cfg.FrankaSingleObjSceneLiftEnvCfg}:config.yaml",
    },
    disable_env_checker=True,
)
