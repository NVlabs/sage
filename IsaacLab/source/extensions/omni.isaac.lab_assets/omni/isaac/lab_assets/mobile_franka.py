# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

OMRON_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            # rigid_body_enabled=True,
            # max_depenetration_velocity=5.0,
        ),
        # mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        asset_path="/home/hongchix/main/curobo/src/curobo/content/assets/robot/omron_franka/composed_robot.urdf",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_mobile_forward": 2.0,
            "joint_mobile_side": 2.0,
            "joint_mobile_yaw": 0.0,
            "joint_torso_height": 0.0,
        },
    ),
    actuators={
        "mobile_base_translate": ImplicitActuatorCfg(
            joint_names_expr=["joint_mobile_forward", "joint_mobile_side", "joint_mobile_yaw"],
            velocity_limit=100.0,
            effort_limit=10000.0,
            stiffness=0.0,
            damping=1e5,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["joint_torso_height"],
            effort_limit=5000.0,
            velocity_limit=2.175,
            stiffness=5000.0,
            damping=1000.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""
FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0

MOBILE_FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            # rigid_body_enabled=True,
            # max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.001,       # Increase from 0.001 to 0.001
            rest_offset=0.001,          # Add small rest offset 
            torsional_patch_radius=0.1, # Add for rotational friction
        ),
        # mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        asset_path="/home/hongchix/main/curobo/src/curobo/content/assets/robot/omron_franka/composed_robot.urdf",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=16, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "base_joint_mobile_forward": 2.0,
            "base_joint_mobile_side": 2.0,
            "base_joint_mobile_yaw": 0.0,
            "base_joint_torso_height": 0.0,
            "arm_joint1": 0.0,
            "arm_joint2": -0.569,
            "arm_joint3": 0.0,
            "arm_joint4": -2.810,
            "arm_joint5": 0.0,
            "arm_joint6": 3.037,
            "arm_joint7": 0.741,
            "gripper_finger_joint1": 0.04,
            "gripper_finger_joint2": -0.04,
        },
    ),
    actuators={
        "mobile_base": ImplicitActuatorCfg(
            joint_names_expr=["base_joint_mobile_side", "base_joint_mobile_forward"],
            velocity_limit=30.0,
            effort_limit=50000.0,
            stiffness=10000.0,
            damping=1000.0,
        ),
        "mobile_base_rotate": ImplicitActuatorCfg(
            joint_names_expr=["base_joint_mobile_yaw"],
            velocity_limit=30.0,
            effort_limit=50000.0,
            stiffness=10000.0,
            damping=2000.0,
        ),
        # "torso": ImplicitActuatorCfg(
        #     joint_names_expr=["base_joint_torso_height"],
        #     effort_limit=5000.0,
        #     velocity_limit=2.175,
        #     stiffness=1000.0,
        #     damping=1000.0,
        # ),
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["base_joint_torso_height", "arm_joint[1-4]"],
            # effort_limit=87.0,
            # velocity_limit=2.175,
            # stiffness=400.0,
            # damping=80.0,
            effort_limit=200.0,
            velocity_limit=2.175,
            stiffness=1000.0,
            damping=8.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[5-7]"],
            effort_limit=200.0,
            velocity_limit=2.61,
            stiffness=1000.0,
            damping=8.0,
        ),
        "finger": ImplicitActuatorCfg(
            joint_names_expr=["gripper_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


# # FRANKA_PANDA_HIGH_PD_CFG = MOBILE_FRANKA_PANDA_CFG.copy()
# # FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# # FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
# # FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
# # FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
# # FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
# """Configuration of Franka Emika Panda robot with stiffer PD control.

# This configuration is useful for task-space control using differential IK.
# """
