# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.lift_single_obj_scene import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift_single_obj_scene.lift_single_obj_scene_env_cfg import LiftSingleObjSceneEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg, RigidObjectCollectionCfg, ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
# Explicit imports for action configurations to help linter
from omni.isaac.lab.envs.mdp import JointPositionActionCfg, BinaryJointPositionActionCfg
from typing import Optional

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import json

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
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from dataclasses import MISSING, Field, dataclass, field, replace
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
import yaml

def build_camera(camera_pos, camera_lookat, camera_up):
    """
    Build camera pose from position, lookat target, and up vector.
    
    Args:
        camera_pos: [x, y, z] position of camera
        camera_lookat: [x, y, z] point the camera should look at
        camera_up: [x, y, z] up direction vector
        
    Returns:
        tuple: (camera_pos, camera_rot) where camera_rot is [w, x, y, z] quaternion
    """
    # Convert to numpy arrays for easier computation
    pos = np.array(camera_pos)
    target = np.array(camera_lookat)
    up = np.array(camera_up)
    
    # Calculate camera coordinate system
    # Forward vector (camera looking direction) - points from camera to target
    forward = target - pos
    forward = forward / np.linalg.norm(forward)
    
    # Right vector - perpendicular to forward and up
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Recalculate up vector to ensure orthogonality
    up_corrected = np.cross(right, forward)
    up_corrected = up_corrected / np.linalg.norm(up_corrected)
    
    # Build rotation matrix (camera coordinate system)
    # Note: In ROS convention:
    # - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - right axis: ``+X``
    rotation_matrix = np.array([
        [right[0], -up_corrected[0], forward[0]],
        [right[1], -up_corrected[1], forward[1]], 
        [right[2], -up_corrected[2], forward[2]]
    ])

    print(f"rotation_matrix: {rotation_matrix}")
    
    # Convert rotation matrix to quaternion using scipy
    scipy_rotation = R.from_matrix(rotation_matrix)
    quaternion = scipy_rotation.as_quat()  # Returns [x, y, z, w] format
    
    # Convert to [w, x, y, z] format as requested
    camera_rot = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    
    return camera_pos, camera_rot

@configclass
class FrankaSingleObjSceneLiftEnvCfg(LiftSingleObjSceneEnvCfg):

    config_yaml: Optional[str] = field(default=None)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        if self.config_yaml is None:
            raise ValueError("config_yaml must be provided")
        
        with open(self.config_yaml, 'r') as file:
            env_cfg = yaml.safe_load(file)

        base_pos = env_cfg["base_pos"]

        scene_save_dir = env_cfg["scene_save_dir"]
        mass_dict = env_cfg.get("mass_dict", {})

        def get_mass(object_name):
            mass = mass_dict.get(object_name, None).get("mass", None)
            print(f"[DEBUG] object_name: {object_name}, mass: {mass}")
            # mass = mass * 0.5
            mass = max(mass, 0.1)
            return mass

        usd_collection_dir = env_cfg.get("usd_collection_dir", f"{scene_save_dir}/usd_collection")
        print(f"[DEBUG] usd_collection_dir: {usd_collection_dir}")

        # environment as below
        # # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # set the robot position
        self.scene.robot.init_state.pos = (base_pos[0], base_pos[1], base_pos[2])
        self.scene.robot.actuators["panda_hand"] = ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        )

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.],
                        # pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )



        # set cameras
        camera_lookat = [3.0, 0.0, 0.0]
        camera_pos = [0., 0.4, 1.2]
        camera_up = [0., 0., 1.]

        # Get camera pose using the build_camera function
        camera_pos, camera_rot = build_camera(camera_pos, camera_lookat, camera_up)

        self.scene.camera_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/camera_left",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos, rot=camera_rot, convention="ros"),
        )

        # self.observations.policy.points_left = ObsTerm(
        #     func=mdp.points,
        #     params={"camera_name": "camera_left"},
        # )

        self.observations.policy.rgb_left = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_left"},
        )

        self.observations.policy.depth_left = ObsTerm(
            func=mdp.image_depth,
            params={"camera_name": "camera_left"},
        )


        rigid_object_property_dict_path = os.path.join(usd_collection_dir, "rigid_object_property_dict.json")
        with open(rigid_object_property_dict_path, 'r') as file:
            rigid_object_property_dict = json.load(file)


        # set cameras
        camera_lookat = [3.0, 0.0, 0.0]
        camera_pos = [0., 0.4, 1.2]
        camera_up = [0., 0., 1.]

        # Get camera pose using the build_camera function
        camera_pos, camera_rot = build_camera(camera_pos, camera_lookat, camera_up)

        self.scene.camera_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/camera_right",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos, rot=camera_rot, convention="ros"),
        )

        # self.observations.policy.points_right = ObsTerm(
        #     func=mdp.points,
        #     params={"camera_name": "camera_right"},
        # )

        self.observations.policy.rgb_right = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_right"},
        )

        self.observations.policy.depth_right = ObsTerm(
            func=mdp.image_depth,
            params={"camera_name": "camera_right"},
        )


        # set wrist camera
        camera_lookat = [0.0, 0.0, 10.0]
        camera_pos = [0., 0., 0.08]
        camera_up = [0., 1., 0.]

        # Get camera pose using the build_camera function
        camera_pos, camera_rot = build_camera(camera_pos, camera_lookat, camera_up)

        self.scene.camera_wrist = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/camera_wrist",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos, rot=camera_rot, convention="ros"),
        )

        # self.observations.policy.points_wrist = ObsTerm(
        #     func=mdp.points,
        #     params={"camera_name": "camera_wrist"},
        # )

        self.observations.policy.rgb_wrist = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_wrist"},
        )

        self.observations.policy.depth_wrist = ObsTerm(
            func=mdp.image_depth,
            params={"camera_name": "camera_wrist"},
        )

        self.observations.policy.points = ObsTerm(
            func=mdp.points,
            params={"camera_name_list": ["camera_left", "camera_right", "camera_wrist"]},
        )

        # mount
        for usd_file_name in os.listdir(usd_collection_dir):
            if not usd_file_name.endswith(".usdz"):
                continue
            
            if (usd_file_name.startswith("floor_") and "ceiling" not in usd_file_name) or usd_file_name.startswith("wall_") \
                or usd_file_name.startswith('window_'):

                setattr(
                    self.scene, 
                    os.path.splitext(usd_file_name)[0], 
                    AssetBaseCfg(
                        prim_path="{ENV_REGEX_NS}/"+os.path.splitext(usd_file_name)[0],
                        spawn=sim_utils.UsdFileCfg(
                            usd_path=f"{usd_collection_dir}/{usd_file_name}",
                        ),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
                    )
                )

            elif usd_file_name.startswith('door_'):

                setattr(
                    self.scene, 
                    os.path.splitext(usd_file_name)[0], 
                    AssetBaseCfg(
                        prim_path="{ENV_REGEX_NS}/"+os.path.splitext(usd_file_name)[0],
                        spawn=sim_utils.UsdFileCfg(
                            usd_path=f"{usd_collection_dir}/{usd_file_name}",
                        ),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
                    )
                )


            elif os.path.splitext(usd_file_name)[0] in mass_dict:
                rigid_object_property = rigid_object_property_dict.get(os.path.splitext(usd_file_name)[0], None)
                if rigid_object_property is None:
                    print(f"[DEBUG] rigid_object_property is not found for {os.path.splitext(usd_file_name)[0]}")
                    continue
                # assert rigid_object_property is not None, f"rigid_object_property is not found for {os.path.splitext(usd_file_name)[0]}"

                static = rigid_object_property["static"]

                print("adding: ", os.path.splitext(usd_file_name)[0])

                if static:
                    setattr(
                        self.scene, 
                        os.path.splitext(usd_file_name)[0], 
                        RigidObjectCfg(
                            prim_path="{ENV_REGEX_NS}/"+os.path.splitext(usd_file_name)[0],
                            spawn=sim_utils.UsdFileCfg(
                                usd_path=f"{usd_collection_dir}/{usd_file_name}",
                                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                    kinematic_enabled=True,
                                    # ADD/INCREASE these for stability:
                                    disable_gravity=True,
                                    linear_damping=1e5,
                                    angular_damping=1e5,
                                    
                                    # REDUCE these to prevent bouncing:
                                    max_linear_velocity=0.5,    # Current: 1.0 - much too high for grasping
                                    max_angular_velocity=0.5,   # Current: 1.0 - much too high for grasping 
                                    max_depenetration_velocity=50.0,  # Current: not set - add this limit
                                    
                                    # INCREASE solver iterations for better contact stability:
                                    solver_position_iteration_count=16,  # Current: 16 - double it
                                    solver_velocity_iteration_count=1,   # Current: 1 - increase
                                ),
                                collision_props=sim_utils.CollisionPropertiesCfg(
                                    collision_enabled=True,
                                    contact_offset=0.005,       # Increase from 0.001 to 0.005
                                    rest_offset=0.001,          # Add small rest offset 
                                    torsional_patch_radius=0.01, # Add for rotational friction
                                ),
                            ),
                            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
                        )
                    )
                else:
                    setattr(
                        self.scene, 
                        os.path.splitext(usd_file_name)[0], 
                        RigidObjectCfg(
                            prim_path="{ENV_REGEX_NS}/"+os.path.splitext(usd_file_name)[0],
                            spawn=sim_utils.UsdFileCfg(
                                usd_path=f"{usd_collection_dir}/{usd_file_name}",
                                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                    # ADD/INCREASE these for stability:
                                    linear_damping=10.0,        # Current: 20.0 - try reducing to 5.0-15.0  
                                    angular_damping=10.0,       # Current: 20.0 - try reducing to 5.0-15.0
                                    
                                    # REDUCE these to prevent bouncing:
                                    max_linear_velocity=5.0,    # Current: 1.0 - much too high for grasping
                                    max_angular_velocity=5.0,   # Current: 1.0 - much too high for grasping 
                                    max_depenetration_velocity=500.0,  # Current: not set - add this limit
                                    
                                    # INCREASE solver iterations for better contact stability:
                                    solver_position_iteration_count=32,  # Current: 16 - double it
                                    solver_velocity_iteration_count=2,   # Current: 1 - increase
                                ),
                                mass_props=sim_utils.MassPropertiesCfg(mass=get_mass(os.path.splitext(usd_file_name)[0])),
                                collision_props=sim_utils.CollisionPropertiesCfg(
                                    collision_enabled=True,
                                    contact_offset=0.005,       # Increase from 0.001 to 0.001
                                    rest_offset=0.001,          # Add small rest offset 
                                    torsional_patch_radius=100.0, # Add for rotational friction
                                ),
                            ),
                            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
                        )
                    )
        