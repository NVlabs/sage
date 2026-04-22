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

from omni.isaac.lab_tasks.manager_based.manipulation.mobile_manipulation_obj_scene_v2 import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.mobile_manipulation_obj_scene_v2.mobile_manipulation_obj_scene_env_cfg import MobileManipulationObjSceneEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.mobile_franka import (
    FRANKA_PANDA_HIGH_PD_CFG,  # isort: skip
    OMRON_PANDA_CFG,
    MOBILE_FRANKA_PANDA_CFG,
)
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg, RigidObjectCollectionCfg, ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
# Explicit imports for action configurations to help linter
from omni.isaac.lab.envs.mdp import JointPositionActionCfg, BinaryJointPositionActionCfg
from typing import Optional

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

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
import json

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
class RobotMobileManipulationObjSceneEnvCfg(MobileManipulationObjSceneEnvCfg):

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
        room_dict = env_cfg.get("room_dict", {})


        def get_mass(object_name):
            mass = mass_dict.get(object_name, None).get("mass", None)
            print(f"[DEBUG] object_name: {object_name}, mass: {mass}")
            # mass = mass * 0.2
            return mass

        usd_collection_dir = env_cfg.get("usd_collection_dir", f"{scene_save_dir}/usd_collection")
        print(f"[DEBUG] usd_collection_dir: {usd_collection_dir}")

        # environment as below
        # # Set Franka as robot
        self.scene.robot = MOBILE_FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # set the robot position
        self.scene.robot.init_state.pos = (base_pos[0], base_pos[1], base_pos[2])
        

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/arm_right_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.],
                        # pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        self.scene.fixed_support_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/base_fixed_support",
                    name="fixed_support",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.],
                        # pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        self.scene.support_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/base_support",
                    name="support",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.],
                        # pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        camera_pos_front = [0.3, 0.0, 1.5]
        camera_lookat_front = [1.0, 0.0, 1.0]
        camera_up_front = [0.0, 0.0, 1.0]

        camera_pos_back = [-0.3, 0.0, 1.5]
        camera_lookat_back = [-1.0, 0.0, 1.0]
        camera_up_back = [0.0, 0.0, 1.0]

        camera_pos_left = [0.0, 0.4, 0.9]
        camera_lookat_left = [0.75, 0.0, 0.0]
        camera_up_left = [0.0, 0.0, 1.0]
        
        camera_pos_right = [0.0, -0.4, 0.9]
        camera_lookat_right = [0.75, 0.0, 0.0]
        camera_up_right = [0.0, 0.0, 1.0]

        camera_lookat_wrist = [0.0, 0.0, 0.5]
        camera_pos_wrist = [0., 0.0, 0.07125]
        camera_up_wrist = [0., 1., 0.]


        def get_camera_quat(camera_pos, camera_lookat, camera_up):

            camera_pos = np.array(camera_pos)
            camera_lookat = np.array(camera_lookat)
            camera_up = np.array(camera_up)

            axis_z = camera_pos - camera_lookat

            axis_z = axis_z / np.linalg.norm(axis_z)

            axis_x = np.cross(camera_up, axis_z)
            axis_x = axis_x / np.linalg.norm(axis_x)

            axis_y = np.cross(axis_z, axis_x)
            axis_y = axis_y / np.linalg.norm(axis_y)

            rotation_matrix = np.concatenate([axis_x.reshape(3, 1), axis_y.reshape(3, 1), axis_z.reshape(3, 1)], axis=1)

            scipy_rotation = R.from_matrix(rotation_matrix)
            quaternion = scipy_rotation.as_quat()  # Returns [x, y, z, w] format

            return [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]

        camera_rot_front = get_camera_quat(camera_pos_front, camera_lookat_front, camera_up_front)
        camera_rot_back = get_camera_quat(camera_pos_back, camera_lookat_back, camera_up_back)
        camera_rot_left = get_camera_quat(camera_pos_left, camera_lookat_left, camera_up_left)
        camera_rot_right = get_camera_quat(camera_pos_right, camera_lookat_right, camera_up_right)
        camera_rot_wrist = get_camera_quat(camera_pos_wrist, camera_lookat_wrist, camera_up_wrist)

        image_resolution = 128

        self.scene.camera_front = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_fixed_support/camera_front",
            update_period=0.0,
            height=image_resolution,
            width=image_resolution,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.FisheyeCameraCfg(
                fisheye_max_fov=240.0,
                fisheye_polynomial_b=0.0015
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_front, rot=camera_rot_front, convention="opengl"),
        )

        self.scene.camera_back = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_fixed_support/camera_back",
            update_period=0.0,
            height=image_resolution,
            width=image_resolution,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.FisheyeCameraCfg(
                fisheye_max_fov=240.0,
                fisheye_polynomial_b=0.0015
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_back, rot=camera_rot_back, convention="opengl"),
        )

        self.scene.camera_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_support/camera_left",
            update_period=0.0,
            height=image_resolution,
            width=image_resolution,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=15.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_left, rot=camera_rot_left, convention="opengl"),
        )

        self.scene.camera_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_support/camera_right",
            update_period=0.0,
            height=image_resolution,
            width=image_resolution,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=15.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_right, rot=camera_rot_right, convention="opengl"),
        )



        self.scene.camera_wrist = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/arm_right_hand/camera_wrist",
            update_period=0.0,
            height=image_resolution,
            width=image_resolution,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=7.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_wrist, rot=camera_rot_wrist, convention="opengl"),
        )

        # light 
        for room_id, room in room_dict.items():
            setattr(
                self.scene, 
                f"light_{room['id']}_1", 
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/"+f"light_{room['id']}_1",
                    init_state=AssetBaseCfg.InitialStateCfg(pos=[room["x_min"] + 0.2, room["y_min"] + 0.2, room["height"] - 0.2]),
                    spawn=sim_utils.SphereLightCfg(color=(0.75, 0.75, 0.75), intensity=100000.0, treat_as_point=True),
                )
            )

            setattr(
                self.scene, 
                f"light_{room['id']}_2", 
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/"+f"light_{room['id']}_2",
                    init_state=AssetBaseCfg.InitialStateCfg(pos=[room["x_max"] - 0.2, room["y_min"] + 0.2, room["height"] - 0.2]),
                    spawn=sim_utils.SphereLightCfg(color=(0.75, 0.75, 0.75), intensity=100000.0, treat_as_point=True),
                )
            )

            setattr(
                self.scene, 
                f"light_{room['id']}_3", 
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/"+f"light_{room['id']}_3",
                    init_state=AssetBaseCfg.InitialStateCfg(pos=[room["x_min"] + 0.2, room["y_max"] - 0.2, room["height"] - 0.2]),
                    spawn=sim_utils.SphereLightCfg(color=(0.75, 0.75, 0.75), intensity=100000.0, treat_as_point=True),
                )
            )

            setattr(
                self.scene, 
                f"light_{room['id']}_4", 
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/"+f"light_{room['id']}_4",
                    init_state=AssetBaseCfg.InitialStateCfg(pos=[room["x_max"] - 0.2, room["y_max"] - 0.2, room["height"] - 0.2]),
                    spawn=sim_utils.SphereLightCfg(color=(0.75, 0.75, 0.75), intensity=100000.0, treat_as_point=True),
                )
            )


        self.observations.policy.rgb_front = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_front"},
        )

        self.observations.policy.depth_front = ObsTerm(
            func=mdp.image_depth,
            params={"camera_name": "camera_front", "depth_max": 4.0},
        )

        self.observations.policy.rgb_back = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_back"},
        )

        self.observations.policy.depth_back = ObsTerm(
            func=mdp.image_depth,
            params={"camera_name": "camera_back", "depth_max": 4.0},
        )

        self.observations.policy.rgb_left = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_left"},
        )

        self.observations.policy.depth_left = ObsTerm(
            func=mdp.image_depth,
            params={"camera_name": "camera_left"},
        )

        self.observations.policy.rgb_right = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_right"},
        )

        self.observations.policy.depth_right = ObsTerm(
            func=mdp.image_depth,
            params={"camera_name": "camera_right"},
        )

        self.observations.policy.rgb_wrist = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_wrist"},
        )

        self.observations.policy.depth_wrist = ObsTerm(
            func=mdp.image_depth,
            params={"camera_name": "camera_wrist", "depth_max": 1.0},
        )

        # mount
        rigid_object_property_dict_path = os.path.join(usd_collection_dir, "rigid_object_property_dict.json")
        with open(rigid_object_property_dict_path, 'r') as file:
            rigid_object_property_dict = json.load(file)

        for usd_file_name in os.listdir(usd_collection_dir):
            if not usd_file_name.endswith(".usdz"):
                continue
            
            if usd_file_name.startswith("floor_") or usd_file_name.startswith("wall_") \
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

                continue
                setattr(
                    self.scene, 
                    os.path.splitext(usd_file_name)[0], 
                    ArticulationCfg(
                        prim_path="{ENV_REGEX_NS}/"+os.path.splitext(usd_file_name)[0],
                        spawn=sim_utils.UsdFileCfg(
                            usd_path=f"{usd_collection_dir}/{usd_file_name}",
                        ),
                        init_state=ArticulationCfg.InitialStateCfg(),
                        actuators={
                            "door_joint": ImplicitActuatorCfg(
                                joint_names_expr=[".*"],
                                effort_limit=100.0,
                                velocity_limit=1.0,            
                                stiffness=80.0,
                                damping=4.0,
                            )
                        }
                    )
                )

            else:
                rigid_object_property = rigid_object_property_dict.get(os.path.splitext(usd_file_name)[0], None)
                assert rigid_object_property is not None, f"rigid_object_property is not found for {os.path.splitext(usd_file_name)[0]}"

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
                                    # ADD/INCREASE these for stability:
                                    disable_gravity=False,
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
                                # mass_props=sim_utils.MassPropertiesCfg(mass=1000.0 if "rubiks" not in usd_file_name else 0.10),
                                mass_props=sim_utils.MassPropertiesCfg(mass=get_mass(os.path.splitext(usd_file_name)[0])),
                                # physics_material=sim_utils.RigidBodyMaterialCfg(),
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
                                    solver_position_iteration_count=16,  # Current: 16 - double it
                                    solver_velocity_iteration_count=1,   # Current: 1 - increase
                                ),
                                # mass_props=sim_utils.MassPropertiesCfg(mass=1000.0 if "rubiks" not in usd_file_name else 0.10),
                                mass_props=sim_utils.MassPropertiesCfg(mass=get_mass(os.path.splitext(usd_file_name)[0])),
                                # physics_material=sim_utils.RigidBodyMaterialCfg(),
                                collision_props=sim_utils.CollisionPropertiesCfg(
                                    collision_enabled=True,
                                    contact_offset=0.005,       # Increase from 0.001 to 0.005
                                    rest_offset=0.001,          # Add small rest offset 
                                    torsional_patch_radius=0.04, # Add for rotational friction
                                ),
                            ),
                            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
                        )
                    )
        