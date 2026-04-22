# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import DeformableObjectCfg
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointVelocityActionCfg,
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
    BinaryJointPositionActionCfg
)
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sim.spawners import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

import omni.isaac.lab_tasks.manager_based.manipulation.lift.mdp as mdp

from . import joint_pos_env_cfg, joint_pos_env_cfg_vis, joint_pos_env_cfg_vis_teaser

##
# Pre-defined configs
##
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
import numpy as np
from scipy.spatial.transform import Rotation as R
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from typing import Optional
from dataclasses import MISSING, Field, dataclass, field, replace
import omni.isaac.lab.sim as sim_utils
import yaml

##
# Rigid object lift environment.
##



def get_full_view_camera_sampling(room_center, room_scales, resolution, horizontal_angle, vertical_angle, fov=35.0):
    """
    Sample camera position and lookat position to ensure the entire room is visible.
    
    Uses optimization to find the minimum radius that ensures all room corners
    are visible at all horizontal angles.
    
    Args:
        room_center: (x, y, z) center of the room
        room_scales: (width, length, height) dimensions of the room
        resolution: (height, width) render resolution tuple
        horizontal_angle: horizontal rotation angle in degrees
        vertical_angle: vertical elevation angle in degrees (from horizontal plane)
        fov: field of view in degrees
    
    Returns:
        camera_pos: (x, y, z) camera position
        lookat_pos: (x, y, z) lookat position
        fov: field of view in degrees (returned as-is)
    """
    width, length, height = room_scales
    res_height, res_width = resolution
    aspect_ratio = res_width / res_height
    
    # Step 1: Calculate lookat position at the center of the room
    lookat_pos = np.array([room_center[0], room_center[1], room_center[2]])
    
    # Step 2: Calculate FOV parameters
    # Note: 'fov' parameter is the VERTICAL FOV in degrees
    fov_rad = np.radians(fov)
    fov_vertical = fov_rad
    # Calculate horizontal FOV from vertical FOV and aspect ratio
    fov_horizontal = 2 * np.arctan(np.tan(fov_vertical / 2) * aspect_ratio)
    
    vertical_angle_rad = np.radians(vertical_angle)
    
    # Step 3: Get all 8 corners of the room relative to lookat
    corners_relative = []
    for dx in [-width/2, width/2]:
        for dy in [-length/2, length/2]:
            for dz in [-height/2, height/2]:
                corners_relative.append(np.array([dx, dy, dz]))
    
    # Step 4: Function to check if all corners are visible from a given radius and horizontal angle
    def corners_fit_in_view(radius, horiz_angle_rad):
        """
        Check if all room corners fit within the camera's FOV at the given radius and angle.
        Projects corners to pixel coordinates and checks if they're within [0, W] x [0, H].
        Returns the maximum pixel overflow (negative if all fit, positive if any exceed bounds).
        """
        # Camera position relative to lookat
        cam_offset = np.array([
            radius * np.cos(vertical_angle_rad) * np.cos(horiz_angle_rad),
            radius * np.cos(vertical_angle_rad) * np.sin(horiz_angle_rad),
            radius * np.sin(vertical_angle_rad)
        ])
        
        # Camera forward direction (from camera to lookat)
        forward = -cam_offset / np.linalg.norm(cam_offset)
        
        # Camera right and up vectors
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Calculate focal length from FOV (using vertical FOV)
        # focal_length = (image_height / 2) / tan(fov_vertical / 2)
        focal_length_y = (res_height / 2.0) / np.tan(fov_vertical / 2)
        focal_length_x = (res_width / 2.0) / np.tan(fov_horizontal / 2)
        
        max_overflow = 0.0
        
        for corner in corners_relative:
            # Vector from camera to corner in world space
            to_corner = corner - cam_offset
            
            # Transform to camera space
            x_cam = np.dot(to_corner, right)
            y_cam = np.dot(to_corner, up)
            z_cam = np.dot(to_corner, forward)
            
            # Skip if behind camera
            if z_cam <= 0:
                return float('inf')  # Corner is behind camera, this radius doesn't work
            
            # Project to image plane (perspective projection)
            # pixel_x = focal_length_x * (x_cam / z_cam) + center_x
            # pixel_y = focal_length_y * (y_cam / z_cam) + center_y
            center_x = res_width / 2.0
            center_y = res_height / 2.0
            
            pixel_x = focal_length_x * (x_cam / z_cam) + center_x
            pixel_y = center_y - focal_length_y * (y_cam / z_cam)  # Subtract because image y is inverted
            
            # Check if pixel is within bounds [0, width] x [0, height]
            overflow_left = -pixel_x
            overflow_right = pixel_x - res_width
            overflow_top = -pixel_y
            overflow_bottom = pixel_y - res_height
            
            max_overflow = max(max_overflow, overflow_left, overflow_right, overflow_top, overflow_bottom)
        
        return max_overflow
    
    # Step 5: Find minimum radius that works for all horizontal angles
    # We'll sample several horizontal angles to find the worst case
    # Version 1 (16 points): test_angles = np.linspace(0, 2 * np.pi, 16)
    # Version 2 (4 cardinal directions): Use 0°, 90°, 180°, 270°
    # Version 3 (longest edges only): Use angles perpendicular to longest dimension
    # If width > length, looking from sides (90°, 270°) shows longest dimension
    # If length > width, looking from ends (0°, 180°) shows longest dimension
    if width >= length:
        test_angles = np.array([90, 270]) * np.pi / 180  # Perpendicular to width (longest)
        print(f"Using longest edge mode: width ({width:.2f}) >= length ({length:.2f})")
        print(f"Test angles: 90° and 270° (perpendicular to longest dimension)")
    else:
        test_angles = np.array([0, 180]) * np.pi / 180  # Perpendicular to length (longest)
        print(f"Using longest edge mode: length ({length:.2f}) > width ({width:.2f})")
        print(f"Test angles: 0° and 180° (perpendicular to longest dimension)")
    
    def objective(radius_candidate):
        """Objective: maximum pixel overflow across all test angles (want this <= 0)"""
        max_overflow = 0.0
        for test_angle in test_angles:
            overflow = corners_fit_in_view(radius_candidate, test_angle)
            max_overflow = max(max_overflow, overflow)
        return max_overflow
    
    # Initial guess: use the room's bounding sphere radius scaled by FOV
    initial_radius = np.sqrt(width**2 + length**2 + height**2) / (2 * np.tan(fov_vertical / 2))
    
    # Debug: Print room and camera parameters
    print(f"\n=== Camera Sampling Debug Info ===")
    print(f"Room dimensions (W x L x H): {width:.2f} x {length:.2f} x {height:.2f}")
    print(f"Room center: ({room_center[0]:.2f}, {room_center[1]:.2f}, {room_center[2]:.2f})")
    print(f"Lookat position: ({lookat_pos[0]:.2f}, {lookat_pos[1]:.2f}, {lookat_pos[2]:.2f})")
    print(f"Vertical angle: {vertical_angle:.1f}°, Horizontal angle: {horizontal_angle:.1f}°")
    print(f"FOV: {fov:.1f}° (vertical), {np.degrees(fov_horizontal):.1f}° (horizontal)")
    print(f"Aspect ratio: {aspect_ratio:.2f}")
    print(f"Initial radius guess: {initial_radius:.2f}")
    
    # Find the minimum radius where objective(radius) <= 0
    # Use binary search for efficiency
    radius_min = initial_radius * 0.5
    radius_max = initial_radius * 3.0
    
    print(f"\nBinary search range: [{radius_min:.2f}, {radius_max:.2f}]")
    print(f"Testing pixel overflows at {len(test_angles)} horizontal angles...")
    
    for iteration in range(20):  # Binary search iterations
        radius_mid = (radius_min + radius_max) / 2
        overflow = objective(radius_mid)
        
        if iteration % 5 == 0 or iteration == 19:  # Print every 5th iteration and last
            print(f"  Iter {iteration:2d}: radius={radius_mid:.2f}, max_overflow={overflow:.1f}px {'✗' if overflow > 0 else '✓'}")
        
        if overflow > 0:
            # Overflow still exists, need larger radius
            radius_min = radius_mid
        else:
            # All corners fit, try smaller radius
            radius_max = radius_mid
    
    radius = radius_max * 1.05  # Add 5% safety margin
    
    print(f"\nFinal radius (with 5% margin): {radius:.2f}")
    
    # Verify the final radius works for the current angle and show corner projections
    final_overflow = corners_fit_in_view(radius, np.radians(horizontal_angle))
    print(f"Verification at current angle: overflow={final_overflow:.1f}px {'✗ FAIL' if final_overflow > 0 else '✓ PASS'}")
    
    # Show actual corner pixel coordinates for debugging
    print(f"\nCorner projections for current angle (resolution: {res_width}x{res_height}):")
    horizontal_angle_rad = np.radians(horizontal_angle)
    cam_offset = np.array([
        radius * np.cos(vertical_angle_rad) * np.cos(horizontal_angle_rad),
        radius * np.cos(vertical_angle_rad) * np.sin(horizontal_angle_rad),
        radius * np.sin(vertical_angle_rad)
    ])
    forward = -cam_offset / np.linalg.norm(cam_offset)
    world_up = np.array([0, 0, 1])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    focal_length_y = (res_height / 2.0) / np.tan(fov_vertical / 2)
    focal_length_x = (res_width / 2.0) / np.tan(fov_horizontal / 2)
    center_x = res_width / 2.0
    center_y = res_height / 2.0
    
    for i, corner in enumerate(corners_relative):
        to_corner = corner - cam_offset
        x_cam = np.dot(to_corner, right)
        y_cam = np.dot(to_corner, up)
        z_cam = np.dot(to_corner, forward)
        
        if z_cam > 0:
            pixel_x = focal_length_x * (x_cam / z_cam) + center_x
            pixel_y = center_y - focal_length_y * (y_cam / z_cam)
            in_bounds = (0 <= pixel_x <= res_width) and (0 <= pixel_y <= res_height)
            print(f"  Corner {i}: ({pixel_x:7.1f}, {pixel_y:7.1f}) {'✓' if in_bounds else '✗ OUT'}")
        else:
            print(f"  Corner {i}: BEHIND CAMERA")
    
    # Step 6: Calculate camera position for the specific horizontal angle (already computed above)
    dx = radius * np.cos(vertical_angle_rad) * np.cos(horizontal_angle_rad)
    dy = radius * np.cos(vertical_angle_rad) * np.sin(horizontal_angle_rad)
    dz = radius * np.sin(vertical_angle_rad)
    
    camera_pos = lookat_pos + np.array([dx, dy, dz])
    
    print(f"Camera position: ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})")
    print(f"Camera-to-lookat distance: {np.linalg.norm(camera_pos - lookat_pos):.2f}")
    print(f"===================================\n")
    
    return camera_pos, lookat_pos, fov



@configclass
class RobotMobileManipulationObjSceneEnvCfg(joint_pos_env_cfg.RobotMobileManipulationObjSceneEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        # self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)


        self.actions.base_action_transform = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=['base_joint_mobile_side', 'base_joint_mobile_forward', 'base_joint_mobile_yaw'],
            body_name="base_fixed_support",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.]),
        )

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["base_joint_torso_height", "arm_joint.*"],
            body_name="arm_right_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.]),
            # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_finger_joint.*"],
            open_command_expr={"gripper_finger_joint1": 0.04, "gripper_finger_joint2": -0.04},
            close_command_expr={"gripper_finger_joint1": 0.0, "gripper_finger_joint2": 0.0},
        )

@configclass
class RobotMobileManipulationObjSceneEnvCfg_Vis(joint_pos_env_cfg_vis_teaser.RobotMobileManipulationObjSceneEnvCfg):
    config_yaml: Optional[str] = field(default=None)
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        if self.config_yaml is None:
            raise ValueError("config_yaml must be provided")
        
        with open(self.config_yaml, 'r') as file:
            env_cfg = yaml.safe_load(file)

        room_dict = env_cfg.get("room_dict", {})

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        # self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)


        self.actions.base_action_transform = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=['base_joint_mobile_side', 'base_joint_mobile_forward', 'base_joint_mobile_yaw'],
            body_name="base_fixed_support",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.]),
        )

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["base_joint_torso_height", "arm_joint.*"],
            body_name="arm_right_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.]),
            # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_finger_joint.*"],
            open_command_expr={"gripper_finger_joint1": 0.04, "gripper_finger_joint2": -0.04},
            close_command_expr={"gripper_finger_joint1": 0.0, "gripper_finger_joint2": 0.0},
        )
         
        around_radius = 1.0
        around_height = 1.0

        camera_lookat_center_vis = [0.3, 0.0, 0.2]
        camera_up_vis = [0.0, 0.0, 1.0]

        camera_pos_front_left_vis = [around_radius, -around_radius, around_height]
        camera_pos_front_right_vis = [around_radius, around_radius, around_height]

        camera_pos_back_left_vis = [-around_radius, -around_radius, around_height]
        camera_pos_back_right_vis = [-around_radius, around_radius, around_height]

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

        image_resolution_h = 1080
        image_resolution_w = 1920
        focal_length = 15.0

        camera_rot_front_left_vis = get_camera_quat(camera_pos_front_left_vis, camera_lookat_center_vis, camera_up_vis)
        camera_rot_front_right_vis = get_camera_quat(camera_pos_front_right_vis, camera_lookat_center_vis, camera_up_vis)
        camera_rot_back_left_vis = get_camera_quat(camera_pos_back_left_vis, camera_lookat_center_vis, camera_up_vis)
        camera_rot_back_right_vis = get_camera_quat(camera_pos_back_right_vis, camera_lookat_center_vis, camera_up_vis)

        self.scene.camera_front_left_vis = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_support/camera_front_left_vis",
            update_period=0.0,
            height=image_resolution_h,
            width=image_resolution_w,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_front_left_vis, rot=camera_rot_front_left_vis, convention="opengl"),
        )

        self.scene.camera_front_right_vis = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_support/camera_front_right_vis",
            update_period=0.0,
            height=image_resolution_h,
            width=image_resolution_w,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_front_right_vis, rot=camera_rot_front_right_vis, convention="opengl"),
        )
        
        self.scene.camera_back_left_vis = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_support/camera_back_left_vis",
            update_period=0.0,
            height=image_resolution_h,
            width=image_resolution_w,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_back_left_vis, rot=camera_rot_back_left_vis, convention="opengl"),
        )

        self.scene.camera_back_right_vis = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_support/camera_back_right_vis",
            update_period=0.0,
            height=image_resolution_h,
            width=image_resolution_w,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_back_right_vis, rot=camera_rot_back_right_vis, convention="opengl"),
        )
        
        # use the first room
        room = room_dict[list(room_dict.keys())[0]]
        room_x_min = room["x_min"]
        room_x_max = room["x_max"]
        room_y_min = room["y_min"]
        room_y_max = room["y_max"]
        room_height = room["height"]

        room_center = [(room_x_min + room_x_max) / 2, (room_y_min + room_y_max) / 2, room_height / 2]
        room_scales = [room_x_max - room_x_min, room_y_max - room_y_min, room_height]

        max_room_edge_length = max(room_x_max - room_x_min, room_y_max - room_y_min)

        top_view_camera_center_pos = [(room_x_min + room_x_max) / 2, (room_y_min + room_y_max) / 2, room_height * (1.0 + max_room_edge_length / 2.0)]
        top_view_camera_lookat_pos = [(room_x_min + room_x_max) / 2, (room_y_min + room_y_max) / 2, 0.0] 
        top_view_camera_up_vec = [0.0, 1.0, 0.0]

        top_view_camera_rot = get_camera_quat(top_view_camera_center_pos, top_view_camera_lookat_pos, top_view_camera_up_vec)


        self.scene.camera_top_view = CameraCfg(
            prim_path="{ENV_REGEX_NS}/camera_top_view",
            update_period=0.0,
            height=image_resolution_h,
            width=image_resolution_w,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=top_view_camera_center_pos, rot=top_view_camera_rot, convention="opengl"),
        )

        self.observations.policy.rgb_front_left_vis = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_front_left_vis"},
        )

        self.observations.policy.rgb_front_right_vis = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_front_right_vis"},
        )

        self.observations.policy.rgb_back_left_vis = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_back_left_vis"},
        )

        self.observations.policy.rgb_back_right_vis = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_back_right_vis"},
        )

        self.observations.policy.rgb_top_view = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_top_view"},
        )

        vertical_angle = 45.0
        fov = 35.0
        camera_pos_out_of_wall_0, camera_lookat_out_of_wall_0, _ = get_full_view_camera_sampling(
            room_center, room_scales, (image_resolution_h, image_resolution_w), 0.0, vertical_angle, fov)
        
        camera_pos_out_of_wall_1, camera_lookat_out_of_wall_1, _ = get_full_view_camera_sampling(
            room_center, room_scales, (image_resolution_h, image_resolution_w), 90.0, vertical_angle, fov)
        
        camera_pos_out_of_wall_2, camera_lookat_out_of_wall_2, _ = get_full_view_camera_sampling(
            room_center, room_scales, (image_resolution_h, image_resolution_w), 180.0, vertical_angle, fov)

        camera_pos_out_of_wall_3, camera_lookat_out_of_wall_3, _ = get_full_view_camera_sampling(
            room_center, room_scales, (image_resolution_h, image_resolution_w), 270.0, vertical_angle, fov)

        camera_rot_out_of_wall_0 = get_camera_quat(camera_pos_out_of_wall_0, camera_lookat_out_of_wall_0, camera_up_vis)
        camera_rot_out_of_wall_1 = get_camera_quat(camera_pos_out_of_wall_1, camera_lookat_out_of_wall_1, camera_up_vis)
        camera_rot_out_of_wall_2 = get_camera_quat(camera_pos_out_of_wall_2, camera_lookat_out_of_wall_2, camera_up_vis)
        camera_rot_out_of_wall_3 = get_camera_quat(camera_pos_out_of_wall_3, camera_lookat_out_of_wall_3, camera_up_vis)

        
        self.scene.camera_out_vis_0 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/camera_out_vis_0",
            update_period=0.0,
            height=image_resolution_h,
            width=image_resolution_w,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_out_of_wall_0, rot=camera_rot_out_of_wall_0, convention="opengl"),
        )

        self.scene.camera_out_vis_1 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/camera_out_vis_1",
            update_period=0.0,
            height=image_resolution_h,
            width=image_resolution_w,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_out_of_wall_1, rot=camera_rot_out_of_wall_1, convention="opengl"),
        )

        self.scene.camera_out_vis_2 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/camera_out_vis_2",
            update_period=0.0,
            height=image_resolution_h,
            width=image_resolution_w,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_out_of_wall_2, rot=camera_rot_out_of_wall_2, convention="opengl"),
        )

        self.scene.camera_out_vis_3 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/camera_out_vis_3",
            update_period=0.0,
            height=image_resolution_h,
            width=image_resolution_w,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.001, 1.0e3)
            ),
            offset=CameraCfg.OffsetCfg(pos=camera_pos_out_of_wall_3, rot=camera_rot_out_of_wall_3, convention="opengl"),
        )

        self.observations.policy.rgb_out_vis_0 = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_out_vis_0"},
        )

        self.observations.policy.rgb_out_vis_1 = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_out_vis_1"},
        )

        self.observations.policy.rgb_out_vis_2 = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_out_vis_2"},
        )

        self.observations.policy.rgb_out_vis_3 = ObsTerm(
            func=mdp.image_rgb,
            params={"camera_name": "camera_out_vis_3"},
        )

