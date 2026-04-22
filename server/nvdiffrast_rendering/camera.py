# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch

def get_intrinsic(fov, H, W):
    """
    Args:
        fov: float, the field of view in degrees
        H: int, the height of the image
        W: int, the width of the image
        near: float, the near plane distance
        far: float, the far plane distance
    """

    fov_rad = fov * np.pi / 180.0
    fx = 0.5 * H / np.tan(fov_rad / 2.0)
    fy = 0.5 * H / np.tan(fov_rad / 2.0)
    cx = W / 2.0
    cy = H / 2.0

    return fx, fy, cx, cy



def get_full_view_camera_sampling(room_center, room_scales, resolution, 
    horizontal_angle, vertical_angle, fov=35.0, mode="longest"
):
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
        mode: "longest" or "adjustable"

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
    if mode == "longest":
        if width >= length:
            test_angles = np.array([90, 270]) * np.pi / 180  # Perpendicular to width (longest)
        else:
            test_angles = np.array([0, 180]) * np.pi / 180  # Perpendicular to length (longest)
    else:
        test_angles = np.array([horizontal_angle, horizontal_angle + 180.0]) * np.pi / 180.0

    def objective(radius_candidate):
        """Objective: maximum pixel overflow across all test angles (want this <= 0)"""
        max_overflow = 0.0
        for test_angle in test_angles:
            overflow = corners_fit_in_view(radius_candidate, test_angle)
            max_overflow = max(max_overflow, overflow)
        return max_overflow
    
    # Initial guess: use the room's bounding sphere radius scaled by FOV
    initial_radius = np.sqrt(width**2 + length**2 + height**2) / (2 * np.tan(fov_vertical / 2))
    
    # Find the minimum radius where objective(radius) <= 0
    # Use binary search for efficiency
    radius_min = initial_radius * 0.5
    radius_max = initial_radius * 3.0
    
    for iteration in range(20):  # Binary search iterations
        radius_mid = (radius_min + radius_max) / 2
        overflow = objective(radius_mid)
        
        if iteration % 5 == 0 or iteration == 19:  # Print every 5th iteration and last
            # print(f"  Iter {iteration:2d}: radius={radius_mid:.2f}, max_overflow={overflow:.1f}px {'✗' if overflow > 0 else '✓'}")
            pass
        
        if overflow > 0:
            # Overflow still exists, need larger radius
            radius_min = radius_mid
        else:
            # All corners fit, try smaller radius
            radius_max = radius_mid
    
    radius = radius_max * 1.05  # Add 5% safety margin
    
    # Verify the final radius works for the current angle and show corner projections
    final_overflow = corners_fit_in_view(radius, np.radians(horizontal_angle))
    
    # Show actual corner pixel coordinates for debugging
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
        else:
            pass
    
    # Step 6: Calculate camera position for the specific horizontal angle (already computed above)
    dx = radius * np.cos(vertical_angle_rad) * np.cos(horizontal_angle_rad)
    dy = radius * np.cos(vertical_angle_rad) * np.sin(horizontal_angle_rad)
    dz = radius * np.sin(vertical_angle_rad)
    
    camera_pos = lookat_pos + np.array([dx, dy, dz])
    
    return camera_pos, lookat_pos, fov



def get_intrinsic_matrix(fx, fy, cx, cy):
    return torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]).float()

def get_camera_perspective_projection_matrix(fx, fy, cx, cy, H, W, near, far):
    projection = np.zeros((4, 4))
    projection[0, 0] = 2.0 * fx / W
    projection[1, 1] = 2.0 * fy / H
    projection[0, 2] = 2.0 * (cx / W - 0.5)
    projection[1, 2] = 2.0 * (cy / H - 0.5)
    projection[2, 2] = (far + near) / (far - near)
    projection[2, 3] = -2.0 * far * near / (far - near)
    projection[3, 2] = 1.0

    return torch.from_numpy(projection.astype(np.float32)).float()

def get_camera_orthogonal_projection_matrix(near, far, scale_x=1.0, scale_y=1.0):
    projection = np.eye(4)
    projection[2, 2] = 2.0 / (far - near)
    projection[2, 3] = -(far + near) / (far - near)

    projection[0, 0] = (1 / scale_x)
    projection[1, 1] = (1 / scale_y)

    return torch.from_numpy(projection.astype(np.float32)).float()

def build_camera_matrix(camera_pos, camera_lookat, camera_up):
    camera_z = camera_lookat - camera_pos
    camera_z = camera_z / torch.norm(camera_z)
    camera_x = torch.cross(-camera_up, camera_z)
    camera_x = camera_x / torch.norm(camera_x)
    camera_y = torch.cross(camera_z, camera_x)
    camera_y = camera_y / torch.norm(camera_y)

    camera_matrix = torch.eye(4)
    camera_matrix[:3] = torch.stack([camera_x, camera_y, camera_z, camera_pos], dim=1)
    return camera_matrix.float()

def get_mvp_matrix(camera_matrix, projection_matrix):
    return projection_matrix @ camera_matrix.inverse()


def sample_cameras_around_object(
        vertices, num_samples=30, 
        radius=1.0, fov=80.0, H=720, W=1280,
        phi=60.0):
    
    # get the bounding box of the object
    print("vertices: ", vertices.shape)
    min_x = torch.min(vertices[:, 0])
    max_x = torch.max(vertices[:, 0])
    min_y = torch.min(vertices[:, 1])
    max_y = torch.max(vertices[:, 1])
    min_z = torch.min(vertices[:, 2])
    max_z = torch.max(vertices[:, 2])
    
    # get the center of the object
    center = torch.tensor([(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2])
    
    # get the scale of the object
    scale = torch.max(torch.tensor([max_x - min_x, max_y - min_y, max_z - min_z])) * 0.5
    
    # print("scale: ", scale)
    # print("center: ", center)

    phi_rad = phi * np.pi / 180.0

    # calculate intrinsic parameters
    fx, fy, cx, cy = get_intrinsic(fov, H, W)
    near = 0.001
    far = 100.0

    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]).float().cuda()

    camera_poses = []
    camera_mvps = []
    
    projection_matrix = get_camera_perspective_projection_matrix(fx, fy, cx, cy, H, W, near, far)


    for theta in np.linspace(0, 2 * np.pi, num_samples, endpoint=False):
        camera_pos = center + radius * scale * torch.tensor([
            np.cos(theta) * np.sin(phi_rad),
            np.sin(theta) * np.sin(phi_rad),
            np.cos(phi_rad)
        ]).float()

        camera_lookat = center.float()
        camera_up = torch.tensor([0.0, 0.0, 1.0]).float()

        camera_matrix = build_camera_matrix(camera_pos, camera_lookat, camera_up)
        mvp_matrix = get_mvp_matrix(camera_matrix, projection_matrix)

        camera_poses.append(camera_matrix.cuda())
        camera_mvps.append(mvp_matrix.cuda())
        
    return camera_poses, camera_mvps, K

