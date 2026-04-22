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
import argparse
import os
import sys
import json
import numpy as np
import imageio
from pathlib import Path
import copy
from tqdm import tqdm
import bpy
import mathutils
from scipy.optimize import minimize

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FloorPlan
from tex_utils import dict_to_floor_plan, export_single_room_layout_to_mesh_dict_list
from constants import RESULTS_DIR
from PIL import Image


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


def get_camera_view_direction(camera_pos, lookat_pos):
    """Calculate normalized view direction from camera to lookat position."""
    direction = lookat_pos - camera_pos
    direction = direction / np.linalg.norm(direction)
    return direction


def get_wall_normal(wall, room):
    """Calculate the outward normal vector of a wall."""
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    
    # Wall direction vector (along the wall)
    wall_dir = end - start
    wall_dir = wall_dir / np.linalg.norm(wall_dir)
    
    # Get perpendicular vector (normal to wall, pointing outward)
    # Assuming walls are vertical (z component is 0 for normal in xy plane)
    # Cross product with up vector to get outward normal
    up = np.array([0, 0, 1])
    normal = np.cross(wall_dir, up)
    
    # Determine if normal points inward or outward
    # Check if normal points away from room center
    room_center = np.array([
        room.position.x + room.dimensions.width / 2,
        room.position.y + room.dimensions.length / 2,
        room.position.z
    ])
    wall_center = (start + end) / 2
    to_center = room_center - wall_center
    
    # If normal points toward center, flip it
    if np.dot(normal[:2], to_center[:2]) > 0:
        normal = -normal
    
    return normal


def should_exclude_wall(wall, room, camera_pos, lookat_pos):
    """Determine if a wall should be excluded based on camera view direction."""
    # Get wall normal (outward facing)
    wall_normal = get_wall_normal(wall, room)
    
    # Get camera view direction
    view_dir = get_camera_view_direction(camera_pos, lookat_pos)
    
    # If view direction and wall normal are opposing (dot product < 0),
    # the camera is looking at the wall from outside, so we should exclude it
    dot_product = np.dot(view_dir[:2], wall_normal[:2])
    
    # Exclude walls that the camera is facing (negative dot product means facing the wall)
    return dot_product < -0.3  # threshold to handle corner cases


def filter_mesh_dict_by_walls(mesh_info_dict, walls_to_exclude):
    """Remove specific walls from the mesh_info_dict."""
    filtered_dict = {}
    excluded_wall_ids = {wall.id for wall in walls_to_exclude}
    
    for mesh_id, mesh_info in mesh_info_dict.items():
        # Check if this is a wall mesh and if it should be excluded
        is_excluded_wall = any(wall_id in mesh_id for wall_id in excluded_wall_ids)
        
        if not is_excluded_wall:
            filtered_dict[mesh_id] = mesh_info
    
    return filtered_dict


def get_or_create_collection(collection_name):
    """Get or create a collection"""
    if collection_name in bpy.data.collections:
        return bpy.data.collections[collection_name]
    
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def clear_blender_scene():
    """Clear all objects from Blender scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear all collections except the default Scene Collection
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def load_scene_meshes_into_blender(room, layout):
    """Load room layout meshes from files into Blender"""
    
    # Clear all existing Blender assets before loading new ones
    clear_blender_scene()
    
    # Get mesh info dict
    mesh_info_dict = export_single_room_layout_to_mesh_dict_list(layout, room.id)
    
    # Create collection for scene objects
    scene_collection = get_or_create_collection("scene_objects")
    
    # Import each mesh
    for mesh_id, mesh_info in mesh_info_dict.items():
        # if mesh_id.startswith("door") or mesh_id.startswith("window"):
        #     continue
        
        trimesh_mesh = mesh_info["mesh"]
        
        # Convert trimesh to Blender mesh
        vertices = trimesh_mesh.vertices
        faces = trimesh_mesh.faces
        
        # Create new mesh data
        mesh_data = bpy.data.meshes.new(name=f"mesh_{mesh_id}")
        mesh_data.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh_data.update()
        
        # Create object from mesh
        obj = bpy.data.objects.new(mesh_id, mesh_data)
        scene_collection.objects.link(obj)
        
        # Load and apply texture if available
        texture_info = mesh_info.get("texture")
        if texture_info and texture_info.get("texture_map_path"):
            texture_path = texture_info["texture_map_path"]
            if os.path.exists(texture_path):
                # Create material with texture
                mat = bpy.data.materials.new(name=f"mat_{mesh_id}")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                nodes.clear()
                
                # Create shader nodes
                bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                tex_image = nodes.new(type='ShaderNodeTexImage')
                output = nodes.new(type='ShaderNodeOutputMaterial')
                
                # Load texture image
                tex_image.image = bpy.data.images.load(texture_path)
                tex_image.image.colorspace_settings.name = 'sRGB'  # Ensure correct color space
                
                # Configure BSDF for clean, slightly glossy appearance like reference images
                bsdf.inputs['Roughness'].default_value = 0.6  # Slight gloss
                bsdf.inputs['Specular'].default_value = 0.3  # Subtle specularity
                bsdf.inputs['Sheen Tint'].default_value = 0.0  # No sheen
                
                # Connect nodes
                mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
                mat.node_tree.links.new(output.inputs['Surface'], bsdf.outputs['BSDF'])
                
                # Apply material to object
                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)
                
                # Set UV coordinates if available
                vts = texture_info.get("vts")
                fts = texture_info.get("fts")
                if vts is not None and fts is not None:
                    # Create UV layer
                    uv_layer = obj.data.uv_layers.new(name="UVMap")
                    for face_idx, face in enumerate(fts):
                        for vert_idx in range(len(face)):
                            loop_idx = face_idx * len(face) + vert_idx
                            if loop_idx < len(uv_layer.data):
                                uv = vts[face[vert_idx]]
                                uv_layer.data[loop_idx].uv = (uv[0], uv[1])
    
    print(f"Loaded {len(mesh_info_dict)} meshes into Blender scene")
    return mesh_info_dict


def setup_camera_look_at(camera, camera_pos, lookat_pos):
    """Position camera and make it look at target position"""
    # Set camera position
    camera.location = camera_pos
    
    # Calculate direction vector
    direction = mathutils.Vector(lookat_pos) - mathutils.Vector(camera_pos)
    
    # Point camera to look at target
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def hide_walls_based_on_camera(room, camera_pos, lookat_pos, scene_objects):
    """Hide/show walls based on camera view direction"""
    # Determine which walls to exclude
    walls_to_exclude = [
        wall for wall in room.walls 
        if should_exclude_wall(wall, room, camera_pos, lookat_pos)
    ]
    
    excluded_wall_ids = {wall.id for wall in walls_to_exclude}
    
    # Hide/show wall objects
    for obj in scene_objects:
        is_wall = any(wall_id in obj.name for wall_id in excluded_wall_ids)
        obj.hide_render = is_wall
        obj.hide_viewport = is_wall


def render_room_circular_video(layout: FloorPlan, room_id: str, output_path: str, 
                                resolution=512, angle_step=10, radius_scale=1.5, fps=24):
    """
    Render views from the two longest edges of a room using Blender bpy.
    
    Automatically determines the longest dimension (width or length) and renders
    two views perpendicular to that dimension for optimal room coverage.
    
    Args:
        layout: FloorPlan object
        room_id: ID of the room to visualize
        output_path: Path to save the video
        resolution: Image resolution (square)
        angle_step: Degrees between each frame (default 10, unused in this version)
        radius_scale: Scale factor for camera distance from room center (unused in this version)
        fps: Frames per second for output video
    """
    # Get the room
    all_rooms = layout.rooms
    room = next((r for r in all_rooms if r.id == room_id), None)
    if room is None:
        raise ValueError(f"Room {room_id} not found in layout")
    
    # Calculate room center and dimensions
    room_position = np.array([room.position.x, room.position.y, room.position.z])
    room_center = room_position + np.array([
        room.dimensions.width / 2,
        room.dimensions.length / 2,
        room.dimensions.height / 2
    ])
    room_scales = (room.dimensions.width, room.dimensions.length, room.dimensions.height)
    
    # Load scene meshes into Blender (do this once)
    print("Loading scene meshes into Blender...")
    mesh_info_dict = load_scene_meshes_into_blender(room, layout)

    # Get scene objects collection
    scene_collection = bpy.data.collections.get("scene_objects")
    scene_objects = list(scene_collection.objects) if scene_collection else []
    
    # Setup Blender scene
    scene = bpy.context.scene
    
    # Create or get camera
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
    else:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    
    scene.camera = camera
    
    # Set camera to perspective (FOV will be set per-frame)
    camera.data.type = 'PERSP'
    # Set sensor fit to vertical so we control vertical FOV explicitly
    camera.data.sensor_fit = 'VERTICAL'
    
    # Set up render engine and settings
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64  # Higher samples for smoother, softer lighting
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # Better denoising for soft lighting
    

    # Set world background to light color for ambient bounce lighting
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    world_nodes = scene.world.node_tree.nodes
    world_nodes.clear()
    world_bg = world_nodes.new(type='ShaderNodeBackground')
    world_bg.inputs[0].default_value = (1, 1, 1, 1)  # White background
    world_bg.inputs[1].default_value = 1.0
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    scene.world.node_tree.links.new(world_output.inputs['Surface'], world_bg.outputs['Background'])
    
    # Add warm ambient light
    if "WarmAmbientLight" in bpy.data.objects:
        light = bpy.data.objects["WarmAmbientLight"]
    else:
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        light = bpy.context.active_object
        light.name = "WarmAmbientLight"
    
    # Configure warm color and intensity
    light.data.energy = 10.0  # Moderate intensity
    light.data.color = (1.0, 0.9, 0.7)  # Warm yellowish color
    light.data.angle = np.radians(10)  # Soft shadows
    
    # Set render resolution (16:9 aspect ratio)
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution * 9 // 16
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"  # Enable alpha channel
    scene.render.film_transparent = True  # Enable transparent background
    
    # Adjust film settings for better light capture
    # scene.cycles.film_exposure = 1.0 # Standard exposure
    
    # Enable compositor for proper alpha channel handling
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_combined = True
    
    # Set color management for natural, bright look matching reference images
    # scene.view_settings.view_transform = 'Standard'  # More natural than Filmic
    # scene.view_settings.look = 'None'  # No additional color grading
    # scene.view_settings.exposure = 1.0  # Brighter to match reference images
    # scene.view_settings.gamma = 1.0
    
    # Adjust color management for more vibrant colors
    # scene.display_settings.display_device = 'sRGB'
    # scene.sequencer_colorspace_settings.name = 'sRGB'
    
    # Create directory for frames (alongside the mp4)
    output_dir = os.path.dirname(output_path)
    output_basename = os.path.splitext(os.path.basename(output_path))[0]
    frames_dir = os.path.join(output_dir, f"{output_basename}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Determine which angles to render based on longest dimension
    width, length, height = room_scales
    if width >= length:
        # Width is longer, render from sides (perpendicular to width)
        render_angles = [90, 270]
        print(f"Room width ({width:.2f}) >= length ({length:.2f})")
        print(f"Rendering 2 views from longest edges: 90° and 270°")
    else:
        # Length is longer, render from ends (perpendicular to length)
        render_angles = [0, 180]
        print(f"Room length ({length:.2f}) > width ({width:.2f})")
        print(f"Rendering 2 views from longest edges: 0° and 180°")
    
    # Generate frames
    frames = []
    num_frames = len(render_angles)
    
    print(f"\nRendering {num_frames} frames...")
    for frame_idx in tqdm(range(num_frames), desc="Rendering frames", unit="frame"):
        horizontal_angle = render_angles[frame_idx]  # in degrees
        vertical_angle = 45.0  # elevated view angle in degrees
        
        # Use the full view camera sampling function
        camera_pos, lookat_pos, fov = get_full_view_camera_sampling(
            room_center=room_center,
            room_scales=room_scales,
            resolution=(scene.render.resolution_y, scene.render.resolution_x),
            horizontal_angle=horizontal_angle,
            vertical_angle=vertical_angle
        )
        
        # Convert to numpy arrays for compatibility
        camera_pos = np.array(camera_pos)
        lookat_pos = np.array(lookat_pos)
        
        # Update camera position, orientation, and FOV
        setup_camera_look_at(camera, camera_pos, lookat_pos)
        # Set vertical FOV (since sensor_fit is 'VERTICAL', angle maps to vertical FOV)
        # Can also use camera.data.angle_y for explicit vertical FOV
        camera.data.angle = np.radians(fov)
        
        # Hide walls based on camera view direction
        hide_walls_based_on_camera(room, camera_pos, lookat_pos, scene_objects)
        
        # Render frame and save to frames directory
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
        scene.render.filepath = frame_path
        bpy.ops.render.render(write_still=True)
        
        # Load rendered frame for video
        frame_img = np.array(Image.open(frame_path))
        frames.append(frame_img)
    
    print(f"Frames saved to: {os.path.abspath(frames_dir)}")
    
    # Save video with the 2 frames (repeat each frame 3 times for visibility)
    frames_repeated = [frame for frame in frames for _ in range(3)]
    imageio.mimsave(output_path, frames_repeated, fps=fps)
    print(f"Video saved to: {os.path.abspath(output_path)}")
    print(f"Total frames in video: {len(frames_repeated)} (2 views × 3 repetitions)")


def find_layout_dir(room_id, results_dir=RESULTS_DIR):
    """Search for layout directory containing the room_id."""
    if not os.path.exists(results_dir):
        raise ValueError(f"Results directory not found: {results_dir}")
    
    # Search through layout directories
    for layout_dir in os.listdir(results_dir):
        layout_path = os.path.join(results_dir, layout_dir)
        if not os.path.isdir(layout_path):
            continue
        
        # Check for layout JSON file
        json_path = os.path.join(layout_path, f"{layout_dir}.json")
        if not os.path.exists(json_path):
            continue
        
        # Load and check if it contains the room_id
        try:
            with open(json_path, 'r') as f:
                layout_data = json.load(f)
            
            # Check if room_id exists in this layout
            for room in layout_data.get("rooms", []):
                if room.get("id") == room_id:
                    return layout_path, json_path
        except:
            continue
    
    raise ValueError(f"No layout found containing room_id: {room_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a room from its two longest edges")
    parser.add_argument("--resolution", type=int, default=1920, help="Video width resolution (default: 1920 for 16:9)")
    parser.add_argument("--angle-step", type=int, default=45, help="Degrees between frames (default: 45, unused)")
    parser.add_argument("--fps", type=int, default=24, help="Video FPS (default: 24)")
    
    args = parser.parse_args()
    
    try:
        # Find the layout directory containing this room
        # if args.room_id.startswith("room_"):
        #     print(f"Searching for room {args.room_id}...")
        #     layout_dir, json_path = find_layout_dir(args.room_id)
        #     print(f"Found layout at: {layout_dir}")
        # else:
        #     layout_dir = os.path.join(RESULTS_DIR, args.room_id)
        #     json_path = os.path.join(layout_dir, f"{args.room_id}.json")

        layout_dir = "/home/hongchix/main/server/results/layout_7821b099/"
        json_path = "/home/hongchix/main/server/results/layout_7821b099/layout_7821b099_stand_pillow.json"
        
        # Load the layout
        print("Loading layout...")
        with open(json_path, 'r') as f:
            layout_data = json.load(f)
        layout = dict_to_floor_plan(layout_data)
        args.room_id = "room_2adee539"
        
        # Create output path
        layout_id = os.path.basename(layout_dir)
        output_path = os.path.join(layout_dir, f"{args.room_id}_longest_edges_video_stand_pillow.mp4")
        
        # Render video
        print(f"Rendering longest edge views for room {args.room_id}...")
        render_room_circular_video(
            layout, 
            args.room_id, 
            output_path,
            resolution=args.resolution,
            angle_step=args.angle_step,
            fps=args.fps
        )
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)