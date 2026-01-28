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
import pickle
import os
import sys
import json
import argparse
import numpy as np
import imageio
from tqdm import tqdm
import bpy
import mathutils
from PIL import Image
from scipy.optimize import minimize

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FloorPlan
from tex_utils import dict_to_floor_plan
from visualize_room_bpy import (
    load_scene_meshes_into_blender,
    setup_camera_look_at,
    hide_walls_based_on_camera,
    get_or_create_collection,
    clear_blender_scene
)

RESULTS_DIR = "/home/hongchix/main/server/results"


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
    # Version 3 (longest edges only): Use angles perpendicular to longest dimension
    # If width > length, looking from sides (90°, 270°) shows longest dimension
    # If length > width, looking from ends (0°, 180°) shows longest dimension
    if width >= length:
        test_angles = np.array([90, 270]) * np.pi / 180  # Perpendicular to width (longest)
        print(f"Using longest edge mode: width ({width:.2f}) >= length ({height:.2f})")
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


def apply_random_bright_colors(objects):
    """Apply random bright colors to objects."""
    import random
    
    for obj in objects:
        if obj.type != 'MESH':
            continue
            
        # Create new material with random bright color
        mat = bpy.data.materials.new(name=f"Material_{obj.name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        # Create Principled BSDF shader
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        # Generate random bright color (high saturation, high value)
        hue = random.random()  # Random hue [0, 1]
        saturation = 0.7 + random.random() * 0.3  # Saturation [0.7, 1.0]
        value = 0.8 + random.random() * 0.2  # Value [0.8, 1.0]
        
        # Convert HSV to RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Set base color
        bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.5
        bsdf.inputs['Metallic'].default_value = 0.0
        
        # Create material output
        output = nodes.new(type='ShaderNodeOutputMaterial')
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        # Assign material to object
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        
        print(f"  Applied color RGB({r:.2f}, {g:.2f}, {b:.2f}) to '{obj.name}'")

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


def update_object_transforms(scene_objects, traced_data_all, frame_idx):
    """Update object transforms based on physics simulation data for the given frame."""
    updated_count = 0
    
    for obj in scene_objects:
        # Check if this object has physics data
        prim_path = f"/World/{obj.name}"
        
        if prim_path in traced_data_all:
            trace_data = traced_data_all[prim_path]
            position_traj = trace_data["position_traj"]
            orientation_traj = trace_data["orientation_traj"]
            
            # Make sure frame_idx is within bounds
            if frame_idx < len(position_traj):
                # Update position
                position = position_traj[frame_idx]
                obj.location = mathutils.Vector(position)
                
                # Update orientation (quaternion: w, x, y, z in Isaac -> x, y, z, w in Blender)
                # Isaac Sim uses (w, x, y, z) format, Blender uses (w, x, y, z) for mathutils.Quaternion
                quat = orientation_traj[frame_idx]
                # Convert from (w, x, y, z) to Blender's Quaternion(w, x, y, z)
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = mathutils.Quaternion((quat[0], quat[1], quat[2], quat[3]))
                
                # Debug print
                print(f"  [Frame {frame_idx}] Updated '{obj.name}':")
                print(f"    Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
                print(f"    Quaternion: ({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")
                
                updated_count += 1
    
    if updated_count > 0:
        print(f"  Total objects updated: {updated_count}/{len(scene_objects)}")
    else:
        print(f"  No objects updated (no matching physics data found)")


def render_physics_simulation(layout: FloorPlan, room_id: str, traced_data_all: dict,
                               output_path: str, resolution=512, fps=24, 
                               camera_mode="circular", angle_step=2, radius_scale=1.5):
    """
    Render physics simulation video using Blender bpy.
    
    Args:
        layout: FloorPlan object
        room_id: ID of the room to visualize
        traced_data_all: Dictionary of physics simulation traces
        output_path: Path to save the video
        resolution: Image resolution (height for 16:9 aspect ratio)
        fps: Frames per second for output video
        camera_mode: "circular" for circular camera path, "fixed" for static camera
        angle_step: Degrees between each frame for circular mode
        radius_scale: Scale factor for camera distance from room center
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
    
    # Determine longest edge for camera orientation
    width = room.dimensions.width
    length = room.dimensions.length
    longest_is_width = width > length
    
    # Load scene meshes into Blender (do this once)
    print("Loading scene meshes into Blender...")
    mesh_info_dict = load_scene_meshes_into_blender(room, layout)
    
    # Get scene objects collection
    scene_collection = bpy.data.collections.get("scene_objects")
    scene_objects = list(scene_collection.objects) if scene_collection else []
    
    # Apply random bright colors to all objects
    print("Applying random bright colors to objects...")
    all_objects = list(bpy.data.objects)
    apply_random_bright_colors(all_objects)
    
    # Determine number of frames from traced data
    num_frames = 0
    print("\n=== Physics Data Available ===")
    for prim_path, trace_data in traced_data_all.items():
        if "position_traj" in trace_data:
            traj_len = len(trace_data["position_traj"])
            num_frames = max(num_frames, traj_len)
            print(f"  {prim_path}: {traj_len} frames")
    
    if num_frames == 0:
        raise ValueError("No physics trajectory data found")
    
    print(f"\nTotal simulation frames: {num_frames}")
    print(f"Scene objects loaded: {len(scene_objects)}")
    print("Scene object names:", [obj.name for obj in scene_objects])
    
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
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    
    # Set world background to white with increased brightness
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    world_nodes = scene.world.node_tree.nodes
    world_nodes.clear()
    world_bg = world_nodes.new(type='ShaderNodeBackground')
    world_bg.inputs[0].default_value = (1, 1, 1, 1)  # White background
    world_bg.inputs[1].default_value = 2.5  # Increased brightness
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    scene.world.node_tree.links.new(world_output.inputs['Surface'], world_bg.outputs['Background'])
    
    # Add area light for better illumination
    bpy.ops.object.light_add(type='AREA', location=(room_center[0], room_center[1], room_position[2] + room.dimensions.height))
    light = bpy.context.active_object
    light.name = "AreaLight"
    light.data.energy = 500
    light.data.size = max(room.dimensions.width, room.dimensions.length) * 0.8
    
    # Set render resolution to 16:9 aspect ratio
    scene.render.resolution_y = resolution
    scene.render.resolution_x = int(resolution * 16 / 9)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"  # Enable alpha channel
    scene.render.film_transparent = True  # Enable transparent background
    
    # Create directory for frames (persistent, not temporary)
    frames_dir = os.path.join(os.path.dirname(output_path), f"{room_id}_physics_frames")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Saving frames to: {frames_dir}")
    
    # Generate frames
    frames = []
    
    # debug
    # num_frames = 10
    print(f"Rendering {num_frames} frames with physics simulation...")
    for frame_idx in tqdm(range(num_frames), desc="Rendering frames", unit="frame"):
        print(f"\n=== Rendering Frame {frame_idx}/{num_frames-1} ===")
        
        # Update object transforms based on physics simulation
        update_object_transforms(scene_objects, traced_data_all, frame_idx)
        
        # Update camera position
        if camera_mode == "circular":
            # Calculate horizontal angle based on frame index
            horizontal_angle = frame_idx * angle_step  # in degrees
        else:  # fixed camera
            # Position camera to look along the longest edge
            if longest_is_width:
                # Width is longer, camera looks from the side (90 degrees)
                horizontal_angle = 90.0 + 180.0
            else:
                # Length is longer, camera looks from the end (0 degrees)
                horizontal_angle = 0.0 + 180.0
        
        # Set vertical angle (elevation)
        vertical_angle = 45.0  # degrees
        
        # Use the full view camera sampling function
        camera_pos, lookat_pos, fov = get_full_view_camera_sampling(
            room_center=room_center,
            room_scales=(room.dimensions.width, room.dimensions.length, room.dimensions.height),
            resolution=(scene.render.resolution_y, scene.render.resolution_x),
            horizontal_angle=horizontal_angle,
            vertical_angle=vertical_angle,
            fov=35.0  # Use 35 degree FOV to match visualize_room_bpy_frames_longest.py
        )
        
        # Convert to numpy arrays for compatibility
        camera_pos = np.array(camera_pos)
        lookat_pos = np.array(lookat_pos)
        
        # Update camera position, orientation, and FOV
        setup_camera_look_at(camera, camera_pos, lookat_pos)
        # Set vertical FOV (since sensor_fit is 'VERTICAL', angle maps to vertical FOV)
        camera.data.angle = np.radians(fov)
        
        # Hide walls based on camera view direction
        hide_walls_based_on_camera(room, camera_pos, lookat_pos, scene_objects)
        
        # Render frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
        scene.render.filepath = frame_path
        bpy.ops.render.render(write_still=True)
        
        # Load rendered frame
        frame_img = np.array(Image.open(frame_path))
        frames.append(frame_img)
    
    # Save video
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"\nVideo saved to: {os.path.abspath(output_path)}")
    print(f"Frames saved to: {os.path.abspath(frames_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize physics simulation with rendering")
    parser.add_argument("layout_id", type=str, help="Layout ID or room ID to visualize")
    parser.add_argument("--resolution", type=int, default=1024, help="Video resolution (default: 1024)")
    parser.add_argument("--fps", type=int, default=24, help="Video FPS (default: 24)")
    parser.add_argument("--camera-mode", type=str, default="fixed", 
                        choices=["circular", "fixed"], help="Camera mode (default: fixed)")
    parser.add_argument("--angle-step", type=int, default=2, 
                        help="Degrees between frames for circular camera (default: 2)")
    
    args = parser.parse_args()
    
    layout_id = args.layout_id
    
    # Determine if it's a room_id or layout_id
    if layout_id.startswith("room_"):
        print(f"Searching for room {layout_id}...")
        layout_path, json_path = find_layout_dir(layout_id)
        room_id = layout_id
        layout_id = os.path.basename(layout_path)
    else:
        layout_path = os.path.join(RESULTS_DIR, layout_id)
        json_path = os.path.join(layout_path, f"{layout_id}.json")
        if not os.path.exists(json_path):
            raise ValueError(f"Layout JSON not found: {json_path}")
        
        with open(json_path, 'r') as f:
            layout_data = json.load(f)
        room_id = layout_data["rooms"][0]["id"]
    
    print(f"Layout ID: {layout_id}, Room ID: {room_id}")
    print(f"Layout path: {layout_path}")
    
    # Load layout
    print("Loading layout...")
    with open(json_path, 'r') as f:
        layout_data = json.load(f)
    layout = dict_to_floor_plan(layout_data)
    
    # Load physics simulation data
    sim_result_path = os.path.join(layout_path, f"{room_id}_simulation_result.pkl")
    if not os.path.exists(sim_result_path):
        raise ValueError(f"Simulation result not found: {sim_result_path}")
    
    print("Loading physics simulation data...")
    with open(sim_result_path, "rb") as f:
        traced_data_all = pickle.load(f)
    
    print(f"Loaded physics data for {len(traced_data_all)} objects")
    
    # Create output path
    output_path = os.path.join(layout_path, f"{room_id}_physics_render.mp4")
    
    # Render video
    print(f"Rendering physics simulation for room {room_id}...")
    render_physics_simulation(
        layout, 
        room_id,
        traced_data_all,
        output_path,
        resolution=args.resolution,
        fps=args.fps,
        camera_mode=args.camera_mode,
        angle_step=args.angle_step
    )
    
    print("Done!")
