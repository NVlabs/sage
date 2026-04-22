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

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FloorPlan
from tex_utils import dict_to_floor_plan, export_single_room_layout_to_mesh_dict_list
from constants import RESULTS_DIR
from PIL import Image


def get_layout_bounds(layout: FloorPlan):
    """
    Calculate the min/max x/y coordinates across all rooms in the layout.
    
    Args:
        layout: FloorPlan object
        
    Returns:
        Dictionary with min_x, max_x, min_y, max_y, min_z, max_z
    """
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    min_z = float('inf')
    max_z = float('-inf')
    
    for room in layout.rooms:
        # Get room bounds
        room_min_x = room.position.x
        room_max_x = room.position.x + room.dimensions.width
        room_min_y = room.position.y
        room_max_y = room.position.y + room.dimensions.length
        room_min_z = room.position.z
        room_max_z = room.position.z + room.dimensions.height
        
        # Update global bounds
        min_x = min(min_x, room_min_x)
        max_x = max(max_x, room_max_x)
        min_y = min(min_y, room_min_y)
        max_y = max(max_y, room_max_y)
        min_z = min(min_z, room_min_z)
        max_z = max(max_z, room_max_z)
    
    return {
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'min_z': min_z,
        'max_z': max_z,
        'center_x': (min_x + max_x) / 2,
        'center_y': (min_y + max_y) / 2,
        'center_z': (min_z + max_z) / 2,
        'width': max_x - min_x,
        'length': max_y - min_y,
        'height': max_z - min_z
    }


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


def load_layout_meshes_into_blender(layout: FloorPlan):
    """Load all room layout meshes from files into Blender"""
    
    # Clear all existing Blender assets before loading new ones
    clear_blender_scene()
    
    # Create collection for scene objects
    scene_collection = get_or_create_collection("scene_objects")
    
    total_meshes = 0
    
    # Import meshes for each room
    for room in layout.rooms:
        # Get mesh info dict for this room
        mesh_info_dict = export_single_room_layout_to_mesh_dict_list(layout, room.id)
        
        # Import each mesh
        for mesh_id, mesh_info in mesh_info_dict.items():
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
                    
                    # Configure BSDF for clean, slightly glossy appearance
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
            
            total_meshes += 1
    
    print(f"Loaded {total_meshes} meshes from {len(layout.rooms)} rooms into Blender scene")
    return total_meshes


def setup_camera_look_at(camera, camera_pos, lookat_pos):
    """Position camera and make it look at target position"""
    # Set camera position
    camera.location = camera_pos
    
    # Calculate direction vector
    direction = mathutils.Vector(lookat_pos) - mathutils.Vector(camera_pos)
    
    # Point camera to look at target
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def render_layout_circular_video(layout: FloorPlan, output_path: str, 
                                   resolution=1920, angle_step=10, radius_scale=1.0, fps=24):
    """
    Render a circular video around an entire layout using Blender bpy.
    
    Args:
        layout: FloorPlan object
        output_path: Path to save the video
        resolution: Image width resolution (default 1920 for 16:9)
        angle_step: Degrees between each frame (default 10)
        radius_scale: Scale factor for camera distance from layout center
        fps: Frames per second for output video
    """
    # Calculate layout bounds
    print("Calculating layout bounds...")
    bounds = get_layout_bounds(layout)
    
    layout_center = np.array([bounds['center_x'], bounds['center_y'], bounds['center_z']])
    layout_width = bounds['width']
    layout_length = bounds['length']
    layout_height = bounds['height']
    
    print(f"Layout bounds: width={layout_width:.2f}, length={layout_length:.2f}, height={layout_height:.2f}")
    print(f"Layout center: ({bounds['center_x']:.2f}, {bounds['center_y']:.2f}, {bounds['center_z']:.2f})")
    
    # Calculate camera trajectory radius (ensure whole layout is visible)
    radius = max(layout_width, layout_length) * radius_scale
    # Camera lifted higher above the layout for elevated perspective
    camera_height = layout_height * (1.0 + max(layout_width, layout_length) * 0.8)
    
    # Load all room meshes into Blender (do this once)
    print("Loading layout meshes into Blender...")
    load_layout_meshes_into_blender(layout)
    
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
    
    # Set camera to perspective with narrower FOV
    camera.data.type = 'PERSP'
    camera.data.angle = np.radians(40)  # 60 degree FOV (narrower for more zoom)
    
    # Set up render engine and settings
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 16  # Higher samples for smoother, softer lighting
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
    
    # Enable compositor for proper alpha channel handling
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_combined = True
    
    # Create directory for frames (alongside the mp4)
    output_dir = os.path.dirname(output_path)
    output_basename = os.path.splitext(os.path.basename(output_path))[0]
    frames_dir = os.path.join(output_dir, f"{output_basename}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Generate frames
    frames = []
    num_frames = 360 // angle_step
    
    print(f"Rendering {num_frames} frames...")
    for frame_idx in tqdm(range(num_frames), desc="Rendering frames", unit="frame"):
        angle = np.radians(frame_idx * angle_step)
        
        # Calculate camera position on circular trajectory
        camera_x = layout_center[0] + radius * np.cos(angle)
        camera_y = layout_center[1] + radius * np.sin(angle)
        camera_z = bounds['min_z'] + camera_height
        camera_pos = np.array([camera_x, camera_y, camera_z])
        
        # Camera looks down at layout center from above (looking at floor level)
        lookat_pos = np.array([
            layout_center[0] - radius * 0.05 * np.cos(angle), 
            layout_center[1] - radius * 0.05 * np.sin(angle), 
            bounds['min_z']
        ])
        
        # Update camera position and orientation
        setup_camera_look_at(camera, camera_pos, lookat_pos)
        
        # Render frame and save to frames directory
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
        scene.render.filepath = frame_path
        bpy.ops.render.render(write_still=True)
        
        # Load rendered frame for video
        frame_img = np.array(Image.open(frame_path))
        
        # Set transparent pixels (alpha=0) to white RGB
        if frame_img.shape[-1] == 4:  # Check if image has alpha channel
            alpha_mask = frame_img[:, :, 3] == 0
            frame_img[alpha_mask, 0:3] = 255  # Set RGB to white where alpha is 0
        
        frames.append(frame_img)
    
    print(f"Frames saved to: {os.path.abspath(frames_dir)}")
    
    # Save video (repeat frames 3 times for smoother loop)
    frames_repeated = frames
    imageio.mimsave(output_path, frames_repeated, fps=fps)
    print(f"Video saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize entire layout with a circular video")
    parser.add_argument("layout_id", type=str, help="Layout ID to visualize")
    parser.add_argument("--resolution", type=int, default=1920, help="Video width resolution (default: 1920 for 16:9)")
    parser.add_argument("--angle-step", type=int, default=2, help="Degrees between frames (default: 45)")
    parser.add_argument("--fps", type=int, default=24, help="Video FPS (default: 24)")
    
    args = parser.parse_args()
    
    try:
        # Construct layout directory path
        layout_dir = os.path.join(RESULTS_DIR, args.layout_id)
        json_path = os.path.join(layout_dir, f"{args.layout_id}.json")
        
        if not os.path.exists(json_path):
            raise ValueError(f"Layout file not found: {json_path}")
        
        # Load the layout
        print(f"Loading layout from {json_path}...")
        with open(json_path, 'r') as f:
            layout_data = json.load(f)
        layout = dict_to_floor_plan(layout_data)
        
        print(f"Loaded layout with {len(layout.rooms)} rooms")
        
        # Create output path
        output_path = os.path.join(layout_dir, f"{args.layout_id}_circular_video.mp4")
        
        # Render video
        print(f"Rendering circular video for layout {args.layout_id}...")
        render_layout_circular_video(
            layout, 
            output_path,
            resolution=args.resolution,
            angle_step=args.angle_step,
            fps=args.fps
        )
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

