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
    
    # Clear default objects
    if "Cube" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)
    
    # Setup lighting if not exists
    if "Scene_Light" not in bpy.data.objects:
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        light = bpy.context.active_object
        light.name = "Scene_Light"
        light.data.energy = 2.0
    
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
    Render a circular video around a room using Blender bpy.
    
    Args:
        layout: FloorPlan object
        room_id: ID of the room to visualize
        output_path: Path to save the video
        resolution: Image resolution (square)
        angle_step: Degrees between each frame (default 10)
        radius_scale: Scale factor for camera distance from room center
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
    
    # Calculate camera trajectory radius
    radius = max(room.dimensions.width, room.dimensions.length) * radius_scale
    camera_height = room.dimensions.height * 1.5  # Camera at 150% of room height
    
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
    
    # Set camera to perspective with FOV
    camera.data.type = 'PERSP'
    camera.data.angle = np.radians(60)  # 60 degree FOV
    
    # Set up render engine and settings
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    
    # Set world background to white
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
    
    # Set render resolution
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    
    # Create temp directory for frames
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate frames
    frames = []
    num_frames = 360 // angle_step
    
    print(f"Rendering {num_frames} frames...")
    for frame_idx in tqdm(range(num_frames), desc="Rendering frames", unit="frame"):
        angle = np.radians(frame_idx * angle_step)
        
        # Calculate camera position on circle
        camera_x = room_center[0] + radius * np.cos(angle)
        camera_y = room_center[1] + radius * np.sin(angle)
        camera_z = room_position[2] + camera_height
        camera_pos = np.array([camera_x, camera_y, camera_z])
        
        # Camera looks at room center
        lookat_pos = room_center.copy()
        
        # Update camera position and orientation
        setup_camera_look_at(camera, camera_pos, lookat_pos)
        
        # Hide walls based on camera view direction
        hide_walls_based_on_camera(room, camera_pos, lookat_pos, scene_objects)
        
        # Render frame
        frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
        scene.render.filepath = frame_path
        bpy.ops.render.render(write_still=True)
        
        # Load rendered frame
        frame_img = np.array(Image.open(frame_path))
        frames.append(frame_img)
        
        # Clean up frame file
        os.remove(frame_path)
    
    # Clean up temp directory
    os.rmdir(temp_dir)
    
    # Save video (repeat frames 3 times for smoother loop)
    frames_repeated = [frame for _ in range(3) for frame in frames]
    imageio.mimsave(output_path, frames_repeated, fps=fps)
    print(f"Video saved to: {os.path.abspath(output_path)}")


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
    parser = argparse.ArgumentParser(description="Visualize a room with a circular video")
    parser.add_argument("room_id", type=str, help="Room ID to visualize")
    parser.add_argument("--resolution", type=int, default=1024, help="Video resolution (default: 1024)")
    parser.add_argument("--angle-step", type=int, default=2, help="Degrees between frames (default: 2)")
    parser.add_argument("--fps", type=int, default=24, help="Video FPS (default: 24)")
    
    args = parser.parse_args()
    
    try:
        # Find the layout directory containing this room
        print(f"Searching for room {args.room_id}...")
        layout_dir, json_path = find_layout_dir(args.room_id)
        print(f"Found layout at: {layout_dir}")
        
        # Load the layout
        print("Loading layout...")
        with open(json_path, 'r') as f:
            layout_data = json.load(f)
        layout = dict_to_floor_plan(layout_data)
        
        # Create output path
        layout_id = os.path.basename(layout_dir)
        output_path = os.path.join(layout_dir, f"{args.room_id}_circular_video.mp4")
        
        # Render video
        print(f"Rendering circular video for room {args.room_id}...")
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