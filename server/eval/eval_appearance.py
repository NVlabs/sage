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
"""
Room Appearance Evaluation using Blender Python (bpy) Rendering

This script evaluates room layouts by:
1. Loading a room layout from JSON
2. Rendering the scene using Blender's native bpy rendering (two-pass approach):
   - First pass: Render the actual 3D scene
   - Second pass: Render transparent overlay with 3D annotations (bboxes, arrows, coordinates)
   - Merge: Composite the two renders into final annotated image
3. Sending the annotated image to GPT for evaluation across multiple criteria

The annotation approach follows the reference implementation from infinigen_examples,
creating 3D Blender objects for bounding boxes, direction arrows, and coordinate grids
that are rendered separately and composited with the base scene.
"""

import asyncio
import json
import sys
import os
import numpy as np
from PIL import Image
import base64
import re
import time
import bpy
import mathutils

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from dataclasses import asdict
from models import Object, Point3D, Euler, Dimensions
from layout import (
    get_layout_from_json,
    add_one_object_with_condition_in_room,
    get_current_layout,
    generate_room_layout,
    place_objects_in_room,
    room_physics_critic
)
from isaacsim.isaac_mcp.server import (
    get_isaac_connection,
    create_room_layout_scene,
    simulate_the_scene,
    create_robot,
    move_robot_to_target,
    create_physics_scene,
    get_room_layout_scene_usd,
    create_single_room_layout_scene,
    get_room_layout_scene_usd_separate_from_layout
)
from tex_utils import export_layout_to_mesh_dict_list
from glb_utils import (
    create_glb_scene,
    add_textured_mesh_to_glb_scene,
    save_glb_scene
)
from objects.object_on_top_placement import (
    get_random_placements_on_target_object, 
    filter_placements_by_physics_critic,
)
from utils import dict_to_floor_plan
from constants import RESULTS_DIR
import argparse
from key import setup_oai_client
from openai import OpenAI

def dict2str(d, indent=0):
    """
    Convert a dictionary into a formatted string.

    Parameters:
    - d: dict, the dictionary to convert.
    - indent: int, the current indentation level (used for nested structures).

    Returns:
    - str: The string representation of the dictionary.
    """
    if not isinstance(d, dict):
        raise ValueError("Input must be a dictionary")

    result = []
    indent_str = " " * (indent * 4)  # Indentation for nested levels

    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            result.append(
                f"{indent_str}{key}: {{\n{dict2str(value, indent + 1)}\n{indent_str}}}"
            )
        elif isinstance(value, list):
            # Handle lists
            # list_str = ", ".join(
            #     dict2str(item, indent + 1) if isinstance(item, dict) else str(item)
            #     for item in value
            # )
            list_str = ", ".join(
                dict2str(item, indent + 1)
                if isinstance(item, dict)
                else f"{item:.2f}"
                if isinstance(item, float)
                else str(item)
                for item in value
            )
            result.append(f"{indent_str}{key}: [{list_str}]")
        else:
            # Handle other types
            result.append(f"{indent_str}{key}: {repr(value)}")

    return "{" + ",\n".join(result) + "}"

def get_descendants_description(target_object, all_objects) -> str:
    """
    Find all descendants of target_object in all_objects and return a description.
    Descendants are objects that are placed on target_object directly or indirectly.
    
    Returns a string with the count and object IDs of descendants.
    """
    descendants = []
    
    # Find all descendants by checking if each object's placement chain leads to target_object
    for obj in all_objects:
        if obj.id == target_object.id:
            # Skip the target object itself
            continue
        
        # Trace up the placement hierarchy to see if this object depends on target_object
        current_obj = obj
        while True:
            # If we've reached the base placement (floor or wall), this object is not a descendant
            if current_obj.place_id == "floor" or current_obj.place_id == "wall":
                break
            
            # If the current object is placed on target_object, it's a descendant
            if current_obj.place_id == target_object.id:
                descendants.append(obj.id)
                break
            
            # Find the parent object and continue tracing up
            parent_obj = next((o for o in all_objects if o.id == current_obj.place_id), None)
            if parent_obj is None:
                # Parent not found, can't continue tracing
                break
            current_obj = parent_obj
    
    # Format the result
    if not descendants:
        return ""
    
    count = len(descendants)
    ids_str = ", ".join(descendants)
    if count == 0:
        return ""
    return f"Object on top:  Totally {count} object{'s' if count > 1 else ''} ({ids_str})"


def get_object_description_list(objects, add_object_descendants=False):
    """
    Get object description list
    """
    object_dict = {}
    def map_constraint_name_to_description(constraint_name):
        if constraint_name == "edge":
            return "against wall"
        if constraint_name == "middle":
            return "placed in"
        else:
            return constraint_name
    for obj in objects:
        other_relationships = [[map_constraint_name_to_description(constraint["constraint"]), constraint.get("target", obj.id[:len("room_xxxxxxxx")])] for constraint in obj.placement_constraints[-1]] if obj.placement_constraints else []
        object_dict[obj.id] = {
            "location": [f"{obj.position.x:.2f}", f"{obj.position.y:.2f}", f"{obj.position.z:.2f}"],
            "rotation": [f"{np.radians(obj.rotation.x):.2f}", f"{np.radians(obj.rotation.y):.2f}", f"{(np.radians(obj.rotation.z + 90) % 360):.2f}"],
            "size": [f"{obj.dimensions.width:.2f}", f"{obj.dimensions.length:.2f}", f"{obj.dimensions.height:.2f}"],
            "relation": [["on top of", obj.place_id]] + other_relationships
        }
    return object_dict

def get_room_description(room):
    """
    Get room description
    """
    room_description = dict2str(get_object_description_list(room.objects))
    return room_description


def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def delete_collection_and_objects(collection_name):
    """Delete a collection and all objects in it"""
    if collection_name not in bpy.data.collections:
        return
    
    collection = bpy.data.collections[collection_name]
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


def get_or_create_collection(collection_name):
    """Get or create a collection"""
    if collection_name in bpy.data.collections:
        return bpy.data.collections[collection_name]
    
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def create_bbox_objects(room):
    """Create 3D bounding box wireframes for objects on floor/wall using Wireframe modifier"""
    collection = get_or_create_collection("mark")
    
    # Set mark collection as active so new objects are created directly in it
    layer_collection = bpy.context.view_layer.layer_collection
    mark_layer = None
    for lc in layer_collection.children:
        if lc.collection == collection:
            mark_layer = lc
            break
    
    if mark_layer:
        bpy.context.view_layer.active_layer_collection = mark_layer
    
    for obj in room.objects:
        if obj.place_id not in ["floor", "wall"]:
            continue
            
        # Get object dimensions and position
        obj_x = obj.position.x
        obj_y = obj.position.y
        obj_z = obj.position.z
        
        width = obj.dimensions.width
        length = obj.dimensions.length
        height = obj.dimensions.height
        
        rotation_z = np.radians(obj.rotation.z)
        
        # Create cube for bounding box at origin first
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
        bbox_obj = bpy.context.active_object
        bbox_obj.name = f"bbox_{obj.id}"
        
        # Scale it to match object dimensions
        bbox_obj.scale = (width, length, height)
        
        # Apply scale and rotation, then position
        center = mathutils.Vector((obj_x, obj_y, obj_z + height/2))
        scale_matrix = mathutils.Matrix.Diagonal((width, length, height)).to_4x4()
        rotation_matrix = mathutils.Matrix.Rotation(rotation_z, 4, 'Z')
        translation_matrix = mathutils.Matrix.Translation(center)
        
        bbox_obj.matrix_world = translation_matrix @ rotation_matrix @ scale_matrix
        
        # Add wireframe modifier to create 3D edges
        mod = bbox_obj.modifiers.new(name="Wireframe", type='WIREFRAME')
        mod.thickness = 0.03  # 3cm edge thickness
        
        # Create material for blue wireframe
        mat = bpy.data.materials.new(name=f"bbox_mat_{obj.id}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.0, 0.3, 1.0, 1.0)  # Blue
            bsdf.inputs["Specular"].default_value = 0.0  # Less shiny
        bbox_obj.data.materials.append(mat)
        bbox_obj.display_type = 'WIRE'
        bbox_obj.hide_render = False
        
        # Create text label
        bpy.ops.object.text_add(location=(0, 0, 0))
        text_obj = bpy.context.active_object
        text_obj.name = f"label_{obj.id}"
        text_obj.data.body = obj.type
        text_obj.rotation_euler = (0, 0, 0)
        text_scale = 0.15
        text_obj.scale = (text_scale, text_scale, text_scale)
        
        # Create material for white text with emission
        text_mat = bpy.data.materials.new(name=f"text_mat_{obj.id}")
        text_mat.use_nodes = True
        text_bsdf = text_mat.node_tree.nodes.get("Principled BSDF")
        if text_bsdf:
            text_bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)  # White
            text_bsdf.inputs["Emission"].default_value = (1, 1, 1, 1)
            text_bsdf.inputs["Emission Strength"].default_value = 2.0
        text_obj.data.materials.append(text_mat)
        
        # Position text above bbox
        bbox_center = bbox_obj.matrix_world.translation
        bbox_size = bbox_obj.dimensions
        bpy.context.view_layer.update()
        text_size = text_obj.dimensions
        text_offset = mathutils.Vector((
            -text_size.x * text_scale / 2,
            0.1,
            bbox_size.z / 2 + 0.02
        ))
        text_obj.location = bbox_center + text_offset
        
        # Create background plane for text
        bpy.ops.mesh.primitive_plane_add(size=1)
        bg_plane = bpy.context.active_object
        bg_plane.name = f"text_bg_{obj.id}"
        
        padding = 0.05
        bg_plane.scale.x = (text_size.x + padding) / text_obj.scale[0]
        bg_plane.scale.y = (text_size.y + padding) / text_obj.scale[1]
        bg_plane.location.x = (text_size.x / 2) / text_obj.scale[0]
        bg_plane.location.y = (text_size.y / 2) / text_obj.scale[1]
        bg_plane.location.z = -0.01
        
        # Blue material for background
        bg_mat = bpy.data.materials.new(name=f"bg_mat_{obj.id}")
        bg_mat.use_nodes = True
        bg_bsdf = bg_mat.node_tree.nodes.get("Principled BSDF")
        if bg_bsdf:
            bg_bsdf.inputs["Base Color"].default_value = (0.0, 0.3, 1.0, 1.0)  # Blue
            bg_bsdf.inputs["Roughness"].default_value = 1.0
        bg_plane.data.materials.append(bg_mat)
        
        # Parent background to text
        bg_plane.parent = text_obj


def create_arrow_objects(room):
    """Create 3D arrows showing object orientations"""
    collection = get_or_create_collection("mark")
    
    # Set mark collection as active so new objects are created directly in it
    layer_collection = bpy.context.view_layer.layer_collection
    mark_layer = None
    for lc in layer_collection.children:
        if lc.collection == collection:
            mark_layer = lc
            break
    
    if mark_layer:
        bpy.context.view_layer.active_layer_collection = mark_layer
    
    for obj in room.objects:
        if obj.place_id not in ["floor", "wall"]:
            continue
            
        obj_x = obj.position.x
        obj_y = obj.position.y
        obj_z = obj.position.z
        height = obj.dimensions.height
        
        # Get object's center at top surface
        start = mathutils.Vector((obj_x, obj_y, obj_z + height / 2))
        
        # Get local Y axis (front direction) in world coordinates
        # Default (0°) points +Y, 90° points -X
        rotation_z = np.radians(obj.rotation.z)
        local_y = mathutils.Vector((0, 1, 0))  # Default front is +Y
        rotation_matrix = mathutils.Matrix.Rotation(rotation_z, 3, 'Z')
        direction = rotation_matrix @ local_y
        direction.normalize()
        
        # Arrow parameters (use min of dimensions to avoid too-long arrows)
        shaft_length = min(obj.dimensions.width, obj.dimensions.length) * 0.75
        shaft_radius = 0.02
        head_length = 0.1
        head_radius = 0.1
        
        # Calculate shaft end position
        shaft_end = start + direction * shaft_length
        
        # Create shaft (cylinder)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=shaft_radius,
            depth=shaft_length,
            location=(start + shaft_end) / 2
        )
        shaft = bpy.context.active_object
        shaft.name = f"arrow_shaft_{obj.id}"
        
        # Align shaft to direction vector
        shaft.rotation_mode = 'QUATERNION'
        shaft.rotation_quaternion = direction.to_track_quat('Z', 'Y')
        
        # Create material for red/orange color
        mat = bpy.data.materials.new(name=f"arrow_mat_{obj.id}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1, 0.5, 0, 1)  # Orange
        shaft.data.materials.append(mat)
        
        # Create arrow head (cone)
        bpy.ops.mesh.primitive_cone_add(
            radius1=head_radius,
            depth=head_length,
            location=shaft_end + direction * (head_length / 2)
        )
        head = bpy.context.active_object
        head.name = f"arrow_head_{obj.id}"
        head.rotation_mode = 'QUATERNION'
        head.rotation_quaternion = direction.to_track_quat('Z', 'Y')
        head.data.materials.append(mat)
        
        # Join shaft and head
        bpy.ops.object.select_all(action='DESELECT')
        shaft.select_set(True)
        head.select_set(True)
        bpy.context.view_layer.objects.active = shaft
        bpy.ops.object.join()
        arrow = bpy.context.active_object
        arrow.name = f"arrow_{obj.id}"


def create_coordinate_grid(room):
    """Create coordinate grid with red circles and labels, plus RGB axis arrows"""
    collection = get_or_create_collection("mark")
    
    # Set mark collection as active so new objects are created directly in it
    layer_collection = bpy.context.view_layer.layer_collection
    mark_layer = None
    for lc in layer_collection.children:
        if lc.collection == collection:
            mark_layer = lc
            break
    
    if mark_layer:
        bpy.context.view_layer.active_layer_collection = mark_layer
    
    room_width = room.dimensions.width
    room_length = room.dimensions.length
    room_x = room.position.x
    room_y = room.position.y
    z = 0.01  # Just above floor
    
    # Helper function to create axis arrows
    def create_axis_arrow(start, direction, length, color, name):
        shaft_end = start + direction * length
        shaft_radius = 0.02
        head_length = 0.2
        head_radius = 0.08
        
        # Create shaft
        bpy.ops.mesh.primitive_cylinder_add(
            radius=shaft_radius,
            depth=length,
            location=(start + shaft_end) / 2
        )
        shaft = bpy.context.active_object
        shaft.rotation_mode = 'QUATERNION'
        shaft.rotation_quaternion = direction.to_track_quat('Z', 'Y')
        
        # Create head
        bpy.ops.mesh.primitive_cone_add(
            radius1=head_radius,
            depth=head_length,
            location=shaft_end + direction * (head_length / 2)
        )
        head = bpy.context.active_object
        head.rotation_mode = 'QUATERNION'
        head.rotation_quaternion = direction.to_track_quat('Z', 'Y')
        
        # Join and apply material
        bpy.ops.object.select_all(action='DESELECT')
        shaft.select_set(True)
        head.select_set(True)
        bpy.context.view_layer.objects.active = shaft
        bpy.ops.object.join()
        arrow = bpy.context.active_object
        arrow.name = name
        
        mat = bpy.data.materials.new(name=f"{name}_mat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = color
        arrow.data.materials.append(mat)
    
    # Create RGB axis arrows at origin
    origin = mathutils.Vector((room_x, room_y, z))
    create_axis_arrow(origin, mathutils.Vector((1, 0, 0)), 1.0, (1, 0, 0, 1), "axis_X")  # Red
    create_axis_arrow(origin, mathutils.Vector((0, 1, 0)), 1.0, (0, 1, 0, 1), "axis_Y")  # Green
    create_axis_arrow(origin, mathutils.Vector((0, 0, 1)), 1.0, (0, 0, 1, 1), "axis_Z")  # Blue
    
    # Create coordinate grid markers with red circles and labels
    for x_m in range(0, int(room_width) + 1):
        for y_m in range(0, int(room_length) + 1):
            # Create red circle marker
            bpy.ops.mesh.primitive_circle_add(
                vertices=64,
                radius=0.05,
                fill_type='NGON',
                location=(room_x + x_m, room_y + y_m, z)
            )
            circle = bpy.context.active_object
            circle.name = f"circle_{x_m}_{y_m}"
            
            # Red emissive material for circle
            circle_mat = bpy.data.materials.new(name=f"circle_mat_{x_m}_{y_m}")
            circle_mat.use_nodes = True
            bsdf = circle_mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (1, 0, 0, 1)  # Red
                bsdf.inputs["Emission"].default_value = (1, 0, 0, 1)
                bsdf.inputs["Emission Strength"].default_value = 2.0
                bsdf.inputs["Roughness"].default_value = 0.5
            circle.data.materials.append(circle_mat)
            
            # Create coordinate label
            label_text = f"({x_m},{y_m})"
            bpy.ops.object.text_add(location=(0, 0, 0))
            coord_text = bpy.context.active_object
            coord_text.name = f"coord_{x_m}_{y_m}"
            coord_text.data.body = label_text
            coord_text.rotation_euler = (0, 0, 0)
            text_scale = 0.2
            coord_text.scale = (text_scale, text_scale, text_scale)
            
            # Red emissive material for text
            text_mat = bpy.data.materials.new(name=f"text_mat_{x_m}_{y_m}")
            text_mat.use_nodes = True
            text_bsdf = text_mat.node_tree.nodes.get("Principled BSDF")
            if text_bsdf:
                text_bsdf.inputs["Base Color"].default_value = (1, 0, 0, 1)  # Red
                text_bsdf.inputs["Emission"].default_value = (1, 0, 0, 1)
                text_bsdf.inputs["Emission Strength"].default_value = 2.0
            coord_text.data.materials.append(text_mat)
            
            # Position text near circle
            bpy.context.view_layer.update()
            text_offset = mathutils.Vector((
                -coord_text.dimensions.x * text_scale / 2,
                0.1,
                circle.dimensions.z / 2 + 0.02
            ))
            coord_text.location = circle.location + text_offset


def place_camera_overhead(camera, room):
    """
    Position camera for top-down perspective view.
    Calculate optimal height based on room dimensions and fixed FOV to see full room.
    """
    room_width = room.dimensions.width
    room_length = room.dimensions.length
    room_height = room.dimensions.height
    room_x = room.position.x
    room_y = room.position.y
    
    # Fixed FOV (field of view angle in degrees)
    fov_degrees = 60.0  # Common FOV for perspective views
    fov_radians = np.radians(fov_degrees)
    
    # Calculate room floor diagonal (what we need to fit in view)
    floor_diagonal = max(room_width, room_length) * np.sqrt(2)
    
    # Calculate required camera height above room floor
    # For a camera looking straight down with FOV, the visible radius at floor is:
    # radius = height * tan(FOV/2)
    # We need: 2 * radius >= diagonal (to see full room)
    # So: height >= diagonal / (2 * tan(FOV/2))
    
    required_height_above_floor = floor_diagonal / (2 * np.tan(fov_radians / 2))
    
    # Add margin (20% extra) for safety and better framing
    margin_factor = 2.0
    camera_height_above_floor = required_height_above_floor * margin_factor
    
    # Total camera Z position (room floor is at room.position.z)
    camera_z = room.position.z + camera_height_above_floor
    
    # Position camera above center of room
    camera.location = (
        room_x + room_width / 2,
        room_y + room_length / 2,
        camera_z
    )
    
    # Point camera straight down
    camera.rotation_euler = (0, 0, 0)
    
    # Set to perspective camera with fixed FOV
    camera.data.type = 'PERSP'
    camera.data.angle = fov_radians  # Set FOV directly
    
    print(f"Camera positioned at height {camera_height_above_floor:.2f}m above floor")
    print(f"FOV: {fov_degrees}°, Room diagonal: {floor_diagonal:.2f}m")


def merge_two_images(background_imgfile, foreground_imgfile, transparent=False):
    """Merge base render with annotation overlay"""
    from PIL import Image
    
    # Load background image
    bg_image = Image.open(background_imgfile).convert("RGBA")
    
    # Load foreground (annotation) image
    fg_image = Image.open(foreground_imgfile).convert("RGBA")
    fg_image = fg_image.resize(bg_image.size)
    
    # Composite
    combined = Image.alpha_composite(bg_image, fg_image)
    
    # Save as PNG with high quality
    filename = background_imgfile.replace(".png", "_marked.png")
    combined.save(filename, "PNG", compress_level=3)  # Lower compression for better quality
    
    return filename


def create_foreground_mask_with_black_background(input_image_path, output_image_path):
    """
    Create a foreground object mask with black background.
    Uses the alpha channel from a transparent render to identify foreground objects.
    
    Args:
        input_image_path: Path to the image with transparent background (RGBA)
        output_image_path: Path to save the masked image with black background
        
    Returns:
        Path to the saved masked image
    """
    from PIL import Image
    
    # Load image with alpha channel
    img = Image.open(input_image_path).convert("RGBA")
    
    # Create a new image with black background
    black_bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    
    # Composite the image over black background using alpha channel
    result = Image.alpha_composite(black_bg, img)
    
    # Convert to RGB (removes alpha channel)
    result_rgb = result.convert("RGB")
    
    # Save the result
    result_rgb.save(output_image_path, "PNG", compress_level=3)
    
    print(f"Created foreground mask with black background: {output_image_path}")
    return output_image_path


def render_foreground_objects_with_black_background(room, layout, output_path):
    """
    Render only the foreground objects with a black background (no annotations).
    This creates a clean render of objects with everything else black.
    
    Args:
        room: Room object containing the scene
        layout: Layout object
        output_path: Path to save the output image
        
    Returns:
        Path to the saved image
    """
    # Load scene meshes first
    load_scene_meshes_into_blender(room, layout)
    
    # Setup scene
    scene = bpy.context.scene
    
    # Create or get camera
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
    else:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    
    # Position camera
    place_camera_overhead(camera, room)
    scene.camera = camera
    
    # Set up render engine and settings
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    
    # Set world background to pure black
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    world_nodes = scene.world.node_tree.nodes
    world_nodes.clear()
    world_bg = world_nodes.new(type='ShaderNodeBackground')
    world_bg.inputs[0].default_value = (0, 0, 0, 1)  # Pure black
    world_bg.inputs[1].default_value = 1.0
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    scene.world.node_tree.links.new(world_output.inputs['Surface'], world_bg.outputs['Background'])
    
    # Hide annotation collection if it exists
    if "mark" in bpy.data.collections:
        bpy.data.collections["mark"].hide_render = True
        bpy.data.collections["mark"].hide_viewport = True
    
    # Set render settings
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False  # No transparency, pure black background
    
    # Render
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered foreground objects with black background to {output_path}")
    
    return output_path


def load_scene_meshes_into_blender(room, layout):
    """Load room layout meshes from files into Blender"""
    from tex_utils import export_single_room_layout_to_mesh_dict_list
    import os
    
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
        if mesh_id.startswith("door") or mesh_id.startswith("wall") or mesh_id.startswith("window"):
            continue
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


def render_scene_with_bpy(room, layout, output_path, transparent=False, render_black_bg=True):
    """
    Render scene using Blender's bpy rendering with two-pass approach:
    1. Load scene meshes into Blender
    2. Render the actual scene
    3. Render transparent overlay with annotations
    4. Merge the two images
    5. Optionally render foreground objects with black background
    
    Args:
        room: Room object
        layout: Layout object
        output_path: Path to save the base render
        transparent: Whether to render with transparent background
        render_black_bg: Whether to also render foreground objects with black background
    """
    # Load scene meshes first
    load_scene_meshes_into_blender(room, layout)
    
    # Setup scene
    scene = bpy.context.scene
    
    # Create or get camera
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
    else:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    
    # Position camera
    place_camera_overhead(camera, room)
    scene.camera = camera
    
    # Set up render engine and settings
    scene.render.engine = 'CYCLES'  # Use Cycles for better quality
    scene.cycles.samples = 64  # Lower samples for speed, increase for quality
    scene.cycles.use_denoising = True
    
    # Set world background to black
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    world_nodes = scene.world.node_tree.nodes
    world_nodes.clear()
    world_bg = world_nodes.new(type='ShaderNodeBackground')
    world_bg.inputs[0].default_value = (1, 1, 1, 1)  # White background
    world_bg.inputs[1].default_value = 1.0  # Strength
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    scene.world.node_tree.links.new(world_output.inputs['Surface'], world_bg.outputs['Background'])
    
    # First pass: Render the scene without annotations
    # Hide annotation collection if it exists
    if "mark" in bpy.data.collections:
        bpy.data.collections["mark"].hide_render = True
        bpy.data.collections["mark"].hide_viewport = True
    
    # Set render settings
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    
    # Always use PNG for better quality
    scene.render.image_settings.file_format = "PNG"
    if transparent:
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.film_transparent = True
    else:
        scene.render.image_settings.color_mode = "RGB"
        scene.render.film_transparent = False
    
    base_output = output_path
    scene.render.filepath = base_output
    bpy.ops.render.render(write_still=True)
    print(f"Rendered base scene to {base_output}")
    
    # Second pass: Render annotations on transparent background
    # Hide all objects except annotations
    delete_collection_and_objects("mark")
    create_bbox_objects(room)
    create_arrow_objects(room)
    create_coordinate_grid(room)
    
    # Hide everything except mark collection
    if "mark" in bpy.data.collections:
        mark_collection = bpy.data.collections["mark"]
        mark_collection.hide_render = False
        mark_collection.hide_viewport = False
        
        # Get list of object names in mark collection
        mark_obj_names = set(obj.name for obj in mark_collection.objects)
        
        # Hide all other objects
        for obj in bpy.data.objects:
            if obj.name != "Camera" and obj.name not in mark_obj_names:
                obj.hide_render = True
    
    # Render with transparency
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    
    annotation_output = output_path.replace(".png", "_bbox.png")
    
    scene.render.filepath = annotation_output
    bpy.ops.render.render(write_still=True)
    print(f"Rendered annotations to {annotation_output}")
    
    # Restore visibility
    for obj in bpy.data.objects:
        obj.hide_render = False
    
    # Merge the two renders
    final_output = merge_two_images(base_output, annotation_output, transparent)
    print(f"Merged final image to {final_output}")
    
    # Optionally render foreground objects with black background
    if render_black_bg:
        black_bg_output = output_path.replace(".png", "_black_bg.png")
        
        # Hide annotation collection
        if "mark" in bpy.data.collections:
            bpy.data.collections["mark"].hide_render = True
            bpy.data.collections["mark"].hide_viewport = True
        
        # Ensure scene objects are visible
        if "scene_objects" in bpy.data.collections:
            bpy.data.collections["scene_objects"].hide_render = False
            bpy.data.collections["scene_objects"].hide_viewport = False
        
        # Set world background to pure black
        world_nodes = scene.world.node_tree.nodes
        world_nodes.clear()
        world_bg = world_nodes.new(type='ShaderNodeBackground')
        world_bg.inputs[0].default_value = (0, 0, 0, 1)  # Pure black
        world_bg.inputs[1].default_value = 1.0
        world_output = world_nodes.new(type='ShaderNodeOutputWorld')
        scene.world.node_tree.links.new(world_output.inputs['Surface'], world_bg.outputs['Background'])
        
        # Render with black background
        scene.render.image_settings.color_mode = "RGB"
        scene.render.film_transparent = False
        scene.render.filepath = black_bg_output
        bpy.ops.render.render(write_still=True)
        print(f"Rendered foreground objects with black background to {black_bg_output}")
        
        # Create marked version with black background
        black_bg_marked = merge_two_images(black_bg_output, annotation_output, transparent=False)
        print(f"Created marked image with black background: {black_bg_marked}")
    
    return final_output


def annotate_top_down_view_with_objects(rgb_image, room):
    """
    [LEGACY/FALLBACK] Annotate the top-down view image with bounding boxes and arrows for objects.
    
    This function uses PIL for 2D image annotation and is kept for reference/fallback.
    The main rendering pipeline now uses render_scene_with_bpy() which creates 3D annotations
    in Blender and renders them separately for better quality and consistency.
    
    Args:
        rgb_image: numpy array of the rendered top-down view (uint8, shape HxWx3)
        room: Room object containing the objects to annotate
    
    Returns:
        Annotated numpy array image
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont
    
    # Convert numpy array to PIL Image
    img = PILImage.fromarray(rgb_image)
    draw = ImageDraw.Draw(img)
    
    # Get image dimensions
    img_height, img_width = rgb_image.shape[:2]
    
    # Room dimensions in meters
    room_width_m = room.dimensions.width
    room_length_m = room.dimensions.length
    
    # Scale factors to convert from meters to pixels
    scale_x = img_width / room_width_m
    scale_y = img_height / room_length_m
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw coordinate axes at very bottom-left corner
    # Draw X axis (red) pointing right
    axis_length = 50
    axis_offset = 5  # Very small offset from edge
    draw.line([(axis_offset, img_height - axis_offset), 
               (axis_offset + axis_length, img_height - axis_offset)], 
              fill='red', width=3)
    draw.text((axis_offset + axis_length + 5, img_height - axis_offset - 8), 
              'X', fill='red', font=font)
    
    # Draw Y axis (green) pointing up
    draw.line([(axis_offset, img_height - axis_offset), 
               (axis_offset, img_height - axis_offset - axis_length)], 
              fill='green', width=3)
    draw.text((axis_offset + 5, img_height - axis_offset - axis_length - 5), 
              'Y', fill='green', font=font)
    
    # Draw origin label
    draw.text((axis_offset + 10, img_height - axis_offset - 20), 
              '[0,0]', fill='white', font=small_font)
    
    
    
    # Annotate each object
    for obj in room.objects:

        
        # Get object position relative to room
        obj_x = obj.position.x - room.position.x  # meters
        obj_y = obj.position.y - room.position.y  # meters
        
        # Convert to pixel coordinates (origin at bottom-left)
        pixel_x = obj_x * scale_x
        pixel_y = img_height - (obj_y * scale_y)  # Flip Y axis
        
        # Get object dimensions
        obj_width = obj.dimensions.width
        obj_length = obj.dimensions.length
        
        # Handle rotation - adjust bounding box dimensions
        rotation = obj.rotation.z
        if rotation == 0 or rotation == 180:
            bbox_width_px = obj_width * scale_x
            bbox_length_px = obj_length * scale_y
        elif rotation == 90 or rotation == 270:
            bbox_width_px = obj_length * scale_x
            bbox_length_px = obj_width * scale_y
        else:
            # For non-standard rotations, use the maximum extent
            bbox_width_px = obj_width * scale_x
            bbox_length_px = obj_length * scale_y
        
        # Calculate bounding box corners (centered at object position)
        x1 = pixel_x - bbox_width_px / 2
        y1 = pixel_y - bbox_length_px / 2
        x2 = pixel_x + bbox_width_px / 2
        y2 = pixel_y + bbox_length_px / 2
        

        label = obj.type
        if obj.place_id == "floor" or obj.place_id == "wall":
            # Draw bounding box in blue
            draw.rectangle([x1, y1, x2, y2], outline='blue', width=5)
        
        
        
        # Draw arrow to indicate facing direction (red)
        arrow_length = min(45, bbox_width_px * 1.8, bbox_length_px * 1.8)
        arrow_width = 8

        head_length = 20
        head_width = 10
        
        # Calculate arrow direction based on rotation
        # Default facing is +Y direction (rotation = 0)
        if rotation == 0:
            # Face +Y (up in image, which is negative pixel Y)
            arrow_end_x = pixel_x
            arrow_end_y = pixel_y - (arrow_length - head_length)

            arrow_end_x_stick = pixel_x
            arrow_end_y_stick = pixel_y - arrow_length
        elif abs(rotation - 90) < 1:
            # Face -X (left)
            arrow_end_x = pixel_x - (arrow_length - head_length)
            arrow_end_y = pixel_y

            arrow_end_x_stick = pixel_x - arrow_length
            arrow_end_y_stick = pixel_y
        elif abs(rotation - 180) < 1:
            # Face -Y (down in image, which is positive pixel Y)
            arrow_end_x = pixel_x
            arrow_end_y = pixel_y + (arrow_length - head_length)

            arrow_end_x_stick = pixel_x
            arrow_end_y_stick = pixel_y + arrow_length
        elif abs(rotation - 270) < 1:
            # Face +X (right)
            arrow_end_x = pixel_x + (arrow_length - head_length)
            arrow_end_y = pixel_y

            arrow_end_x_stick = pixel_x + arrow_length
            arrow_end_y_stick = pixel_y
        else:
            # For other angles, calculate based on rotation in radians
            rotation_rad = np.radians(rotation)
            # Default is +Y, so we start with angle pointing up (90 degrees in standard coords)
            arrow_end_x = pixel_x + (arrow_length - head_length) * np.sin(rotation_rad)
            arrow_end_y = pixel_y - (arrow_length - head_length) * np.cos(rotation_rad)

            arrow_end_x_stick = pixel_x + arrow_length * np.sin(rotation_rad)
            arrow_end_y_stick = pixel_y - arrow_length * np.cos(rotation_rad)
        # Draw arrow line
        # if obj.place_id == "floor":
        if obj.place_id == "floor" or obj.place_id == "wall":
            draw.line([(pixel_x, pixel_y), (arrow_end_x, arrow_end_y)], 
                    fill='red', width=arrow_width)
        
            # Draw arrow head
            # Calculate perpendicular vectors for arrow head
            dx = arrow_end_x_stick - pixel_x
            dy = arrow_end_y_stick - pixel_y
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx /= length
                dy /= length
                # Arrow head points
                
                # Back from tip
                base_x = arrow_end_x_stick - dx * head_length
                base_y = arrow_end_y_stick - dy * head_length
                # Perpendicular
                perp_x = -dy * head_width
                perp_y = dx * head_width
                # Triangle points
                p1 = (arrow_end_x_stick, arrow_end_y_stick)
                p2 = (base_x + perp_x, base_y + perp_y)
                p3 = (base_x - perp_x, base_y - perp_y)
                draw.polygon([p1, p2, p3], fill='red', outline='black')

        if obj.place_id == "floor" or obj.place_id == "wall":
            # Draw text with blue background
            # Get text bounding box
            text_bbox = draw.textbbox((pixel_x, pixel_y), label, font=small_font, anchor='mb')
            # Add padding to the background rectangle
            padding = 2
            text_bg_bbox = [
                text_bbox[0] - padding,
                text_bbox[1] - padding,
                text_bbox[2] + padding,
                text_bbox[3] + padding
            ]
            # Draw blue rectangle background
            draw.rectangle(text_bg_bbox, fill='blue')
            # Draw text on top
            draw.text((pixel_x, pixel_y), label, fill='white', font=small_font, 
                        anchor='mb')

                
    # Draw coordinate grid every 1 meter
    grid_color = (200, 200, 200)  # Light gray, more visible
    label_color = (255, 255, 255)  # White for labels
    bg_color = (0, 0, 0)  # Black background for labels
    
    # Draw vertical grid lines (parallel to Y axis) every 1 meter along X
    for x_m in range(0, int(room_width_m) + 1):
        pixel_x = x_m * scale_x
        draw.line([(pixel_x, 0), (pixel_x, img_height)], 
                  fill=grid_color, width=2)
        # # Add numeric labels along the bottom (X axis)
        # if x_m > 0:  # Skip 0 since we already have origin label
        #     text = str(x_m)
        #     bbox = draw.textbbox((pixel_x - 5, img_height - 25), text, font=small_font)
        #     draw.rectangle(bbox, fill=bg_color)
        #     draw.text((pixel_x - 5, img_height - 25), 
        #               text, fill=label_color, font=small_font)
    
    # Draw horizontal grid lines (parallel to X axis) every 1 meter along Y
    for y_m in range(0, int(room_length_m) + 1):
        pixel_y = img_height - (y_m * scale_y)  # Flip Y axis
        draw.line([(0, pixel_y), (img_width, pixel_y)], 
                  fill=grid_color, width=2)
        # Add numeric labels along the left side (Y axis)
        # if y_m > 0:  # Skip 0 since we already have origin label
        #     text = str(y_m)
        #     bbox = draw.textbbox((10, pixel_y - 10), text, font=small_font)
        #     draw.rectangle(bbox, fill=bg_color)
        #     draw.text((10, pixel_y - 10), 
        #               text, fill=label_color, font=small_font)
    
    # Add coordinate labels at every grid intersection
    for x_m in range(0, int(room_width_m) + 1):
        for y_m in range(0, int(room_length_m) + 1):
            if x_m == 0 and y_m == 0:
                continue  # Skip origin, already labeled
            
            pixel_x = x_m * scale_x + 10
            pixel_y = max(10, img_height - (y_m * scale_y) - 10)  # Flip Y axis
            
            text = f"({x_m},{y_m})"
            # Calculate text position (centered on intersection)
            bbox = draw.textbbox((pixel_x, pixel_y), text, font=small_font, anchor="mm")
            draw.rectangle(bbox, fill=bg_color)
            draw.text((pixel_x, pixel_y), text, fill=label_color, font=small_font, anchor="mm")
    
    # Convert back to numpy array
    return np.array(img)


def find_layout_id_by_room_id(room_id, results_dir=RESULTS_DIR):
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
                    return os.path.basename(layout_path)
        except:
            continue
    
    raise ValueError(f"No layout found containing room_id: {room_id}")


async def test_load_layout(layout_id: str, user_demand: str):
    """Test loading layout from JSON file"""
    
    try:

        if layout_id.startswith("room_"):
            layout_id = find_layout_id_by_room_id(layout_id)

        model="gpt-4.1"
        client = setup_oai_client()

        from PIL import Image as PILImage
        
        layout_save_path = os.path.join(RESULTS_DIR, layout_id, layout_id+".json")
        scene_save_dir = os.path.join(RESULTS_DIR, layout_id)

        with open(layout_save_path, 'r') as f:
            layout_data = json.load(f)
        layout = dict_to_floor_plan(layout_data)

        preview_save_dir = os.path.join(scene_save_dir, "preview")
        os.makedirs(preview_save_dir, exist_ok=True)

        room = layout.rooms[0]
        
        # Use Blender bpy rendering with two-pass approach
        # Render base scene + annotations overlay, then merge
        image_path_raw = os.path.join(preview_save_dir, "top_view_raw.png")
        
        # This will create:
        # 1. top_view_raw.png (base render with white background)
        # 2. top_view_raw_bbox.png (annotation overlay)
        # 3. top_view_raw_marked.png (merged final image with white background)
        # 4. top_view_raw_black_bg.png (base render with black background)
        # 5. top_view_raw_black_bg_marked.png (merged final image with black background)
        final_image_path = render_scene_with_bpy(room, layout, image_path_raw, transparent=False, render_black_bg=True)
        
        print(f"Rendered scene with annotations to {final_image_path}")
        
        # Use the black background marked image for evaluation
        image_path = image_path_raw.replace(".png", "_black_bg_marked.png")
        print(f"Using black background marked image for evaluation: {image_path}")

        # get room description
        room_description = get_room_description(room)
        
        # Get user demand from layout data if available
        user_demand = args.user_demand
        # user_demand = "Design me an office."
        # user_demand = layout.policy_analysis["scene_requirements"]
        
        # Prepare the evaluation prompt
        example_json = """
```json
{
  "realism": {
    "grade": your grade as int,
    "comment": "Your comment and suggestion, give exact evidence for the grade."
  },
  "functionality": {
    "grade": your grade as int,
    "comment": "Your comment and suggestion, give exact evidence for the grade."
  },
  "layout": {
    "grade": your grade as int,
    "comment": "Your comment and suggestion, give exact evidence for the grade."
  },
  "completion": {
    "grade": your grade as int,
    "comment": "Your comment and suggestion, give exact evidence for the grade."
  }
}
```
"""

        prompting_text_user = f"""
You are given a top-down room render image and the corresponding layout of each object. 
Your task is to evaluate how well they align with the user's preferences (provided in triple backticks) across the four criteria listed below.
For each criterion, assign a score from 0 to 10, and provide a brief justification for your rating.

Scoring must be strict. If any critical issue is found (such as missing key objects, obvious layout errors, or unrealistic elements), the score should be significantly lowered, even if other aspects are fine.

**Score Guidelines**:
- Score 10: Fully meets or exceeds expectations; no major improvements needed.
- Score 5: Partially meets expectations; some obvious flaws exist that limit usefulness or quality.
- Score 0: Completely fails to meet expectations; the aspect is absent, wrong, or contradicts user needs.

**Evaluation Criteria**:

1. **Realism**: How realistic the room appears. *Ignore texture, lighting, and doors.*
    - **Good (8-10)**: The layout (position, rotation, and size) is believable, and common daily objects make the room feel lived-in. Rich of daily furniture and objects.
    - **Bad (0-3)**: Unusual objects or strange placements make the room unrealistic.
    - **Note**: If object types or combinations defy real-world logic (e.g., bathtubs in bedrooms), score should be below 5.

2. **Functionality**: How well the room supports the intended activities (e.g., sleeping, working).
    - **Good (8-10)**: Contains the necessary furniture and setup for the specified function.
    - **Bad (0-3)**: Missing key objects or contains mismatched furniture (e.g., no bed in a bedroom).
    - **Note**: Even one missing critical item should lower the score below 6.

3. **Layout**: Whether the furniture is arranged logically in good pose and aligns with the user's preferences.
    - **Good (8-10)**: Each objects is in **reasonable size**, neatly placed, objects of the same category are well aglined, relationships are reasonable (e.g., chairs face desks), sufficient space exists for walking, and **orientations must** be correct. 
    - **Bad (0-3)**: Floating objects, crowded floor, **abnormal size**, objects with collision, incorrect **orientation**, or large items placed oddly (e.g., sofa not against the wall). Large empty space. Blocker in front of furniture.
    - **Note**: If the room has layout issues that affect use, it should not score above 5.

4. **Completion**: How complete and finished the room feels.
    - **Good (8-10)**: All necessary large and small items are present. Has rich details. Each shelf is full of objects (>5) inside. Each supporter (e.g. table, desk, and shelf) has small objects on it. Empty area is less than 50%. The room feels done.
    - **Bad (0-3)**: Room is sparse or empty, lacks decor or key elements.
    - **Note**: If more than 30% of the room is blank or lacks detail, score under 5.


Use the following user preferences as reference (enclosed in triple backticks):
User Preference:
```{user_demand}```

Room layout:
{room_description}

The Layout include each object's X-Y-Z Position, Z rotation, size (x_dim, y_dim, z_dim), as well as relation info with parents.
Each key in layout is the name for each object, consisting of a random number and the category name, such as "3142143_table". 
Note different category name can represent the same category, such as ChairFactory, armchair and chair can represent chair simultaneously.
Count objects carefully! Do not miss any details. 
Pay more attention to the orientation of each objects.

Return the results in the following JSON format, the "comment" should be short:
{example_json}

For the image:
Each object is marked with a 3D bounding box and its category label. You must count the object carefully with the given image and layout.

You are working in a 3D scene environment with the following conventions:

- Right-handed coordinate system.
- The X-Y plane is the floor.
- X axis (red) points right, Y axis (green) points top, Z axis (blue) points up.
- For the location [x,y,z], x,y means the location of object's center in x- and y-axis, z means the location of the object's bottom in z-axis.
- All asset local origins are centered in X-Y and at the bottom in Z.
- By default, assets face the +X direction.
- A rotation of [0, 0, 1.57] in Euler angles will turn the object to face +Y.
- All bounding boxes are aligned with the local frame and marked in blue with category labels.
- The front direction of objects are marked with yellow arrow.
- Coordinates in the image are marked from [0, 0] at bottom-left of the room.

"""
        
        # Encode image to base64
        image_base64 = encode_image_to_base64(image_path)
        
        # Call GPT API with image
        print("Calling GPT API for evaluation...")
        
        grades = {"realism": [], "functionality": [], "layout": [], "completion": []}
        
        for iteration in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompting_text_user
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2000
                )
                
                grading_str = response.choices[0].message.content
            except Exception as e:
                print(f"API call failed: {e}, retrying in 30 seconds...")
                time.sleep(30)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompting_text_user
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=6000
                )
                grading_str = response.choices[0].message.content
            
            print("GPT Response:")
            print(grading_str)
            print("-" * 50)
            
            # Parse JSON from response
            pattern = r"```json(.*?)```"
            matches = re.findall(pattern, grading_str, re.DOTALL)
            json_content = matches[0].strip() if matches else None
            
            if json_content is None:
                grading = json.loads(grading_str)
            else:
                grading = json.loads(json_content)
            
            for key in grades:
                grades[key].append(grading[key]["grade"])

            grading["input_text"] = prompting_text_user
        
        # Save the detailed grading
        grading_save_path = os.path.join(scene_save_dir, "grading.json")
        with open(grading_save_path, "w") as f:
            json.dump(grading, f, indent=4)
        print(f"Saved grading to {grading_save_path}")
        
        # Calculate mean and std of the grades
        for key in grades:
            grades[key] = {
                "mean": round(sum(grades[key]) / len(grades[key]), 2),
                "std": round(np.std(grades[key]), 2),
            }
        
        # Save the evaluation summary
        eval_save_path = os.path.join(scene_save_dir, "evaluation.json")
        with open(eval_save_path, "w") as f:
            json.dump(grades, f, indent=4)
        print(f"Saved evaluation summary to {eval_save_path}")
        
        print("\nEvaluation Results:")
        print(json.dumps(grades, indent=2))
        
        return grades, grading
        
    except Exception as e:
        print(f"ERROR: Exception occurred during test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    parser = argparse.ArgumentParser()
    parser.add_argument("layout_id", type=str)
    parser.add_argument("user_demand", type=str)
    args = parser.parse_args()
    asyncio.run(test_load_layout(args.layout_id, args.user_demand))
