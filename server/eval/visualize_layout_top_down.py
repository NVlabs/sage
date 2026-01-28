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
import bpy
import mathutils

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FloorPlan
from tex_utils import dict_to_floor_plan, export_single_room_layout_to_mesh_dict_list
from constants import RESULTS_DIR


def get_layout_bounding_box(layout: FloorPlan):
    """
    Calculate the bounding box of the entire layout across all rooms.
    
    Args:
        layout: FloorPlan object
    
    Returns:
        min_x, max_x, min_y, max_y, min_z, max_z
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
        
        min_x = min(min_x, room_min_x)
        max_x = max(max_x, room_max_x)
        min_y = min(min_y, room_min_y)
        max_y = max(max_y, room_max_y)
        min_z = min(min_z, room_min_z)
        max_z = max(max_z, room_max_z)
    
    return min_x, max_x, min_y, max_y, min_z, max_z


def get_top_down_camera_sampling(layout_bbox, resolution, fov=35.0):
    """
    Sample camera position for top-down view of entire layout.
    
    Camera looks straight down (or slightly angled) from above the layout center.
    Camera up vector is chosen from 4 candidates to maximize layout width in view.
    
    Args:
        layout_bbox: tuple of (min_x, max_x, min_y, max_y, min_z, max_z)
        resolution: (height, width) render resolution tuple
        fov: field of view in degrees (fixed at 35)
    
    Returns:
        camera_pos: (x, y, z) camera position
        lookat_pos: (x, y, z) lookat position (center of layout at ground level)
        fov: field of view in degrees (returned as-is)
        best_up_vector: (x, y, z) best camera up vector for maximizing width
    """
    min_x, max_x, min_y, max_y, min_z, max_z = layout_bbox
    
    # Calculate layout center
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2  # Use middle height for lookat
    
    # Layout dimensions
    width = max_x - min_x
    length = max_y - min_y
    height = max_z - min_z
    
    res_height, res_width = resolution
    aspect_ratio = res_width / res_height
    
    # FOV parameters
    fov_rad = np.radians(fov)
    fov_vertical = fov_rad
    fov_horizontal = 2 * np.arctan(np.tan(fov_vertical / 2) * aspect_ratio)
    
    # Lookat position at the center of the layout
    lookat_pos = np.array([center_x, center_y, center_z])
    
    # Get all 4 corners of the layout bounding box at ground level
    corners_2d = [
        np.array([min_x, min_y, center_z]),
        np.array([max_x, min_y, center_z]),
        np.array([min_x, max_y, center_z]),
        np.array([max_x, max_y, center_z])
    ]
    
    print(f"\n=== Top-Down Camera Sampling ===")
    print(f"Layout bbox: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}], Z=[{min_z:.2f}, {max_z:.2f}]")
    print(f"Layout dimensions (W x L x H): {width:.2f} x {length:.2f} x {height:.2f}")
    print(f"Layout center: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")
    print(f"Lookat position: ({lookat_pos[0]:.2f}, {lookat_pos[1]:.2f}, {lookat_pos[2]:.2f})")
    print(f"FOV: {fov:.1f}° (vertical), {np.degrees(fov_horizontal):.1f}° (horizontal)")
    print(f"Aspect ratio: {aspect_ratio:.2f}")
    print(f"Resolution: {res_width}x{res_height}")
    
    # Test all 4 possible up vectors and choose the one with largest width
    print(f"\n=== Testing Up Vectors (with optimal height for each) ===")
    candidate_up_vectors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0)
    ]
    
    def find_optimal_height_and_width(camera_up_vec):
        """
        For a given up vector, find the optimal camera height and calculate the resulting width.
        Returns: (optimal_camera_height, width, overflow_status)
        """
        camera_up = np.array(camera_up_vec)
        
        def check_corners_fit(camera_height):
            """
            Check if all layout corners fit within the camera's FOV at the given height.
            Returns the maximum pixel overflow.
            """
            camera_pos = np.array([center_x, center_y, camera_height])
            
            # Camera forward direction
            forward = lookat_pos - camera_pos
            forward = forward / np.linalg.norm(forward)
            
            # Right vector
            right = np.cross(forward, camera_up)
            if np.linalg.norm(right) < 1e-6:
                return float('inf')  # Invalid configuration
            right = right / np.linalg.norm(right)
            
            # Recompute up to ensure orthogonality
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            # Calculate focal lengths
            focal_length_y = (res_height / 2.0) / np.tan(fov_vertical / 2)
            focal_length_x = (res_width / 2.0) / np.tan(fov_horizontal / 2)
            
            center_x_px = res_width / 2.0
            center_y_px = res_height / 2.0
            
            max_overflow = 0.0
            
            for corner in corners_2d:
                to_corner = corner - camera_pos
                x_cam = np.dot(to_corner, right)
                y_cam = np.dot(to_corner, up)
                z_cam = np.dot(to_corner, forward)
                
                if z_cam <= 0:
                    return float('inf')
                
                pixel_x = focal_length_x * (x_cam / z_cam) + center_x_px
                pixel_y = center_y_px - focal_length_y * (y_cam / z_cam)
                
                overflow_left = -pixel_x
                overflow_right = pixel_x - res_width
                overflow_top = -pixel_y
                overflow_bottom = pixel_y - res_height
                
                max_overflow = max(max_overflow, overflow_left, overflow_right, overflow_top, overflow_bottom)
            
            return max_overflow
        
        # Binary search for optimal camera height
        max_dim = max(width, length)
        initial_height = max_dim / (2 * np.tan(fov_vertical / 2)) + center_z
        
        height_min = initial_height * 0.5
        height_max = initial_height * 3.0
        
        for iteration in range(20):
            height_mid = (height_min + height_max) / 2
            overflow = check_corners_fit(height_mid)
            
            if overflow > 0:
                height_min = height_mid
            else:
                height_max = height_mid
        
        # Final height with 5% margin
        optimal_height = height_max * 1.05
        
        # Calculate width at this optimal height
        camera_pos = np.array([center_x, center_y, optimal_height])
        forward = lookat_pos - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, camera_up)
        if np.linalg.norm(right) < 1e-6:
            return optimal_height, -1, float('inf')
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        focal_length_x = (res_width / 2.0) / np.tan(fov_horizontal / 2)
        center_x_px = res_width / 2.0
        
        projected_x = []
        
        for corner in corners_2d:
            to_corner = corner - camera_pos
            x_cam = np.dot(to_corner, right)
            y_cam = np.dot(to_corner, up)
            z_cam = np.dot(to_corner, forward)
            
            if z_cam > 0:
                pixel_x = focal_length_x * (x_cam / z_cam) + center_x_px
                projected_x.append(pixel_x)
        
        if len(projected_x) == 0:
            return optimal_height, -1, float('inf')
        
        width_px = max(projected_x) - min(projected_x)
        final_overflow = check_corners_fit(optimal_height)
        
        return optimal_height, width_px, final_overflow
    
    best_up_vector = None
    best_width = -1
    best_camera_height = None
    
    for up_vec in candidate_up_vectors:
        optimal_height, width_px, overflow = find_optimal_height_and_width(up_vec)
        status = '✓' if overflow <= 0 else '✗ FAIL'
        print(f"  Up vector {up_vec}: height={optimal_height:.2f}, width={width_px:.1f}px {status}")
        if width_px > best_width:
            best_width = width_px
            best_up_vector = up_vec
            best_camera_height = optimal_height
    
    print(f"\nBest configuration:")
    print(f"  Up vector: {best_up_vector}")
    print(f"  Camera height: {best_camera_height:.2f}")
    print(f"  Width: {best_width:.1f}px")

    best_camera_height = best_camera_height * 1.05
    
    # Create final camera position with best configuration
    camera_pos = np.array([center_x, center_y, best_camera_height])
    
    print(f"\nCamera position: ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})")
    print(f"Camera-to-lookat distance: {np.linalg.norm(camera_pos - lookat_pos):.2f}")
    print(f"===================================\n")
    
    return camera_pos, lookat_pos, fov, best_up_vector




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


def load_all_rooms_meshes_into_blender(layout):
    """Load all rooms' meshes from layout into Blender"""
    
    # Clear all existing Blender assets before loading new ones
    clear_blender_scene()
    
    # Create collection for scene objects
    scene_collection = get_or_create_collection("scene_objects")
    
    total_meshes = 0
    
    # Load meshes for each room
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
            
            total_meshes += 1
    
    print(f"Loaded {total_meshes} meshes from {len(layout.rooms)} rooms into Blender scene")
    return total_meshes


def setup_camera_look_at(camera, camera_pos, lookat_pos, up_vector=(0, 1, 0)):
    """Position camera and make it look at target position with specified up vector"""
    # Set camera position
    camera.location = camera_pos
    
    # Build camera coordinate system manually
    # Forward: from camera to lookat (Blender camera looks along -Z)
    forward = mathutils.Vector(lookat_pos) - mathutils.Vector(camera_pos)
    forward.normalize()
    
    # Up vector (desired image up direction)
    up = mathutils.Vector(up_vector)
    up.normalize()
    
    # Right: perpendicular to forward and up
    right = forward.cross(up)
    if right.length < 1e-6:
        # Forward and up are parallel, use a different up vector
        up = mathutils.Vector((1, 0, 0)) if abs(up_vector[0]) < 0.9 else mathutils.Vector((0, 1, 0))
        right = forward.cross(up)
    right.normalize()
    
    # Recalculate up to ensure orthogonality
    up = right.cross(forward)
    up.normalize()
    
    # Build rotation matrix
    # Blender camera coordinate system: X=right, Y=up, Z=-forward
    # We need to map our vectors to Blender's camera space
    rot_matrix = mathutils.Matrix((
        (right.x, up.x, -forward.x),
        (right.y, up.y, -forward.y),
        (right.z, up.z, -forward.z)
    )).transposed()
    
    camera.rotation_euler = rot_matrix.to_euler()


def render_layout_top_down(layout: FloorPlan, output_path: str, resolution=1920):
    """
    Render a top-down view of the entire layout using Blender bpy.
    
    Args:
        layout: FloorPlan object
        output_path: Path to save the image
        resolution: Image width resolution (default: 1920 for 16:9)
    """
    # Calculate layout bounding box
    print("Calculating layout bounding box...")
    layout_bbox = get_layout_bounding_box(layout)
    
    # Load all rooms' meshes into Blender
    print("Loading all rooms' meshes into Blender...")
    load_all_rooms_meshes_into_blender(layout)
    
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
    
    # Set camera to perspective with vertical FOV control
    camera.data.type = 'PERSP'
    camera.data.sensor_fit = 'VERTICAL'
    
    # Set up render engine and settings
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    
    # Set world background to white for clean ambient lighting
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
    
    # Add warm ambient light from above
    if "TopDownLight" in bpy.data.objects:
        light = bpy.data.objects["TopDownLight"]
    else:
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        light = bpy.context.active_object
        light.name = "TopDownLight"
    
    # Configure light for top-down view
    light.data.energy = 10.0
    light.data.color = (1.0, 0.9, 0.7)  # Warm yellowish color
    light.data.angle = np.radians(10)  # Soft shadows
    light.rotation_euler = (0, 0, 0)  # Point straight down
    
    # Set render resolution (16:9 aspect ratio)
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution * 9 // 16
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    
    # Enable compositor for proper alpha channel handling
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_combined = True
    
    # Sample camera position for top-down view
    print("Sampling camera position...")
    camera_pos, lookat_pos, fov, best_up_vector = get_top_down_camera_sampling(
        layout_bbox=layout_bbox,
        resolution=(scene.render.resolution_y, scene.render.resolution_x),
        fov=35.0
    )
    
    # Set up camera with the best up vector
    print(f"Setting up camera with up vector: {best_up_vector}")
    setup_camera_look_at(camera, camera_pos, lookat_pos, up_vector=best_up_vector)
    camera.data.angle = np.radians(fov)
    
    # Render the image
    print(f"\nRendering top-down view...")
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    
    print(f"Image saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize entire layout from a top-down view")
    parser.add_argument("layout_id", type=str, help="Layout ID to visualize (e.g., layout_xxxxx)")
    parser.add_argument("--resolution", type=int, default=1920, help="Image width resolution (default: 1920 for 16:9)")
    
    args = parser.parse_args()
    
    try:
        # Find the layout directory
        layout_dir = os.path.join(RESULTS_DIR, args.layout_id)
        json_path = os.path.join(layout_dir, f"{args.layout_id}.json")
        
        if not os.path.exists(json_path):
            raise ValueError(f"Layout JSON not found: {json_path}")
        
        # Load the layout
        print(f"Loading layout from: {json_path}")
        with open(json_path, 'r') as f:
            layout_data = json.load(f)
        layout = dict_to_floor_plan(layout_data)
        
        print(f"Layout contains {len(layout.rooms)} rooms")
        
        # Create output path
        output_path = os.path.join(layout_dir, f"{args.layout_id}_top_down.png")
        
        # Render top-down view
        print(f"Rendering top-down view of layout {args.layout_id}...")
        render_layout_top_down(
            layout, 
            output_path,
            resolution=args.resolution
        )
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)