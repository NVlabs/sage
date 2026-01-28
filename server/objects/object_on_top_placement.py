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
import trimesh
import tempfile
from constants import SERVER_ROOT_DIR, RESULTS_DIR, PHYSICS_CRITIC_ENABLED, SEMANTIC_CRITIC_ENABLED
from models import FloorPlan, Object, Room
from tex_utils import (
    get_textured_object_mesh,
    get_textured_object_mesh_from_object,
    apply_object_transform_direct,
)
from vlm import call_vlm
from utils import extract_json_from_response
import numpy as np
from datetime import datetime
import json
from isaacsim.isaac_mcp.server import test_object_placements_in_single_room
from typing import Dict, List
import open3d as o3d
import sys
from dataclasses import asdict
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from scipy.spatial.transform import Rotation as R

def get_random_placements_on_target_object(
    layout: FloorPlan, room: Room, target_object_id: str, 
    object_to_place: Object, sample_count: int = 100, 
    place_location: str = None,
    do_mobile_manipulator_reachability_check: bool = False,
    enable_reachability_visualization: bool = False,
    reachability_debug_dir: str = None,
    layout_file_name: str = None,
    regular_rotation: bool = False,
):
    room_id = room.id
    mesh_info_dict = get_textured_object_mesh(layout, room, room_id, target_object_id)
    mesh = mesh_info_dict["mesh"]
    vts = mesh_info_dict["texture"]["vts"]
    fts = mesh_info_dict["texture"]["fts"]
    texture_map_path = mesh_info_dict["texture"]["texture_map_path"]

    mesh_and_objects_on_it_list = [mesh]
    mesh_and_objects_on_it_list_ids = [target_object_id]
    for object_in_room in room.objects:
        # recursively check the object.place_id to see whether it is on the target object, until the place_id is not a object_id (floor or wall)
        object_in_room_place_id_original = object_in_room.id
        while True:
            object_in_room_place_id = object_in_room.place_id
            if object_in_room_place_id == "floor" or object_in_room_place_id == "wall":
                break
            else:
                if object_in_room_place_id == target_object_id:
                    mesh_and_objects_on_it_list.append(get_textured_object_mesh(layout, room, room_id, object_in_room_place_id_original)["mesh"])
                    mesh_and_objects_on_it_list_ids.append(object_in_room_place_id_original)
                    break
                else:
                    object_in_room = next((object for object in room.objects if object.id == object_in_room_place_id), None)
                    if object_in_room is None:
                        raise ValueError(f"Object with id {object_in_room_place_id} not found")
    
    face_normals = mesh.face_normals.reshape(-1, 3)
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)
    up_axis = np.array([0, 0, 1]).reshape(1, 3)
    
    support_faces = face_normals @ up_axis.T > 0.5

    support_faces_idxs = np.where(support_faces)[0]
    support_mesh = mesh.submesh([support_faces_idxs], append=True)
    samples, sample_face_idxs = trimesh.sample.sample_surface(support_mesh, sample_count)
    sample_face_normals = support_mesh.face_normals[sample_face_idxs]

    # get random placement location on the support faces
    placement_locations = samples + sample_face_normals * 0.01
    if regular_rotation:
        # sample from only 0, 90, 180, 270 degrees for all samples
        regular_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        placement_z_rotations = np.random.choice(regular_angles, size=sample_count)
    else:
        placement_z_rotations = np.random.uniform(0, 2 * np.pi, sample_count)


    # # debug
    # # export the placement locations pt cloud to ply
    # placement_locations_o3d = o3d.geometry.PointCloud()
    # placement_locations_o3d.points = o3d.utility.Vector3dVector(placement_locations)
    # o3d.io.write_point_cloud(f"{SERVER_ROOT_DIR}/place_info/placement_locations.ply", placement_locations_o3d)
    # # export the mesh to ply
    # mesh.export(f"{SERVER_ROOT_DIR}/place_info/mesh.ply")
    # assert False

    object_to_place_mesh_dict = get_textured_object_mesh_from_object(layout, object_to_place)
    object_to_place_mesh = object_to_place_mesh_dict["mesh"]

    # use vlm to give the placement location
    target_object = next((object for object in room.objects if object.id == target_object_id), None)

    detailed_prompt = f"""You are an expert interior designer analyzing object placement relationships. You need to determine the most appropriate placement location for one object relative to another object.

PLACEMENT OPTIONS:
1. "top" - The object should be placed ON TOP of the target object
2. "inside" - The object should be placed INSIDE the target object  
3. "both" - Either placement is acceptable with no strong preference

TARGET OBJECT INFORMATION:
- Type: {target_object.type}
- Description: {target_object.description}
- Dimensions: {target_object.dimensions.width}m × {target_object.dimensions.length}m × {target_object.dimensions.height}m
- Place ID: {target_object.place_id}
- Place Guidance: {target_object.place_guidance}

OBJECT TO PLACE INFORMATION:
- Type: {object_to_place.type}
- Description: {object_to_place.description}
- Dimensions: {object_to_place.dimensions.width}m × {object_to_place.dimensions.length}m × {object_to_place.dimensions.height}m
- Place Guidance: {object_to_place.place_guidance}

PLACEMENT EXAMPLES:

TOP PLACEMENT (place_location: "top"):
- Laptop on top of desk
- Pillow on top of sofa/bed
- Vase on top of table
- Book on top of nightstand
- Remote control on top of coffee table
- Plate on top of dining table
- Lamp on top of side table
- Phone on top of counter
- Decorative items on top of shelf
- Cup on top of coaster

INSIDE PLACEMENT (place_location: "inside"):
- Books inside bookshelf
- Food inside refrigerator
- Clothes inside wardrobe/closet
- Dishes inside dishwasher
- Toiletries inside medicine cabinet
- Files inside filing cabinet
- Shoes inside shoe rack
- Utensils inside drawer
- Groceries inside pantry
- Tools inside toolbox

BOTH PLACEMENT (place_location: "both"):
- Storage containers on/in storage shelf
- Decorative items on/in display cabinet
- Linens on/in linen closet
- Office supplies on/in desk organizer
- Cleaning supplies on/in utility cabinet
- Toys on/in toy chest
- Craft supplies on/in craft cabinet
- Personal items on/in dresser

DECISION CRITERIA:
- Consider the functional relationship between objects
- Think about typical usage patterns and accessibility
- Consider safety and stability
- Think about the intended purpose of both objects
- Consider size compatibility and physical constraints
- Follow common interior design and organizational principles

Analyze the target object and object to place, then determine the most appropriate placement location.

Return your response as a JSON object with this exact format:
```json
{{
    "place_location": "top|inside|both",
    "reason": "Brief explanation of why this placement choice makes the most sense based on the object types, dimensions, functionality, and typical usage patterns"
}}
```
"""

    if place_location is None:
        claude_response = call_vlm(
            vlm_type="openai",
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": detailed_prompt
                        }
                    ]
                }
            ]
        )
        response_text = claude_response.content[0].text

        response_text = response_text.strip()
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError("Could not extract JSON content from Claude response")
        response = json.loads(response_text)

        place_location = response["place_location"]



    placements = []
    for i in range(sample_count):
        placement_location = placement_locations[i]
        placement_z_rotation = placement_z_rotations[i]
        placement = {
            "position": {
                "x": float(placement_location[0]),
                "y": float(placement_location[1]),
                "z": float(placement_location[2])
            },
            "rotation": {
                "x": 0.0,
                "y": 0.0,
                "z": float(placement_z_rotation)
            }
        }
        if filter_placements_by_support_rays_and_collisions(
            mesh_and_objects_on_it_list, 
            object_to_place_mesh, 
            placement,
            place_location=place_location
        ):
            placements.append(placement)
    # print(f"Number of placements: {len(placements)}", file=sys.stderr)
    if do_mobile_manipulator_reachability_check:
        # Get target object name for visualization
        target_obj = next((obj for obj in room.objects if obj.id == target_object_id), None)
        target_object_name = target_obj.type if target_obj else target_object_id
        
        placements = mobile_manipulator_reachability_check(
            layout, room, placements,
            enable_visualization=enable_reachability_visualization,
            target_object_name=target_object_name,
            debug_dir=reachability_debug_dir,
            layout_file_name=layout_file_name,
        )
    # print(f"Number of reachable placements: {len(placements)}", file=sys.stderr)
        
    return placements

def visualize_placement_reachability(
    room_bounds, occupancy_grid, grid_x, grid_y, grid_res,
    placements, placement_reachability_data,
    layout_name="", room_id="", target_object_name="",
    save_path=None
):
    """
    Visualize placement reachability check including room occupancy, placements, 
    and robot positions color-coded by collision status.
    
    Args:
        room_bounds: tuple of (min_x, min_y, max_x, max_y) for the room
        occupancy_grid: 2D numpy array of boolean occupancy for scene objects
        grid_x, grid_y: arrays defining room grid coordinates
        grid_res: room grid resolution
        placements: list of placement dictionaries with "position" keys
        placement_reachability_data: dict mapping placement index to dict with:
            - "reachable_robot_points": array of robot positions within reach
            - "collision_status": array of boolean collision status for each robot position
            - "is_reachable": boolean indicating if placement is reachable
        layout_name, room_id, target_object_name: strings for labeling
        save_path: path to save the PNG file
    """
    from isaaclab.omron_franka_occupancy import occupancy_map, get_forward_side_from_support_point_and_yaw
    
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 1. Draw room boundaries
    room_rect = patches.Rectangle(
        (room_min_x, room_min_y), 
        room_max_x - room_min_x, 
        room_max_y - room_min_y,
        linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.2,
        label='Room Boundary'
    )
    ax.add_patch(room_rect)
    
    # 2. Visualize room occupancy grid
    if occupancy_grid is not None and occupancy_grid.size > 0:
        extent = [room_min_x, room_max_x, room_min_y, room_max_y]
        
        # Create custom colormap: white for free, red for occupied
        colors = ['white', 'red']
        cmap = ListedColormap(colors)
        
        # Display room occupancy grid
        im = ax.imshow(
            occupancy_grid.T,  # Transpose for correct orientation
            extent=extent,
            origin='lower',
            cmap=cmap,
            alpha=0.4,
            aspect='equal'
        )
    
    # 3. Draw placements
    for i, placement in enumerate(placements):
        placement_pos = placement["position"]
        placement_xy = [placement_pos["x"], placement_pos["y"]]
        
        # Check if this placement is reachable
        is_reachable = i in placement_reachability_data and placement_reachability_data[i]["is_reachable"]
        
        # Color code: green for reachable, red for unreachable
        marker_color = 'green' if is_reachable else 'red'
        marker_edge = 'darkgreen' if is_reachable else 'darkred'
        
        ax.scatter(
            placement_xy[0], placement_xy[1],
            c=marker_color, s=200, marker='*',
            edgecolors=marker_edge, linewidth=2,
            alpha=0.8, zorder=10
        )
        
        # Add placement index annotation
        ax.annotate(
            f'P{i}', 
            (placement_xy[0], placement_xy[1]),
            xytext=(8, 8), textcoords='offset points',
            fontsize=9, fontweight='bold',
            color='white',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=marker_color, alpha=0.9, edgecolor=marker_edge)
        )
    
    # 4. Draw robot positions for each placement with collision status
    for placement_idx, reach_data in placement_reachability_data.items():
        robot_points = reach_data["reachable_robot_points"]
        collision_status = reach_data["collision_status"]
        
        if len(robot_points) == 0:
            continue
        
        # Separate into collision and no-collision points
        no_collision_mask = ~collision_status
        collision_mask = collision_status
        
        no_collision_points = robot_points[no_collision_mask]
        collision_points = robot_points[collision_mask]
        
        # Draw no-collision points (green)
        if len(no_collision_points) > 0:
            ax.scatter(
                no_collision_points[:, 0], no_collision_points[:, 1],
                c='green', s=20, alpha=0.6, marker='o',
                edgecolors='darkgreen', linewidth=0.5,
                label=f'P{placement_idx} No Collision' if placement_idx == 0 else ''
            )
        
        # Draw collision points (purple)
        if len(collision_points) > 0:
            ax.scatter(
                collision_points[:, 0], collision_points[:, 1],
                c='purple', s=20, alpha=0.4, marker='x',
                linewidth=0.5,
                label=f'P{placement_idx} Collision' if placement_idx == 0 else ''
            )
    
    # 5. Setup plot
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Create title
    title = f'Placement Reachability Check\n'
    if layout_name:
        title += f'Layout: {layout_name}'
    if room_id:
        title += f' | Room: {room_id}'
    if target_object_name:
        title += f'\nTarget Object: {target_object_name}'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 6. Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='white', edgecolor='red', alpha=0.6, label='Occupied Space'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='green', 
               markersize=12, markeredgecolor='darkgreen', markeredgewidth=2, 
               label='Reachable Placement'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
               markersize=12, markeredgecolor='darkred', markeredgewidth=2, 
               label='Unreachable Placement'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=8, markeredgecolor='darkgreen', markeredgewidth=1, 
               label='Robot Pos (No Collision)'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='purple', 
               markersize=8, markeredgewidth=1, 
               label='Robot Pos (Collision)'),
    ]
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=10, framealpha=0.95, edgecolor='black')
    
    # 7. Save or show
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Placement reachability visualization saved to: {save_path}", file=sys.stderr)
    else:
        plt.show()
    
    plt.close(fig)


def mobile_manipulator_reachability_check(
    layout, room, placements, 
    enable_visualization=False,
    target_object_name="",
    debug_dir=None,
    layout_file_name=None
):
    """
    Filter placements to keep only those reachable by a mobile manipulator.
    
    Algorithm:
    1. Create occupancy grid and sample valid robot positions (once for all placements)
    2. For each placement:
       - Find robot positions within reachability distance (< 0.8m)
       - Test robot occupancy collision for reachable positions
       - Keep placement if at least one valid robot position exists
    3. Return filtered list of reachable placements

    Args:
        layout: FloorPlan object
        room: Room object
        placements: List of placement dictionaries with "position" and "rotation" keys
        enable_visualization: If True, generate visualization of reachability check
        target_object_name: Name of target object for visualization labeling
        debug_dir: Directory to save visualization (if None, will auto-generate)
        layout_file_name: Optional layout file name (without .json) to use for loading layout.
                         If None, uses layout.id (useful for temp files during correction)
        
    Returns:
        List of placements that are reachable by the mobile manipulator
        
    Take a look at the code in function sample_pick_object_pose_with_mobile_franka_occupancy()
    at object_mobile_manipulation_utils.py for reference
    """
    from objects.object_mobile_manipulation_utils import (
        create_unified_occupancy_grid,
        create_unified_scene_occupancy_fn,
        check_unified_robot_collision,
        CollisionCheckingConfig
    )
    
    if not placements:
        return []
    
    # Get scene information
    layout_id = layout.id
    room_id = room.id
    scene_save_dir = f"{RESULTS_DIR}/{layout_id}"
    
    # Use custom layout file name if provided (e.g., for temp files), otherwise use layout_id
    layout_name_to_use = layout_file_name if layout_file_name is not None else layout_id
    
    # Create unified occupancy grid for robot collision checking (once for all placements)
    occupancy_grid, grid_x, grid_y, room_bounds, _, _, _ = create_unified_occupancy_grid(
        scene_save_dir, layout_name_to_use, room_id
    )
    scene_occupancy_fn = create_unified_scene_occupancy_fn(occupancy_grid, grid_x, grid_y, room_bounds)
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds
    
    # Sample robot positions in room (once for all placements)
    num_robot_sample_points = 50000
    sample_points = np.random.uniform(
        low=[room_min_x, room_min_y], 
        high=[room_max_x, room_max_y], 
        size=(num_robot_sample_points, 2)
    )
    
    # Filter robot positions by room edges
    dist_to_edges = np.minimum.reduce([
        sample_points[:, 0] - room_min_x,
        room_max_x - sample_points[:, 0],
        sample_points[:, 1] - room_min_y,
        room_max_y - sample_points[:, 1]
    ])
    
    edge_valid_mask = dist_to_edges >= CollisionCheckingConfig.ROBOT_MIN_DIST_TO_ROOM_EDGE
    edge_valid_points = sample_points[edge_valid_mask]
    
    if len(edge_valid_points) == 0:
        return []
    
    # Filter by occupancy grid
    valid_robot_points = []
    grid_coords = np.floor((edge_valid_points - [room_min_x, room_min_y]) / CollisionCheckingConfig.GRID_RES).astype(int)
    
    grid_valid_mask = (
        (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < len(grid_x)) &
        (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < len(grid_y)) &
        (~occupancy_grid[grid_coords[:, 0], grid_coords[:, 1]])
    )
    
    grid_valid_points = edge_valid_points[grid_valid_mask]
    
    if len(grid_valid_points) == 0:
        return []
    
    # Check distance to occupied cells
    occupied_indices = np.where(occupancy_grid)
    if len(occupied_indices[0]) > 0:
        occupied_positions = np.column_stack([
            room_min_x + occupied_indices[0] * CollisionCheckingConfig.GRID_RES + CollisionCheckingConfig.GRID_RES/2,
            room_min_y + occupied_indices[1] * CollisionCheckingConfig.GRID_RES + CollisionCheckingConfig.GRID_RES/2
        ])
        
        # Batched distance checking
        batch_size = 1000
        valid_points_list = []
        
        for i in range(0, len(grid_valid_points), batch_size):
            batch_end = min(i + batch_size, len(grid_valid_points))
            batch_points = grid_valid_points[i:batch_end]
            
            distances = np.linalg.norm(
                batch_points[:, np.newaxis, :] - occupied_positions[np.newaxis, :, :], 
                axis=2
            )
            min_distances = np.min(distances, axis=1)
            distance_valid_mask = min_distances >= CollisionCheckingConfig.ROBOT_MIN_DIST_TO_OBJECT
            valid_points_list.append(batch_points[distance_valid_mask])
        
        if valid_points_list:
            valid_robot_points = np.concatenate(valid_points_list, axis=0)
        else:
            return []
    else:
        valid_robot_points = grid_valid_points
    
    if len(valid_robot_points) == 0:
        return []
    
    # Filter each placement for reachability
    max_reachability_distance = 0.8
    reachable_placements = []
    
    # For visualization: track reachability data for each placement
    placement_reachability_data = {}
    
    for placement_idx, placement in enumerate(placements):
        # Get placement position (object center position)
        placement_position = placement["position"]
        object_xy = np.array([placement_position["x"], placement_position["y"]])
        
        # Filter by reachability (distance < 0.8m)
        distances_to_object = np.linalg.norm(valid_robot_points - object_xy, axis=1)
        reachable_mask = distances_to_object <= max_reachability_distance
        reachable_robot_points = valid_robot_points[reachable_mask]
        reachable_distances = distances_to_object[reachable_mask]

        # For visualization: collect robot points and collision status
        if enable_visualization and len(reachable_robot_points) > 0:
            # Check collision for ALL reachable robot points (for visualization)
            collision_status = []
            for robot_pos_2d in reachable_robot_points:
                direction_to_object = object_xy - robot_pos_2d
                yaw = np.arctan2(direction_to_object[1], direction_to_object[0])
                robot_quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
                robot_pos_3d = np.array([robot_pos_2d[0], robot_pos_2d[1], 0])
                collision = check_unified_robot_collision(robot_pos_3d, robot_quat, scene_occupancy_fn, room_bounds)
                collision_status.append(collision)
            
            collision_status = np.array(collision_status)
        
        if len(reachable_robot_points) == 0:
            # For visualization: record unreachable placement
            if enable_visualization:
                placement_reachability_data[placement_idx] = {
                    "reachable_robot_points": np.array([]),
                    "collision_status": np.array([], dtype=bool),
                    "is_reachable": False
                }
            continue
        
        # Sort by distance (closest first)
        sorted_indices = np.argsort(reachable_distances)
        
        # Try robot positions until we find one without collision
        placement_is_reachable = False
        for idx in sorted_indices:
            robot_pos_2d = reachable_robot_points[idx]
            
            # Calculate robot orientation toward object
            direction_to_object = object_xy - robot_pos_2d
            yaw = np.arctan2(direction_to_object[1], direction_to_object[0])
            
            # Create quaternion
            robot_quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
            robot_pos_3d = np.array([robot_pos_2d[0], robot_pos_2d[1], 0])
            
            # Check robot collision
            collision = check_unified_robot_collision(robot_pos_3d, robot_quat, scene_occupancy_fn, room_bounds)
            
            if not collision:
                # Found at least one valid robot position for this placement!
                placement_is_reachable = True
                break
        
        # For visualization: record reachability data
        if enable_visualization:
            placement_reachability_data[placement_idx] = {
                "reachable_robot_points": reachable_robot_points,
                "collision_status": collision_status,
                "is_reachable": placement_is_reachable
            }
        
        if placement_is_reachable:
            reachable_placements.append(placement)
    
    # Generate visualization if enabled
    if enable_visualization:
        # Determine save path for visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if debug_dir is None:
            debug_dir = f"{scene_save_dir}/placement_reachability_debug"
            os.makedirs(debug_dir, exist_ok=True)
        
        save_path = os.path.join(
            debug_dir,
            f"placement_reachability_{layout_id}_{room_id}_{timestamp}.png"
        )
        
        visualize_placement_reachability(
            room_bounds=room_bounds,
            occupancy_grid=occupancy_grid,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_res=CollisionCheckingConfig.GRID_RES,
            placements=placements,
            placement_reachability_data=placement_reachability_data,
            layout_name=layout_id,
            room_id=room_id,
            target_object_name=target_object_name,
            save_path=save_path
        )
    
    return reachable_placements

def filter_placements_by_support_rays_and_collisions(
        target_object_mesh_list: List[trimesh.Trimesh], 
        object_to_place_mesh: trimesh.Trimesh, 
        placement: Dict, place_location="both"
    ):

    

    # get the placement location
    placement_location = placement["position"]

    # get the placement rotation
    placement_rotation = placement["rotation"]

    transformed_object_to_place_mesh = apply_object_transform_direct(
        object_to_place_mesh,
        placement_location,
        placement_rotation
    )

    transformed_vertices = transformed_object_to_place_mesh.vertices

    ray_origins = transformed_vertices
    ray_directions = np.array([0, 0, -1]).reshape(1, 3).repeat(len(transformed_vertices), axis=0)

    locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(target_object_mesh_list[0]).intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )

    num_vertices = len(transformed_vertices)

    intersected_num_rays = index_ray.shape[0]
    if intersected_num_rays < num_vertices:
        return False

    # no ray collisions with other objects
    if len(target_object_mesh_list) > 1:
        ray_origins = transformed_vertices
        ray_directions = np.array([0, 0, -1]).reshape(1, 3).repeat(len(transformed_vertices), axis=0)

        locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(
            trimesh.util.concatenate(target_object_mesh_list)
        ).intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )

        # num_vertices = len(transformed_vertices)

        # intersected_num_rays = index_ray.shape[0]
        # if intersected_num_rays > 0:
        #     return False
        if np.max(index_tri) >= target_object_mesh_list[0].faces.shape[0]:
            return False


    if place_location=="top":

        ray_origins = transformed_vertices
        ray_directions = np.array([0, 0, 1]).reshape(1, 3).repeat(len(transformed_vertices), axis=0)

        locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(target_object_mesh_list[0]).intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )

        num_vertices = len(transformed_vertices)

        intersected_num_rays = index_ray.shape[0]
        if intersected_num_rays > 0:
            return False
        
    elif place_location=="inside":

        ray_origins = transformed_vertices
        ray_directions = np.array([0, 0, 1]).reshape(1, 3).repeat(len(transformed_vertices), axis=0)

        locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(target_object_mesh_list[0]).intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )

        num_vertices = len(transformed_vertices)

        intersected_num_rays = index_ray.shape[0]
        if intersected_num_rays == 0:
            return False

    is_collided = detect_collision(target_object_mesh_list, transformed_object_to_place_mesh)
    if is_collided:
        return False

    return True


def detect_collision(base_meshes, test_mesh):
    """
    Detect collisions between a test mesh and a series of base meshes.
    Uses edge-based ray casting to detect intersections.

    Parameters:
    -----------
    base_meshes : List of trimesh.Trimesh
        List of base meshes to check against.


    test_mesh : trimesh.Trimesh
        The mesh to test for collisions.

    Returns:
    --------
    is_collided: bool
        True if the test mesh is collided with any of the base meshes, False otherwise
    """

    # Extract edges from test mesh
    edges = test_mesh.edges_unique

    # Get edge vertices
    edge_points = test_mesh.vertices[edges]

    # Create ray origins and directions from edges
    ray_origins = edge_points[:, 0]
    ray_directions = edge_points[:, 1] - edge_points[:, 0]

    # Normalize ray directions
    ray_lengths = np.linalg.norm(ray_directions, axis=1)
    ray_directions = ray_directions / ray_lengths[:, np.newaxis]

    # Check collision with each base mesh
    is_collided = False
    for base_trimesh in base_meshes:

        locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(base_trimesh).intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        if len(locations) > 0:
            # Calculate distances from ray origins to intersection points
            distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)

            # Only consider intersections that fall within the edge length
            valid_indices = distances <= ray_lengths[index_ray]

            if np.any(valid_indices):
                is_collided = True
                break

    return is_collided





def filter_placements_by_physics_critic(layout: FloorPlan, room: Room, object: Object, placements: list):
    room_id = room.id

    placements_info = {
        "placements": placements,
        "object": {
            "source": object.source,
            "source_id": object.source_id,
            "mass": getattr(object, "mass", 1.0)
        }
    }

    if not PHYSICS_CRITIC_ENABLED:
        return placements
    
    layout_id = layout.id
    scene_save_dir = f"{RESULTS_DIR}/{layout_id}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        placements_info_path = f.name
        json.dump(placements_info, f, indent=4)

    # export room into dict
    room_dict = asdict(room)
    room_dict_save_path = os.path.join(scene_save_dir, f"{room_id}.json")
    with open(room_dict_save_path, "w") as f:
        json.dump(room_dict, f, indent=4)

    sim_result = test_object_placements_in_single_room(room_dict_save_path, placements_info_path, only_need_one=True)
    safe_placements_path = sim_result["safe_placements_path"]

    with open(safe_placements_path, "r") as f:
        safe_placements = json.load(f)

    # delete two temporary files
    if os.path.exists(placements_info_path):
        os.remove(placements_info_path)
    if os.path.exists(safe_placements_path):
        os.remove(safe_placements_path)

    return safe_placements