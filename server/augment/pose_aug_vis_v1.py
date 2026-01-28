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
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import trimesh
import argparse

# Add parent directory to Python path to import constants
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR, ROBOMIMIC_ROOT_DIR, M2T2_ROOT_DIR

# Add path to server modules
sys.path.insert(0, SERVER_ROOT_DIR)
from utils import dict_to_floor_plan
from tex_utils import get_textured_object_mesh

def create_room_occupancy_grid(scene_save_dir, layout_name, room_id, aug_name):
    """Create room occupancy grid using ray casting, similar to sample_robot_location function"""
    
    grid_res = 0.02
    
    layout_json_path = os.path.join(scene_save_dir, aug_name, f"{layout_name}.json")
    with open(layout_json_path, "r") as f:
        layout_info = json.load(f)
    
    floor_plan = dict_to_floor_plan(layout_info)
    target_room = next(room for room in floor_plan.rooms if room.id == room_id)
    assert target_room is not None, f"target_room {room_id} not found in floor_plan"

    # Get room rectangle bounds
    room_min_x = target_room.position.x
    room_min_y = target_room.position.y
    room_max_x = target_room.position.x + target_room.dimensions.width
    room_max_y = target_room.position.y + target_room.dimensions.length
    
    # Get all object meshes in the room for occupancy calculation
    object_meshes = []
    for obj in target_room.objects:
        try:
            mesh_info = get_textured_object_mesh(floor_plan, target_room, room_id, obj.id)
            if mesh_info and mesh_info["mesh"] is not None:
                object_meshes.append(mesh_info["mesh"])
        except Exception as e:
            print(f"Warning: Could not load mesh for object {obj.id}: {e}")
            continue
    
    # Combine all object meshes for ray casting
    if object_meshes:
        combined_mesh = trimesh.util.concatenate(object_meshes)
    else:
        # If no meshes available, create an empty mesh
        combined_mesh = trimesh.Trimesh()
    
    # Create occupancy grid using ray casting
    grid_x = np.arange(room_min_x, room_max_x, grid_res)
    grid_y = np.arange(room_min_y, room_max_y, grid_res)
    
    # Create ray origins at grid centers, elevated above the room
    grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x + grid_res/2, grid_y + grid_res/2, indexing='ij')
    ray_origins = np.stack([
        grid_x_mesh.flatten(), 
        grid_y_mesh.flatten(), 
        np.full(grid_x_mesh.size, 10.0)
    ], axis=1).astype(np.float32)
    
    ray_directions = np.tile([0, 0, -1], (len(ray_origins), 1)).astype(np.float32)  # All pointing down
    
    # Perform ray casting to detect object occupancy
    occupancy_grid = np.zeros((len(grid_x), len(grid_y)), dtype=bool)
    
    if len(object_meshes) > 0:
        try:
            locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(combined_mesh).intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions
            )
            
            # Mark occupied grid cells using vectorized operations
            if len(index_ray) > 0:
                grid_i = index_ray // len(grid_y)
                grid_j = index_ray % len(grid_y)
                
                # Filter valid indices
                valid_mask = (grid_i < len(grid_x)) & (grid_j < len(grid_y))
                valid_i = grid_i[valid_mask]
                valid_j = grid_j[valid_mask]
                
                # Mark all valid cells as occupied at once
                occupancy_grid[valid_i, valid_j] = True
        except Exception as e:
            print(f"Warning: Ray casting failed: {e}")
    
    return {
        'room_bounds': (room_min_x, room_min_y, room_max_x, room_max_y),
        'occupancy_grid': occupancy_grid,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'grid_res': grid_res
    }

def add_room_and_occupancy_to_plot(ax, room_info):
    """Add room rectangle and occupancy grid to a matplotlib axis"""
    
    room_min_x, room_min_y, room_max_x, room_max_y = room_info['room_bounds']
    occupancy_grid = room_info['occupancy_grid']
    grid_x = room_info['grid_x']
    grid_y = room_info['grid_y']
    grid_res = room_info['grid_res']
    
    # Draw room rectangle
    room_rect = patches.Rectangle(
        (room_min_x, room_min_y), 
        room_max_x - room_min_x, 
        room_max_y - room_min_y,
        linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3
    )
    ax.add_patch(room_rect)
    
    # Draw occupancy grid
    for i in range(len(grid_x)):
        for j in range(len(grid_y)):
            if occupancy_grid[i, j]:
                x = room_min_x + i * grid_res
                y = room_min_y + j * grid_res
                occupied_rect = patches.Rectangle(
                    (x, y), grid_res, grid_res,
                    facecolor='gray', alpha=0.6
                )
                ax.add_patch(occupied_rect)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_id", type=str, required=True)
    parser.add_argument("--room_id", type=str, required=True)
    parser.add_argument("--target_object_name", type=str, required=True)
    parser.add_argument("--place_object_name", type=str, required=True)
    parser.add_argument("--table_object_name", type=str, default="")
    parser.add_argument("--aug_name", type=str, required=True)

    args = parser.parse_args()

    layout_id = args.layout_id
    scene_save_dir = os.path.join(SERVER_ROOT_DIR, f"results/{layout_id}")
    
    room_id = args.room_id
    target_object_name = args.target_object_name
    place_object_name = args.place_object_name
    table_object_name = args.table_object_name

    aug_name = args.aug_name
    aug_pose_json_name = f"all_augmented_layouts_info_{aug_name}.json"
    usd_collection_dir = f"usd_collection_{aug_name}"

    with open(os.path.join(scene_save_dir, aug_pose_json_name), "r") as f:
        all_augmented_layouts_info = json.load(f)

    mass_dict = all_augmented_layouts_info["mass_dict"]
    usd_collection_dir = all_augmented_layouts_info["usd_collection_dir"]
    object_transform_layouts_dict = all_augmented_layouts_info["object_transform_dict"]

    layout_aug_ids = sorted(list(object_transform_layouts_dict.keys()))

    print(f"Total layouts number: {len(object_transform_layouts_dict)}")

    # Create visualization directory
    vis_dir = os.path.join(os.path.dirname(__file__), "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # Get room and occupancy information from a reference layout
    print("Creating room occupancy grid...")
    reference_layout = layout_aug_ids[0]  # Use first layout as reference for room geometry
    room_info = create_room_occupancy_grid(scene_save_dir, reference_layout, room_id, aug_name)

    # Extract target and place object positions from the dataset
    target_positions = []
    place_positions = []

    # Process all data
    for layout_aug_id, layout_data in object_transform_layouts_dict.items():
        if target_object_name in layout_data:
            target_pos = layout_data[target_object_name]["position"]
            target_positions.append([target_pos[0], target_pos[1]])  # x, y coordinates
        
        if place_object_name in layout_data:
            place_pos = layout_data[place_object_name]["position"]
            place_positions.append([place_pos[0], place_pos[1]])  # x, y coordinates

    # Convert to numpy arrays for easier manipulation
    target_positions = np.array(target_positions)
    place_positions = np.array(place_positions)

    # Create the visualization with room context
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot 1: Target object positions with room context
    add_room_and_occupancy_to_plot(ax1, room_info)
    
    if len(target_positions) > 0:
        ax1.scatter(target_positions[:, 0], target_positions[:, 1], 
                   c='red', alpha=0.8, s=50, label=f'Target ({len(target_positions)})', 
                   marker='o', edgecolors='darkred', linewidth=1, zorder=5)

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title(f'Target Object Positions with Room Context\n({target_object_name})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: Place object positions with room context
    add_room_and_occupancy_to_plot(ax2, room_info)
    
    if len(place_positions) > 0:
        ax2.scatter(place_positions[:, 0], place_positions[:, 1], 
                   c='blue', alpha=0.8, s=50, label=f'Place ({len(place_positions)})', 
                   marker='s', edgecolors='darkblue', linewidth=1, zorder=5)

    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title(f'Place Object Positions with Room Context\n({place_object_name})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()
    
    # Save the visualization
    output_path = os.path.join(vis_dir, f'pose_augmentation_visualization_{aug_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create a combined visualization showing both target and place positions in one plot with room context
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    # Add room rectangle and occupancy grid
    add_room_and_occupancy_to_plot(ax, room_info)

    # Plot target positions
    if len(target_positions) > 0:
        ax.scatter(target_positions[:, 0], target_positions[:, 1], 
                  c='red', alpha=0.9, s=60, label=f'Target ({len(target_positions)})', 
                  marker='o', edgecolors='darkred', linewidth=1.5, zorder=5)

    # Plot place positions
    if len(place_positions) > 0:
        ax.scatter(place_positions[:, 0], place_positions[:, 1], 
                  c='blue', alpha=0.9, s=60, label=f'Place ({len(place_positions)})', 
                  marker='s', edgecolors='darkblue', linewidth=1.5, zorder=5)

    # Set plot bounds based on room with some padding
    room_min_x, room_min_y, room_max_x, room_max_y = room_info['room_bounds']
    ax.set_xlim(room_min_x - 0.5, room_max_x + 0.5)
    ax.set_ylim(room_min_y - 0.5, room_max_y + 0.5)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Object Position Distribution with Room Context - {aug_name}\nRoom: {room_id}\nGray areas: Object occupancy grid')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    
    # Save the combined visualization
    combined_output_path = os.path.join(vis_dir, f'pose_augmentation_combined_{aug_name}.png')
    plt.savefig(combined_output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print(f"\nVisualization Summary:")
    print(f"- Target positions: {len(target_positions)}")
    print(f"- Place positions: {len(place_positions)}")
    print(f"\nVisualization saved to:")
    print(f"- {output_path}")
    print(f"- {combined_output_path}")

    # Calculate and print position statistics
    if len(target_positions) > 0:
        target_mean = np.mean(target_positions, axis=0)
        target_std = np.std(target_positions, axis=0)
        print(f"\nTarget Statistics:")
        print(f"- Mean position: ({target_mean[0]:.3f}, {target_mean[1]:.3f})")
        print(f"- Std deviation: ({target_std[0]:.3f}, {target_std[1]:.3f})")

    if len(place_positions) > 0:
        place_mean = np.mean(place_positions, axis=0)
        place_std = np.std(place_positions, axis=0)
        print(f"\nPlace Statistics:")
        print(f"- Mean position: ({place_mean[0]:.3f}, {place_mean[1]:.3f})")
        print(f"- Std deviation: ({place_std[0]:.3f}, {place_std[1]:.3f})")

    print(f"\nRoom bounds: ({room_min_x:.2f}, {room_min_y:.2f}) to ({room_max_x:.2f}, {room_max_y:.2f})")
    print(f"Room dimensions: {room_max_x - room_min_x:.2f}m x {room_max_y - room_min_y:.2f}m") 