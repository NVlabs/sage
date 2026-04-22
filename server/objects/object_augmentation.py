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
from models import FloorPlan, Room, Object, Point3D, Euler
import copy
import os
import shutil
import numpy as np
from typing import List, Dict, Tuple, Union
import trimesh.transformations as tf
from constants import RESULTS_DIR
from tex_utils import get_textured_object_mesh_from_layout_id, get_textured_object_mesh, create_room_meshes_with_openings
from objects.object_on_top_placement import detect_collision
from utils import export_layout_to_json, generate_unique_id, extract_json_from_response
from isaacsim.isaac_mcp.server import (
    create_single_room_layout_scene,
    simulate_the_scene,
    create_single_room_layout_scene_from_room,
    create_room_groups_layouts,
    simulate_the_scene_groups
)
import json
from dataclasses import asdict
import random
import trimesh
import gc
import pdb
from vlm import call_vlm
from math import ceil
from objects.object_selection_planner import select_objects
from tqdm import tqdm
import itertools
import uuid
from constants import PHYSICS_CRITIC_ENABLED, SEMANTIC_CRITIC_ENABLED

# Cache for parent support meshes keyed by layout_id:room_id:parent_object_id
_SUPPORT_MESH_CACHE: Dict[str, trimesh.Trimesh] = {}


def _check_reachability(layout_copy: FloorPlan, room_copy: Room, all_objects_post_order: List[Object], reach_threshold: float) -> bool:
    """
    Check if objects in the room are reachable by a robot.
    
    Args:
        layout_copy: The floor plan layout
        room_copy: The room containing the objects
        all_objects_post_order: List of objects to check for reachability
        reach_threshold: Maximum reach distance for the robot
        
    Returns:
        True if at least one robot location can reach all non-floor objects, False otherwise
    """
    # Parameters (same as in sample_robot_location)
    robot_min_dist_to_room_edge = 0.5
    robot_min_dist_to_object = 0.15
    grid_res = 0.02
    num_sample_points = 10000
    
    # Get room rectangle bounds
    room_min_x = room_copy.position.x
    room_min_y = room_copy.position.y
    room_max_x = room_copy.position.x + room_copy.dimensions.width
    room_max_y = room_copy.position.y + room_copy.dimensions.length
    
    # Get all object meshes in the room for occupancy calculation
    object_meshes = []
    for obj in room_copy.objects:
        try:
            mesh_info = get_textured_object_mesh(layout_copy, room_copy, room_copy.id, obj.id)
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
    
    # Sample random points in the room rectangle
    sample_points = np.random.uniform(
        low=[room_min_x, room_min_y], 
        high=[room_max_x, room_max_y], 
        size=(num_sample_points, 2)
    )
    
    # Filter points based on constraints using vectorized operations
    # Check distance to room edges for all points at once
    dist_to_edges = np.minimum.reduce([
        sample_points[:, 0] - room_min_x,  # distance to left edge
        room_max_x - sample_points[:, 0],  # distance to right edge
        sample_points[:, 1] - room_min_y,  # distance to bottom edge
        room_max_y - sample_points[:, 1]   # distance to top edge
    ])
    
    # Filter out points too close to room edges
    edge_valid_mask = dist_to_edges >= robot_min_dist_to_room_edge
    edge_valid_points = sample_points[edge_valid_mask]
    
    if len(edge_valid_points) == 0:
        return False  # No valid robot positions
    
    # Convert to grid coordinates
    grid_coords = np.floor((edge_valid_points - [room_min_x, room_min_y]) / grid_res).astype(int)
    
    # Check if points are within grid bounds and not in occupied cells
    grid_valid_mask = (
        (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < len(grid_x)) &
        (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < len(grid_y)) &
        (~occupancy_grid[grid_coords[:, 0], grid_coords[:, 1]])
    )
    
    grid_valid_points = edge_valid_points[grid_valid_mask]
    
    if len(grid_valid_points) == 0:
        return False  # No valid robot positions after grid filtering
    
    # Check distance to occupied cells using vectorized operations
    occupied_indices = np.where(occupancy_grid)
    if len(occupied_indices[0]) > 0:
        occupied_positions = np.column_stack([
            room_min_x + occupied_indices[0] * grid_res + grid_res/2,
            room_min_y + occupied_indices[1] * grid_res + grid_res/2
        ])
        
        # Calculate distances from each valid point to all occupied cells
        distances = np.linalg.norm(
            grid_valid_points[:, np.newaxis, :] - occupied_positions[np.newaxis, :, :], 
            axis=2
        )
        
        # Find minimum distance to any occupied cell for each point
        min_distances = np.min(distances, axis=1)
        
        # Filter points that are far enough from occupied cells
        distance_valid_mask = min_distances >= robot_min_dist_to_object
        valid_robot_points = grid_valid_points[distance_valid_mask]
    else:
        # No occupied cells, all grid-valid points are valid
        valid_robot_points = grid_valid_points
    
    if len(valid_robot_points) == 0:
        return False  # No valid robot positions after distance filtering
    
    # Check reachability: for each object not on the floor, check if any robot location is within reach
    objects_to_check = [obj for obj in all_objects_post_order if obj.place_id != "floor"]
    
    if len(objects_to_check) == 0:
        return True  # No objects to check, consider reachable
    
    # Get object positions (2D)
    object_positions = np.array([[obj.position.x, obj.position.y] for obj in objects_to_check])
    
    # Calculate distances from each robot location to each object (10k x num_objects)
    # Shape: (n_robot_points, n_objects)
    distances_to_objects = np.linalg.norm(
        valid_robot_points[:, np.newaxis, :] - object_positions[np.newaxis, :, :], 
        axis=2
    )
    
    # Check if any robot location is within reach_threshold of all objects
    # For each robot location, check if it can reach all objects
    robot_can_reach_all = np.all(distances_to_objects <= reach_threshold, axis=1)
    
    # Return True if at least one robot location can reach all objects
    is_reachable = np.any(robot_can_reach_all)
    
    print(f"Reach test: {len(valid_robot_points)} valid robot locations, {len(objects_to_check)} objects to check, reachable: {is_reachable}")
    
    return is_reachable


def _object_pose_to_matrix(obj: Object) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from an object's position and Euler rotation (degrees).
    Rotation order matches apply_object_transform: Z @ Y @ X (Euler xyz).
    """
    rx = np.radians(obj.rotation.x)
    ry = np.radians(obj.rotation.y)
    rz = np.radians(obj.rotation.z)

    rot_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1],
    ])
    rot_y = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1],
    ])
    rot_z = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    rot = rot_z @ rot_y @ rot_x

    trans = np.eye(4)
    trans[:3, 3] = np.array([obj.position.x, obj.position.y, obj.position.z])

    return trans @ rot


def _matrix_to_pose(matrix: np.ndarray) -> Tuple[Point3D, Euler]:
    """
    Convert a 4x4 transform into position (meters) and Euler rotation (degrees, xyz order).
    """
    pos = matrix[:3, 3]
    rot_mat = matrix[:3, :3]
    # trimesh.transformations expects a 4x4 matrix for euler_from_matrix
    rot4 = np.eye(4)
    rot4[:3, :3] = rot_mat
    # 'sxyz' corresponds to static axes X->Y->Z; matches our construction
    ex, ey, ez = tf.euler_from_matrix(rot4, axes='sxyz')
    euler_deg = np.degrees([ex, ey, ez])
    return Point3D(float(pos[0]), float(pos[1]), float(pos[2])), Euler(float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2]))


def _sample_pose_offsets(unit_length: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate up to 30 small relative pose offsets.
    Returns a list of (T_pos, T_rot) where both are 4x4 matrices.
    """
    max_count = 20
    poses: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(max_count):
        # Ensure at least one of x/y/z translation or z-rotation changes
        while True:
            random_x = random.random() < 0.5
            random_y = random.random() < 0.5
            random_z_move = random.random() < 0.0  # keep z translation 0 as before
            random_z_rot = random.random() < 0.5
            if random_x or random_y or random_z_move or random_z_rot:
                break
        if random_x:
            tx = np.random.uniform(-0.5 * unit_length, 0.5 * unit_length)
        else:
            tx = 0.0
        if random_y:
            ty = np.random.uniform(-0.5 * unit_length, 0.5 * unit_length)
        else:
            ty = 0.0
        tz = 0.0

        T_pos = np.eye(4)
        T_pos[:3, 3] = np.array([tx, ty, tz])

        # Rotation offset: keep rx, ry = 0; vary rz optionally
        rx = 0.0
        ry = 0.0
        rz = np.deg2rad(np.random.uniform(-5.0, 5.0)) if random_z_rot else 0.0

        rot_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1],
        ])
        rot_y = np.array([
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1],
        ])
        rot_z = np.array([
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        T_rot = rot_z @ rot_y @ rot_x

        poses.append((T_pos, T_rot))
    return poses


def _sample_pose_offsets_condition(aug_obj, room_copy, layout_id, layout_copy_id, scene_save_dir, unit_length) -> List[Tuple[np.ndarray, np.ndarray]]:

    if aug_obj.place_id == "floor":
        sampled_poses = _sample_pose_offsets(unit_length)

        # Get the aug_obj mesh for collision testing
        try:
            aug_obj_mesh_info = get_textured_object_mesh_from_layout_id(layout_id, room_copy, room_copy.id, aug_obj.id)
            aug_obj_mesh = aug_obj_mesh_info["mesh"]
        except Exception as e:
            print(f"Warning: Could not load mesh for object {aug_obj.id}: {e}")
            return sampled_poses  # Return unfiltered if mesh loading fails
        
        # Get current pose matrix of aug_obj
        T_curr = _object_pose_to_matrix(aug_obj)
        
        # Get room structural meshes (walls, doors, windows)
        processed_doors = set()
        processed_windows = set()
        try:
            room_wall_meshes, room_door_meshes, room_window_meshes, room_wall_ids, room_door_ids, room_window_ids = create_room_meshes_with_openings(
                room_copy, processed_doors, processed_windows
            )
            # Combine all structural meshes for collision testing
            structural_meshes = room_wall_meshes + room_door_meshes + room_window_meshes
        except Exception as e:
            print(f"Warning: Could not create room structural meshes: {e}")
            return sampled_poses  # Return unfiltered if structural mesh creation fails

        # Filter poses by collision testing
        sampled_poses_filtered = []
        for T_pos, T_rot in sampled_poses:
            # Build candidate transform: apply rotation first, then translation
            T_candidate = T_curr @ T_rot @ T_pos
            
            # Transform the aug_obj mesh to the candidate pose
            transformed_mesh = aug_obj_mesh.copy()
            transformed_mesh.apply_transform(T_candidate @ np.linalg.inv(T_curr))
            
            # Test for collision with structural meshes
            if len(structural_meshes) > 0:
                is_collided = detect_collision(structural_meshes, transformed_mesh)
                if not is_collided:
                    sampled_poses_filtered.append((T_pos, T_rot))
            else:
                # No structural meshes to collide with, keep the pose
                sampled_poses_filtered.append((T_pos, T_rot))

        return sampled_poses_filtered
    elif aug_obj.place_id == "wall":
        assert False, "not implemented"
    else:
        unit_length = 10000.0
        aug_obj_parent: Object = next((o for o in room_copy.objects if o.id == aug_obj.place_id), None)
        aug_obj_parent_mesh_info_dict = get_textured_object_mesh_from_layout_id(layout_id, room_copy, room_copy.id, aug_obj_parent.id)
        aug_obj_parent_mesh = aug_obj_parent_mesh_info_dict["mesh"]

        # Build a map for hierarchy checks
        id_to_obj: Dict[str, Object] = {o.id: o for o in room_copy.objects}

        def is_on_parent(candidate: Object, parent_id: str) -> bool:
            current_parent = candidate.place_id
            visited = set()
            while isinstance(current_parent, str) and current_parent not in visited:
                if current_parent == parent_id:
                    return True
                visited.add(current_parent)
                parent_obj = id_to_obj.get(current_parent)
                if parent_obj is None:
                    return False
                current_parent = parent_obj.place_id
            return False

        # List A: meshes on the parent surface, excluding aug_obj and its descendants
        base_meshes: List[trimesh.Trimesh] = [aug_obj_parent_mesh]
        for obj in room_copy.objects:
            if obj.id == aug_obj.id:
                continue
            # exclude aug_obj's descendants
            if _is_descendant(aug_obj.id, obj, id_to_obj):
                continue
            if is_on_parent(obj, aug_obj_parent.id):
                mesh_info = get_textured_object_mesh_from_layout_id(layout_id, room_copy, room_copy.id, obj.id)
                base_meshes.append(mesh_info["mesh"])

        # Sample candidate points on upward-facing faces of the parent mesh
        cache_key = f"{layout_id}:{room_copy.id}:{aug_obj_parent.id}"
        support_mesh = _SUPPORT_MESH_CACHE.get(cache_key)
        if support_mesh is None:
            face_normals = aug_obj_parent_mesh.face_normals.reshape(-1, 3)
            face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)
            up_axis = np.array([0.0, 0.0, 1.0]).reshape(1, 3)
            support_faces_mask = (face_normals @ up_axis.T) > 0.5
            support_faces_idxs = np.where(support_faces_mask)[0]
            if support_faces_idxs.size == 0:
                return []
            support_mesh = aug_obj_parent_mesh.submesh([support_faces_idxs], append=True)
            _SUPPORT_MESH_CACHE[cache_key] = support_mesh

        sample_count = 100
        samples, sample_face_idxs = trimesh.sample.sample_surface(support_mesh, sample_count)
        sample_face_normals = support_mesh.face_normals[sample_face_idxs]
        candidate_positions = samples + sample_face_normals * 0.01

        # Filter by radius of unit_length in XY from current object's position
        curr_xy = np.array([aug_obj.position.x, aug_obj.position.y])
        cand_xy = candidate_positions[:, :2]
        dist_xy = np.linalg.norm(cand_xy - curr_xy.reshape(1, 2), axis=1)
        valid_idx = np.where(dist_xy <= float(unit_length))[0]
        if valid_idx.size == 0:
            valid_idx = np.arange(candidate_positions.shape[0])
        np.random.shuffle(valid_idx)

        # Prepare the mesh of the aug_obj to transform for collision checks
        aug_obj_mesh_info = get_textured_object_mesh_from_layout_id(layout_id, room_copy, room_copy.id, aug_obj.id)
        aug_obj_mesh: trimesh.Trimesh = aug_obj_mesh_info["mesh"]

        # Current pose matrix
        T_curr = _object_pose_to_matrix(aug_obj)

        # Collect up to 5 valid relative poses
        poses: List[Tuple[np.ndarray, np.ndarray]] = []
        # print(f"valid_idx: {len(valid_idx)}")
        for idx in valid_idx:
            if len(poses) >= 5:
                break
            placement_location = candidate_positions[idx]
            # Keep x,y rotation, perturb z by small random within [-30, 30] deg
            rx = 0
            ry = 0
            rz = np.deg2rad(np.random.uniform(-180.0, 180.0))

            # Build transform and transform mesh
            T_candidate = np.eye(4)
            T_candidate[:3, :3] = tf.euler_matrix(rx, ry, rz, axes='sxyz')[:3, :3]
            T_candidate[:3, 3] = placement_location

            transformed_mesh = aug_obj_mesh.copy()
            transformed_mesh.apply_transform(T_candidate @ np.linalg.inv(T_curr))

            # Support test: all vertices must have a support ray hit on the parent mesh
            transformed_vertices = transformed_mesh.vertices
            ray_origins = transformed_vertices
            ray_directions = np.tile(np.array([0.0, 0.0, -1.0]), (len(transformed_vertices), 1))
            _, index_ray, _ = trimesh.ray.ray_pyembree.RayMeshIntersector(base_meshes[0]).intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                multiple_hits=False
            )
            # print("lower normal ray test: ", index_ray.shape[0], transformed_vertices.shape[0])
            if index_ray.shape[0] < transformed_vertices.shape[0]:
                continue

            ray_directions = np.tile(np.array([0.0, 0.0, 1.0]), (len(transformed_vertices), 1))
            _, index_ray, _ = trimesh.ray.ray_pyembree.RayMeshIntersector(base_meshes[0]).intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                multiple_hits=False
            )
            # print("upper normal ray test: ", index_ray.shape[0], transformed_vertices.shape[0])
            if index_ray.shape[0] > 0:
                continue

            # Collision test with objects on parent (edge-based ray casting)
            edges = transformed_mesh.edges_unique
            edge_points = transformed_mesh.vertices[edges]
            ray_origins_edges = edge_points[:, 0]
            ray_directions_edges = edge_points[:, 1] - edge_points[:, 0]
            ray_lengths = np.linalg.norm(ray_directions_edges, axis=1) + 1e-12
            ray_directions_edges = ray_directions_edges / ray_lengths[:, np.newaxis]

            collided = False
            for base_m in base_meshes:
                locs, idx_ray, idx_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(base_m).intersects_location(
                    ray_origins=ray_origins_edges,
                    ray_directions=ray_directions_edges
                )

                if len(locs) > 0:
                    dists = np.linalg.norm(locs - ray_origins_edges[idx_ray], axis=1)
                    valid = dists <= ray_lengths[idx_ray]
                    if np.any(valid):
                        collided = True
                        break
            if collided:
                continue

            # Compute relative offsets: T_delta = inv(T_curr) @ T_candidate = T_rot @ T_pos
            T_delta = np.linalg.inv(T_curr) @ T_candidate
            R_delta = T_delta[:3, :3]
            t_delta = T_delta[:3, 3]

            T_rot = np.eye(4)
            T_rot[:3, :3] = R_delta

            # Make T_pos so that T_rot @ T_pos == T_delta => T_pos translation = inv(R_delta) @ t_delta
            T_pos = np.eye(4)
            T_pos[:3, 3] = np.linalg.inv(R_delta) @ t_delta

            poses.append((T_pos, T_rot))

        if len(poses) == 0:
            # print(f"no pose candidates for {aug_obj.id}")
            return []
        # print(f"pose candidates for {aug_obj.id}: {len(poses)}")
        return poses


def _build_object_id_map(room: Room) -> Dict[str, Object]:
    return {obj.id: obj for obj in room.objects}


def _is_descendant(target_object_id: str, candidate: Object, id_to_object: Dict[str, Object]) -> bool:
    """
    Return True if candidate is a descendant of target_object_id via place_id chain.
    """
    current_parent = candidate.place_id
    visited = set()
    while isinstance(current_parent, str) and current_parent not in visited:
        visited.add(current_parent)
        if current_parent == target_object_id:
            return True
        parent_obj = id_to_object.get(current_parent)
        if parent_obj is None:
            # Reached room/wall or invalid ID
            return False
        current_parent = parent_obj.place_id
    return False


def _update_descendants_transforms(room: Room, target_object_id: str, T_obj_old: np.ndarray, T_obj_new: np.ndarray):
    """
    Update transforms of all descendants of target_object_id according to:
    T_child_new = T_obj_new @ inv(T_obj_old) @ T_child_old
    """
    id_to_obj = _build_object_id_map(room)
    T_obj_old_inv = np.linalg.inv(T_obj_old)

    for obj in room.objects:
        if obj.id == target_object_id:
            # Will be updated separately by caller
            continue
        if not _is_descendant(target_object_id, obj, id_to_obj):
            continue

        T_child_old = _object_pose_to_matrix(obj)
        T_child_new = T_obj_new @ T_obj_old_inv @ T_child_old
        new_pos, new_rot = _matrix_to_pose(T_child_new)
        obj.position = new_pos
        obj.rotation = new_rot
        # print(f"updated {obj.id} to {new_pos}, {new_rot}")


def _validate_room_stability(scene_save_dir: str, room: Room, layout_aug_id: str) -> bool:

    # Create and simulate the single-room scene
    room_dict_save_path = os.path.join(scene_save_dir, f"{layout_aug_id}_{room.id}.json")
    with open(room_dict_save_path, "w") as f:
        json.dump(asdict(room), f)

    result_create = create_single_room_layout_scene_from_room(scene_save_dir, room_dict_save_path)
    if not isinstance(result_create, dict) or result_create.get("status") != "success":
        return False
    # pdb.set_trace()
    result_sim = simulate_the_scene()
    if not isinstance(result_sim, dict) or result_sim.get("status") != "success":
        return False
    next_step = result_sim.get("next_step", "") or ""

    # clean the room_dict_save_path
    os.remove(room_dict_save_path)
    return "stable" in next_step.lower() and "unstable" not in next_step.lower()

def _validate_room_stability_groups(scene_save_dir: str, aug_name: str, room_list: List[Room], layout_aug_id_list: List[str]) -> bool:

    # Create and simulate the single-room scene
    room_dict_save_paths = []
    os.makedirs(os.path.join(scene_save_dir, aug_name), exist_ok=True)
    for room, layout_aug_id in zip(room_list, layout_aug_id_list):
        room_dict_save_path = os.path.join(scene_save_dir, aug_name, f"{layout_aug_id}_{room.id}.json")
        with open(room_dict_save_path, "w") as f:
            json.dump(asdict(room), f)
        room_dict_save_paths.append(room_dict_save_path)

    # Generate unique UUID to avoid conflicts between multiple processes
    unique_id = str(uuid.uuid4())[:8]
    room_list_save_path = os.path.join(scene_save_dir, f"{aug_name}_aug_room_list_{unique_id}.json")
    with open(room_list_save_path, "w") as f:
        json.dump(room_dict_save_paths, f, indent=4)

    result_create = create_room_groups_layouts(scene_save_dir, room_list_save_path)
    if not isinstance(result_create, dict) or result_create.get("status") != "success":
        return False
    # pdb.set_trace()
    results_sim = simulate_the_scene_groups()
    # print(f"results_sim: {results_sim}")
    if not isinstance(results_sim, dict) or results_sim.get("status") != "success":
        return False

    stable_list = json.loads(results_sim["group_stable_list"])
    # print(f"stable_list: {stable_list}")

    os.remove(room_list_save_path)
    for room_dict_save_path in room_dict_save_paths:
        os.remove(room_dict_save_path)

    return stable_list


def object_augmentation_pose_single_object(layout: FloorPlan, room: Room, object: Object, aug_num: int, aug_name: str):

    augmented_layouts_info = []

    # Precompute the parent object's original transform
    T_obj_original = _object_pose_to_matrix(object)

    # Build mapping for descendant checks once
    id_to_obj = _build_object_id_map(room)

    for aug_iter in range(max(0, int(aug_num))):
        # Deep copy layout
        layout_copy: FloorPlan = copy.deepcopy(layout)
        # Find the corresponding room and object in the copied layout
        room_copy = next((r for r in layout_copy.rooms if r.id == room.id), None)
        if room_copy is None:
            continue
        obj_copy = next((o for o in room_copy.objects if o.id == object.id), None)
        if obj_copy is None:
            continue

        # New layout id with aug tag
        layout_copy_id = generate_unique_id(f"{layout.id}_{aug_name}")

        # Sample augmentation offsets (pick first candidate)
        pose_candidates = _sample_pose_offsets(unit_length=min(object.dimensions.x, object.dimensions.y))
        if len(pose_candidates) == 0:
            continue
        T_pos, T_rot = pose_candidates[0]
        T_aug = T_obj_original @ T_rot @ T_pos

        # Update parent object's pose
        new_pos, new_rot = _matrix_to_pose(T_aug)
        obj_copy.position = new_pos
        obj_copy.rotation = new_rot

        # Update all descendants inside the copied room
        _update_descendants_transforms(room_copy, obj_copy.id, T_obj_original, T_aug)

        # Validate physics stability for the room
        # Use original layout folder for assets during validation
        scene_save_dir = os.path.join(RESULTS_DIR, layout.id)
        is_stable = _validate_room_stability(scene_save_dir, room_copy, layout_copy_id)

        print(f"augmentation iteration {aug_iter} is stable: {is_stable}")
        if is_stable:
            # update the objects in room of layout_copy
            for layout_room in layout_copy.rooms:
                if layout_room.id == room.id:
                    # Replace all objects with the placed objects (includes existing + new)
                    layout_room.objects = room_copy.objects
                    break

            # save the augmented layout
            layout_copy_path = os.path.join(RESULTS_DIR, layout.id, f"{layout_copy_id}.json")
            with open(layout_copy_path, "w") as f:
                json.dump(asdict(layout_copy), f)
            augmented_layouts_info.append((layout_copy_id, layout_copy))

    return augmented_layouts_info


def object_augmentation_pose_object_tree(
    layout: FloorPlan, room: Room, object: Object, aug_num: int, aug_name: str
):

    augmented_layouts_info = []

    # Precompute the parent object's original transform (root object)
    T_obj_original = _object_pose_to_matrix(object)

    # Build mapping for descendant checks once
    id_to_obj = _build_object_id_map(room)

    # Use original layout folder for assets during validation
    scene_save_dir = os.path.join(RESULTS_DIR, layout.id)

    for aug_iter in range(max(0, int(aug_num))):
        # Deep copy layout
        layout_copy: FloorPlan = copy.deepcopy(layout)
        # Find the corresponding room and object in the copied layout
        room_copy = next((r for r in layout_copy.rooms if r.id == room.id), None)
        if room_copy is None:
            continue

        # todo: get all child objects of the object
        # Build children map for objects inside the room copy
        object_map_copy = {o.id: o for o in room_copy.objects}
        children_map: Dict[str, List[Object]] = {}
        for o in room_copy.objects:
            parent_id = o.place_id
            if isinstance(parent_id, str) and parent_id in object_map_copy:
                children_map.setdefault(parent_id, []).append(o)

        # and sort them from the leaves to the root (a parent node is visited until all its children are visited)
        def collect_post_order(root_id: str) -> List[Object]:
            result: List[Object] = []
            visited: set = set()

            def dfs(node_id: str):
                # Traverse children first
                for child in children_map.get(node_id, []):
                    if child.id not in visited:
                        dfs(child.id)
                # Append after visiting children; exclude the root itself to return only child objects
                if node_id != root_id and node_id in object_map_copy:
                    visited.add(node_id)
                    result.append(object_map_copy[node_id])

            dfs(root_id)
            return result

        child_objects_post_order: List[Object] = collect_post_order(object.id)
        all_objects_post_order: List[Object] = child_objects_post_order + [object]

        # New layout id with aug tag
        layout_copy_id = generate_unique_id(f"{layout.id}_{aug_name}")

        # For each child object + parent object, sample pose augmentations and try each
        safe_aug = True
        for aug_i, aug_obj in enumerate(all_objects_post_order):
            pose_candidates = _sample_pose_offsets_condition(aug_obj, room_copy, layout.id, layout_copy_id, scene_save_dir, unit_length=min(aug_obj.dimensions.width, aug_obj.dimensions.length))
            if len(pose_candidates) == 0:
                safe_aug = False
                print(f"no pose candidates for {aug_obj.id}")
                break

            # Previous transform of current object in the evolving room_copy state
            T_prev = _object_pose_to_matrix(aug_obj)

            accepted = False
            for pose_i, (T_pos, T_rot) in enumerate(pose_candidates):
                T_aug = T_prev @ T_rot @ T_pos

                # Apply transform to aug_obj
                new_pos, new_rot = _matrix_to_pose(T_aug)
                old_pos, old_rot = _matrix_to_pose(T_prev)


                # Update all descendants inside the copied room
                _update_descendants_transforms(room_copy, aug_obj.id, T_prev, T_aug)
                for obj in room_copy.objects:
                    if obj.id == aug_obj.id:
                        obj.position = new_pos
                        obj.rotation = new_rot
                        break

                # Validate physics stability for the room
                is_stable = _validate_room_stability(scene_save_dir, room_copy, layout_copy_id)
                print(f"aug {aug_i}th object {aug_obj.id} pose {pose_i} is stable: {is_stable}")

                room_copy_ckpt = copy.deepcopy(room_copy)

                # Revert transforms after validation
                _update_descendants_transforms(room_copy, aug_obj.id, T_aug, T_prev)
                for obj in room_copy.objects:
                    if obj.id == aug_obj.id:
                        obj.position = old_pos
                        obj.rotation = old_rot
                        break
                
                # pdb.set_trace()
                if is_stable:
                    accepted = True
                    room_copy = room_copy_ckpt
                    
                    break


            if not accepted:
                safe_aug = False
                print(f"aug failed at {aug_i}th object")
                break

        print(f"augmentation iteration {aug_iter} is stable: {safe_aug}")
        if not safe_aug:
            continue

        # update the objects in room of layout_copy
        for layout_room in layout_copy.rooms:
            if layout_room.id == room.id:
                # Replace all objects with the placed objects (includes existing + new)
                layout_room.objects = room_copy_ckpt.objects
                break

        # save the augmented layout
        layout_copy_path = os.path.join(RESULTS_DIR, layout.id, f"{layout_copy_id}.json")
        with open(layout_copy_path, "w") as f:
            json.dump(asdict(layout_copy), f)
        augmented_layouts_info.append((layout_copy_id, layout_copy))

    return augmented_layouts_info

def object_augmentation_pose_object_tree_sim_correction(
    layout: FloorPlan, room: Room, parent_object: Object,
    reachable_object_ids: List[str],
    layout_file_name: str = None,
    always_return_room: bool = False
):
    """
    It will place all objects belonging to the descandant of the parent_object
    without changing the original pose of the parent_object.
    and follow the original support tree of objects.
    
    Args:
        layout: The floor plan layout
        room: The room containing the parent object
        parent_object: The parent object whose descendants need to be re-placed
        reachable_object_ids: List of object IDs that must be reachable by mobile manipulator
        layout_file_name: Optional layout file name (without .json) for loading occupancy grid.
                         If None, uses layout.id (useful for temp files during correction)
        
    Returns:
        Tuple of (new_room_objects, success) where new_room_objects is the updated
        object list if successful, None otherwise, and success is a boolean
    """
    from objects.object_on_top_placement import (
        get_random_placements_on_target_object,
        filter_placements_by_physics_critic
    )
    import sys
    
    # Step 1: Make a copy of the room and objects
    room_copy = copy.deepcopy(room)
    
    # Build object map and children map for the original room structure
    object_map: Dict[str, Object] = {o.id: o for o in room.objects}
    children_map: Dict[str, List[Object]] = {}
    for o in room.objects:
        parent_id = o.place_id
        if isinstance(parent_id, str) and parent_id in object_map:
            children_map.setdefault(parent_id, []).append(o)
    
    # Step 2: Collect all descendants in DFS order (depth-first)
    # Prioritize objects in reachable_object_ids while maintaining DFS order
    
    # First, build a descendant map: for each object, collect all its descendants
    descendant_map: Dict[str, set] = {}
    
    def build_descendants(obj_id: str) -> set:
        """Recursively build set of all descendants for an object."""
        if obj_id in descendant_map:
            return descendant_map[obj_id]
        
        descendants = set()
        for child in children_map.get(obj_id, []):
            descendants.add(child.id)
            descendants.update(build_descendants(child.id))
        
        descendant_map[obj_id] = descendants
        return descendants
    
    # Build descendant map for all objects
    for obj in room.objects:
        build_descendants(obj.id)
    
    # Now collect descendants in DFS order, prioritizing branches with reachable objects
    dfs_descendants: List[Object] = []
    visited: set = set()
    
    def dfs_collect(current_id: str):
        """DFS helper to collect descendants, prioritizing reachable objects."""
        if current_id in visited:
            return
        visited.add(current_id)
        
        # Get children of current object
        children = children_map.get(current_id, [])
        
        # Sort children by priority:
        # 1. Children that are themselves reachable
        # 2. Children whose subtree contains reachable objects
        # 3. Children that has larger width and length dimension
        # 4. Other children
        def child_priority(child: Object) -> tuple:
            is_reachable = child.id in reachable_object_ids
            has_reachable_descendants = bool(descendant_map.get(child.id, set()) & set(reachable_object_ids))
            object_dimensions = child.dimensions
            object_area = object_dimensions.width * object_dimensions.length
            # Return tuple for sorting: (not is_reachable, not has_reachable_descendants, -object_area)
            # This way, reachable objects come first (0, _, _), then objects with reachable descendants (1, 0, _), 
            # then among objects at the same level, those with larger area come first (negative area for descending sort)
            return (not is_reachable, not has_reachable_descendants, -object_area)
        
        sorted_children = sorted(children, key=child_priority)
        
        for child in sorted_children:
            if child.id not in visited:
                # Add child before recursing (pre-order DFS)
                dfs_descendants.append(child)
                # Recurse to process this child's descendants
                dfs_collect(child.id)
    
    # Start DFS from parent object (but don't include parent itself)
    dfs_collect(parent_object.id)
    
    # If no descendants, return success immediately
    if len(dfs_descendants) == 0:
        return room.objects, True

    print("dfs_descendants: ", [obj.id for obj in dfs_descendants])
    
    print(f"Found {len(dfs_descendants)} descendants of {parent_object.id} to re-place", file=sys.stderr)
    # Step 3: Remove all descendants from the room copy
    descendant_ids = {obj.id for obj in dfs_descendants}
    room_copy.objects = [o for o in room_copy.objects if o.id not in descendant_ids]
    
    # Step 4: Re-place descendants one by one in DFS order
    # Track which objects have been successfully placed for updating place_id references
    # old_id_to_new_id: Dict[str, str] = {}
    all_successfully_placed = True
    successfully_placed_ids: set = set()  # Track successfully placed object IDs
    skipped_object_ids: set = set()  # Track skipped objects and their descendants
    
    # Scene save directory for physics validation
    scene_save_dir = os.path.join(RESULTS_DIR, layout.id)
    
    for obj in dfs_descendants:
        try:
            # Skip this object if it's a descendant of a skipped object
            if obj.id in skipped_object_ids:
                print(f"Skipping {obj.id} (descendant of skipped object)", file=sys.stderr)
                continue
            
            # Determine the target object ID (parent in support tree)
            target_object_id = obj.place_id
            
            # # If the parent was re-placed, use its new ID
            # if target_object_id in old_id_to_new_id:
            #     target_object_id = old_id_to_new_id[target_object_id]
            
            # Verify that the target object exists in the room
            target_object = next((room_obj for room_obj in room_copy.objects if room_obj.id == target_object_id), None)
            if target_object is None:
                print(f"Target object {target_object_id} not found for {obj.id}", file=sys.stderr)
                all_successfully_placed = False
                break
            
            # Create a fresh copy of the object to place
            obj_to_place = copy.deepcopy(obj)
            # obj_to_place.place_id = target_object_id  # Update place_id in case it changed
            
            # 1. Sample placements on target object
            placements = get_random_placements_on_target_object(
                layout, room_copy, target_object_id, obj_to_place, sample_count=100,
                place_location="top" if obj_to_place.id in reachable_object_ids else "both",
                do_mobile_manipulator_reachability_check=obj_to_place.id in reachable_object_ids,
                enable_reachability_visualization=True,
                layout_file_name=layout_file_name,
            )
            
            if not placements:
                print(f"No valid placements found for {obj.id} on {target_object_id}", file=sys.stderr)
                # Check if all reachable objects have been placed successfully
                if always_return_room and len(reachable_object_ids) == 0:
                    print(f"No reachable objects. Skipping {obj.id} and its descendants", file=sys.stderr)
                    skipped_object_ids.add(obj.id)
                    # Add all descendants to skipped set
                    for desc in dfs_descendants:
                        if _is_descendant(obj.id, desc, object_map):
                            skipped_object_ids.add(desc.id)
                    continue
                if len(reachable_object_ids) > 0:
                    reachable_placed = set(reachable_object_ids) & successfully_placed_ids
                    if reachable_placed == set(reachable_object_ids):
                        # All reachable objects placed, skip this object and descendants
                        print(f"All reachable objects placed successfully. Skipping {obj.id} and its descendants", file=sys.stderr)
                        skipped_object_ids.add(obj.id)
                        # Add all descendants to skipped set
                        for desc in dfs_descendants:
                            if _is_descendant(obj.id, desc, object_map):
                                skipped_object_ids.add(desc.id)
                        continue
                all_successfully_placed = False
                break
            
            # 2. Filter placements by physics critic
            safe_placements = filter_placements_by_physics_critic(
                layout, room_copy, obj_to_place, placements
            )
            
            if not safe_placements:
                print(f"No safe placements found for {obj.id} after physics validation", file=sys.stderr)
                if always_return_room and len(reachable_object_ids) == 0:
                    print(f"No reachable objects. Skipping {obj.id} and its descendants", file=sys.stderr)
                    skipped_object_ids.add(obj.id)
                    # Add all descendants to skipped set
                    for desc in dfs_descendants:
                        if _is_descendant(obj.id, desc, object_map):
                            skipped_object_ids.add(desc.id)
                    continue
                # Check if all reachable objects have been placed successfully
                elif len(reachable_object_ids) > 0:
                    reachable_placed = set(reachable_object_ids) & successfully_placed_ids
                    if reachable_placed == set(reachable_object_ids):
                        # All reachable objects placed, skip this object and descendants
                        print(f"All reachable objects placed successfully. Skipping {obj.id} and its descendants", file=sys.stderr)
                        skipped_object_ids.add(obj.id)
                        # Add all descendants to skipped set
                        for desc in dfs_descendants:
                            if _is_descendant(obj.id, desc, object_map):
                                skipped_object_ids.add(desc.id)
                        continue
                all_successfully_placed = False
                break
            
            # 3. Use the first safe placement
            best_placement = safe_placements[0]
            position_placed = best_placement["position"]
            rotation_placed = best_placement["rotation"]
            
            # 4. Create placed object with new pose
            placed_obj = Object(
                id=obj.id,
                room_id=room.id,
                type=obj.type,
                description=obj.description if hasattr(obj, 'description') else f"Placed {obj.type}",
                position=Point3D(
                    x=position_placed["x"],
                    y=position_placed["y"],
                    z=position_placed["z"]
                ),
                rotation=Euler(
                    x=rotation_placed["x"] * 180 / np.pi,  # Convert from radians to degrees
                    y=rotation_placed["y"] * 180 / np.pi,
                    z=rotation_placed["z"] * 180 / np.pi
                ),
                dimensions=obj.dimensions,
                source=obj.source if hasattr(obj, 'source') else "placement",
                source_id=obj.source_id if hasattr(obj, 'source_id') else obj.id,
                place_id=target_object_id,
                mass=getattr(obj, 'mass', 1.0)
            )
            
            # 5. Add to room copy for subsequent placements
            room_copy.objects.append(placed_obj)
            
            # 6. Validate stability with physics simulation
            if PHYSICS_CRITIC_ENABLED:
                print(f"Validating stability for {obj.id}", file=sys.stderr)
                room_dict_save_path = os.path.join(scene_save_dir, f"{room_copy.id}_correction_temp.json")
                with open(room_dict_save_path, "w") as f:
                    json.dump(asdict(room_copy), f)
                
                result_create = create_single_room_layout_scene_from_room(
                    scene_save_dir,
                    room_dict_save_path
                )
                if not isinstance(result_create, dict) or result_create.get("status") != "success":
                    print(f"Failed to create scene for {obj.id}", file=sys.stderr)
                    all_successfully_placed = False
                    # Clean up temp file
                    if os.path.exists(room_dict_save_path):
                        os.remove(room_dict_save_path)
                    break
                
                result_sim = simulate_the_scene()
                if not isinstance(result_sim, dict) or result_sim.get("status") != "success":
                    print(f"Failed to simulate scene for {obj.id}", file=sys.stderr)
                    all_successfully_placed = False
                    # Clean up temp file
                    if os.path.exists(room_dict_save_path):
                        os.remove(room_dict_save_path)
                    break
                
                # Clean up temp file
                if os.path.exists(room_dict_save_path):
                    os.remove(room_dict_save_path)
                
                unstable_object_ids = result_sim.get("unstable_objects", [])
                if len(unstable_object_ids) > 0:
                    print(f"Object {obj.id} or scene became unstable: {unstable_object_ids}", file=sys.stderr)
                    # Remove unstable objects from room copy
                    room_copy.objects = [o for o in room_copy.objects if o.id not in unstable_object_ids]
                    
                    # Check if all reachable objects have been placed successfully
                    if always_return_room and len(reachable_object_ids) == 0:
                        print(f"No reachable objects. Skipping {obj.id} and its descendants", file=sys.stderr)
                        skipped_object_ids.add(obj.id)
                        # Add all descendants to skipped set
                        for desc in dfs_descendants:
                            if _is_descendant(obj.id, desc, object_map):
                                skipped_object_ids.add(desc.id)
                        continue

                    if len(reachable_object_ids) > 0:
                        reachable_placed = set(reachable_object_ids) & successfully_placed_ids
                        if reachable_placed == set(reachable_object_ids):
                            # All reachable objects placed, skip this object and descendants
                            print(f"All reachable objects placed successfully. Skipping {obj.id} and its descendants due to instability", file=sys.stderr)
                            skipped_object_ids.add(obj.id)
                            # Add all descendants to skipped set
                            for desc in dfs_descendants:
                                if _is_descendant(obj.id, desc, object_map):
                                    skipped_object_ids.add(desc.id)
                            continue
                    
                    all_successfully_placed = False
                    break
            
            # Track the placement for updating child references
            # old_id_to_new_id[obj.id] = placed_obj.id
            successfully_placed_ids.add(obj.id)  # Track successfully placed object
            print(f"Successfully placed {obj.id} on {target_object_id}", file=sys.stderr)
            
        except Exception as e:
            print(f"Exception while placing {obj.id}: {str(e)}", file=sys.stderr)
            all_successfully_placed = False
            break
    
    # Return result
    # Success if either all placed OR if all reachable objects were placed
    if all_successfully_placed or always_return_room:
        print(f"Successfully re-placed all {len(dfs_descendants)} descendants", file=sys.stderr)
        return room_copy.objects, True
    elif len(reachable_object_ids) > 0:
        reachable_placed = set(reachable_object_ids) & successfully_placed_ids
        if reachable_placed == set(reachable_object_ids):
            num_placed = len(successfully_placed_ids)
            num_skipped = len(skipped_object_ids)
            print(f"Successfully placed all {len(reachable_object_ids)} reachable objects ({num_placed} total placed, {num_skipped} skipped)", file=sys.stderr)
            return room_copy.objects, True
    
    print(f"Failed to re-place all descendants", file=sys.stderr)
    return None, False



def object_augmentation_pose_object_tree_with_reach_test(
    layout: FloorPlan, room: Room, object: Object, aug_num: int, aug_name: str,
    reach_threshold: float = 0.4
):

    augmented_layouts_info = []

    # Build children map and BFS order from parent to children (root first) - like type augmentation
    object_map: Dict[str, Object] = {o.id: o for o in room.objects}
    children_map: Dict[str, List[Object]] = {}
    for o in room.objects:
        parent_id = o.place_id
        if isinstance(parent_id, str) and parent_id in object_map:
            children_map.setdefault(parent_id, []).append(o)

    bfs_objects: List[Object] = []
    queue: List[str] = [object.id]
    visited: set = set()
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        if current_id in object_map:
            bfs_objects.append(object_map[current_id])
        for child in children_map.get(current_id, []):
            queue.append(child.id)

    # Use original layout folder for assets during validation
    scene_save_dir = os.path.join(RESULTS_DIR, layout.id)

    os.makedirs(os.path.join(RESULTS_DIR, layout.id, aug_name), exist_ok=True)

    for aug_iter in range(max(0, int(aug_num))):
        # Deep copy the layout and locate the room copy
        layout_copy: FloorPlan = copy.deepcopy(layout)
        layout_copy_id = generate_unique_id(f"{layout.id}_{aug_name}")
        room_copy = next((r for r in layout_copy.rooms if r.id == room.id), None)
        if room_copy is None:
            continue

        # Remove the original root and all its descendants from the room copy - like type augmentation
        ids_to_remove = {obj.id for obj in bfs_objects}
        room_copy.objects = [o for o in room_copy.objects if o.id not in ids_to_remove]

        # Map from original object id -> object copy for tracking
        new_id_map: Dict[str, str] = {}

        # Process each object in BFS order (root first, then children)
        aug_successful = True
        for aug_i, orig_obj in enumerate(bfs_objects):
            # Create a copy of the original object
            aug_obj = copy.deepcopy(orig_obj)
            
            # Update placement location for children to use their parent's new id
            if orig_obj.id == object.id:
                # Root object keeps its original location
                aug_obj.place_id = orig_obj.place_id
            else:
                parent_original_id = orig_obj.place_id
                if not isinstance(parent_original_id, str) or parent_original_id not in new_id_map:
                    aug_successful = False
                    break
                aug_obj.place_id = new_id_map[parent_original_id]

            # Insert into room
            room_copy.objects.insert(0, aug_obj)

            # Sample pose candidates for this object
            pose_candidates = _sample_pose_offsets_condition(
                aug_obj,
                room_copy,
                layout.id,
                layout_copy_id,
                scene_save_dir,
                unit_length=min(aug_obj.dimensions.width, aug_obj.dimensions.length) if aug_obj.place_id == "floor" else 1000.0,
            )
            if len(pose_candidates) == 0:
                print(f"no pose candidates for {aug_obj.id}")
                aug_successful = False
                break

            T_prev = _object_pose_to_matrix(aug_obj)
            pose_accepted = False

            for pose_i, (T_pos, T_rot) in enumerate(pose_candidates):
                T_aug = T_prev @ T_rot @ T_pos

                new_pos, new_rot = _matrix_to_pose(T_aug)
                old_pos, old_rot = _matrix_to_pose(T_prev)

                # Update all descendants inside the copied room
                _update_descendants_transforms(room_copy, aug_obj.id, T_prev, T_aug)
                for obj_copy in room_copy.objects:
                    if obj_copy.id == aug_obj.id:
                        obj_copy.position = new_pos
                        obj_copy.rotation = new_rot
                        break

                # Validate physics stability for the room
                is_stable = _validate_room_stability(scene_save_dir, room_copy, layout_copy_id)
                print(f"aug iter {aug_i} object {aug_obj.id} pose {pose_i} is stable: {is_stable}")

                # Check reachability during augmentation process
                is_reachable = True
                if is_stable:
                    # Get current objects placed so far for reachability test
                    current_objects = [o for o in room_copy.objects if o.place_id != "floor"]
                    is_reachable = _check_reachability(layout_copy, room_copy, current_objects, reach_threshold)
                    print(f"aug iter {aug_i} object {aug_obj.id} pose {pose_i} is reachable: {is_reachable}")

                room_copy_ckpt = copy.deepcopy(room_copy)

                # Revert transforms after validation
                _update_descendants_transforms(room_copy, aug_obj.id, T_aug, T_prev)
                for obj_copy in room_copy.objects:
                    if obj_copy.id == aug_obj.id:
                        obj_copy.position = old_pos
                        obj_copy.rotation = old_rot
                        break

                if is_stable and is_reachable:
                    pose_accepted = True
                    room_copy = room_copy_ckpt
                    break

            if not pose_accepted:
                aug_successful = False
                print(f"pose augmentation failed at object {orig_obj.id}")
                break

            # Record the new id so children can attach to this parent
            new_id_map[orig_obj.id] = aug_obj.id

        print(f"augmentation iteration {aug_iter} is successful: {aug_successful}")
        if not aug_successful:
            continue

        # Save successful augmentation
        for layout_room in layout_copy.rooms:
            if layout_room.id == room.id:
                layout_room.objects = room_copy.objects
                break

        # save the augmented layout
        layout_copy_path = os.path.join(RESULTS_DIR, layout.id, aug_name, f"{layout_copy_id}.json")
        with open(layout_copy_path, "w") as f:
            json.dump(asdict(layout_copy), f)
        augmented_layouts_info.append((layout_copy_id, layout_copy))

    return augmented_layouts_info

def object_augmentation_pose_object_tree_with_reach_test_parallel(
    layout: FloorPlan, room: Room, object: Object, aug_num: int, aug_name: str,
    reach_threshold: float = 0.4, group_size: int = 10
):

    augmented_layouts_info = []

    # Build children map and BFS order from parent to children (root first) - like type augmentation
    object_map: Dict[str, Object] = {o.id: o for o in room.objects}
    children_map: Dict[str, List[Object]] = {}
    for o in room.objects:
        parent_id = o.place_id
        if isinstance(parent_id, str) and parent_id in object_map:
            children_map.setdefault(parent_id, []).append(o)

    bfs_objects: List[Object] = []
    queue: List[str] = [object.id]
    visited: set = set()
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        if current_id in object_map:
            bfs_objects.append(object_map[current_id])
        for child in children_map.get(current_id, []):
            queue.append(child.id)

    # Use original layout folder for assets during validation
    scene_save_dir = os.path.join(RESULTS_DIR, layout.id)

    os.makedirs(os.path.join(RESULTS_DIR, layout.id, aug_name), exist_ok=True)

    # Process augmentations in groups
    for group_start in range(0, aug_num, group_size):
        gc.collect()
        group_end = min(group_start + group_size, aug_num)
        current_group_size = group_end - group_start
        
        print(f"Processing augmentation group {group_start}-{group_end-1} (size: {current_group_size})")
        
        # Initialize group data structures
        group_rooms = []  # List of room copies for each aug in the group
        group_layouts = []  # List of layout copies for each aug in the group
        group_layout_ids = []  # List of layout IDs for each aug in the group
        group_new_id_maps = []  # List of new_id_maps for each aug in the group
        group_successful_mask = []  # Boolean mask for successful augmentations
        
        # Initialize base rooms for each augmentation in the group
        for aug_iter in range(group_start, group_end):
            layout_copy: FloorPlan = copy.deepcopy(layout)
            layout_copy_id = generate_unique_id(f"{layout.id}_{aug_name}_{aug_iter}")
            room_copy = next((r for r in layout_copy.rooms if r.id == room.id), None)
            if room_copy is None:
                group_successful_mask.append(False)
                continue

            # Remove the original root and all its descendants from the room copy
            ids_to_remove = {obj.id for obj in bfs_objects}
            room_copy.objects = [o for o in room_copy.objects if o.id not in ids_to_remove]

            group_rooms.append(room_copy)
            group_layouts.append(layout_copy)
            group_layout_ids.append(layout_copy_id)
            group_new_id_maps.append({})  # Map from original object id -> object copy for tracking
            group_successful_mask.append(True)

        if not any(group_successful_mask):
            print(f"No successful base rooms in group {group_start}-{group_end-1}")
            continue

        # Process each object in the tree (BFS order) across all augmentations
        for aug_i, orig_obj in enumerate(bfs_objects):
            print(f"Processing object {aug_i}: {orig_obj.id} across {sum(group_successful_mask)} remaining augmentations")
            
            # Collect pose candidates for this object across all successful augmentations
            group_pose_candidates = []  # List of pose_candidates for each successful aug
            group_aug_objects = []  # List of created objects for each successful aug
            current_indices = []  # Indices of currently successful augmentations
            
            for i, (room_copy, layout_copy, layout_copy_id, new_id_map, is_successful) in enumerate(tqdm(zip(
                group_rooms, group_layouts, group_layout_ids, group_new_id_maps, group_successful_mask
            ), desc=f"Sampling poses for object {orig_obj.id}", total=len(group_rooms), leave=False)):
                if not is_successful:
                    continue
                    
                # Create a copy of the original object
                aug_obj = copy.deepcopy(orig_obj)
                
                # Update placement location for children to use their parent's new id
                if orig_obj.id == object.id:
                    # Root object keeps its original location
                    aug_obj.place_id = orig_obj.place_id
                else:
                    parent_original_id = orig_obj.place_id
                    if not isinstance(parent_original_id, str) or parent_original_id not in new_id_map:
                        group_successful_mask[i] = False
                        continue
                    aug_obj.place_id = new_id_map[parent_original_id]

                # Insert into room
                room_copy.objects.insert(0, aug_obj)

                # Sample pose candidates for this object
                pose_candidates = _sample_pose_offsets_condition(
                    aug_obj,
                    room_copy,
                    layout.id,
                    layout_copy_id,
                    scene_save_dir,
                    unit_length=min(aug_obj.dimensions.width, aug_obj.dimensions.length) if aug_obj.place_id == "floor" else 1000.0,
                )
                if len(pose_candidates) == 0:
                    print(f"No pose candidates for {aug_obj.id} in aug {i}")
                    group_successful_mask[i] = False
                    continue

                group_pose_candidates.append(pose_candidates)
                group_aug_objects.append(aug_obj)
                current_indices.append(i)

            if not current_indices:
                print(f"No successful augmentations remaining after object {orig_obj.id}")
                break

            # Find the maximum number of pose candidates for this object
            max_pose_candidates = max(len(candidates) for candidates in group_pose_candidates)
            print(f"Object {orig_obj.id}: max {max_pose_candidates} pose candidates across {len(current_indices)} augs")

            # Try poses until all augmentations find a successful pose or exhaust all poses
            object_successful_indices = []
            remaining_aug_indices = list(range(len(current_indices)))  # Track which augs still need poses
            
            for pose_idx in range(max_pose_candidates):
                if not remaining_aug_indices:  # All augmentations found successful poses
                    break
                    
                print(f"  Trying pose {pose_idx} for object {orig_obj.id}, {len(remaining_aug_indices)} augs remaining")
                
                # Apply pose_idx only to augmentations that haven't found a successful pose yet
                test_rooms = []
                test_layout_ids = []
                test_indices_mapping = []  # Maps to remaining_aug_indices
                
                for j in remaining_aug_indices:
                    pose_candidates = group_pose_candidates[j]
                    aug_obj = group_aug_objects[j]
                    global_idx = current_indices[j]
                    
                    if pose_idx >= len(pose_candidates):
                        continue  # This augmentation doesn't have this pose
                    
                    room_copy = group_rooms[global_idx]
                    layout_copy_id = group_layout_ids[global_idx]
                    
                    # Deep copy room for this pose test
                    room_copy_test = copy.deepcopy(room_copy)
                    
                    # Find the object and apply the pose transformation
                    aug_obj_test = next((o for o in room_copy_test.objects if o.id == aug_obj.id), None)
                    if aug_obj_test is None:
                        continue
                    
                    T_prev = _object_pose_to_matrix(aug_obj_test)
                    T_pos, T_rot = pose_candidates[pose_idx]
                    T_aug = T_prev @ T_rot @ T_pos
                    
                    new_pos, new_rot = _matrix_to_pose(T_aug)
                    
                    # Update all descendants inside the copied room
                    _update_descendants_transforms(room_copy_test, aug_obj_test.id, T_prev, T_aug)
                    for obj_copy in room_copy_test.objects:
                        if obj_copy.id == aug_obj_test.id:
                            obj_copy.position = new_pos
                            obj_copy.rotation = new_rot
                            break
                    
                    test_rooms.append(room_copy_test)
                    test_layout_ids.append(layout_copy_id)
                    test_indices_mapping.append(j)
                
                if not test_rooms:
                    continue
                
                # Validate all rooms with this pose using group validation
                print(f"  Validating {len(test_rooms)} rooms with pose {pose_idx} for object {orig_obj.id}")
                group_aug_name = f"{aug_name}_obj{aug_i}_pose{pose_idx}"
                stable_list = _validate_room_stability_groups(scene_save_dir, aug_name, test_rooms, test_layout_ids)
                
                if not isinstance(stable_list, list):
                    print(f"  Group validation failed for pose {pose_idx}")
                    continue
                
                # Process results: accept poses that are both stable and reachable
                newly_successful = []
                for k, (room_copy_test, is_stable) in enumerate(zip(test_rooms, stable_list)):
                    j = test_indices_mapping[k]  # Index into remaining_aug_indices
                    global_idx = current_indices[j]  # Global index into group arrays
                    
                    if is_stable:
                        print(f"  Aug {global_idx} object {orig_obj.id} pose {pose_idx} is stable")
                        
                        # Check reachability at this intermediate stage
                        current_objects = [o for o in room_copy_test.objects if o.place_id != "floor"]
                        is_reachable = _check_reachability(group_layouts[global_idx], room_copy_test, current_objects, reach_threshold)
                        
                        if is_reachable:
                            print(f"  Aug {global_idx} object {orig_obj.id} pose {pose_idx} is ACCEPTED (stable and reachable)")
                            
                            # Update the actual room copy with the successful pose
                            group_rooms[global_idx] = room_copy_test
                            
                            # Record the new id mapping for children
                            aug_obj = group_aug_objects[j]
                            group_new_id_maps[global_idx][orig_obj.id] = aug_obj.id
                            
                            # Mark this augmentation as successful for this object
                            object_successful_indices.append(global_idx)
                            newly_successful.append(j)  # Remove from remaining list
                        else:
                            print(f"  Aug {global_idx} object {orig_obj.id} pose {pose_idx} is stable but not reachable - will try next pose")
                    else:
                        print(f"  Aug {global_idx} object {orig_obj.id} pose {pose_idx} is not stable - will try next pose")
                
                # Remove successfully placed augmentations from remaining list
                for j in newly_successful:
                    remaining_aug_indices.remove(j)

            # Update success mask: only keep augmentations that found a working pose for this object
            for i in range(len(group_successful_mask)):
                if group_successful_mask[i] and i not in object_successful_indices:
                    group_successful_mask[i] = False
                    print(f"Aug {i} failed at object {orig_obj.id}")

            if not any(group_successful_mask):
                print(f"No augmentations remain successful after object {orig_obj.id}")
                break

        # Save successful augmentations (already filtered for stability and reachability)
        for i, is_successful in enumerate(group_successful_mask):
            if not is_successful:
                continue
                
            room_copy = group_rooms[i]
            layout_copy = group_layouts[i]
            layout_copy_id = group_layout_ids[i]
            
            print(f"Aug {i} completed successfully - saving")
            
            # Save successful augmentation
            for layout_room in layout_copy.rooms:
                if layout_room.id == room.id:
                    layout_room.objects = room_copy.objects
                    break

            # save the augmented layout
            layout_copy_path = os.path.join(RESULTS_DIR, layout.id, aug_name, f"{layout_copy_id}.json")
            with open(layout_copy_path, "w") as f:
                json.dump(asdict(layout_copy), f)
            augmented_layouts_info.append((layout_copy_id, layout_copy))

        print(f"Group {group_start}-{group_end-1} completed: {len(augmented_layouts_info)} total successful so far")

    print(f"Parallel augmentation completed: {len(augmented_layouts_info)} successful layouts generated")
    return augmented_layouts_info

from models import Dimensions

def augment_description(object_type: str, object_description: str, aug_num: int, dimensions: Dimensions = None, return_object_type: bool = False) -> Union[List[str], Tuple[str, List[str]]]:
	if aug_num <= 0:
		if return_object_type:
			return object_type, []
		return []

	num_batches = int(ceil(aug_num / 10))
	all_descriptions: List[str] = []
	returned_object_type = object_type  # Default to input object_type

	for batch_idx in range(num_batches):
		# Build dimension information if provided
		dimension_info = ""
		if dimensions is not None:
			dimension_info = f"""
- approximate_dimensions: width={dimensions.width:.2f}m, length={dimensions.length:.2f}m, height={dimensions.height:.2f}m"""

		prompt = f"""
You are generating diverse but physically plausible descriptions for a single object category.

Inputs:
- object_description: "{object_description}"

Task:
Given the object description:

1. Infer the object type from the object description.
2. Generate 10 augmented descriptions for the object type, and you need to:
Vary the shape, geometry, color, style, materials, and finish to ensure the diversity of the object


Output JSON only, no extra text, in a single object with this schema:
```json
{{
  "object_type": "the object type inferred from the object description",
  "base_description": "{object_description}",
  "augmented_descriptions": [
    "...10 items..."
  ]
}}

Notes:

You need to follow the Object description rules to generate the augmented descriptions.
- Start with the full name of the object (e.g. "A wooden office chair", "A glass coffee table", "A ceramic table lamp")
- Focus on the physical characteristics of the object, including object shape 
(important since it will be used to generate the 3D model, so please include the shape of the object in the description clearly, 
e.g. rectangular, square, circular, oval, triangular, cubic, cylindrical, spherical, conical, curved, angular, straight-edged,
and elongated, oblong, squat, slender, chunky, low-slung, compact, sprawling, towering, leggy, bulky, streamlined, narrow, wide, deep, shallow, stocky, lanky, flat, oversized), 
color, finish, style, material, etc.
- If you are describing a wall mounted **thin** object including but not limited to paintings, posters, tv screens, clocks, mirrors, artworks, etc., you must add adjective "single piece of", "thin" and "upright" in the description.
e.g. "A single piece of thin and upright painting with ...", "A single piece of thin and upright tv screen with ...", "A single piece of thin and upright clock with ...", etc.
- Use your imagination to come up with at least one unique physical feature of the object. This should distinguish the object from other object with the same type.
- Do not include the size of the object in the description. e.g. width is xx cm, height is xx cm, etc. are all not allowed in object description.
- Do not include the usage of the object in the description, including what is it for, where to place the object, etc.
The presence of other object names in the description would lead to the failure of 3D model generation. 
(e.g. if you describe a "cushion" on the sofa, you can't say "cushion on the sofa", or "cushion colored with sofa's color", etc. in the description. Focus only on the object itself and its physical characteristics.)
- Each description should be 1 concise sentence (10-25 words).

```
"""

		response = call_vlm(
			vlm_type="openai",
			model="openai/gpt-oss-120b",
			max_tokens=5000,
			temperature=1.0,
			messages=[
				{
					"role": "user",
					"content": prompt
				}
			]
		)

		try:
			response_text = response.content[0].text.strip()
			extracted = extract_json_from_response(response_text) or response_text
			data = json.loads(extracted)
			descriptions = data.get("augmented_descriptions") or data.get("descriptions") or []
			
			# Extract object_type if return_object_type is True and this is the first batch
			if return_object_type and batch_idx == 0:
				extracted_object_type = data.get("object_type")
				if extracted_object_type and isinstance(extracted_object_type, str):
					returned_object_type = extracted_object_type.strip()
			
			if isinstance(descriptions, list):
                # shuffle the descriptions
				random.shuffle(descriptions)
				for d in descriptions:
					if isinstance(d, (str, int, float)):
						all_descriptions.append(str(d).strip())
		except Exception:
			# Skip this batch if parsing failed
			pass

	# Deduplicate while preserving order
	seen = set()
	unique_descriptions: List[str] = []
	for d in all_descriptions:
		if not d:
			continue
		key = d.lower()
		if key in seen:
			continue
		seen.add(key)
		unique_descriptions.append(d)

	# Ensure exactly aug_num items (truncate or cycle if needed)
	final_descriptions = []
	if len(unique_descriptions) >= aug_num:
		final_descriptions = unique_descriptions[:aug_num]
	elif len(unique_descriptions) > 0:
		final_descriptions = unique_descriptions[:]
		while len(final_descriptions) < aug_num:
			final_descriptions.append(unique_descriptions[len(final_descriptions) % len(unique_descriptions)])
	else:
		final_descriptions = []

	if return_object_type:
		return returned_object_type, final_descriptions
	else:
		return final_descriptions

def object_augmentation_type_single_object_leaf(layout: FloorPlan, room: Room, object: Object, aug_num: int, aug_name: str):
    object_type = object.type
    object_description = object.description

    augmented_descriptions = augment_description(object_type, object_description, aug_num)
    print(f"augmented descriptions: ")
    for d in augmented_descriptions:
        print(f"- {d}")

    augmented_layouts_info = []
    scene_save_dir = os.path.join(RESULTS_DIR, layout.id)

    for aug_iter in range(max(0, int(aug_num))):
        # Deep copy layout
        layout_copy: FloorPlan = copy.deepcopy(layout)
        layout_copy_id = generate_unique_id(f"{layout.id}_{aug_name}")
        # Find the corresponding room and object in the copied layout
        room_copy = next((r for r in layout_copy.rooms if r.id == room.id), None)
        if room_copy is None:
            assert False

        object_info_dict = {
            object_type: {
                "description": augmented_descriptions[aug_iter],
                "location": object.place_id,
                "size": [object.dimensions.width * 100, object.dimensions.length * 100, object.dimensions.height * 100],
                "quantity": 1,
                "variance_type": "same"
            }
        }

        selected_objects, _ = select_objects(object_info_dict, room_copy, None, layout_copy)
        aug_obj = selected_objects[0]
        aug_obj.position = object.position
        aug_obj.rotation = object.rotation

        aug_obj_parent_id = aug_obj.place_id
        aug_obj_parent = next((o for o in room_copy.objects if o.id == aug_obj_parent_id), None)
        if aug_obj_parent is None:
            assert False

        # delete the object from the room_copy
        room_copy.objects.remove(object)

        # insert the aug_obj to the room_copy
        room_copy.objects.insert(0, aug_obj)

        pose_candidates = _sample_pose_offsets_condition(aug_obj, room_copy, layout.id, layout_copy_id, scene_save_dir, unit_length=1000.0)
        if len(pose_candidates) == 0:
            safe_aug = False
            print(f"no pose candidates for {aug_obj.id}")
            break

        # Previous transform of current object in the evolving room_copy state
        T_prev = _object_pose_to_matrix(aug_obj)

        accepted = False

        for pose_i, (T_pos, T_rot) in enumerate(pose_candidates):
            T_aug = T_prev @ T_rot @ T_pos

            # Apply transform to aug_obj
            new_pos, new_rot = _matrix_to_pose(T_aug)
            old_pos, old_rot = _matrix_to_pose(T_prev)


            for obj in room_copy.objects:
                if obj.id == aug_obj.id:
                    obj.position = new_pos
                    obj.rotation = new_rot
                    break

            # Validate physics stability for the room
            is_stable = _validate_room_stability(scene_save_dir, room_copy, layout_copy_id)
            print(f"aug {aug_iter}th object {aug_obj.id} pose {pose_i} is stable: {is_stable}")

            room_copy_ckpt = copy.deepcopy(room_copy)

            # Revert transforms after validation
            for obj in room_copy.objects:
                if obj.id == aug_obj.id:
                    obj.position = old_pos
                    obj.rotation = old_rot
                    break
            
            # pdb.set_trace()
            if is_stable:
                accepted = True
                room_copy = room_copy_ckpt
                
                break
        
        print(f"augmentation iteration {aug_iter} is stable: {accepted}")
        if not accepted:
            continue

        # update the objects in room of layout_copy
        for layout_room in layout_copy.rooms:
            if layout_room.id == room.id:
                # Replace all objects with the placed objects (includes existing + new)
                layout_room.objects = room_copy_ckpt.objects
                break

        # save the augmented layout
        layout_copy_path = os.path.join(RESULTS_DIR, layout.id, f"{layout_copy_id}.json")
        with open(layout_copy_path, "w") as f:
            json.dump(asdict(layout_copy), f)
        augmented_layouts_info.append((layout_copy_id, layout_copy))
    
    return augmented_layouts_info

def object_augmentation_type_object_tree(layout: FloorPlan, room: Room, object: Object, aug_num: int, aug_name: str):
    # Build children map and BFS order from parent to children (root first)
    object_map: Dict[str, Object] = {o.id: o for o in room.objects}
    children_map: Dict[str, List[Object]] = {}
    for o in room.objects:
        parent_id = o.place_id
        if isinstance(parent_id, str) and parent_id in object_map:
            children_map.setdefault(parent_id, []).append(o)

    bfs_objects: List[Object] = []
    queue: List[str] = [object.id]
    visited: set = set()
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        if current_id in object_map:
            bfs_objects.append(object_map[current_id])
        for child in children_map.get(current_id, []):
            queue.append(child.id)

    # Precompute augmented descriptions for each object in the tree
    aug_descriptions_by_id: Dict[str, List[str]] = {}
    for obj in bfs_objects:
        aug_descriptions_by_id[obj.id] = augment_description(obj.type, obj.description, aug_num)

    augmented_layouts_info = []
    scene_save_dir = os.path.join(RESULTS_DIR, layout.id)

    for aug_iter in range(max(0, int(aug_num))):
        # Deep copy the layout and locate the room copy
        layout_copy: FloorPlan = copy.deepcopy(layout)
        layout_copy_id = generate_unique_id(f"{layout.id}_{aug_name}")
        room_copy = next((r for r in layout_copy.rooms if r.id == room.id), None)
        if room_copy is None:
            assert False

        # Remove the original root and all its descendants from the room copy
        ids_to_remove = {obj.id for obj in bfs_objects}
        room_copy.objects = [o for o in room_copy.objects if o.id not in ids_to_remove]

        # Map from original object id -> newly created object's id
        new_id_map: Dict[str, str] = {}

        safe_aug = True
        for orig_obj in bfs_objects:
            # Determine placement location: root keeps its original location; children use their parent's new id
            if orig_obj.id == object.id:
                location = orig_obj.place_id
            else:
                parent_original_id = orig_obj.place_id
                if not isinstance(parent_original_id, str) or parent_original_id not in new_id_map:
                    safe_aug = False
                    break
                location = new_id_map[parent_original_id]

            # Select the new object instance based on augmented description
            obj_descs = aug_descriptions_by_id.get(orig_obj.id, [])
            if len(obj_descs) == 0:
                safe_aug = False
                break

            object_info_dict = {
                orig_obj.type: {
                    "description": obj_descs[aug_iter % len(obj_descs)],
                    "location": location,
                    "size": [orig_obj.dimensions.width * 100, orig_obj.dimensions.length * 100, orig_obj.dimensions.height * 100],
                    "quantity": 1,
                    "variance_type": "same",
                }
            }

            selected_objects, _ = select_objects(object_info_dict, room_copy, None, layout_copy)
            if not selected_objects:
                safe_aug = False
                break
            aug_obj = selected_objects[0]

            # Initialize pose to match the original object
            aug_obj.position = orig_obj.position
            aug_obj.rotation = orig_obj.rotation

            # Insert into room
            room_copy.objects.insert(0, aug_obj)

            # Pose sampling and stability validation
            pose_candidates = _sample_pose_offsets_condition(
                aug_obj,
                room_copy,
                layout.id,
                layout_copy_id,
                scene_save_dir,
                unit_length=min(aug_obj.dimensions.width, aug_obj.dimensions.length) if location == "floor" else 1000.0,
            )
            if len(pose_candidates) == 0:
                print(f"no pose candidates for {aug_obj.id}")
                safe_aug = False
                break

            T_prev = _object_pose_to_matrix(aug_obj)
            accepted_current = False

            for pose_i, (T_pos, T_rot) in enumerate(pose_candidates):
                T_aug = T_prev @ T_rot @ T_pos

                new_pos, new_rot = _matrix_to_pose(T_aug)
                old_pos, old_rot = _matrix_to_pose(T_prev)

                for obj_copy in room_copy.objects:
                    if obj_copy.id == aug_obj.id:
                        obj_copy.position = new_pos
                        obj_copy.rotation = new_rot
                        break

                is_stable = _validate_room_stability(scene_save_dir, room_copy, layout_copy_id)
                print(f"aug iter {aug_iter} object {aug_obj.id} pose {pose_i} is stable: {is_stable}")

                room_copy_ckpt = copy.deepcopy(room_copy)

                # Revert
                for obj_copy in room_copy.objects:
                    if obj_copy.id == aug_obj.id:
                        obj_copy.position = old_pos
                        obj_copy.rotation = old_rot
                        break

                # pdb.set_trace()
                if is_stable:
                    accepted_current = True
                    room_copy = room_copy_ckpt
                    break

            if not accepted_current:
                safe_aug = False
                print(f"aug failed at object {orig_obj.id}")
                break

            # Record the new id so children can attach to this parent
            new_id_map[orig_obj.id] = aug_obj.id

        print(f"augmentation iteration {aug_iter} is stable: {safe_aug}")
        if not safe_aug:
            continue

        # Save successful augmented layout
        for layout_room in layout_copy.rooms:
            if layout_room.id == room.id:
                layout_room.objects = room_copy.objects
                break

        layout_copy_path = os.path.join(RESULTS_DIR, layout.id, f"{layout_copy_id}.json")
        with open(layout_copy_path, "w") as f:
            json.dump(asdict(layout_copy), f)
        augmented_layouts_info.append((layout_copy_id, layout_copy))

    return augmented_layouts_info

def object_augmentation_type_object_tree_with_pose_augmentation(layout: FloorPlan, room: Room, 
    object: Object, aug_type_num: int, aug_pose_num: int, aug_name: str):
    """
    Combined type and pose augmentation for object trees.
    First augments object types, then for each successful type augmentation, performs pose augmentation.
    
    Args:
        layout: The floor plan containing the room
        room: The room containing the object tree
        object: The root object of the tree to augment
        aug_type_num: Number of type augmentations to perform
        aug_pose_num: Number of pose augmentations to perform for each type augmentation
        aug_name: Name prefix for the augmented layouts
    
    Returns:
        List of (layout_id, layout) tuples for all successful augmentations
    """
    # Build children map and BFS order from parent to children (root first)
    object_map: Dict[str, Object] = {o.id: o for o in room.objects}
    children_map: Dict[str, List[Object]] = {}
    for o in room.objects:
        parent_id = o.place_id
        if isinstance(parent_id, str) and parent_id in object_map:
            children_map.setdefault(parent_id, []).append(o)

    bfs_objects: List[Object] = []
    queue: List[str] = [object.id]
    visited: set = set()
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        if current_id in object_map:
            bfs_objects.append(object_map[current_id])
        for child in children_map.get(current_id, []):
            queue.append(child.id)

    # Precompute augmented descriptions for each object in the tree
    aug_descriptions_by_id: Dict[str, List[str]] = {}
    for obj in bfs_objects:
        aug_descriptions_by_id[obj.id] = augment_description(obj.type, obj.description, aug_type_num)

    augmented_layouts_info = []
    scene_save_dir = os.path.join(RESULTS_DIR, layout.id)

    os.makedirs(os.path.join(RESULTS_DIR, layout.id, aug_name), exist_ok=True)

    # Outer loop: Type augmentation
    for type_aug_iter in range(max(0, int(aug_type_num))):
        print(f"Starting type augmentation iteration {type_aug_iter}")
        
        # Deep copy the layout and locate the room copy
        type_layout_copy: FloorPlan = copy.deepcopy(layout)
        type_room_copy = next((r for r in type_layout_copy.rooms if r.id == room.id), None)
        if type_room_copy is None:
            continue

        # Remove the original root and all its descendants from the room copy
        ids_to_remove = {obj.id for obj in bfs_objects}
        type_room_copy.objects = [o for o in type_room_copy.objects if o.id not in ids_to_remove]

        # Map from original object id -> newly created object's id
        new_id_map: Dict[str, str] = {}
        new_objects: List[Object] = []

        # Create new objects with augmented types (BFS order ensures parents are created before children)
        type_aug_successful = True
        for orig_obj in bfs_objects:
            # Determine placement location: root keeps its original location; children use their parent's new id
            if orig_obj.id == object.id:
                location = orig_obj.place_id
            else:
                parent_original_id = orig_obj.place_id
                if not isinstance(parent_original_id, str) or parent_original_id not in new_id_map:
                    type_aug_successful = False
                    break
                location = new_id_map[parent_original_id]

            # Select the new object instance based on augmented description
            obj_descs = aug_descriptions_by_id.get(orig_obj.id, [])
            if len(obj_descs) == 0:
                type_aug_successful = False
                break

            object_info_dict = {
                orig_obj.type: {
                    "description": obj_descs[type_aug_iter % len(obj_descs)],
                    "location": location,
                    "size": [orig_obj.dimensions.width * 100, orig_obj.dimensions.length * 100, orig_obj.dimensions.height * 100],
                    "quantity": 1,
                    "variance_type": "same",
                }
            }

            selected_objects, _ = select_objects(object_info_dict, type_room_copy, None, type_layout_copy)
            if not selected_objects:
                type_aug_successful = False
                break
            type_aug_obj = selected_objects[0]

            # Initialize pose to match the original object
            type_aug_obj.position = orig_obj.position
            type_aug_obj.rotation = orig_obj.rotation

            # Insert into room and track new id
            type_room_copy.objects.insert(0, type_aug_obj)
            new_id_map[orig_obj.id] = type_aug_obj.id
            new_objects.append(type_aug_obj)

        if not type_aug_successful:
            print(f"Type augmentation {type_aug_iter} failed during object creation")
            continue

        # Now perform pose augmentation on the type-augmented objects
        # Inner loop: Pose augmentation
        for pose_aug_iter in range(max(0, int(aug_pose_num))):
            print(f"  Starting pose augmentation iteration {pose_aug_iter} for type aug {type_aug_iter}")
            
            # Deep copy the type-augmented layout
            pose_layout_copy: FloorPlan = copy.deepcopy(type_layout_copy)
            pose_layout_copy_id = generate_unique_id(f"{layout.id}_{aug_name}_type{type_aug_iter}_pose{pose_aug_iter}")
            
            pose_room_copy = next((r for r in pose_layout_copy.rooms if r.id == room.id), None)
            if pose_room_copy is None:
                continue

            # Find the corresponding new objects in the pose layout copy
            pose_new_objects: List[Object] = []
            for new_obj in new_objects:
                pose_obj = next((o for o in pose_room_copy.objects if o.id == new_obj.id), None)
                if pose_obj is None:
                    break
                pose_new_objects.append(pose_obj)
            
            if len(pose_new_objects) != len(new_objects):
                continue

            # Perform pose augmentation on each object in the tree (reverse order for dependencies)
            pose_aug_successful = True
            for pose_obj_idx, pose_aug_obj in enumerate(reversed(pose_new_objects)):
                pose_candidates = _sample_pose_offsets_condition(
                    pose_aug_obj,
                    pose_room_copy,
                    layout.id,
                    pose_layout_copy_id,
                    scene_save_dir,
                    unit_length=min(pose_aug_obj.dimensions.width, pose_aug_obj.dimensions.length) if pose_aug_obj.place_id == "floor" else 1000.0,
                )
                if len(pose_candidates) == 0:
                    print(f"    No pose candidates for {pose_aug_obj.id}")
                    pose_aug_successful = False
                    break

                T_prev = _object_pose_to_matrix(pose_aug_obj)
                pose_accepted = False

                for pose_i, (T_pos, T_rot) in enumerate(pose_candidates):
                    T_aug = T_prev @ T_rot @ T_pos

                    new_pos, new_rot = _matrix_to_pose(T_aug)
                    old_pos, old_rot = _matrix_to_pose(T_prev)

                    # Update all descendants inside the copied room
                    _update_descendants_transforms(pose_room_copy, pose_aug_obj.id, T_prev, T_aug)
                    for obj_copy in pose_room_copy.objects:
                        if obj_copy.id == pose_aug_obj.id:
                            obj_copy.position = new_pos
                            obj_copy.rotation = new_rot
                            break

                    # Validate physics stability for the room
                    is_stable = _validate_room_stability(scene_save_dir, pose_room_copy, pose_layout_copy_id)
                    print(f"    Aug type {type_aug_iter} pose {pose_aug_iter} object {pose_aug_obj.id} pose {pose_i} is stable: {is_stable}")

                    pose_room_copy_ckpt = copy.deepcopy(pose_room_copy)

                    # Revert transforms after validation
                    _update_descendants_transforms(pose_room_copy, pose_aug_obj.id, T_aug, T_prev)
                    for obj_copy in pose_room_copy.objects:
                        if obj_copy.id == pose_aug_obj.id:
                            obj_copy.position = old_pos
                            obj_copy.rotation = old_rot
                            break

                    if is_stable:
                        pose_accepted = True
                        pose_room_copy = pose_room_copy_ckpt
                        break

                if not pose_accepted:
                    pose_aug_successful = False
                    print(f"    Pose augmentation failed at object {pose_aug_obj.id}")
                    break

            print(f"  Type {type_aug_iter} pose {pose_aug_iter} augmentation is stable: {pose_aug_successful}")
            if not pose_aug_successful:
                continue

            # Save successful combined augmentation
            for layout_room in pose_layout_copy.rooms:
                if layout_room.id == room.id:
                    layout_room.objects = pose_room_copy.objects
                    break

            layout_copy_path = os.path.join(RESULTS_DIR, layout.id, aug_name, f"{pose_layout_copy_id}.json")
            with open(layout_copy_path, "w") as f:
                json.dump(asdict(pose_layout_copy), f)
            augmented_layouts_info.append((pose_layout_copy_id, pose_layout_copy))

    print(f"Combined type+pose augmentation generated {len(augmented_layouts_info)} successful layouts")
    return augmented_layouts_info

def generate_type_augmentation_candidates(
    layout: FloorPlan, room: Room, object: Object, aug_type_num: int, aug_name: str
) -> List[Tuple[str, FloorPlan, Room, List[Object]]]:
    """
    Generate k sets of type augmentation candidates for an object tree.
    Each set contains objects with the same type but different geometry and textures.
    
    Args:
        layout: The floor plan containing the room
        room: The room containing the object tree
        object: The root object of the tree to augment
        aug_type_num: Number of type augmentation sets to generate
        aug_name: Name prefix for the augmented layouts
    
    Returns:
        List of (layout_id, layout_copy, room_copy, new_objects) tuples for each type augmentation set
    """
    # Build children map and BFS order from parent to children (root first)
    object_map: Dict[str, Object] = {o.id: o for o in room.objects}
    children_map: Dict[str, List[Object]] = {}
    for o in room.objects:
        parent_id = o.place_id
        if isinstance(parent_id, str) and parent_id in object_map:
            children_map.setdefault(parent_id, []).append(o)

    bfs_objects: List[Object] = []
    queue: List[str] = [object.id]
    visited: set = set()
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        if current_id in object_map:
            bfs_objects.append(object_map[current_id])
        for child in children_map.get(current_id, []):
            queue.append(child.id)

    # Precompute augmented descriptions for each object in the tree
    aug_descriptions_by_id: Dict[str, List[str]] = {}
    aug_object_types_by_id: Dict[str, str] = {}
    for obj in bfs_objects:
        object_type, descriptions = augment_description(obj.type, obj.description, aug_type_num, obj.dimensions, return_object_type=True)
        aug_descriptions_by_id[obj.id] = descriptions
        aug_object_types_by_id[obj.id] = object_type

    type_augmentation_candidates = []

    os.makedirs(os.path.join(RESULTS_DIR, layout.id, aug_name), exist_ok=True)

    # Generate type augmentation candidates
    for type_aug_iter in range(max(0, int(aug_type_num))):
        print(f"Generating type augmentation candidate {type_aug_iter}")
        
        # Deep copy the layout and locate the room copy
        type_layout_copy: FloorPlan = copy.deepcopy(layout)
        type_layout_copy_id = f"{layout.id}_{aug_name}_type{type_aug_iter}"
        type_room_copy = next((r for r in type_layout_copy.rooms if r.id == room.id), None)
        if type_room_copy is None:
            continue

        # Remove the original root and all its descendants from the room copy
        ids_to_remove = {obj.id for obj in bfs_objects}
        type_room_copy.objects = [o for o in type_room_copy.objects if o.id not in ids_to_remove]

        # Map from original object id -> newly created object's id
        new_id_map: Dict[str, str] = {}
        new_objects: List[Object] = []

        # Create new objects with augmented types (BFS order ensures parents are created before children)
        type_aug_successful = True
        for orig_obj in bfs_objects:
            # Determine placement location: root keeps its original location; children use their parent's new id
            if orig_obj.id == object.id:
                location = orig_obj.place_id
            else:
                parent_original_id = orig_obj.place_id
                if not isinstance(parent_original_id, str) or parent_original_id not in new_id_map:
                    type_aug_successful = False
                    break
                location = new_id_map[parent_original_id]

            # Select the new object instance based on augmented description
            obj_descs = aug_descriptions_by_id.get(orig_obj.id, [])
            if len(obj_descs) == 0:
                type_aug_successful = False
                break

            # Get the augmented object type
            augmented_object_type = aug_object_types_by_id.get(orig_obj.id, orig_obj.type)

            object_info_dict = {
                augmented_object_type: {
                    "description": obj_descs[type_aug_iter % len(obj_descs)],
                    "location": location,
                    "size": [orig_obj.dimensions.width * 100, orig_obj.dimensions.length * 100, orig_obj.dimensions.height * 100],
                    "quantity": 1,
                    "variance_type": "same",
                }
            }

            selected_objects, _ = select_objects(object_info_dict, type_room_copy, None, type_layout_copy)
            if not selected_objects:
                type_aug_successful = False
                break
            type_aug_obj = selected_objects[0]

            # Initialize pose to match the original object
            type_aug_obj.position = orig_obj.position
            type_aug_obj.rotation = orig_obj.rotation

            # Insert into room and track new id
            type_room_copy.objects.insert(0, type_aug_obj)
            new_id_map[orig_obj.id] = type_aug_obj.id
            new_objects.append(type_aug_obj)

        if type_aug_successful:
            print(f"Type augmentation candidate {type_aug_iter} generated successfully with {len(new_objects)} objects")
            
            # Create inverse mapping: new_id -> old_id
            new_id_to_old_id_map = {new_id: old_id for old_id, new_id in new_id_map.items()}
            
            # Save the type augmentation candidate
            candidate_path = os.path.join(RESULTS_DIR, layout.id, aug_name, f"{type_layout_copy_id}_type_candidate.json")
            candidate_data = {
                "layout_id": type_layout_copy_id,
                "layout": asdict(type_layout_copy),
                "room_id": type_room_copy.id,
                "new_object_ids": [obj.id for obj in new_objects],
                "old_id_to_new_id_map": new_id_map,
                "new_id_to_old_id_map": new_id_to_old_id_map
            }
            with open(candidate_path, "w") as f:
                json.dump(candidate_data, f)
            
            type_augmentation_candidates.append((type_layout_copy_id, type_layout_copy, type_room_copy, new_objects))
        else:
            print(f"Type augmentation candidate {type_aug_iter} failed during object creation")

    return type_augmentation_candidates

def generate_type_augmentation_candidates_mixed(
    layout: FloorPlan, room: Room, object: Object, aug_type_num: int, aug_name: str
) -> List[Tuple[str, FloorPlan, Room, List[Object]]]:
    """
    Generate all possible combinations of type augmentation candidates for an object tree.
    Unlike generate_type_augmentation_candidates which creates aug_type_num layouts where all objects
    use the same description index, this function creates aug_type_num^(num_objects) layouts by
    generating all possible combinations where each object can use any of its aug_type_num descriptions.
    
    Args:
        layout: The floor plan containing the room
        room: The room containing the object tree
        object: The root object of the tree to augment
        aug_type_num: Number of type augmentation variations per object (results in aug_type_num^num_objects total layouts)
        aug_name: Name prefix for the augmented layouts
    
    Returns:
        List of (layout_id, layout_copy, room_copy, new_objects) tuples for each type augmentation combination
    """
    # Build children map and BFS order from parent to children (root first)
    object_map: Dict[str, Object] = {o.id: o for o in room.objects}
    children_map: Dict[str, List[Object]] = {}
    for o in room.objects:
        parent_id = o.place_id
        if isinstance(parent_id, str) and parent_id in object_map:
            children_map.setdefault(parent_id, []).append(o)

    bfs_objects: List[Object] = []
    queue: List[str] = [object.id]
    visited: set = set()
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        if current_id in object_map:
            bfs_objects.append(object_map[current_id])
        for child in children_map.get(current_id, []):
            queue.append(child.id)

    # Precompute augmented descriptions for each object in the tree
    aug_descriptions_by_id: Dict[str, List[str]] = {}
    aug_object_types_by_id: Dict[str, str] = {}
    for obj in bfs_objects:
        object_type, descriptions = augment_description(obj.type, obj.description, aug_type_num, obj.dimensions, return_object_type=True)
        aug_descriptions_by_id[obj.id] = descriptions
        aug_object_types_by_id[obj.id] = object_type

        print("--------------------------------")
        print("original object:")
        print(obj.id, obj.type, obj.description, obj.dimensions)
        print("--------------------------------")
        print("augmented object:")
        print(object_type, descriptions)
        print("--------------------------------")

    os.makedirs(os.path.join(RESULTS_DIR, layout.id, aug_name), exist_ok=True)

    # STEP 1: Pre-generate all object variations for each object (m variations per object)
    print(f"Pre-generating {aug_type_num} variations for each of {len(bfs_objects)} objects")
    
    # Store all generated object variations: object_id -> [list of m Object variations]
    object_variations_by_id: Dict[str, List[Object]] = {}
    
    # Create a temporary layout and room for object generation
    temp_layout_copy: FloorPlan = copy.deepcopy(layout)
    temp_room_copy = next((r for r in temp_layout_copy.rooms if r.id == room.id), None)
    if temp_room_copy is None:
        return []
    
    # Remove the original objects from temp room for clean generation
    ids_to_remove = {obj.id for obj in bfs_objects}
    temp_room_copy.objects = [o for o in temp_room_copy.objects if o.id not in ids_to_remove]
    
    # Helper function to generate variations for a single object
    def generate_variations_for_object(orig_obj, aug_descriptions_by_id, aug_object_types_by_id, aug_type_num, temp_room_copy, temp_layout_copy):
        """Generate all variations for a single object."""
        print(f"Generating {aug_type_num} variations for object {orig_obj.id} ({orig_obj.type})")
        
        obj_variations = []
        obj_descs = aug_descriptions_by_id.get(orig_obj.id, [])
        augmented_object_type = aug_object_types_by_id.get(orig_obj.id, orig_obj.type)
        
        for desc_idx in range(aug_type_num):
            if desc_idx >= len(obj_descs):
                # If we don't have enough descriptions, cycle through them
                selected_description = obj_descs[desc_idx % len(obj_descs)]
            else:
                selected_description = obj_descs[desc_idx]
            
            # Use a placeholder location for generation (will be updated during composition)
            object_info_dict = {
                orig_obj.type: {
                    "description": selected_description,
                    "location": orig_obj.place_id,  # Placeholder location
                    "size": [orig_obj.dimensions.width * 100, orig_obj.dimensions.length * 100, orig_obj.dimensions.height * 100],
                    "quantity": 1,
                    "variance_type": "same",
                    "limit_size": True
                }
            }
            
            selected_objects, _ = select_objects(object_info_dict, temp_room_copy, None, temp_layout_copy)
            if selected_objects:
                variation_obj = selected_objects[0]
                # Initialize pose to match the original object
                variation_obj.position = orig_obj.position
                variation_obj.rotation = orig_obj.rotation
                # Store the original object ID for later reference
                variation_obj._original_id = orig_obj.id
                obj_variations.append(variation_obj)
            else:
                print(f"Warning: Failed to generate variation {desc_idx} for object {orig_obj.id}")
        
        return orig_obj.id, obj_variations
    
    # Parallelize object variation generation using ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_obj_id = {}
        for orig_obj in bfs_objects:
            future = executor.submit(
                generate_variations_for_object,
                orig_obj,
                aug_descriptions_by_id,
                aug_object_types_by_id,
                aug_type_num,
                temp_room_copy,
                temp_layout_copy
            )
            future_to_obj_id[future] = orig_obj.id
        
        # Collect results as they complete
        for future in as_completed(future_to_obj_id):
            obj_id = future_to_obj_id[future]
            try:
                result_obj_id, obj_variations = future.result()
                
                if len(obj_variations) > 0:
                    object_variations_by_id[result_obj_id] = obj_variations
                    print(f"Generated {len(obj_variations)} variations for object {result_obj_id}")
                else:
                    print(f"Error: No variations generated for object {result_obj_id}")
                    return []  # Cannot proceed without variations for any object
            except Exception as exc:
                print(f"Object {obj_id} generated an exception: {exc}")
                return []

    # STEP 2: Generate all possible combinations of descriptions for each object
    # Each object has aug_type_num descriptions, so we get aug_type_num^(num_objects) combinations
    description_indices_lists = [list(range(len(aug_descriptions_by_id[obj.id]))) for obj in bfs_objects]
    all_combinations = list(itertools.product(*description_indices_lists))
    if len(all_combinations) > 64:
        random.shuffle(all_combinations)
        all_combinations = all_combinations[:64]
    
    print(f"Composing {len(all_combinations)} mixed type augmentation candidates ({aug_type_num}^{len(bfs_objects)} combinations)")
    
    type_augmentation_candidates = []
    
    # STEP 3: Compose layouts by selecting appropriate pre-generated objects for each combination
    for combo_idx, description_indices in enumerate(all_combinations):
        print(f"Composing mixed type augmentation candidate {combo_idx + 1}/{len(all_combinations)}")
        
        # Deep copy the layout and locate the room copy
        type_layout_copy: FloorPlan = copy.deepcopy(layout)
        type_layout_copy_id = f"{layout.id}_{aug_name}_mixed{combo_idx}"
        type_room_copy = next((r for r in type_layout_copy.rooms if r.id == room.id), None)
        if type_room_copy is None:
            continue

        # Remove the original root and all its descendants from the room copy
        ids_to_remove = {obj.id for obj in bfs_objects}
        type_room_copy.objects = [o for o in type_room_copy.objects if o.id not in ids_to_remove]

        # Map from original object id -> newly created object's id
        new_id_map: Dict[str, str] = {}
        new_objects: List[Object] = []

        # Compose objects by selecting the appropriate pre-generated variations
        type_aug_successful = True
        for obj_idx, orig_obj in enumerate(bfs_objects):
            # Get the description index for this object in this combination
            desc_idx = description_indices[obj_idx]
            
            # Get the pre-generated variation for this description index
            obj_variations = object_variations_by_id.get(orig_obj.id, [])
            if desc_idx >= len(obj_variations):
                type_aug_successful = False
                print(f"Error: No variation {desc_idx} available for object {orig_obj.id}")
                break
            
            # Create a copy of the pre-generated variation
            type_aug_obj = copy.deepcopy(obj_variations[desc_idx])
            
            # Determine placement location: root keeps its original location; children use their parent's new id
            if orig_obj.id == object.id:
                location = orig_obj.place_id
            else:
                parent_original_id = orig_obj.place_id
                if not isinstance(parent_original_id, str) or parent_original_id not in new_id_map:
                    type_aug_successful = False
                    break
                location = new_id_map[parent_original_id]
            
            # Update the placement location
            type_aug_obj.place_id = location
            
            # Insert into room and track new id
            type_room_copy.objects.insert(0, type_aug_obj)
            new_id_map[orig_obj.id] = type_aug_obj.id
            new_objects.append(type_aug_obj)

        if type_aug_successful:
            print(f"Mixed type augmentation candidate {combo_idx} composed successfully with {len(new_objects)} objects")
            
            # Create inverse mapping: new_id -> old_id
            new_id_to_old_id_map = {new_id: old_id for old_id, new_id in new_id_map.items()}
            
            # Save the type augmentation candidate
            candidate_path = os.path.join(RESULTS_DIR, layout.id, aug_name, f"{type_layout_copy_id}_type_candidate.json")
            candidate_data = {
                "layout_id": type_layout_copy_id,
                "layout": asdict(type_layout_copy),
                "room_id": type_room_copy.id,
                "new_object_ids": [obj.id for obj in new_objects],
                "old_id_to_new_id_map": new_id_map,
                "new_id_to_old_id_map": new_id_to_old_id_map,
                "description_combination": description_indices  # Track which descriptions were used
            }
            with open(candidate_path, "w") as f:
                json.dump(candidate_data, f)
            
            type_augmentation_candidates.append((type_layout_copy_id, type_layout_copy, type_room_copy, new_objects))
        else:
            print(f"Mixed type augmentation candidate {combo_idx} failed during object composition")

    return type_augmentation_candidates


def generate_type_augmentation_candidates_linear(
    layout: FloorPlan, room: Room, object: Object, aug_type_num: int, aug_name: str
) -> List[Tuple[str, FloorPlan, Room, List[Object]]]:
    """
    Generate linear type augmentation candidates for an object tree.
    This function creates aug_type_num layouts where all objects use the same description index:
    (0,0,0), (1,1,1), (2,2,2), etc. This is different from the mixed version which generates
    all possible combinations where each object can use any of its aug_type_num descriptions.
    
    Args:
        layout: The floor plan containing the room
        room: The room containing the object tree
        object: The root object of the tree to augment
        aug_type_num: Number of type augmentation variations (results in aug_type_num total layouts)
        aug_name: Name prefix for the augmented layouts
    
    Returns:
        List of (layout_id, layout_copy, room_copy, new_objects) tuples for each type augmentation combination
    """
    # Build children map and BFS order from parent to children (root first)
    object_map: Dict[str, Object] = {o.id: o for o in room.objects}
    children_map: Dict[str, List[Object]] = {}
    for o in room.objects:
        parent_id = o.place_id
        if isinstance(parent_id, str) and parent_id in object_map:
            children_map.setdefault(parent_id, []).append(o)

    bfs_objects: List[Object] = []
    queue: List[str] = [object.id]
    visited: set = set()
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        if current_id in object_map:
            bfs_objects.append(object_map[current_id])
        for child in children_map.get(current_id, []):
            queue.append(child.id)

    # Precompute augmented descriptions for each object in the tree
    aug_descriptions_by_id: Dict[str, List[str]] = {}
    aug_object_types_by_id: Dict[str, str] = {}
    for obj in bfs_objects:
        object_type, descriptions = augment_description(obj.type, obj.description, aug_type_num, obj.dimensions, return_object_type=True)
        aug_descriptions_by_id[obj.id] = descriptions
        aug_object_types_by_id[obj.id] = object_type
        print("--------------------------------")
        print("original object:")
        print(obj.id, obj.type, obj.description, obj.dimensions)
        print("--------------------------------")
        print("augmented object:")
        print(object_type, descriptions)
        print("--------------------------------")
    
    os.makedirs(os.path.join(RESULTS_DIR, layout.id, aug_name), exist_ok=True)

    # STEP 1: Pre-generate all object variations for each object (m variations per object)
    print(f"Pre-generating {aug_type_num} variations for each of {len(bfs_objects)} objects")
    
    # Store all generated object variations: object_id -> [list of m Object variations]
    object_variations_by_id: Dict[str, List[Object]] = {}
    
    # Create a temporary layout and room for object generation
    temp_layout_copy: FloorPlan = copy.deepcopy(layout)
    temp_room_copy = next((r for r in temp_layout_copy.rooms if r.id == room.id), None)
    if temp_room_copy is None:
        return []
    
    # Remove the original objects from temp room for clean generation
    ids_to_remove = {obj.id for obj in bfs_objects}
    temp_room_copy.objects = [o for o in temp_room_copy.objects if o.id not in ids_to_remove]
    
    # Helper function to generate variations for a single object
    def generate_variations_for_object(orig_obj, aug_descriptions_by_id, aug_object_types_by_id, aug_type_num, temp_room_copy, temp_layout_copy):
        """Generate all variations for a single object."""
        print(f"Generating {aug_type_num} variations for object {orig_obj.id} ({orig_obj.type})")
        
        obj_variations = []
        obj_descs = aug_descriptions_by_id.get(orig_obj.id, [])
        augmented_object_type = aug_object_types_by_id.get(orig_obj.id, orig_obj.type)
        
        for desc_idx in range(aug_type_num):
            if desc_idx >= len(obj_descs):
                # If we don't have enough descriptions, cycle through them
                selected_description = obj_descs[desc_idx % len(obj_descs)]
            else:
                selected_description = obj_descs[desc_idx]
            
            # Use a placeholder location for generation (will be updated during composition)
            object_info_dict = {
                orig_obj.type: {
                    "description": selected_description,
                    "location": orig_obj.place_id,  # Placeholder location
                    "size": [orig_obj.dimensions.width * 100, orig_obj.dimensions.length * 100, orig_obj.dimensions.height * 100],
                    "quantity": 1,
                    "variance_type": "same",
                    "limit_size": True,
                }
            }
            
            selected_objects, _ = select_objects(object_info_dict, temp_room_copy, None, temp_layout_copy)
            if selected_objects:
                variation_obj = selected_objects[0]
                # Initialize pose to match the original object
                variation_obj.position = orig_obj.position
                variation_obj.rotation = orig_obj.rotation
                # variation_obj.dimensions.width = min(variation_obj.dimensions.width, orig_obj.dimensions.width)
                # variation_obj.dimensions.length = min(variation_obj.dimensions.length, orig_obj.dimensions.length)
                # Store the original object ID for later reference
                variation_obj._original_id = orig_obj.id
                obj_variations.append(variation_obj)
            else:
                print(f"Warning: Failed to generate variation {desc_idx} for object {orig_obj.id}")
        
        return orig_obj.id, obj_variations
    
    # Parallelize object variation generation using ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_obj_id = {}
        for orig_obj in bfs_objects:
            future = executor.submit(
                generate_variations_for_object,
                orig_obj,
                aug_descriptions_by_id,
                aug_object_types_by_id,
                aug_type_num,
                temp_room_copy,
                temp_layout_copy
            )
            future_to_obj_id[future] = orig_obj.id
        
        # Collect results as they complete
        for future in as_completed(future_to_obj_id):
            obj_id = future_to_obj_id[future]
            try:
                result_obj_id, obj_variations = future.result()
                
                if len(obj_variations) > 0:
                    object_variations_by_id[result_obj_id] = obj_variations
                    print(f"Generated {len(obj_variations)} variations for object {result_obj_id}")
                else:
                    print(f"Error: No variations generated for object {result_obj_id}")
                    return []  # Cannot proceed without variations for any object
            except Exception as exc:
                print(f"Object {obj_id} generated an exception: {exc}")
                return []

    # STEP 2: Generate linear combinations where all objects use the same description index
    # Generate aug_type_num combinations: (0,0,0), (1,1,1), (2,2,2), etc.
    all_combinations = []
    for desc_idx in range(aug_type_num):
        # Create a combination where all objects use the same description index
        combination = tuple(desc_idx % len(aug_descriptions_by_id[obj.id]) for obj in bfs_objects)
        all_combinations.append(combination)
    
    print(f"Composing {len(all_combinations)} linear type augmentation candidates (all objects use same description index)")
    
    type_augmentation_candidates = []
    
    # STEP 3: Compose layouts by selecting appropriate pre-generated objects for each combination
    for combo_idx, description_indices in enumerate(all_combinations):
        print(f"Composing linear type augmentation candidate {combo_idx + 1}/{len(all_combinations)}")
        
        # Deep copy the layout and locate the room copy
        type_layout_copy: FloorPlan = copy.deepcopy(layout)
        type_layout_copy_id = f"{layout.id}_{aug_name}_linear{combo_idx}"
        type_room_copy = next((r for r in type_layout_copy.rooms if r.id == room.id), None)
        if type_room_copy is None:
            continue

        # Remove the original root and all its descendants from the room copy
        ids_to_remove = {obj.id for obj in bfs_objects}
        type_room_copy.objects = [o for o in type_room_copy.objects if o.id not in ids_to_remove]

        # Map from original object id -> newly created object's id
        new_id_map: Dict[str, str] = {}
        new_objects: List[Object] = []

        # Compose objects by selecting the appropriate pre-generated variations
        type_aug_successful = True
        for obj_idx, orig_obj in enumerate(bfs_objects):
            # Get the description index for this object in this combination
            desc_idx = description_indices[obj_idx]
            
            # Get the pre-generated variation for this description index
            obj_variations = object_variations_by_id.get(orig_obj.id, [])
            if desc_idx >= len(obj_variations):
                type_aug_successful = False
                print(f"Error: No variation {desc_idx} available for object {orig_obj.id}")
                break
            
            # Create a copy of the pre-generated variation
            type_aug_obj = copy.deepcopy(obj_variations[desc_idx])
            
            # Determine placement location: root keeps its original location; children use their parent's new id
            if orig_obj.id == object.id:
                location = orig_obj.place_id
            else:
                parent_original_id = orig_obj.place_id
                if not isinstance(parent_original_id, str) or parent_original_id not in new_id_map:
                    type_aug_successful = False
                    break
                location = new_id_map[parent_original_id]
            
            # Update the placement location
            type_aug_obj.place_id = location
            
            # Insert into room and track new id
            type_room_copy.objects.insert(0, type_aug_obj)
            new_id_map[orig_obj.id] = type_aug_obj.id
            new_objects.append(type_aug_obj)

        if type_aug_successful:
            print(f"Linear type augmentation candidate {combo_idx} composed successfully with {len(new_objects)} objects")
            
            # Create inverse mapping: new_id -> old_id
            new_id_to_old_id_map = {new_id: old_id for old_id, new_id in new_id_map.items()}
            
            # Save the type augmentation candidate
            candidate_path = os.path.join(RESULTS_DIR, layout.id, aug_name, f"{type_layout_copy_id}_type_candidate.json")
            candidate_data = {
                "layout_id": type_layout_copy_id,
                "layout": asdict(type_layout_copy),
                "room_id": type_room_copy.id,
                "new_object_ids": [obj.id for obj in new_objects],
                "old_id_to_new_id_map": new_id_map,
                "new_id_to_old_id_map": new_id_to_old_id_map,
                "description_combination": description_indices  # Track which descriptions were used
            }
            with open(candidate_path, "w") as f:
                json.dump(candidate_data, f)
            
            type_augmentation_candidates.append((type_layout_copy_id, type_layout_copy, type_room_copy, new_objects))
        else:
            print(f"Linear type augmentation candidate {combo_idx} failed during object composition")

    return type_augmentation_candidates


def object_augmentation_pose_object_tree_with_reach_test_parallel_on_type_augmentation(
    layout_copy: FloorPlan, room_copy: Room, new_objects: List[Object], layout_id: str,
    aug_pose_num: int, aug_name: str, reach_threshold: float = 0.4, group_size: int = 10,
    custom_save_dir: str = None
) -> List[Tuple[str, FloorPlan]]:
    """
    Apply parallel pose augmentation with reachability test on type-augmented objects.
    
    Args:
        layout_copy: The layout containing the type-augmented objects
        room_copy: The room containing the type-augmented objects  
        new_objects: The type-augmented objects to apply pose augmentation on
        layout_id: The original layout ID for asset loading
        aug_pose_num: Number of pose augmentations to perform
        aug_name: Name prefix for the augmented layouts
        reach_threshold: Maximum reach distance for reachability test
        group_size: Size of groups for parallel processing
        custom_save_dir: Optional custom directory to save individual layout files
    
    Returns:
        List of (layout_id, layout) tuples for all successful pose augmentations
    """
    augmented_layouts_info = []
    
    # Build proper support tree order from the type-augmented objects
    # Find the root object (should be the first one that doesn't have a parent in new_objects)
    new_object_ids = {obj.id for obj in new_objects}
    object_map = {obj.id: obj for obj in room_copy.objects}
    children_map: Dict[str, List[Object]] = {}
    
    # Build children map from all objects in the room (not just new_objects)
    for o in room_copy.objects:
        parent_id = o.place_id
        if isinstance(parent_id, str) and parent_id in object_map:
            children_map.setdefault(parent_id, []).append(o)
    
    # Find the root of the new objects tree
    root_object = None
    for obj in new_objects:
        parent_id = obj.place_id
        # Root object either has no parent in new_objects or is placed on floor/wall
        if not isinstance(parent_id, str) or parent_id not in new_object_ids:
            root_object = obj
            break
    
    if root_object is None:
        print("Warning: Could not find root object in type-augmented objects")
        return []
    
    # Build BFS order starting from the root
    bfs_objects: List[Object] = []
    queue: List[str] = [root_object.id]
    visited: set = set()
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        if current_id in object_map and current_id in new_object_ids:
            bfs_objects.append(object_map[current_id])
        for child in children_map.get(current_id, []):
            if child.id in new_object_ids:  # Only include new objects in the traversal
                queue.append(child.id)
    
    print(f"Built BFS order for {len(bfs_objects)} type-augmented objects: {[obj.id for obj in bfs_objects]}")
    
    # Use original layout folder for assets during validation
    scene_save_dir = os.path.join(RESULTS_DIR, layout_id)
    
    # Process augmentations in groups
    for group_start in range(0, aug_pose_num, group_size):
        gc.collect()
        group_end = min(group_start + group_size, aug_pose_num)
        current_group_size = group_end - group_start
        
        print(f"Processing pose augmentation group {group_start}-{group_end-1} (size: {current_group_size})")
        
        # Initialize group data structures
        group_rooms = []  # List of room copies for each aug in the group
        group_layouts = []  # List of layout copies for each aug in the group
        group_layout_ids = []  # List of layout IDs for each aug in the group
        group_new_id_maps = []  # List of new_id_maps for each aug in the group
        group_successful_mask = []  # Boolean mask for successful augmentations
        
        # Initialize base rooms for each augmentation in the group
        for aug_iter in range(group_start, group_end):
            pose_layout_copy: FloorPlan = copy.deepcopy(layout_copy)
            pose_layout_copy_id = generate_unique_id(f"{layout_copy.id}_{aug_name}_pose{aug_iter}")
            pose_room_copy = next((r for r in pose_layout_copy.rooms if r.id == room_copy.id), None)
            if pose_room_copy is None:
                group_successful_mask.append(False)
                continue

            group_rooms.append(pose_room_copy)
            group_layouts.append(pose_layout_copy)
            group_layout_ids.append(pose_layout_copy_id)
            group_new_id_maps.append({})  # Map from original object id -> object copy for tracking
            group_successful_mask.append(True)

        if not any(group_successful_mask):
            print(f"No successful base rooms in group {group_start}-{group_end-1}")
            continue

        # Process each object in the tree (BFS order following support relationships)
        for aug_i, orig_obj in enumerate(bfs_objects):
            print(f"Processing object {aug_i}: {orig_obj.id} across {sum(group_successful_mask)} remaining augmentations")
            
            # Collect pose candidates for this object across all successful augmentations
            group_pose_candidates = []  # List of pose_candidates for each successful aug
            group_aug_objects = []  # List of created objects for each successful aug
            current_indices = []  # Indices of currently successful augmentations
            
            for i, (pose_room_copy, pose_layout_copy, pose_layout_copy_id, new_id_map, is_successful) in enumerate(tqdm(zip(
                group_rooms, group_layouts, group_layout_ids, group_new_id_maps, group_successful_mask
            ), desc=f"Sampling poses for object {orig_obj.id}", total=len(group_rooms), leave=False)):
                if not is_successful:
                    continue
                
                # Find the corresponding object in this room copy
                aug_obj = next((o for o in pose_room_copy.objects if o.id == orig_obj.id), None)
                if aug_obj is None:
                    group_successful_mask[i] = False
                    continue

                # Sample pose candidates for this object
                pose_candidates = _sample_pose_offsets_condition(
                    aug_obj,
                    pose_room_copy,
                    layout_id,
                    pose_layout_copy_id,
                    scene_save_dir,
                    unit_length=min(aug_obj.dimensions.width, aug_obj.dimensions.length) if aug_obj.place_id == "floor" else 1000.0,
                )
                if len(pose_candidates) == 0:
                    print(f"No pose candidates for {aug_obj.id} in aug {i}")
                    group_successful_mask[i] = False
                    continue

                group_pose_candidates.append(pose_candidates)
                group_aug_objects.append(aug_obj)
                current_indices.append(i)

            if not current_indices:
                print(f"No successful augmentations remaining after object {orig_obj.id}")
                break

            # Find the maximum number of pose candidates for this object
            max_pose_candidates = max(len(candidates) for candidates in group_pose_candidates)
            print(f"Object {orig_obj.id}: max {max_pose_candidates} pose candidates across {len(current_indices)} augs")

            # Try poses until all augmentations find a successful pose or exhaust all poses
            object_successful_indices = []
            remaining_aug_indices = list(range(len(current_indices)))  # Track which augs still need poses
            
            for pose_idx in range(max_pose_candidates):
                if not remaining_aug_indices:  # All augmentations found successful poses
                    break
                    
                print(f"  Trying pose {pose_idx} for object {orig_obj.id}, {len(remaining_aug_indices)} augs remaining")
                
                # Apply pose_idx only to augmentations that haven't found a successful pose yet
                test_rooms = []
                test_layout_ids = []
                test_indices_mapping = []  # Maps to remaining_aug_indices
                
                for j in remaining_aug_indices:
                    pose_candidates = group_pose_candidates[j]
                    aug_obj = group_aug_objects[j]
                    global_idx = current_indices[j]
                    
                    if pose_idx >= len(pose_candidates):
                        continue  # This augmentation doesn't have this pose
                    
                    pose_room_copy = group_rooms[global_idx]
                    pose_layout_copy_id = group_layout_ids[global_idx]
                    
                    # Deep copy room for this pose test
                    room_copy_test = copy.deepcopy(pose_room_copy)
                    
                    # Find the object and apply the pose transformation
                    aug_obj_test = next((o for o in room_copy_test.objects if o.id == aug_obj.id), None)
                    if aug_obj_test is None:
                        continue
                    
                    T_prev = _object_pose_to_matrix(aug_obj_test)
                    T_pos, T_rot = pose_candidates[pose_idx]
                    T_aug = T_prev @ T_rot @ T_pos
                    
                    new_pos, new_rot = _matrix_to_pose(T_aug)
                    
                    # Update all descendants inside the copied room
                    _update_descendants_transforms(room_copy_test, aug_obj_test.id, T_prev, T_aug)
                    for obj_copy in room_copy_test.objects:
                        if obj_copy.id == aug_obj_test.id:
                            obj_copy.position = new_pos
                            obj_copy.rotation = new_rot
                            break
                    
                    test_rooms.append(room_copy_test)
                    test_layout_ids.append(pose_layout_copy_id)
                    test_indices_mapping.append(j)
                
                if not test_rooms:
                    continue
                
                # Validate all rooms with this pose using group validation
                print(f"  Validating {len(test_rooms)} rooms with pose {pose_idx} for object {orig_obj.id}")
                group_aug_name = f"{aug_name}_obj{aug_i}_pose{pose_idx}"
                stable_list = _validate_room_stability_groups(scene_save_dir, group_aug_name, test_rooms, test_layout_ids)
                
                if not isinstance(stable_list, list):
                    print(f"  Group validation failed for pose {pose_idx}")
                    continue
                
                # Process results: accept poses that are both stable and reachable
                newly_successful = []
                for k, (room_copy_test, is_stable) in enumerate(zip(test_rooms, stable_list)):
                    j = test_indices_mapping[k]  # Index into remaining_aug_indices
                    global_idx = current_indices[j]  # Global index into group arrays
                    
                    if is_stable:
                        print(f"  Aug {global_idx} object {orig_obj.id} pose {pose_idx} is stable")
                        
                        # Check reachability at this intermediate stage
                        current_objects = [o for o in room_copy_test.objects if o.place_id != "floor"]
                        is_reachable = _check_reachability(group_layouts[global_idx], room_copy_test, current_objects, reach_threshold)
                        
                        if is_reachable:
                            print(f"  Aug {global_idx} object {orig_obj.id} pose {pose_idx} is ACCEPTED (stable and reachable)")
                            
                            # Update the actual room copy with the successful pose
                            group_rooms[global_idx] = room_copy_test
                            
                            # Record the new id mapping for children
                            aug_obj = group_aug_objects[j]
                            group_new_id_maps[global_idx][orig_obj.id] = aug_obj.id
                            
                            # Mark this augmentation as successful for this object
                            object_successful_indices.append(global_idx)
                            newly_successful.append(j)  # Remove from remaining list
                        else:
                            print(f"  Aug {global_idx} object {orig_obj.id} pose {pose_idx} is stable but not reachable - will try next pose")
                    else:
                        print(f"  Aug {global_idx} object {orig_obj.id} pose {pose_idx} is not stable - will try next pose")
                
                # Remove successfully placed augmentations from remaining list
                for j in newly_successful:
                    remaining_aug_indices.remove(j)

            # Update success mask: only keep augmentations that found a working pose for this object
            for i in range(len(group_successful_mask)):
                if group_successful_mask[i] and i not in object_successful_indices:
                    group_successful_mask[i] = False
                    print(f"Aug {i} failed at object {orig_obj.id}")

            if not any(group_successful_mask):
                print(f"No augmentations remain successful after object {orig_obj.id}")
                break

        # Save successful augmentations (already filtered for stability and reachability)
        for i, is_successful in enumerate(group_successful_mask):
            if not is_successful:
                continue
                
            pose_room_copy = group_rooms[i]
            pose_layout_copy = group_layouts[i]
            pose_layout_copy_id = group_layout_ids[i]
            
            print(f"Aug {i} completed successfully - saving")
            
            # Save successful augmentation
            for layout_room in pose_layout_copy.rooms:
                if layout_room.id == room_copy.id:
                    layout_room.objects = pose_room_copy.objects
                    break

            # save the augmented layout
            if custom_save_dir:
                layout_copy_path = os.path.join(custom_save_dir, f"{pose_layout_copy_id}.json")
            else:
                layout_copy_path = os.path.join(RESULTS_DIR, layout_id, aug_name, f"{pose_layout_copy_id}.json")
            
            os.makedirs(os.path.dirname(layout_copy_path), exist_ok=True)
            with open(layout_copy_path, "w") as f:
                json.dump(asdict(pose_layout_copy), f)
            augmented_layouts_info.append((pose_layout_copy_id, pose_layout_copy))

        print(f"Group {group_start}-{group_end-1} completed: {len(augmented_layouts_info)} total successful so far")

    print(f"Parallel pose augmentation on type candidates completed: {len(augmented_layouts_info)} successful layouts generated")
    return augmented_layouts_info

def object_augmentation_pose_support_tree_with_reach_test_parallel_on_type_augmentation(
    layout_copy: FloorPlan, room_copy: Room, object_id: str, old_id_to_new_id_map: Dict[str, str], 
    layout_id: str, aug_pose_num: int, aug_name: str, reach_threshold: float = 0.4, 
    group_size: int = 10, custom_save_dir: str = None
) -> List[Tuple[str, FloorPlan]]:
    """
    Apply parallel pose augmentation with reachability test on support tree based on old object ID.
    
    Args:
        layout_copy: The layout containing the type-augmented objects
        room_copy: The room containing the type-augmented objects  
        object_id: The old object ID to find the support tree for
        old_id_to_new_id_map: Mapping from old object IDs to new object IDs
        layout_id: The original layout ID for asset loading
        aug_pose_num: Number of pose augmentations to perform
        aug_name: Name prefix for the augmented layouts
        reach_threshold: Maximum reach distance for reachability test
        group_size: Size of groups for parallel processing
        custom_save_dir: Optional custom directory to save individual layout files
    
    Returns:
        List of (layout_id, layout) tuples for all successful pose augmentations
    """
    augmented_layouts_info = []
    
    # Find the root object using old_id mapping
    root_new_id = old_id_to_new_id_map.get(object_id)
    if root_new_id is None:
        # Object wasn't type augmented, try to find it directly
        object_map = {obj.id: obj for obj in room_copy.objects}
        if object_id not in object_map:
            raise ValueError(f"Object with old ID {object_id} not found and not in mapping")
        root_object = object_map[object_id]
    else:
        # Find the type-augmented object
        object_map = {obj.id: obj for obj in room_copy.objects}
        if root_new_id not in object_map:
            raise ValueError(f"Mapped object with new ID {root_new_id} not found")
        root_object = object_map[root_new_id]
    
    # Build children map from all objects in the room
    children_map: Dict[str, List[Object]] = {}
    for o in room_copy.objects:
        parent_id = o.place_id
        if isinstance(parent_id, str) and parent_id in object_map:
            children_map.setdefault(parent_id, []).append(o)
    
    # Build BFS order starting from the root to get the complete support tree
    bfs_objects: List[Object] = []
    queue: List[str] = [root_object.id]
    visited: set = set()
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        if current_id in object_map:
            bfs_objects.append(object_map[current_id])
        for child in children_map.get(current_id, []):
            queue.append(child.id)
    
    print(f"Built BFS order for {len(bfs_objects)} objects in support tree: {[obj.id for obj in bfs_objects]}")
    
    # Use original layout folder for assets during validation
    scene_save_dir = os.path.join(RESULTS_DIR, layout_id)
    
    # Process augmentations in groups
    for group_start in range(0, aug_pose_num, group_size):
        gc.collect()
        group_end = min(group_start + group_size, aug_pose_num)
        current_group_size = group_end - group_start
        
        print(f"Processing pose augmentation group {group_start}-{group_end-1} (size: {current_group_size})")
        
        # Initialize group data structures
        group_rooms = []  # List of room copies for each aug in the group
        group_layouts = []  # List of layout copies for each aug in the group
        group_layout_ids = []  # List of layout IDs for each aug in the group
        group_new_id_maps = []  # List of new_id_maps for each aug in the group
        group_successful_mask = []  # Boolean mask for successful augmentations
        
        # Initialize base rooms for each augmentation in the group
        for aug_iter in range(group_start, group_end):
            pose_layout_copy: FloorPlan = copy.deepcopy(layout_copy)
            pose_layout_copy_id = generate_unique_id(f"{layout_copy.id}_{aug_name}_pose{aug_iter}")
            pose_room_copy = next((r for r in pose_layout_copy.rooms if r.id == room_copy.id), None)
            if pose_room_copy is None:
                group_successful_mask.append(False)
                continue

            # Remove the support tree objects from the room copy (similar to reference function)
            support_tree_ids = {obj.id for obj in bfs_objects}
            pose_room_copy.objects = [o for o in pose_room_copy.objects if o.id not in support_tree_ids]

            group_rooms.append(pose_room_copy)
            group_layouts.append(pose_layout_copy)
            group_layout_ids.append(pose_layout_copy_id)
            group_new_id_maps.append({})  # Map from original object id -> object copy for tracking
            group_successful_mask.append(True)

        if not any(group_successful_mask):
            print(f"No successful base rooms in group {group_start}-{group_end-1}")
            continue

        # Process each object in the tree (BFS order following support relationships)
        for aug_i, orig_obj in enumerate(bfs_objects):
            print(f"Processing object {aug_i}: {orig_obj.id} across {sum(group_successful_mask)} remaining augmentations")
            
            # Collect pose candidates for this object across all successful augmentations
            group_pose_candidates = []  # List of pose_candidates for each successful aug
            group_aug_objects = []  # List of created objects for each successful aug
            current_indices = []  # Indices of currently successful augmentations
            
            for i, (pose_room_copy, pose_layout_copy, pose_layout_copy_id, new_id_map, is_successful) in enumerate(tqdm(zip(
                group_rooms, group_layouts, group_layout_ids, group_new_id_maps, group_successful_mask
            ), desc=f"Sampling poses for object {orig_obj.id}", total=len(group_rooms), leave=False)):
                if not is_successful:
                    continue
                
                # Create a copy of the original object (similar to reference function)
                aug_obj = copy.deepcopy(orig_obj)
                
                # Update placement location for children to use their parent's new id
                if orig_obj.id == root_object.id:
                    # Root object keeps its original location
                    aug_obj.place_id = orig_obj.place_id
                else:
                    parent_original_id = orig_obj.place_id
                    if not isinstance(parent_original_id, str) or parent_original_id not in new_id_map:
                        group_successful_mask[i] = False
                        continue
                    aug_obj.place_id = new_id_map[parent_original_id]

                # Insert into room
                pose_room_copy.objects.insert(0, aug_obj)

                # Sample pose candidates for this object
                pose_candidates = _sample_pose_offsets_condition(
                    aug_obj,
                    pose_room_copy,
                    layout_id,
                    pose_layout_copy_id,
                    scene_save_dir,
                    unit_length=min(aug_obj.dimensions.width, aug_obj.dimensions.length) if aug_obj.place_id == "floor" else 1000.0,
                )
                if len(pose_candidates) == 0:
                    print(f"No pose candidates for {aug_obj.id} in aug {i}")
                    group_successful_mask[i] = False
                    continue

                group_pose_candidates.append(pose_candidates)
                group_aug_objects.append(aug_obj)
                current_indices.append(i)

            if not current_indices:
                print(f"No successful augmentations remaining after object {orig_obj.id}")
                break

            # Find the maximum number of pose candidates for this object
            max_pose_candidates = max(len(candidates) for candidates in group_pose_candidates)
            print(f"Object {orig_obj.id}: max {max_pose_candidates} pose candidates across {len(current_indices)} augs")

            # Try poses until all augmentations find a successful pose or exhaust all poses
            object_successful_indices = []
            remaining_aug_indices = list(range(len(current_indices)))  # Track which augs still need poses
            
            for pose_idx in range(max_pose_candidates):
                if not remaining_aug_indices:  # All augmentations found successful poses
                    break
                    
                print(f"  Trying pose {pose_idx} for object {orig_obj.id}, {len(remaining_aug_indices)} augs remaining")
                
                # Apply pose_idx only to augmentations that haven't found a successful pose yet
                test_rooms = []
                test_layout_ids = []
                test_indices_mapping = []  # Maps to remaining_aug_indices
                
                for j in remaining_aug_indices:
                    pose_candidates = group_pose_candidates[j]
                    aug_obj = group_aug_objects[j]
                    global_idx = current_indices[j]
                    
                    if pose_idx >= len(pose_candidates):
                        continue  # This augmentation doesn't have this pose
                    
                    pose_room_copy = group_rooms[global_idx]
                    pose_layout_copy_id = group_layout_ids[global_idx]
                    
                    # Deep copy room for this pose test
                    room_copy_test = copy.deepcopy(pose_room_copy)
                    
                    # Find the object and apply the pose transformation
                    aug_obj_test = next((o for o in room_copy_test.objects if o.id == aug_obj.id), None)
                    if aug_obj_test is None:
                        continue
                    
                    T_prev = _object_pose_to_matrix(aug_obj_test)
                    T_pos, T_rot = pose_candidates[pose_idx]
                    T_aug = T_prev @ T_rot @ T_pos
                    
                    new_pos, new_rot = _matrix_to_pose(T_aug)
                    
                    # Update all descendants inside the copied room
                    _update_descendants_transforms(room_copy_test, aug_obj_test.id, T_prev, T_aug)
                    for obj_copy in room_copy_test.objects:
                        if obj_copy.id == aug_obj_test.id:
                            obj_copy.position = new_pos
                            obj_copy.rotation = new_rot
                            break
                    
                    test_rooms.append(room_copy_test)
                    test_layout_ids.append(pose_layout_copy_id)
                    test_indices_mapping.append(j)
                
                if not test_rooms:
                    continue
                
                # Validate all rooms with this pose using group validation
                print(f"  Validating {len(test_rooms)} rooms with pose {pose_idx} for object {orig_obj.id}")
                group_aug_name = f"{aug_name}_obj{aug_i}_pose{pose_idx}"
                stable_list = _validate_room_stability_groups(scene_save_dir, group_aug_name, test_rooms, test_layout_ids)
                
                if not isinstance(stable_list, list):
                    print(f"  Group validation failed for pose {pose_idx}")
                    continue
                
                # Process results: accept poses that are both stable and reachable
                newly_successful = []
                for k, (room_copy_test, is_stable) in enumerate(zip(test_rooms, stable_list)):
                    j = test_indices_mapping[k]  # Index into remaining_aug_indices
                    global_idx = current_indices[j]  # Global index into group arrays
                    
                    if is_stable:
                        print(f"  Aug {global_idx} object {orig_obj.id} pose {pose_idx} is stable")
                        
                        # Check reachability at this intermediate stage
                        current_objects = [o for o in room_copy_test.objects if o.place_id != "floor"]
                        is_reachable = _check_reachability(group_layouts[global_idx], room_copy_test, current_objects, reach_threshold)
                        
                        if is_reachable:
                            print(f"  Aug {global_idx} object {orig_obj.id} pose {pose_idx} is ACCEPTED (stable and reachable)")
                            
                            # Update the actual room copy with the successful pose
                            group_rooms[global_idx] = room_copy_test
                            
                            # Record the new id mapping for children
                            aug_obj = group_aug_objects[j]
                            group_new_id_maps[global_idx][orig_obj.id] = aug_obj.id
                            
                            # Mark this augmentation as successful for this object
                            object_successful_indices.append(global_idx)
                            newly_successful.append(j)  # Remove from remaining list
                        else:
                            print(f"  Aug {global_idx} object {orig_obj.id} pose {pose_idx} is stable but not reachable - will try next pose")
                    else:
                        print(f"  Aug {global_idx} object {orig_obj.id} pose {pose_idx} is not stable - will try next pose")
                
                # Remove successfully placed augmentations from remaining list
                for j in newly_successful:
                    remaining_aug_indices.remove(j)

            # Update success mask: only keep augmentations that found a working pose for this object
            for i in range(len(group_successful_mask)):
                if group_successful_mask[i] and i not in object_successful_indices:
                    group_successful_mask[i] = False
                    print(f"Aug {i} failed at object {orig_obj.id}")

            if not any(group_successful_mask):
                print(f"No augmentations remain successful after object {orig_obj.id}")
                break

        # Save successful augmentations (already filtered for stability and reachability)
        for i, is_successful in enumerate(group_successful_mask):
            if not is_successful:
                continue
                
            pose_room_copy = group_rooms[i]
            pose_layout_copy = group_layouts[i]
            pose_layout_copy_id = group_layout_ids[i]
            
            print(f"Aug {i} completed successfully - saving")
            
            # Save successful augmentation
            for layout_room in pose_layout_copy.rooms:
                if layout_room.id == room_copy.id:
                    layout_room.objects = pose_room_copy.objects
                    break

            # save the augmented layout
            if custom_save_dir:
                layout_copy_path = os.path.join(custom_save_dir, f"{pose_layout_copy_id}.json")
            else:
                layout_copy_path = os.path.join(RESULTS_DIR, layout_id, aug_name, f"{pose_layout_copy_id}.json")
            
            os.makedirs(os.path.dirname(layout_copy_path), exist_ok=True)
            with open(layout_copy_path, "w") as f:
                json.dump(asdict(pose_layout_copy), f)
            augmented_layouts_info.append((pose_layout_copy_id, pose_layout_copy))

        print(f"Group {group_start}-{group_end-1} completed: {len(augmented_layouts_info)} total successful so far")

    print(f"Parallel pose augmentation on support tree completed: {len(augmented_layouts_info)} successful layouts generated")
    return augmented_layouts_info