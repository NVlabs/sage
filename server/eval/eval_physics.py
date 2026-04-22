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
import asyncio
import carb
import argparse
# import omni.ext
# import omni.ui as ui
import omni
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "multi_gpu": False})

import omni.usd
import threading
import time
import socket
import json
import traceback
import sys
sys.path.append("/home/hongchix/main/server/isaacsim/isaac.sim.mcp_extension")

import gc
from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils, Sdf, UsdShade

import omni
import omni.kit.commands
import omni.physx as _physx
import omni.timeline
from typing import Dict, Any, List, Optional, Union
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
import numpy as np
from omni.isaac.core import World
# Import Beaver3d and USDLoader
from isaac_sim_mcp_extension.gen3d import Beaver3d
from isaac_sim_mcp_extension.usd import USDLoader
from isaac_sim_mcp_extension.usd import USDSearch3d
import requests

from isaac_sim_mcp_extension.usd_utils import (
    convert_mesh_to_usd, convert_mesh_to_usd_simple,
    door_frame_to_usd
)
from isaac_sim_mcp_extension.sim_utils import (
    get_all_prims_with_paths, 
    get_all_prims_with_prim_paths,
    start_simulation_and_track,
    start_simulation_and_track_groups,
    quaternion_angle
)
from pxr import Usd, UsdUtils
from isaac_sim_mcp_extension.scene.utils import (
    dict_to_floor_plan, 
    dict_to_room,
    export_layout_to_mesh_dict_list, 
    export_single_room_layout_to_mesh_dict_list,
    export_single_room_layout_to_mesh_dict_list_from_room, 
    get_single_object_mesh_info_dict,
    apply_object_transform_direct,
    export_layout_to_mesh_dict_list_no_object_transform
)
import os
from tqdm import tqdm
from datetime import datetime
import hashlib
import pdb
import pickle
import trimesh
from typing import Tuple, List

RESULTS_DIR = "/home/hongchix/main/server/results"


def detect_collision(base_meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], test_mesh: Tuple[np.ndarray, np.ndarray]):
    """
    Detect collisions between a test mesh and a series of base meshes.
    Uses edge-based ray casting to detect intersections.

    Parameters:
    -----------
    base_meshes : List of tuples, each containing (vertices, faces)
        List of base meshes to check against.
        vertices: (n, 3) float32 - Vertex coordinates
        faces: (m, 3) int32 - Face indices
        face_normals: (m, 3) float32 - Face normals (optional, can be None)

    test_mesh : Tuple of (vertices, faces)
        The mesh to test for collisions.
        vertices: (n, 3) float32 - Vertex coordinates
        faces: (m, 3) int32 - Face indices

    Returns:
    --------
    contact_points : np.ndarray
        (k, 3) array of contact point coordinates
    contact_mesh_id : np.ndarray
        (k,) array of indices indicating which base mesh had the contact
    contact_face_id : np.ndarray
        (k,) array of face indices in the base mesh
    """
    # Convert test mesh to trimesh object
    test_vertices, test_faces = test_mesh
    test_trimesh = trimesh.Trimesh(vertices=test_vertices, faces=test_faces)

    # Extract edges from test mesh
    edges = test_trimesh.edges_unique

    # Get edge vertices
    edge_points = test_vertices[edges]

    # Create ray origins and directions from edges
    ray_origins = edge_points[:, 0]
    ray_directions = edge_points[:, 1] - edge_points[:, 0]

    # Normalize ray directions
    ray_lengths = np.linalg.norm(ray_directions, axis=1)
    ray_directions = ray_directions / ray_lengths[:, np.newaxis]

    # Lists to store contact information
    all_contact_points = []
    all_contact_mesh_ids = []
    all_contact_face_ids = []
    all_contact_face_normals = []

    # Check collision with each base mesh
    for mesh_id, (base_vertices, base_faces, base_face_normals) in enumerate(base_meshes):
        # Create trimesh object for the base mesh
        base_trimesh = trimesh.Trimesh(vertices=base_vertices, faces=base_faces)

        # Use ray_mesh intersection to find contacts
        # locations, index_ray, index_tri = base_trimesh.ray.intersects_location(
        #     ray_origins=ray_origins,
        #     ray_directions=ray_directions
        # )

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
                contact_points = locations[valid_indices]
                contact_face_ids = index_tri[valid_indices]
                contact_mesh_ids = np.full(len(contact_points), mesh_id)
                contact_normals = base_face_normals[contact_face_ids]

                all_contact_points.append(contact_points)
                all_contact_mesh_ids.append(contact_mesh_ids)
                all_contact_face_ids.append(contact_face_ids)
                all_contact_face_normals.append(contact_normals)

    # Combine results from all base meshes
    if all_contact_points:
        contact_points = np.vstack(all_contact_points)
        contact_mesh_id = np.concatenate(all_contact_mesh_ids)
        contact_face_id = np.concatenate(all_contact_face_ids)
        contact_face_normals = np.vstack(all_contact_face_normals)
    else:
        # Return empty arrays if no contacts found
        contact_points = np.empty((0, 3), dtype=np.float32)
        contact_mesh_id = np.empty(0, dtype=np.int32)
        contact_face_id = np.empty(0, dtype=np.int32)
        contact_face_normals = np.empty((0, 3), dtype=np.float32)

    return contact_points, contact_mesh_id, contact_face_id, contact_face_normals


class EvalPhysics:
    def __init__(self) -> None:
        self.mesh_info_dict = None
        self.collision_record = None

    def create_single_room_layout_scene(self, scene_save_dir: str, room_id: str):
        """
        Create a room layout scene from a dictionary of mesh information.
        """

        # Load JSON data
        current_layout_id = os.path.basename(scene_save_dir)
        json_file_path = os.path.join(scene_save_dir, f"{current_layout_id}.json")

        try:
            with open(json_file_path, 'r') as f:
                layout_data = json.load(f)
        except FileNotFoundError:
            return json.dumps({
                "success": False,
                "error": f"JSON file not found: {json_file_path}"
            })
        
        floor_plan = dict_to_floor_plan(layout_data)
        current_layout = floor_plan
        
        mesh_info_dict = export_single_room_layout_to_mesh_dict_list(current_layout, room_id)
        self.mesh_info_dict = mesh_info_dict  # Store for later use

        stage = Usd.Stage.CreateInMemory()


        world_base_prim = UsdGeom.Xform.Define(stage, "/World")

        # set default prim to World
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

        collision_approximation = "sdf"
        
        self.track_ids = []
        door_ids = []
        door_frame_ids = []

        print(f"mesh_info_dict: {mesh_info_dict.keys()}")

        for mesh_id in mesh_info_dict:
            if mesh_id.startswith("wall_room_") or mesh_id.startswith("window_") or mesh_id.startswith("floor_"):
                usd_internal_path = f"/World/{mesh_id}"
            elif mesh_id.startswith("door_"):
                if mesh_id.endswith("_frame"):
                    door_frame_ids.append(mesh_id)
                else:
                    door_ids.append(mesh_id)
                continue
            else:
                self.track_ids.append(mesh_id)
                usd_internal_path = f"/World/{mesh_id}"
            mesh_dict = mesh_info_dict[mesh_id]
            mesh_obj_i = mesh_dict['mesh']
            static = mesh_dict['static']
            articulation = mesh_dict.get('articulation', None)
            texture = mesh_dict.get('texture', None)
            mass = mesh_dict.get('mass', 1.0)

            print(f"usd_internal_path: {usd_internal_path}")

            stage = convert_mesh_to_usd(stage, usd_internal_path,
                                        mesh_obj_i.vertices, mesh_obj_i.faces,
                                        collision_approximation, static, articulation, mass=mass, physics_iter=(16, 4),
                                        apply_debug_torque=False, debug_torque_value=30.0, texture=texture,
                                        usd_internal_art_reference_path=f"/World/{mesh_id}")

        # TODO create collision record
        # use mesh_info_dict get every object mesh (exclude door and wall and window)
        # calculate the number of objects which has collision with other object meshes.
        # save into collision_record dict

        door_ids = sorted(door_ids)
        door_frame_ids = sorted(door_frame_ids)

        for door_id, door_frame_id in zip(door_ids, door_frame_ids):
            usd_internal_path_door = f"/World/{door_id}"
            usd_internal_path_door_frame = f"/World/{door_frame_id}"


            mesh_dict_door = mesh_info_dict[door_id]
            mesh_obj_door = mesh_dict_door['mesh']
            articulation_door = mesh_dict_door.get('articulation', None)
            texture_door = mesh_dict_door.get('texture', None)

            mesh_dict_door_frame = mesh_info_dict[door_frame_id]
            mesh_obj_door_frame = mesh_dict_door_frame['mesh']
            texture_door_frame = mesh_dict_door_frame.get('texture', None)

            stage = door_frame_to_usd(
                stage,
                usd_internal_path_door,
                usd_internal_path_door_frame,
                mesh_obj_door,
                mesh_obj_door_frame,
                articulation_door,
                texture_door,
                texture_door_frame,
            )

        cache = UsdUtils.StageCache.Get()
        stage_id = cache.Insert(stage).ToLongInt()
        omni.usd.get_context().attach_stage_with_callback(stage_id)

        # Set the world axis of the stage root layer to Z
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        # Perform collision detection before simulation
        print("Detecting collisions in initial placement...")
        collision_record = self.detect_initial_collisions()
        self.collision_record = collision_record

        return {
            "status": "success",
            "message": f"Room layout scene created successfully",
            "collision_record": collision_record
        }



    def detect_initial_collisions(self):
        """
        Detect collisions between objects in their initial placement.
        Uses the mesh data from mesh_info_dict without any transformation.
        
        Returns:
            Dictionary mapping object_id to collision data
        """
        collision_record = {}
        
        # Prepare meshes for all tracked objects
        object_meshes = {}
        for object_id in self.track_ids:
            mesh_dict = self.mesh_info_dict[object_id]
            mesh_obj = mesh_dict['mesh']
            
            vertices = np.array(mesh_obj.vertices, dtype=np.float32)
            faces = np.array(mesh_obj.faces, dtype=np.int32)
            
            # Calculate face normals
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            face_normals = np.cross(v1 - v0, v2 - v0)
            face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10)
            
            object_meshes[object_id] = (vertices, faces, face_normals.astype(np.float32))
        
        # Detect collisions for each object
        for object_id in self.track_ids:
            test_vertices, test_faces, _ = object_meshes[object_id]
            test_mesh = (test_vertices, test_faces)
            
            # Prepare base meshes (all other objects)
            base_meshes = []
            base_object_ids = []
            
            for other_id in self.track_ids:
                if other_id != object_id:
                    base_meshes.append(object_meshes[other_id])
                    base_object_ids.append(other_id)
            
            if len(base_meshes) == 0:
                collision_record[object_id] = {
                    "collision_points": 0,
                    "collision_details": {}
                }
                continue
            
            # Detect collisions
            try:
                contact_points, contact_mesh_id, contact_face_id, contact_face_normals = detect_collision(
                    base_meshes, test_mesh
                )
                
                # Count collision points per object
                collision_details = {}
                for mesh_idx in np.unique(contact_mesh_id):
                    other_object_id = base_object_ids[mesh_idx]
                    count = int(np.sum(contact_mesh_id == mesh_idx))
                    collision_details[other_object_id] = count
                
                collision_record[object_id] = {
                    "collision_points": len(contact_points),
                    "collision_details": collision_details
                }
            except Exception as e:
                print(f"Warning: Collision detection failed for {object_id}: {e}")
                collision_record[object_id] = {
                    "collision_points": 0,
                    "collision_details": {}
                }
        
        return collision_record

    def simulate_the_scene(self):
        """
        Simulate the scene.
        """
        stage = omni.usd.get_context().get_stage()

        prims, prim_paths = get_all_prims_with_paths(self.track_ids)
        traced_data_all = start_simulation_and_track(
            prims, prim_paths, simulation_steps=120, longterm_equilibrium_steps=120,
            early_stop_unstable_exemption_prim_paths=prim_paths
        )

        unstable_prims = []
        unstable_object_ids = []
        for object_id, (prim_path, traced_data) in zip(self.track_ids, traced_data_all.items()):
            if not traced_data["stable"]:
                unstable_prims.append(os.path.basename(prim_path))
                unstable_object_ids.append(object_id)

        if len(unstable_prims) > 0:
            next_step_message = f"""
The scene is unstable. Please check the following prims: {unstable_prims}; 
Suggestions: 
1. use move_one_object_with_condition_in_room(room_id, condition) to adjust the placement of those unstable objects.
2. use place_objects_in_room(room_id, 'remove [object_description]') to remove the unstable objects.
"""
        else:
            next_step_message = "The scene is stable. You can continue to the next step."

        return {
            "status": "success",
            "message": "Scene simulated successfully!",
            "unstable_objects": unstable_object_ids,
            "next_step": next_step_message,
            "traced_data_all": traced_data_all,
            # "simulation_result": {os.path.basename(k): v for k, v in traced_data_all.items()},
        }



def generate_physics_statistics(traced_data_all, track_ids, collision_record):
    """
    Generate physics statistics from traced simulation data and collision record.
    
    Args:
        traced_data_all: Dictionary mapping prim paths to traced data
        track_ids: List of object IDs that were tracked
        collision_record: Pre-computed collision data from initial placement
    
    Returns:
        Dictionary containing physics statistics in the desired format
    """
    statistics = {
        "objects": {},
        "total_objects": 0,
        "stable_objects": 0,
        "unstable_objects": 0,
        "stability_ratio": 0.0
    }
    
    # Generate statistics for each object
    for object_id, (prim_path, traced_data) in zip(track_ids, traced_data_all.items()):
        # Extract data
        initial_pos = traced_data["initial_position"]
        final_pos = traced_data["final_position"]
        initial_orient = traced_data["initial_orientation"]
        final_orient = traced_data["final_orientation"]
        stable = traced_data["stable"]
        
        # Calculate position offset
        position_offset = (final_pos - initial_pos).tolist()
        position_offset_magnitude = float(np.linalg.norm(final_pos - initial_pos))
        
        # Calculate orientation angle offset using quaternion_angle function
        orientation_angle_offset = float(quaternion_angle(initial_orient, final_orient))
        
        # Get collision data from pre-computed record
        obj_collision_data = collision_record.get(object_id, {
            "collision_points": 0,
            "collision_details": {}
        })
        
        # Store statistics for this object
        statistics["objects"][object_id] = {
            "stable": bool(stable),
            "position_offset": position_offset,
            "position_offset_magnitude": position_offset_magnitude,
            "orientation_angle_offset": orientation_angle_offset,
            "collision_points": obj_collision_data["collision_points"],
            "collision_details": obj_collision_data["collision_details"]
        }
        
        # Update counters
        statistics["total_objects"] += 1
        if stable:
            statistics["stable_objects"] += 1
        else:
            statistics["unstable_objects"] += 1
    
    # Calculate stability ratio
    if statistics["total_objects"] > 0:
        statistics["stability_ratio"] = statistics["stable_objects"] / statistics["total_objects"]
    
    # Add collision summary
    total_collision_points = sum(obj["collision_points"] for obj in statistics["objects"].values())
    objects_with_collisions = sum(1 for obj in statistics["objects"].values() if obj["collision_points"] > 0)
    objects_without_collisions = statistics["total_objects"] - objects_with_collisions
    
    statistics["collision_summary"] = {
        "total_objects_with_collisions": objects_with_collisions,
        "total_objects_without_collisions": objects_without_collisions,
        "total_collision_points": total_collision_points
    }
    
    return statistics


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
    parser = argparse.ArgumentParser()
    parser.add_argument("layout_id", type=str)
    args = parser.parse_args()
    
    layout_id = args.layout_id
    if layout_id.startswith("room_"):
        layout_path, json_path = find_layout_dir(layout_id)
        room_id = layout_id
        layout_id = os.path.basename(layout_path)
    else:
        layout_path = os.path.join(RESULTS_DIR, layout_id)
        json_path = os.path.join(layout_path, f"{layout_id}.json")
        with open(json_path, 'r') as f:
            layout_data = json.load(f)
        room_id = layout_data["rooms"][0]["id"]

    print(f"layout_id: {layout_id}, room_id: {room_id}")
    print(f"layout_path: {layout_path}, json_path: {json_path}")
    
    eval_physics = EvalPhysics()

    eval_physics.create_single_room_layout_scene(layout_path, room_id)
    # TODO save the collision record to the physics simulaiton result file as well

    result = eval_physics.simulate_the_scene()
    # print(result)
    unstable_objects = result["unstable_objects"]
    traced_data_all = result["traced_data_all"]

    # Generate physics statistics with pre-computed collision record
    physics_statistics = generate_physics_statistics(traced_data_all, eval_physics.track_ids, eval_physics.collision_record)
    
    # Print summary
    print(f"\n=== Physics Simulation Results ===")
    print(f"Total objects: {physics_statistics['total_objects']}")
    print(f"Stable objects: {physics_statistics['stable_objects']}")
    print(f"Unstable objects: {physics_statistics['unstable_objects']}")
    print(f"Stability ratio: {physics_statistics['stability_ratio']:.2%}")
    print(f"\n=== Collision Summary (Initial Placement) ===")
    print(f"Objects with collisions: {physics_statistics['collision_summary']['total_objects_with_collisions']}")
    print(f"Objects without collisions: {physics_statistics['collision_summary']['total_objects_without_collisions']}")
    print(f"Total collision points: {physics_statistics['collision_summary']['total_collision_points']}")
    
    # Save statistics as JSON file
    stats_json_path = os.path.join(layout_path, f"{room_id}_physics_statistics.json")
    with open(stats_json_path, "w") as f:
        json.dump(physics_statistics, f, indent=2)
    print(f"\nPhysics statistics saved to: {stats_json_path}")
    
    # Save traced_data_all as pickle file (for detailed analysis)
    with open(os.path.join(layout_path, f"{room_id}_simulation_result.pkl"), "wb") as f:
        pickle.dump(traced_data_all, f)
    print(f"Detailed simulation data saved to: {os.path.join(layout_path, f'{room_id}_simulation_result.pkl')}")

