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
from objects.get_objects import get_object_candidates
from models import Room, Object, Point3D, Dimensions, FloorPlan, Euler
from typing import List
import uuid
import os
import numpy as np
import pickle
from PIL import Image
from constants import RESULTS_DIR
import sys
import torch
from pytorch3d.io import save_obj
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

def process_single_object(object_name: str, object_info: dict, room: Room, object_save_dir: str, selection_source = "generation"):
    """
    Process a single object type and return selected objects and updated recommendations.
    
    Args:
        object_name: Name of the object type
        object_info: Dictionary containing object information
        room: Room object with dimensions
        object_save_dir: Directory to save object files
        
    Returns:
        tuple: (selected_objects, updated_recommendations) for this object type
    """
    print(f"Selecting objects for {object_name}", file=sys.stderr)

    object_info["type"] = object_name

    object_description = object_info["description"]
    object_location = object_info["location"]
    object_size = object_info["size"]
    object_quantity = object_info["quantity"]
    object_variance_type = object_info["variance_type"]
    object_place_guidance = object_info.get("place_guidance", f"Standard placement for {object_name}")

    

    candidates = get_object_candidates(object_info, selection_source)

    # in each candidate, we can access the mesh by candidate["mesh"]
    # the mesh is trimesh object, with units in meters

    # we need to select the best objects (object_quantity) from the candidates
    # first filter the candidates by the object sizes of candidates, the size should not larger than the room dimension
    
    # Convert target size from centimeters to meters for comparison
    target_size_m = [s / 100.0 for s in object_size]  # [length, width, height] in meters
    
    # Filter candidates by room dimensions
    room_dims = room.dimensions

    if selection_source == "objaverse":
        filtered_candidates = []
        
        for candidate in candidates:
            mesh = candidate["mesh"]
            # Get mesh bounding box dimensions
            bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            mesh_dims = bounds[1] - bounds[0]  # [width, length, height] in meters
            
            # Check if object fits in room (with some margin for placement flexibility)
            # margin = 0.1
            # if (mesh_dims[0] <= room_dims.width - margin and 
            #     mesh_dims[1] <= room_dims.length - margin and 
            #     mesh_dims[2] <= room_dims.height * 0.9):
            #     filtered_candidates.append(candidate)

            filtered_candidates.append(candidate)
        if len(filtered_candidates) == 0:
            return [], []
    elif selection_source == "generation":
        # replicate the candidates to make sure the quantity is enough
        filtered_candidates = candidates * object_quantity
    else:
        assert False, "Only objaverse and generation are supported for now"

    # then filter the candidates by the object sizes of candidates, choose the most similar sizes to the object_size (object_info["size"], which is a list of three numbers [length, width, height] in centimeters) in object_info
    
    # Rank candidates by size similarity
    def size_similarity_score(candidate):
        mesh = candidate["mesh"]
        bounds = mesh.bounds
        mesh_dims = bounds[1] - bounds[0]  # [width, length, height] in meters
        
        # Calculate similarity score using L1 distance (lower is better)
        score = 0
        for i in range(3):
            score += abs(mesh_dims[i] - target_size_m[i])
        return score
    
    # Sort by similarity score (best matches first)
    filtered_candidates.sort(key=size_similarity_score)
    
    # Select top candidates up to quantity
    selected_candidates = filtered_candidates[:object_quantity]
    
    selected_objects = []
    updated_recommendations = []
    
    # Create Object instances from selected candidates
    for i, candidate in enumerate(selected_candidates):

        mesh = candidate["mesh"]
        bounds = mesh.bounds
        mesh_dims = bounds[1] - bounds[0]
        
        # Create unique object ID
        unique_object_id = f"{room.id}_{object_name}_{str(uuid.uuid4())[:8]}"
        
        obj = Object(
            id=unique_object_id,
            room_id=room.id,
            type=object_name,
            description=object_description,
            position=Point3D(0, 0, 0),  # Will be positioned later by placement logic
            rotation=Euler(0, 0, 0), # will be positioned later by placement logic
            dimensions=Dimensions(
                width=mesh_dims[0],   # x-dimension
                length=mesh_dims[1],  # y-dimension  
                height=mesh_dims[2]   # z-dimension
            ),
            source=candidate["source"],
            source_id=candidate["source_id"],
            place_id=object_location,
            place_guidance=object_place_guidance,
            mass=candidate.get("mass", 1.0)
        )
        selected_objects.append(obj)
        
        # Create updated recommendation with unique object information
        object_info_wo_quantity = object_info.copy()
        object_info_wo_quantity.update({
            "object_id": unique_object_id,  # Include unique object ID
            "type": object_name,  # Ensure type is correct
            "quantity": 1,  # Set to 1 for this specific instance
            "size": [mesh_dims[0] * 100, mesh_dims[1] * 100, mesh_dims[2] * 100],  # Actual mesh size in cm
            "actual_source": candidate["source"],  # Track actual source used
            "actual_source_id": candidate["source_id"]  # Track actual source ID used
        })

        updated_recommendations.append(object_info_wo_quantity)

        # save the object to the object_save_dir
        save_path = os.path.join(object_save_dir, f"{obj.source}", f"{obj.source_id}.ply")
        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            mesh.export(save_path)

            texture = candidate["texture"]
            texture_save_path = os.path.join(object_save_dir, f"{obj.source}", f"{obj.source_id}_texture.png")
            if not os.path.exists(texture_save_path):
                Image.fromarray((texture * 255).astype(np.uint8)).save(texture_save_path)
            
            tex_coords = candidate["tex_coords"]
            tex_coords_save_path = os.path.join(object_save_dir, f"{obj.source}", f"{obj.source_id}_tex_coords.pkl")
            if not os.path.exists(tex_coords_save_path):
                with open(tex_coords_save_path, 'wb') as f:    
                    pickle.dump(tex_coords, f)
            
            if "pbr_parameters" in candidate:
                pbr_parameters = candidate["pbr_parameters"]
                pbr_parameters_save_path = os.path.join(object_save_dir, f"{obj.source}", f"{obj.source_id}_pbr_parameters.json")
                if not os.path.exists(pbr_parameters_save_path):
                    with open(pbr_parameters_save_path, 'w') as f:
                        json.dump(pbr_parameters, f)

            save_obj(
                os.path.join(object_save_dir, f"{obj.source}", f"{obj.source_id}.obj"),
                torch.from_numpy(candidate["mesh"].vertices),
                torch.from_numpy(candidate["mesh"].faces),
                verts_uvs=torch.from_numpy(candidate["tex_coords"]["vts"]),
                faces_uvs=torch.from_numpy(candidate["tex_coords"]["fts"]),
                texture_map=torch.from_numpy(candidate["texture"])
            )
    
    return selected_objects, updated_recommendations

def select_objects(object_info_dict: dict, room: Room, existing_objects: List[Object], current_layout: FloorPlan, selection_source = "generation"):

    object_save_dir = f"{RESULTS_DIR}/{current_layout.id}"
    os.makedirs(object_save_dir, exist_ok=True)

    selected_objects = []
    updated_recommendations = []
    
    # Use ThreadPoolExecutor with maximum 5 threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks and store futures with their corresponding object names to maintain order
        future_to_object_name = {}
        object_names_order = list(object_info_dict.keys())
        
        for object_name in object_names_order:
            object_info = object_info_dict[object_name]
            future = executor.submit(process_single_object, object_name, object_info, room, object_save_dir, selection_source)
            future_to_object_name[future] = object_name
        
        # Collect results in the original order
        results = {}
        for future in as_completed(future_to_object_name):
            object_name = future_to_object_name[future]
            try:
                objects, recommendations = future.result()
                results[object_name] = (objects, recommendations)
                print(f"Object {object_name} selected {len(objects)} objects", file=sys.stderr)
            except Exception as exc:
                print(f'Object {object_name} generated an exception: {exc}', file=sys.stderr)
                results[object_name] = ([], [])
        
        # Add results to final lists in the original order
        for object_name in object_names_order:
            if object_name in results:
                objects, recommendations = results[object_name]
                selected_objects.extend(objects)
                updated_recommendations.extend(recommendations)

    return selected_objects, updated_recommendations
        

        