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
from typing import Dict, Any
from dataclasses import asdict
from models import FloorPlan, Room, Point3D, Dimensions, Door, Window
from utils import generate_unique_id, create_walls_for_room, extract_wall_side_from_id
from validation import validate_llm_response_structure
from llm_client import request_missing_information
import sys
import numpy as np
import os
from PIL import Image
import compress_pickle
import trimesh

def create_door_frame_material(door_texture_path: str, frame_texture_path: str, frame_coords_path: str):
    """
    Create door frame material using the average color of the door texture.
    
    Args:
        door_texture_path: Path to the door texture image
        frame_texture_path: Path where frame texture will be saved
        frame_coords_path: Path where frame texture coordinates will be saved
    """
    try:
        # Load door texture image
        if not os.path.exists(door_texture_path):
            print(f"Warning: Door texture not found at {door_texture_path}", file=sys.stderr)
            return
            
        door_image = np.array(Image.open(door_texture_path)).astype(np.float32) / 255.0
        
        # Calculate average color of the door texture
        # Ignore fully transparent pixels if there's an alpha channel
        if door_image.shape[2] == 4:  # RGBA
            # Use only pixels with alpha > 0.1
            mask = door_image[:, :, 3] > 0.1
            if np.any(mask):
                avg_color = door_image[mask][:, :3].mean(axis=0)
            else:
                avg_color = door_image[:, :, :3].mean(axis=(0, 1))
        else:  # RGB
            avg_color = door_image.mean(axis=(0, 1))
        
        # Create a solid color texture image for the door frame
        # Use a standard texture size
        frame_texture_size = 256
        frame_texture_image = np.ones((frame_texture_size, frame_texture_size, 3), dtype=np.float32)
        frame_texture_image[:, :, :] = avg_color
        
        # Create texture coordinates for the door frame mesh structure
        # The door frame consists of 3 box pieces: left, right, and top frame
        # Each box has 12 triangular faces, so we need coordinates for 36 faces total
        
        # Create a temporary door frame mesh to understand the actual structure
        # We'll use dummy dimensions since we only need the face structure
        temp_left = trimesh.creation.box([0.1, 0.1, 1.0])
        temp_right = trimesh.creation.box([0.1, 0.1, 1.0])
        temp_top = trimesh.creation.box([1.0, 0.1, 0.1])
        
        # Apply dummy offsets
        temp_left.apply_translation([-0.6, 0, 0])
        temp_right.apply_translation([0.6, 0, 0])
        temp_top.apply_translation([0, 0, 0.6])
        
        # Combine to match the actual door frame structure
        temp_frame_meshes = [temp_left, temp_right, temp_top]
        temp_combined_frame = trimesh.util.concatenate(temp_frame_meshes)
        
        # Get face normals from the combined mesh
        face_normals = temp_combined_frame.face_normals
        
        # Create unique texture coordinates for each face
        unique_vts = []
        unique_fts = []
        
        for face_idx, (face, normal) in enumerate(zip(temp_combined_frame.faces, face_normals)):
            # Get vertices for this face
            face_vertices = temp_combined_frame.vertices[face]
            
            # For door frame, we can use simple planar mapping for all faces
            # since it's a solid color texture
            if abs(normal[1]) > 0.9:  # Front or back face (Y-axis - thickness direction)
                # Map to full texture using X (width) and Z (height) coordinates
                min_x, max_x = face_vertices[:, 0].min(), face_vertices[:, 0].max()
                min_z, max_z = face_vertices[:, 2].min(), face_vertices[:, 2].max()
                
                # Avoid division by zero
                if max_x == min_x:
                    u_coords = [0.5, 0.5, 0.5]
                else:
                    u_coords = [(face_vertices[i, 0] - min_x) / (max_x - min_x) for i in range(3)]
                
                if max_z == min_z:
                    v_coords = [0.5, 0.5, 0.5]
                else:
                    v_coords = [(face_vertices[i, 2] - min_z) / (max_z - min_z) for i in range(3)]
                
                tex_coords = np.array([[u_coords[i], v_coords[i]] for i in range(3)])
                
            elif abs(normal[0]) > 0.9:  # Left/right faces (X-axis normals)
                # Map to full texture using Y (thickness) and Z (height) coordinates
                min_y, max_y = face_vertices[:, 1].min(), face_vertices[:, 1].max()
                min_z, max_z = face_vertices[:, 2].min(), face_vertices[:, 2].max()
                
                # Avoid division by zero
                if max_y == min_y:
                    u_coords = [0.5, 0.5, 0.5]
                else:
                    u_coords = [(face_vertices[i, 1] - min_y) / (max_y - min_y) for i in range(3)]
                
                if max_z == min_z:
                    v_coords = [0.5, 0.5, 0.5]
                else:
                    v_coords = [(face_vertices[i, 2] - min_z) / (max_z - min_z) for i in range(3)]
                
                tex_coords = np.array([[u_coords[i], v_coords[i]] for i in range(3)])
                
            elif abs(normal[2]) > 0.9:  # Top/bottom faces (Z-axis normals)
                # Map to full texture using X (width) and Y (thickness) coordinates
                min_x, max_x = face_vertices[:, 0].min(), face_vertices[:, 0].max()
                min_y, max_y = face_vertices[:, 1].min(), face_vertices[:, 1].max()
                
                # Avoid division by zero
                if max_x == min_x:
                    u_coords = [0.5, 0.5, 0.5]
                else:
                    u_coords = [(face_vertices[i, 0] - min_x) / (max_x - min_x) for i in range(3)]
                
                if max_y == min_y:
                    v_coords = [0.5, 0.5, 0.5]
                else:
                    v_coords = [(face_vertices[i, 1] - min_y) / (max_y - min_y) for i in range(3)]
                
                tex_coords = np.array([[u_coords[i], v_coords[i]] for i in range(3)])
            else:
                # Fallback (should not happen with box mesh)
                tex_coords = np.array([
                    [0.0, 0.0],
                    [0.0, 0.0], 
                    [0.0, 0.0]
                ])
            
            # Add to unique arrays
            start_idx = len(unique_vts)
            unique_vts.extend(tex_coords)
            unique_fts.append([start_idx, start_idx + 1, start_idx + 2])
        
        # Convert to numpy arrays
        vts = np.array(unique_vts, dtype=np.float32)
        fts = np.array(unique_fts, dtype=np.int32)
        
        # Flip V coordinates for proper texture mapping
        vts[:, 1] = 1 - vts[:, 1]
        
        # Debug information to verify shapes
        print(f"Door frame texture coordinates - vts shape: {vts.shape}, fts shape: {fts.shape}", file=sys.stderr)
        print(f"Door frame mesh faces: {len(temp_combined_frame.faces)}, expected fts rows: {len(temp_combined_frame.faces)}", file=sys.stderr)
        
        # Verify that fts shape matches the number of faces
        assert fts.shape[0] == len(temp_combined_frame.faces), f"fts shape {fts.shape[0]} doesn't match mesh faces {len(temp_combined_frame.faces)}"
        
        # Save the texture image and coordinates
        tex_coords_dict = {
            "vts": vts,  # np array of shape (N, 2)
            "fts": fts,  # np array of shape (M, 3)
        }
        compress_pickle.dump(tex_coords_dict, frame_coords_path)
        Image.fromarray((frame_texture_image * 255).astype(np.uint8)).save(frame_texture_path)
        
        print(f"Created door frame material: texture={frame_texture_path}, coords={frame_coords_path}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error creating door frame material: {e}", file=sys.stderr)

def parse_rooms_only_to_floor_plan(llm_response: Dict[str, Any], input_text: str) -> FloorPlan:
    """
    Convert rooms-only LLM response to FloorPlan data structure.
    This is for the first stage of layout generation (rooms without doors/windows).
    """
    
    # First validate the response structure for rooms-only
    validation_result = validate_room_only_response_structure(llm_response)
    
    if not validation_result["valid"]:
        error_details = f"Invalid rooms-only LLM response structure:\n"
        if validation_result["missing_fields"]:
            error_details += f"Missing fields: {', '.join(validation_result['missing_fields'])}\n"
        if validation_result["invalid_fields"]:
            error_details += f"Invalid fields: {', '.join(validation_result['invalid_fields'])}\n"
        if validation_result["error"]:
            error_details += f"Critical error: {validation_result['error']}"
        
        raise ValueError(error_details)
    
    rooms = []
    total_area = 0
    
    try:
        for i, room_data in enumerate(llm_response["rooms"]):
            room_id = generate_unique_id("room")
            
            # Create room position and dimensions with error handling
            try:
                position = Point3D(**room_data["position"])
            except (KeyError, TypeError) as e:
                raise ValueError(f"Room {i} position error: {str(e)}")
            
            try:
                dimensions = Dimensions(**room_data["dimensions"])
            except (KeyError, TypeError) as e:
                raise ValueError(f"Room {i} dimensions error: {str(e)}")
            
            total_area += dimensions.width * dimensions.length
            
            # Create walls
            walls = create_walls_for_room(room_data, room_id)
            
            # Create room without doors and windows
            try:
                if "room_type" not in room_data:
                    raise KeyError("Missing room_type")
                    
                room = Room(
                    id=room_id,
                    room_type=str(room_data["room_type"]),
                    position=position,
                    dimensions=dimensions,
                    walls=walls,
                    doors=[],  # Empty doors list for room-only stage
                    objects=[],  # Empty objects list for room-only stage
                    windows=[],  # Empty windows list for room-only stage
                    floor_material="hardwood",  # Default floor material
                    ceiling_height=dimensions.height
                )
                rooms.append(room)
                
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"Room {i} creation error: {str(e)}")
        
        # Create floor plan
        floor_plan = FloorPlan(
            id=generate_unique_id("layout"),
            rooms=rooms,
            total_area=total_area,
            building_style=str(llm_response.get("building_style", "Unknown")),
            description=f"Room structure generated from: {input_text}",
            created_from_text=input_text
        )
        
        return floor_plan
        
    except Exception as e:
        # Re-raise with more context
        raise ValueError(f"Failed to parse rooms-only LLM response to floor plan: {str(e)}")


def add_doors_windows_to_floor_plan(current_floor_plan: FloorPlan, doors_windows_response: Dict[str, Any]) -> FloorPlan:
    """
    Add doors and windows from Claude response to an existing floor plan.
    This is for the second stage of layout generation.
    """
    
    # Validate that we have the required doors/windows response structure
    if "rooms" not in doors_windows_response:
        raise ValueError("Invalid doors/windows LLM response structure: Missing 'rooms' field")
    
    if not isinstance(doors_windows_response["rooms"], list):
        raise ValueError("Invalid doors/windows LLM response structure: 'rooms' must be a list")
    
    if len(doors_windows_response["rooms"]) == 0:
        raise ValueError("Invalid doors/windows LLM response structure: 'rooms' list is empty")
    
    # Verify room count and types match
    response_rooms = doors_windows_response["rooms"]
    if len(response_rooms) != len(current_floor_plan.rooms):
        raise ValueError(f"Room count mismatch: existing {len(current_floor_plan.rooms)}, response {len(response_rooms)}")
    
    try:
        # Update each room with doors and windows
        import os
        import shutil
        from constants import RESULTS_DIR
        from foundation_models import get_clip_models
        from floor_plan_materials.door_material import DoorMaterialSelector
        from floor_plan_materials.window_material import WindowMaterialGenerator

        door_material_save_dir = os.path.join(RESULTS_DIR, current_floor_plan.id, "materials")
        os.makedirs(door_material_save_dir, exist_ok=True)
        
        # Initialize door material selector
        clip_model, clip_preprocess, clip_tokenizer = get_clip_models()
        door_material_selector = DoorMaterialSelector(clip_model, clip_preprocess, clip_tokenizer)

        # Initialize window material generator
        window_material_generator = WindowMaterialGenerator()
        window_material_save_dir = door_material_save_dir

        for i, room_data in enumerate(response_rooms):
            current_room = current_floor_plan.rooms[i]
            
            # Verify room type matches (basic consistency check)
            if room_data["room_type"] != current_room.room_type:
                raise ValueError(f"Room type mismatch at index {i}: existing '{current_room.room_type}', response '{room_data['room_type']}'")
            
            # Clear existing doors and windows
            current_room.doors = []
            current_room.windows = []
            
            # Create doors with validation
            for j, door_data in enumerate(room_data.get("doors", [])):
                try:
                    # Validate required door fields
                    required_fields = ["wall_side", "position_on_wall", "width", "height"]
                    for field in required_fields:
                        if field not in door_data:
                            raise KeyError(f"Missing door field: {field}")
                    
                    # Find the wall for this door                    
                    wall_side = door_data["wall_side"]
                    target_wall = next((w for w in current_room.walls if wall_side in w.id), None)
                    if target_wall:
                        # Get door material using DoorMaterialSelector
                        door_material_description = door_data.get("door_material", "standard wooden door")
                        door_material_id_index = door_material_selector.select_door(door_material_description)
                        door_material_id = door_material_selector.door_ids[int(door_material_id_index)]
                        
                        # Copy material files to save directory
                        from floor_plan_materials.door_material import HOLODECK_BASE_DATA_DIR
                        source_texture_path = os.path.join(HOLODECK_BASE_DATA_DIR, "doors/textures", f"{door_material_id}_texture.png")
                        source_coords_path = os.path.join(HOLODECK_BASE_DATA_DIR, "doors/textures", f"{door_material_id}_tex_coords.pkl")
                        
                        dest_texture_path = os.path.join(door_material_save_dir, f"{door_material_id}_texture.png")
                        dest_coords_path = os.path.join(door_material_save_dir, f"{door_material_id}_tex_coords.pkl")
                        
                        try:
                            if os.path.exists(source_texture_path):
                                shutil.copy2(source_texture_path, dest_texture_path)
                            if os.path.exists(source_coords_path):
                                shutil.copy2(source_coords_path, dest_coords_path)
                        except Exception as e:
                            print(f"Warning: Could not copy door material files for {door_material_id}: {e}", file=sys.stderr)
                        
                        # Create door frame texture using average color of door texture
                        door_frame_texture_path = os.path.join(door_material_save_dir, f"{door_material_id}_frame_texture.png")
                        door_frame_coords_path = os.path.join(door_material_save_dir, f"{door_material_id}_frame_tex_coords.pkl")
                        
                        if not os.path.exists(door_frame_texture_path) or not os.path.exists(door_frame_coords_path):
                            create_door_frame_material(dest_texture_path, door_frame_texture_path, door_frame_coords_path)
                        
                        door = Door(
                            id=generate_unique_id("door"),
                            wall_id=target_wall.id,
                            position_on_wall=float(door_data["position_on_wall"]),
                            width=float(door_data["width"]),
                            height=float(door_data["height"]),
                            door_type=door_data.get("door_type", "standard"),
                            opening=door_data.get("opening", False),  # Handle opening property
                            door_material=door_material_id
                        )
                        current_room.doors.append(door)
                    else:
                        raise ValueError(f"Invalid wall_side '{wall_side}' for door {j} in room {i}")
                        
                except (KeyError, ValueError, TypeError) as e:
                    raise ValueError(f"Room {i} door {j} error: {str(e)}")
            
            # Create windows with validation
            for j, window_data in enumerate([]):
            # for j, window_data in enumerate(room_data.get("windows", [])):
                try:

                    print("window_data", window_data)
                    # Validate required window fields
                    required_fields = ["wall_side", "position_on_wall", "width", "height", "sill_height"]
                    for field in required_fields:
                        if field not in window_data:
                            raise KeyError(f"Missing window field: {field}")
                    
                    # Find the wall for this window
                    wall_side = window_data["wall_side"]
                    target_wall = next((w for w in current_room.walls if wall_side in w.id), None)
                    if target_wall:
                        window_id = generate_unique_id("window")
                        
                        # Generate window material
                        window_grid = window_data.get("window_grid", [1, 1])
                        glass_color = window_data.get("glass_color", [204, 230, 255])
                        frame_color = window_data.get("frame_color", [77, 77, 77])
                        glass_color = [color / 255. for color in glass_color]
                        frame_color = [color / 255. for color in frame_color]
                        window_prompt = window_data.get("window_appearance_description", "standard window")

                        # Ensure grid is a valid tuple
                        if isinstance(window_grid, list) and len(window_grid) == 2:
                            window_grid = tuple(window_grid)
                        else:
                            window_grid = (1, 1)  # Default to fixed window

                        # print(f"window_grid: {window_grid}, glass_color: {glass_color}, frame_color: {frame_color}", file=sys.stderr)
                        
                        # Generate window material
                        window_material = window_material_generator.generate_window_material_color_diffusion(
                            window_grid_xy=window_grid,
                            color_glass=glass_color,
                            color_frame=frame_color
                        )

                        # window_material = window_material_generator.generate_window_material_from_prompt(
                        #     window_grid_xy=window_grid,
                        #     prompt=window_prompt
                        # )

                        # print(f"window_material: {window_material}", file=sys.stderr)
                        
                        # Save window material files
                        window_material_prefix = os.path.join(window_material_save_dir, window_id)
                        try:
                            saved_paths = window_material_generator.save_window_material(
                                window_material, window_material_prefix
                            )
                            # print(f"Saved window material for {window_id}: {saved_paths}", file=sys.stderr)
                        except Exception as e:
                            print(f"Warning: Could not save window material for {window_id}: {e}", file=sys.stderr)
                        
                        window = Window(
                            id=window_id,
                            wall_id=target_wall.id,
                            position_on_wall=float(window_data["position_on_wall"]),
                            width=float(window_data["width"]),
                            height=float(window_data["height"]),
                            sill_height=float(window_data["sill_height"]),
                            window_type=window_data.get("window_type", "standard"),
                            window_material=window_id  # Use window_id as material identifier
                        )
                        current_room.windows.append(window)
                    else:
                        raise ValueError(f"Invalid wall_side '{wall_side}' for window {j} in room {i}")
                        
                except (KeyError, ValueError, TypeError) as e:
                    raise ValueError(f"Room {i} window {j} error: {str(e)}")
        
        # Update floor plan metadata
        current_floor_plan.building_style = str(doors_windows_response.get("building_style", current_floor_plan.building_style))
        current_floor_plan.description = f"Complete layout with doors/windows: {current_floor_plan.created_from_text}"
        
        return current_floor_plan
        
    except Exception as e:
        # Re-raise with more context
        raise ValueError(f"Failed to add doors/windows to floor plan: {str(e)}")


def validate_room_only_response_structure(llm_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the structure of a room-only LLM response.
    """
    missing_fields = []
    invalid_fields = []
    error = None
    
    try:
        # Check top-level fields
        if "building_style" not in llm_response:
            missing_fields.append("building_style")
        elif not isinstance(llm_response["building_style"], str):
            invalid_fields.append("building_style (must be string)")
        
        if "rooms" not in llm_response:
            missing_fields.append("rooms")
            return {
                "valid": False,
                "error": "No rooms array found",
                "missing_fields": missing_fields,
                "invalid_fields": invalid_fields
            }
        
        if not isinstance(llm_response["rooms"], list):
            invalid_fields.append("rooms (must be array)")
            return {
                "valid": False,
                "error": "Rooms must be an array",
                "missing_fields": missing_fields,
                "invalid_fields": invalid_fields
            }
        
        if len(llm_response["rooms"]) == 0:
            return {
                "valid": False,
                "error": "Rooms array is empty",
                "missing_fields": missing_fields,
                "invalid_fields": invalid_fields
            }
        
        # Check each room structure
        for i, room in enumerate(llm_response["rooms"]):
            if not isinstance(room, dict):
                invalid_fields.append(f"room[{i}] (must be object)")
                continue
            
            # Required room fields for room-only stage
            required_room_fields = ["room_type", "dimensions", "position"]
            for field in required_room_fields:
                if field not in room:
                    missing_fields.append(f"room[{i}].{field}")
                elif field == "room_type" and not isinstance(room[field], str):
                    invalid_fields.append(f"room[{i}].{field} (must be string)")
                elif field in ["dimensions", "position"] and not isinstance(room[field], dict):
                    invalid_fields.append(f"room[{i}].{field} (must be object)")
            
            # Check dimensions structure
            if "dimensions" in room and isinstance(room["dimensions"], dict):
                required_dim_fields = ["width", "length", "height"]
                for dim_field in required_dim_fields:
                    if dim_field not in room["dimensions"]:
                        missing_fields.append(f"room[{i}].dimensions.{dim_field}")
                    elif not isinstance(room["dimensions"][dim_field], (int, float)):
                        invalid_fields.append(f"room[{i}].dimensions.{dim_field} (must be number)")
            
            # Check position structure
            if "position" in room and isinstance(room["position"], dict):
                required_pos_fields = ["x", "y", "z"]
                for pos_field in required_pos_fields:
                    if pos_field not in room["position"]:
                        missing_fields.append(f"room[{i}].position.{pos_field}")
                    elif not isinstance(room["position"][pos_field], (int, float)):
                        invalid_fields.append(f"room[{i}].position.{pos_field} (must be number)")
            
            # Ensure doors and windows are NOT present in room-only stage
            if "doors" in room:
                invalid_fields.append(f"room[{i}].doors (not allowed in room-only stage)")
            if "windows" in room:
                invalid_fields.append(f"room[{i}].windows (not allowed in room-only stage)")
    
    except Exception as e:
        error = f"Exception during validation: {str(e)}"
    
    return {
        "valid": len(missing_fields) == 0 and len(invalid_fields) == 0 and error is None,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
        "error": error
    }


def parse_llm_response_to_floor_plan(llm_response: Dict[str, Any], input_text: str) -> FloorPlan:
    """
    Convert LLM response to FloorPlan data structure with comprehensive validation.
    This function handles both complete layouts and room-only layouts.
    """
    
    # Check if this is a room-only response (no doors/windows)
    has_doors_or_windows = any(
        room.get("doors") or room.get("windows") 
        for room in llm_response.get("rooms", [])
    )
    
    if not has_doors_or_windows:
        # This is a room-only response
        return parse_rooms_only_to_floor_plan(llm_response, input_text)
    
    # This is a complete response with doors/windows - use original logic
    # First validate the response structure
    validation_result = validate_llm_response_structure(llm_response)
    
    if not validation_result["valid"]:
        error_details = f"Invalid LLM response structure:\n"
        if validation_result["missing_fields"]:
            error_details += f"Missing fields: {', '.join(validation_result['missing_fields'])}\n"
        if validation_result["invalid_fields"]:
            error_details += f"Invalid fields: {', '.join(validation_result['invalid_fields'])}\n"
        if validation_result["error"]:
            error_details += f"Critical error: {validation_result['error']}"
        
        raise ValueError(error_details)
    
    rooms = []
    total_area = 0
    
    try:
        for i, room_data in enumerate(llm_response["rooms"]):
            room_id = generate_unique_id("room")
            
            # Create room position and dimensions with error handling
            try:
                position = Point3D(**room_data["position"])
            except (KeyError, TypeError) as e:
                raise ValueError(f"Room {i} position error: {str(e)}")
            
            try:
                dimensions = Dimensions(**room_data["dimensions"])
            except (KeyError, TypeError) as e:
                raise ValueError(f"Room {i} dimensions error: {str(e)}")
            
            total_area += dimensions.width * dimensions.length
            
            # Create walls
            walls = create_walls_for_room(room_data, room_id)
            
            # Create doors with validation
            doors = []
            for j, door_data in enumerate(room_data.get("doors", [])):
                try:
                    # Validate required door fields
                    required_fields = ["wall_side", "position_on_wall", "width", "height"]
                    for field in required_fields:
                        if field not in door_data:
                            raise KeyError(f"Missing door field: {field}")
                    
                    # Find the wall for this door
                    wall_side = door_data["wall_side"]
                    target_wall = next((w for w in walls if wall_side in w.id), None)
                    if target_wall:
                        door = Door(
                            id=generate_unique_id("door"),
                            wall_id=target_wall.id,
                            position_on_wall=float(door_data["position_on_wall"]),
                            width=float(door_data["width"]),
                            height=float(door_data["height"]),
                            door_type=door_data.get("door_type", "standard"),
                            opening=door_data.get("opening", False),  # Handle opening property
                            door_material=door_data.get("door_material", "standard")
                        )
                        doors.append(door)
                    else:
                        raise ValueError(f"Invalid wall_side '{wall_side}' for door {j} in room {i}")
                        
                except (KeyError, ValueError, TypeError) as e:
                    raise ValueError(f"Room {i} door {j} error: {str(e)}")
            
            # Create windows with validation
            windows = []
            for j, window_data in enumerate(room_data.get("windows", [])):
                try:
                    # Validate required window fields
                    required_fields = ["wall_side", "position_on_wall", "width", "height", "sill_height"]
                    for field in required_fields:
                        if field not in window_data:
                            raise KeyError(f"Missing window field: {field}")
                    
                    # Find the wall for this window
                    wall_side = window_data["wall_side"]
                    target_wall = next((w for w in walls if wall_side in w.id), None)
                    if target_wall:
                        window = Window(
                            id=generate_unique_id("window"),
                            wall_id=target_wall.id,
                            position_on_wall=float(window_data["position_on_wall"]),
                            width=float(window_data["width"]),
                            height=float(window_data["height"]),
                            sill_height=float(window_data["sill_height"]),
                            window_type=window_data.get("window_type", "standard"),
                            window_material="standard"  # Default material for older function
                        )
                        windows.append(window)
                    else:
                        raise ValueError(f"Invalid wall_side '{wall_side}' for window {j} in room {i}")
                        
                except (KeyError, ValueError, TypeError) as e:
                    raise ValueError(f"Room {i} window {j} error: {str(e)}")
            
            # Create room with validation
            try:
                if "room_type" not in room_data:
                    raise KeyError("Missing room_type")
                    
                room = Room(
                    id=room_id,
                    room_type=str(room_data["room_type"]),
                    position=position,
                    dimensions=dimensions,
                    walls=walls,
                    doors=doors,
                    objects=[],  # Empty objects list for complete stage
                    windows=windows,
                    floor_material="hardwood",  # Default floor material
                    ceiling_height=dimensions.height
                )
                rooms.append(room)
                
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"Room {i} creation error: {str(e)}")
        
        # Create floor plan
        floor_plan = FloorPlan(
            id=generate_unique_id("layout"),
            rooms=rooms,
            total_area=total_area,
            building_style=str(llm_response.get("building_style", "Unknown")),
            description=f"Generated from: {input_text}",
            created_from_text=input_text
        )
        
        return floor_plan
        
    except Exception as e:
        # Re-raise with more context
        raise ValueError(f"Failed to parse LLM response to floor plan: {str(e)}")


def test_response_parsing() -> Dict[str, Any]:
    """
    Test the response parsing with a minimal valid structure.
    
    Returns:
        Detailed information about parsing capabilities
    """
    try:
        # Create a minimal test response
        test_response = {
            "building_style": "Test Layout",
            "rooms": [
                {
                    "room_type": "test room",
                    "dimensions": {"width": 4.0, "length": 5.0, "height": 2.7},
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "doors": [
                        {
                            "width": 0.9,
                            "height": 2.1,
                            "position_on_wall": 0.5,
                            "wall_side": "east",
                            "door_type": "standard"
                        }
                    ],
                    "windows": [
                        {
                            "width": 1.2,
                            "height": 1.0,
                            "position_on_wall": 0.5,
                            "wall_side": "south",
                            "sill_height": 0.9,
                            "window_type": "standard"
                        }
                    ]
                }
            ]
        }
        
        # Test validation
        validation_result = validate_llm_response_structure(test_response)
        
        # Test parsing
        try:
            floor_plan = parse_llm_response_to_floor_plan(test_response, "test input")
            parsing_success = True
            parsing_error = None
        except Exception as e:
            parsing_success = False
            parsing_error = str(e)
        
        return {
            "success": True,
            "validation_result": validation_result,
            "parsing_success": parsing_success,
            "parsing_error": parsing_error,
            "test_response_structure": {
                "has_building_style": "building_style" in test_response,
                "has_rooms": "rooms" in test_response,
                "rooms_count": len(test_response.get("rooms", [])),
                "first_room_fields": list(test_response["rooms"][0].keys()) if test_response.get("rooms") else []
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Test parsing failed: {str(e)}",
            "error_type": str(type(e))
        } 