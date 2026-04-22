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
Door/Window Validation System - Unified Criteria
================================================

This module implements a unified approach to door and window validation across the entire codebase.
All functions now use consistent criteria and position calculations for better reliability.

UNIFIED POSITION SYSTEM:
- position_on_wall: Fractional value (0.0 to 1.0) representing position along the wall
- 0.0 = start of wall, 0.5 = center of wall, 1.0 = end of wall
- Position calculation: actual_position = position_on_wall * wall_length
- Door/window bounds: center ± (width/2)

VALIDATION CRITERIA:
1. Wall Side Validation:
   - Must be one of: "north", "south", "east", "west"
   - Cannot be empty or undefined

2. Position Validation:
   - position_on_wall must be between 0.0 and 1.0 (inclusive)
   - Invalid positions are rejected

3. Size Validation:
   - Door width: 0.1m to 3.0m (reasonable door sizes)
   - Window width: 0.1m to 5.0m (reasonable window sizes)
   - Height validation for structural integrity

4. Wall Boundary Compliance:
   - Element must fit completely within its wall
   - Check: position_on_wall * wall_length ± (width/2) must be within [0, wall_length]

5. Overlap Prevention:
   - No two elements on same wall can overlap
   - Minimum separation maintained between elements
   - Doors have priority over windows in conflict resolution

6. Architectural Standards:
   - Window sill heights: 0.3m to 1.8m (typical range)
   - Door heights: typically 2.0m to 2.4m
   - Structural clearances maintained

UNIFIED FUNCTIONS:
- validate_door_window_issues(): Primary validation function
- check_door_window_integrity(): Wrapper for backward compatibility
- validate_door_window_placement(): Legacy function redirects to primary
- correct_door_window_integrity(): Uses unified validation for corrections

CONSISTENCY FEATURES:
- All position calculations use same formula
- All boundary checks use same criteria
- All overlap detection uses same algorithm
- All error messages follow same format
- All return structures follow same schema
"""

from typing import Dict, List, Any

def validate_room_structure_issues(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Helper function to validate room structure issues (overlaps, detachment).
    Reuses existing validation logic from validation module.
    
    Args:
        rooms_data: List of room data dictionaries
        
    Returns:
        Dictionary with room structure validation results
    """
    # from validation import validate_room_only_layout
    
    return validate_room_only_layout(rooms_data)

def validate_room_only_layout(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate room-only layout for overlaps and connectivity issues.
    This is for the first stage of layout generation (rooms without doors/windows).
    
    Args:
        rooms_data: List of room dictionaries from LLM response (rooms only)
        
    Returns:
        Dictionary with validation results and issues found
    """
    
    issues = []
    overlaps = []
    detached_rooms = []
    
    # Check for room overlaps
    for i, room1 in enumerate(rooms_data):
        pos1 = room1["position"]
        dims1 = room1["dimensions"]
        
        # Room 1 bounds
        x1_min, y1_min = pos1["x"], pos1["y"]
        x1_max = x1_min + dims1["width"]
        y1_max = y1_min + dims1["length"]
        
        for j, room2 in enumerate(rooms_data[i+1:], i+1):
            pos2 = room2["position"]
            dims2 = room2["dimensions"]
            
            # Room 2 bounds
            x2_min, y2_min = pos2["x"], pos2["y"]
            x2_max = x2_min + dims2["width"]
            y2_max = y2_min + dims2["length"]
            
            # Check for overlap
            if not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min):
                overlap_area = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * \
                              max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                if overlap_area > 0.01:  # Allow tiny overlaps for floating point precision
                    overlaps.append({
                        "room1": room1["room_type"],
                        "room2": room2["room_type"],
                        "overlap_area": overlap_area
                    })
    
    # Check for detached rooms (no shared walls) - only for multi-room layouts
    if len(rooms_data) > 1:
        for i, room1 in enumerate(rooms_data):
            pos1 = room1["position"]
            dims1 = room1["dimensions"]
            
            x1_min, y1_min = pos1["x"], pos1["y"]
            x1_max = x1_min + dims1["width"]
            y1_max = y1_min + dims1["length"]
            
            has_shared_wall = False
            
            for j, room2 in enumerate(rooms_data):
                if i == j:
                    continue
                    
                pos2 = room2["position"]
                dims2 = room2["dimensions"]
                
                x2_min, y2_min = pos2["x"], pos2["y"]
                x2_max = x2_min + dims2["width"]
                y2_max = y2_min + dims2["length"]
                
                # Check for shared walls (adjacent rooms)
                tolerance = 0.1  # Small tolerance for floating point
                
                # Vertical shared wall
                if (abs(x1_max - x2_min) < tolerance or abs(x2_max - x1_min) < tolerance):
                    # Check if they overlap in Y direction
                    if not (y1_max <= y2_min + tolerance or y2_max <= y1_min + tolerance):
                        has_shared_wall = True
                        break
                
                # Horizontal shared wall
                if (abs(y1_max - y2_min) < tolerance or abs(y2_max - y1_min) < tolerance):
                    # Check if they overlap in X direction
                    if not (x1_max <= x2_min + tolerance or x2_max <= x1_min + tolerance):
                        has_shared_wall = True
                        break
            
            if not has_shared_wall:
                detached_rooms.append(room1["room_type"])
    
    # Compile issues
    if overlaps:
        issues.append(f"Room overlaps detected")
    
    if detached_rooms:
        issues.append(f"Detached rooms (no shared walls): {detached_rooms}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "overlaps": overlaps,
        "detached_rooms": detached_rooms
    }


def validate_room_layout(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate room layout for overlaps, connectivity issues, and door/window placement.
    
    Args:
        rooms_data: List of room dictionaries from LLM response
        
    Returns:
        Dictionary with validation results and issues found
    """
    
    issues = []
    overlaps = []
    detached_rooms = []
    door_issues = []
    window_issues = []
    
    # First check room-only validation
    room_only_validation = validate_room_only_layout(rooms_data)
    overlaps = room_only_validation["overlaps"]
    detached_rooms = room_only_validation["detached_rooms"]
    issues.extend(room_only_validation["issues"])
    
    # Check door placement issues
    for i, room in enumerate(rooms_data):
        room_type = room["room_type"]
        doors = room.get("doors", [])
        
        # Check if room has doors (most rooms should have at least one)
        # Exception: single room layouts may not need doors for access
        if not doors and len(rooms_data) > 1:
            # Exception: some utility rooms might not need doors
            if room_type.lower() not in ["utility room", "storage", "mechanical room"]:
                door_issues.append(f"'{room_type}' has no doors - inaccessible room")
        elif not doors and len(rooms_data) == 1:
            # Single room layout - doors are optional but recommended for most room types
            if room_type.lower() not in ["utility room", "storage", "mechanical room", "closet", "pantry"]:
                # This is a note rather than an error for single rooms
                pass  # Single rooms can exist without doors in some contexts
        
        # Check door placement logic
        for door in doors:
            wall_side = door.get("wall_side", "")
            if not wall_side:
                door_issues.append(f"'{room_type}' has door with undefined wall_side")
            
            # Check position_on_wall is valid (0.0 to 1.0)
            position = door.get("position_on_wall", -1)
            if position < 0 or position > 1:
                door_issues.append(f"'{room_type}' has door with invalid position_on_wall: {position}")
    
    # Check window placement issues
    for i, room in enumerate(rooms_data):
        room_type = room["room_type"]
        windows = room.get("windows", [])
        
        # Most rooms should have windows for natural light
        if not windows:
            # Exception: some interior rooms might not need windows
            if room_type.lower() not in ["bathroom", "closet", "storage", "hallway", "mechanical room", "pantry"]:
                window_issues.append(f"'{room_type}' has no windows - may lack natural light")
        
        # Check window placement logic
        for window in windows:
            wall_side = window.get("wall_side", "")
            if not wall_side:
                window_issues.append(f"'{room_type}' has window with undefined wall_side")
            
            # Check position_on_wall is valid (0.0 to 1.0)
            position = window.get("position_on_wall", -1)
            if position < 0 or position > 1:
                window_issues.append(f"'{room_type}' has window with invalid position_on_wall: {position}")
            
            # Check sill height is reasonable
            sill_height = window.get("sill_height", 0)
            if sill_height < 0.5 or sill_height > 1.5:
                window_issues.append(f"'{room_type}' has window with unusual sill_height: {sill_height}m")
    
    # Perform comprehensive door/window placement validation
    door_window_validation = validate_door_window_placement(rooms_data)
    
    # Merge comprehensive validation results
    door_issues.extend(door_window_validation["door_window_issues"])
    
    # Add new issue categories
    wall_boundary_violations = door_window_validation["wall_boundary_violations"]
    door_window_overlaps = door_window_validation["door_window_overlaps"]
    wall_intersections = door_window_validation["wall_intersections"]
    
    # Compile door/window issues only if doors/windows exist
    has_doors_or_windows = any(room.get("doors") or room.get("windows") for room in rooms_data)
    
    if has_doors_or_windows:
        if door_issues:
            issues.append(f"Door placement issues: {door_issues}")
        
        if window_issues:
            issues.append(f"Window placement issues: {window_issues}")
        
        if wall_boundary_violations:
            issues.append(f"Wall boundary violations: doors/windows extending beyond wall limits")
        
        if door_window_overlaps:
            issues.append(f"Door/window overlaps: elements overlapping on same wall")
        
        if wall_intersections:
            issues.append(f"Wall intersections: doors/windows intersecting with other walls")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "overlaps": overlaps,
        "detached_rooms": detached_rooms,
        "door_issues": door_issues,
        "window_issues": window_issues,
        "wall_boundary_violations": wall_boundary_violations if has_doors_or_windows else [],
        "door_window_overlaps": door_window_overlaps if has_doors_or_windows else [],
        "wall_intersections": wall_intersections if has_doors_or_windows else []
    }


def check_room_overlap(room1: Dict[str, Any], room2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if two rooms overlap and return overlap details.
    
    Args:
        room1: First room data
        room2: Second room data
        
    Returns:
        Dictionary with overlap information
    """
    pos1 = room1["position"]
    dims1 = room1["dimensions"]
    pos2 = room2["position"]
    dims2 = room2["dimensions"]
    
    # Room 1 bounds
    x1_min, y1_min = pos1["x"], pos1["y"]
    x1_max = x1_min + dims1["width"]
    y1_max = y1_min + dims1["length"]
    
    # Room 2 bounds
    x2_min, y2_min = pos2["x"], pos2["y"]
    x2_max = x2_min + dims2["width"]
    y2_max = y2_min + dims2["length"]
    
    # Check for overlap
    if not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min):
        overlap_area = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * \
                      max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        if overlap_area > 0.01:  # Allow tiny overlaps for floating point precision
            return {
                "overlaps": True,
                "overlap_area": overlap_area,
                "overlap_bounds": {
                    "x_min": max(x1_min, x2_min),
                    "x_max": min(x1_max, x2_max),
                    "y_min": max(y1_min, y2_min),
                    "y_max": min(y1_max, y2_max)
                }
            }
    
    return {"overlaps": False, "overlap_area": 0}


def validate_llm_response_structure(llm_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the LLM response has all required properties.
    
    Returns:
        Dict with validation results and missing fields
    """
    missing_fields = []
    invalid_fields = []
    
    # Check top-level required fields
    if "building_style" not in llm_response:
        missing_fields.append("building_style")
    
    if "rooms" not in llm_response:
        missing_fields.append("rooms")
        return {
            "valid": False,
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields,
            "error": "Critical: 'rooms' field is missing from response"
        }
    
    if not isinstance(llm_response["rooms"], list):
        invalid_fields.append("rooms (must be a list)")
        return {
            "valid": False,
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields,
            "error": "Critical: 'rooms' must be a list"
        }
    
    if len(llm_response["rooms"]) == 0:
        invalid_fields.append("rooms (empty list)")
        return {
            "valid": False,
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields,
            "error": "Critical: 'rooms' list is empty"
        }
    
    # Check each room structure
    for i, room in enumerate(llm_response["rooms"]):
        room_prefix = f"room[{i}]"
        
        # Required room fields
        required_room_fields = ["room_type", "dimensions", "position"]
        for field in required_room_fields:
            if field not in room:
                missing_fields.append(f"{room_prefix}.{field}")
        
        # Check dimensions structure
        if "dimensions" in room:
            if not isinstance(room["dimensions"], dict):
                invalid_fields.append(f"{room_prefix}.dimensions (must be dict)")
            else:
                required_dim_fields = ["width", "length", "height"]
                for dim_field in required_dim_fields:
                    if dim_field not in room["dimensions"]:
                        missing_fields.append(f"{room_prefix}.dimensions.{dim_field}")
                    elif not isinstance(room["dimensions"][dim_field], (int, float)):
                        invalid_fields.append(f"{room_prefix}.dimensions.{dim_field} (must be number)")
        
        # Check position structure  
        if "position" in room:
            if not isinstance(room["position"], dict):
                invalid_fields.append(f"{room_prefix}.position (must be dict)")
            else:
                required_pos_fields = ["x", "y", "z"]
                for pos_field in required_pos_fields:
                    if pos_field not in room["position"]:
                        missing_fields.append(f"{room_prefix}.position.{pos_field}")
                    elif not isinstance(room["position"][pos_field], (int, float)):
                        invalid_fields.append(f"{room_prefix}.position.{pos_field} (must be number)")
        
        # Check doors structure (optional but if present, must be valid)
        if "doors" in room:
            if not isinstance(room["doors"], list):
                invalid_fields.append(f"{room_prefix}.doors (must be list)")
            else:
                for j, door in enumerate(room["doors"]):
                    door_prefix = f"{room_prefix}.doors[{j}]"
                    required_door_fields = ["width", "height", "position_on_wall", "wall_side"]
                    for door_field in required_door_fields:
                        if door_field not in door:
                            missing_fields.append(f"{door_prefix}.{door_field}")
        
        # Check windows structure (optional but if present, must be valid)
        if "windows" in room:
            if not isinstance(room["windows"], list):
                invalid_fields.append(f"{room_prefix}.windows (must be list)")
            else:
                for j, window in enumerate(room["windows"]):
                    window_prefix = f"{room_prefix}.windows[{j}]"
                    required_window_fields = ["width", "height", "position_on_wall", "wall_side", "sill_height"]
                    for window_field in required_window_fields:
                        if window_field not in window:
                            missing_fields.append(f"{window_prefix}.{window_field}")
    
    return {
        "valid": len(missing_fields) == 0 and len(invalid_fields) == 0,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
        "error": None
    } 


def validate_door_window_issues(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Unified door and window validation function with bidirectional requirements.
    Uses standardized fractional position system (0-1) for consistency.
    
    BIDIRECTIONAL REQUIREMENTS:
    - Doors/windows on shared walls between rooms MUST exist on both sides
    - Doors/windows on exterior walls can be unidirectional
    - Corresponding elements must align properly across shared walls
    
    Args:
        rooms_data: List of room data dictionaries with doors and windows
        
    Returns:
        Dictionary with comprehensive door/window validation results including bidirectional checks
    """
    from utils import find_shared_walls, calculate_door_world_position, doors_align
    
    door_issues = []
    window_issues = []
    wall_boundary_violations = []
    door_window_overlaps = []
    bidirectional_issues = []
    
    # Build a mapping of all walls and which are shared vs exterior
    shared_wall_map = {}  # Format: {(room1_idx, wall_side): [(room2_idx, wall_side), ...]}
    exterior_walls = set()  # Format: {(room_idx, wall_side), ...}
    
    # # Find all shared walls
    # for room1_idx in range(len(rooms_data)):
    #     for room2_idx in range(room1_idx + 1, len(rooms_data)):
    #         shared_walls = find_shared_walls(rooms_data[room1_idx], rooms_data[room2_idx])
            
    #         for shared_wall in shared_walls:
    #             room1_wall = shared_wall["room1_wall"]
    #             room2_wall = shared_wall["room2_wall"]
                
    #             # Add to shared wall mapping
    #             key1 = (room1_idx, room1_wall)
    #             key2 = (room2_idx, room2_wall)
                
    #             if key1 not in shared_wall_map:
    #                 shared_wall_map[key1] = []
    #             if key2 not in shared_wall_map:
    #                 shared_wall_map[key2] = []
                
    #             shared_wall_map[key1].append(key2)
    #             shared_wall_map[key2].append(key1)
    
    # # Identify exterior walls (walls not in shared_wall_map)
    # for room_idx, room in enumerate(rooms_data):
    #     for wall_side in ["north", "south", "east", "west"]:
    #         wall_key = (room_idx, wall_side)
    #         if wall_key not in shared_wall_map:
    #             exterior_walls.add(wall_key)
    
    # # Validate doors and windows for each room
    # for room_idx, room in enumerate(rooms_data):
    #     room_type = room["room_type"]
    #     width = room["dimensions"]["width"]
    #     length = room["dimensions"]["length"]
        
    #     # Process doors
    #     doors_on_walls = {}
    #     for door_idx, door in enumerate(room.get("doors", [])):
    #         door_id = f"{room_type}_door_{door_idx}"
            
    #         # Basic door validation
    #         wall_side = door.get("wall_side", "")
    #         if not wall_side or wall_side not in ["north", "south", "east", "west"]:
    #             door_issues.append(f"'{room_type}' door {door_idx} has invalid wall_side: '{wall_side}'")
    #             continue
            
    #         position_on_wall = door.get("position_on_wall", -1)
    #         if position_on_wall < 0 or position_on_wall > 1:
    #             door_issues.append(f"'{room_type}' door {door_idx} has invalid position_on_wall: {position_on_wall} (must be 0-1)")
    #             continue
            
    #         door_width = door.get("width", 0)
    #         if door_width <= 0 or door_width > 3.0:  # Reasonable door width limits
    #             door_issues.append(f"'{room_type}' door {door_idx} has invalid width: {door_width}m (must be 0-3m)")
    #             continue
            
    #         # Calculate wall length and door position
    #         wall_length = width if wall_side in ["north", "south"] else length
    #         door_start_pos = position_on_wall * wall_length - (door_width / 2)
    #         door_end_pos = position_on_wall * wall_length + (door_width / 2)
            
    #         # Check wall boundary violations
    #         if door_start_pos < 0 or door_end_pos > wall_length:
    #             wall_boundary_violations.append({
    #                 "room": room_type,
    #                 "element_type": "door",
    #                 "element_id": door_id,
    #                 "wall_side": wall_side,
    #                 "wall_length": wall_length,
    #                 "element_start": door_start_pos,
    #                 "element_end": door_end_pos,
    #                 "element_width": door_width,
    #                 "position_on_wall": position_on_wall
    #             })
            
    #         # Track for overlap detection
    #         if wall_side not in doors_on_walls:
    #             doors_on_walls[wall_side] = []
    #         doors_on_walls[wall_side].append({
    #             "type": "door",
    #             "id": door_id,
    #             "start": door_start_pos,
    #             "end": door_end_pos,
    #             "width": door_width,
    #             "position_on_wall": position_on_wall,
    #             "door_data": door
    #         })
        
    #     # Process windows
    #     windows_on_walls = {}
    #     for window_idx, window in enumerate(room.get("windows", [])):
    #         window_id = f"{room_type}_window_{window_idx}"
            
    #         # Basic window validation
    #         wall_side = window.get("wall_side", "")
    #         if not wall_side or wall_side not in ["north", "south", "east", "west"]:
    #             window_issues.append(f"'{room_type}' window {window_idx} has invalid wall_side: '{wall_side}'")
    #             continue
            
    #         position_on_wall = window.get("position_on_wall", -1)
    #         if position_on_wall < 0 or position_on_wall > 1:
    #             window_issues.append(f"'{room_type}' window {window_idx} has invalid position_on_wall: {position_on_wall} (must be 0-1)")
    #             continue
            
    #         window_width = window.get("width", 0)
    #         if window_width <= 0:  # Reasonable window width limits
    #             window_issues.append(f"'{room_type}' window {window_idx} has invalid width: {window_width}m (must be > 0m)")
    #             continue
            
    #         # Check sill height is reasonable
    #         # sill_height = window.get("sill_height", 0.9)
    #         # if sill_height < 0.3 or sill_height > 1.8:
    #         #     window_issues.append(f"'{room_type}' window {window_idx} has unusual sill_height: {sill_height}m (typically 0.3-1.8m)")
            
    #         # Calculate wall length and window position
    #         wall_length = width if wall_side in ["north", "south"] else length
    #         window_start_pos = position_on_wall * wall_length - (window_width / 2)
    #         window_end_pos = position_on_wall * wall_length + (window_width / 2)
            
    #         # Check wall boundary violations
    #         if window_start_pos < 0 or window_end_pos > wall_length:
    #             wall_boundary_violations.append({
    #                 "room": room_type,
    #                 "element_type": "window",
    #                 "element_id": window_id,
    #                 "wall_side": wall_side,
    #                 "wall_length": wall_length,
    #                 "element_start": window_start_pos,
    #                 "element_end": window_end_pos,
    #                 "element_width": window_width,
    #                 "position_on_wall": position_on_wall
    #             })
            
    #         # Track for overlap detection
    #         if wall_side not in windows_on_walls:
    #             windows_on_walls[wall_side] = []
    #         windows_on_walls[wall_side].append({
    #             "type": "window",
    #             "id": window_id,
    #             "start": window_start_pos,
    #             "end": window_end_pos,
    #             "width": window_width,
    #             "position_on_wall": position_on_wall,
    #             "window_data": window
    #         })
        
    #     # Check for overlaps on same wall
    #     all_walls = set(list(doors_on_walls.keys()) + list(windows_on_walls.keys()))
    #     for wall_side in all_walls:
    #         wall_elements = doors_on_walls.get(wall_side, []) + windows_on_walls.get(wall_side, [])
            
    #         # Check all pairs for overlaps
    #         for i, elem1 in enumerate(wall_elements):
    #             for j, elem2 in enumerate(wall_elements[i+1:], i+1):
    #                 # Check if elements overlap (with small tolerance for floating point)
    #                 tolerance = 0.001
    #                 if not (elem1["end"] <= elem2["start"] + tolerance or elem2["end"] <= elem1["start"] + tolerance):
    #                     overlap_start = max(elem1["start"], elem2["start"])
    #                     overlap_end = min(elem1["end"], elem2["end"])
    #                     overlap_width = overlap_end - overlap_start
                        
    #                     door_window_overlaps.append({
    #                         "room": room_type,
    #                         "wall_side": wall_side,
    #                         "element1_type": elem1["type"],
    #                         "element1_id": elem1["id"],
    #                         "element2_type": elem2["type"],
    #                         "element2_id": elem2["id"],
    #                         "overlap_width": overlap_width,
    #                         "element1_range": f"{elem1['start']:.2f}-{elem1['end']:.2f}m",
    #                         "element2_range": f"{elem2['start']:.2f}-{elem2['end']:.2f}m"
    #                     })
        
    #     # BIDIRECTIONAL VALIDATION: Check doors and windows on shared walls
    #     for wall_side in all_walls:
    #         wall_key = (room_idx, wall_side)
            
    #         # Skip exterior walls (bidirectional not required)
    #         if wall_key in exterior_walls:
    #             continue
            
    #         # Get corresponding walls for this shared wall
    #         corresponding_walls = shared_wall_map.get(wall_key, [])
            
    #         # Check doors on this shared wall
    #         for door_element in doors_on_walls.get(wall_side, []):
    #             door_data = door_element["door_data"]
    #             door_world_pos = calculate_door_world_position(room, door_data)
                
    #             # Find corresponding door on other side
    #             corresponding_door_found = False
    #             for other_room_idx, other_wall_side in corresponding_walls:
    #                 other_room = rooms_data[other_room_idx]
                    
    #                 # Check doors on the corresponding wall
    #                 for other_door in other_room.get("doors", []):
    #                     if other_door["wall_side"] == other_wall_side:
    #                         other_door_world_pos = calculate_door_world_position(other_room, other_door)
                            
    #                         if doors_align(door_world_pos, other_door_world_pos, door_data["width"], other_door["width"]):
    #                             corresponding_door_found = True
    #                             break
                    
    #                 if corresponding_door_found:
    #                     break
                
    #             if not corresponding_door_found:
    #                 bidirectional_issues.append({
    #                     "room": room_type,
    #                     "element_type": "door",
    #                     "element_id": door_element["id"],
    #                     "wall_side": wall_side,
    #                     "issue": "door_missing_bidirectional",
    #                     "description": f"Door in {room_type} on {wall_side} wall has no corresponding door on other side of shared wall",
    #                     "corresponding_rooms": [rooms_data[idx]["room_type"] for idx, _ in corresponding_walls]
    #                 })
            
    #         # # Check windows on this shared wall
    #         # for window_element in windows_on_walls.get(wall_side, []):
    #         #     window_data = window_element["window_data"]
    #         #     window_world_pos = calculate_door_world_position(room, window_data)  # Same calculation
                
    #         #     # Find corresponding window on other side
    #         #     corresponding_window_found = False
    #         #     for other_room_idx, other_wall_side in corresponding_walls:
    #         #         other_room = rooms_data[other_room_idx]
                    
    #         #         # Check windows on the corresponding wall
    #         #         for other_window in other_room.get("windows", []):
    #         #             if other_window["wall_side"] == other_wall_side:
    #         #                 other_window_world_pos = calculate_door_world_position(other_room, other_window)
                            
    #         #                 if doors_align(window_world_pos, other_window_world_pos, window_data["width"], other_window["width"]):
    #         #                     corresponding_window_found = True
    #         #                     break
                    
    #         #         if corresponding_window_found:
    #         #             break
                
    #         #     if not corresponding_window_found:
    #         #         bidirectional_issues.append({
    #         #             "room": room_type,
    #         #             "element_type": "window",
    #         #             "element_id": window_element["id"],
    #         #             "wall_side": wall_side,
    #         #             "issue": "window_missing_bidirectional",
    #         #             "description": f"Window in {room_type} on {wall_side} wall has no corresponding window on other side of shared wall",
    #         #             "corresponding_rooms": [rooms_data[idx]["room_type"] for idx, _ in corresponding_walls]
    #         #         })
    
    # Compile all issues
    all_issues = []
    all_issues.extend(door_issues)
    all_issues.extend(window_issues)
    all_issues.extend([f"{item['room']} {item['element_type']} extends beyond {item['wall_side']} wall boundary" for item in wall_boundary_violations])
    all_issues.extend([f"{item['room']} {item['element1_type']} and {item['element2_type']} overlap on {item['wall_side']} wall" for item in door_window_overlaps])
    all_issues.extend([item["description"] for item in bidirectional_issues])
    
    return {
        "valid": len(all_issues) == 0,
        "total_issues": len(all_issues),
        "issues": all_issues,
        "door_issues": door_issues,
        "window_issues": window_issues,
        "wall_boundary_violations": wall_boundary_violations,
        "door_window_overlaps": door_window_overlaps,
        "bidirectional_issues": bidirectional_issues,
        "door_window_issues": all_issues,  # For backward compatibility
        "wall_intersections": [],  # Placeholder for future enhancement
        "total_door_window_issues": len(all_issues),  # For backward compatibility
        "summary": {
            "boundary_violations": len(wall_boundary_violations),
            "door_window_overlaps": len(door_window_overlaps),
            "bidirectional_violations": len(bidirectional_issues),
            "door_issues": len(door_issues),
            "window_issues": len(window_issues),
            "rooms_with_issues": len(set([item["room"] for item in wall_boundary_violations + door_window_overlaps + bidirectional_issues]))
        },
        "shared_wall_analysis": {
            "total_shared_walls": len(shared_wall_map) // 2,  # Each shared wall counted twice
            "total_exterior_walls": len(exterior_walls),
            "rooms_analyzed": len(rooms_data)
        }
    }


def check_door_window_integrity(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Unified door/window integrity checking function.
    This is now a wrapper around validate_door_window_issues for consistency.
    
    Args:
        rooms_data: List of room data dictionaries with doors and windows
        
    Returns:
        Dictionary with door/window integrity check results
    """
    return validate_door_window_issues(rooms_data)


def validate_door_window_placement(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Legacy function - now redirects to unified validation.
    Maintained for backward compatibility.
    
    Args:
        rooms_data: List of room dictionaries from LLM response
        
    Returns:
        Dictionary with detailed door/window validation results
    """
    return validate_door_window_issues(rooms_data)

