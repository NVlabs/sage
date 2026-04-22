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
from models import Object, Room, FloorPlan, Point3D, Euler, Dimensions
from typing import List, Dict, Any, Tuple, Optional
import json
from key import ANTHROPIC_API_KEY
import anthropic
from vlm import call_vlm
from .get_objects import get_object_candidates
from .object_selection_planner import select_objects
from .object_placement_planner import (
    get_door_window_placements,
    solution_to_objects,
    get_room_layout_description,
    create_wall_coordinate_systems,
    create_wall_grid_points,
    calculate_impossible_wall_regions,
    filter_valid_wall_points,
    select_best_wall_placement,
    adjust_wall_object_position,
    create_wall_placed_object,
    visualize_wall_placement
)
from utils import extract_wall_side_from_id
import uuid
import os
import numpy as np
from shapely.geometry import Polygon, Point, box, LineString
from rtree import index
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import shutil
import base64
import io
import copy
import random
import time
import math
import sys
from constants import SERVER_ROOT_DIR
from visualizer import RoomVisualizer
from utils import extract_json_from_response


async def analyze_single_object_from_condition(room: Room, current_layout: FloorPlan, condition: str) -> Dict[str, Any]:
    """
    Use Claude API to analyze the condition and determine what single object to add to the room.
    
    Args:
        room: The target room
        current_layout: Current floor plan layout
        condition: Text condition describing what object to add
        
    Returns:
        Dictionary containing object information or error
    """
    api_key = ANTHROPIC_API_KEY
    if not api_key:
        return {"success": False, "error": "ANTHROPIC_API_KEY not available"}
    
    try:
        # Prepare detailed room information including doors, windows, and objects
        doors_info = []
        if room.doors:
            for door in room.doors:
                wall_side = extract_wall_side_from_id(door.wall_id) if door.wall_id else "unknown wall"
                doors_info.append(f"- Door (ID: {door.id}): {door.door_type} door on {wall_side}, {door.width:.1f}m wide, position {door.position_on_wall:.2f} along wall, opens {'inward' if door.opens_inward else 'outward'}")
        doors_text = "\n".join(doors_info) if doors_info else "- No doors"
        
        windows_info = []
        if room.windows:
            for window in room.windows:
                wall_side = extract_wall_side_from_id(window.wall_id) if window.wall_id else "unknown wall"
                windows_info.append(f"- Window (ID: {window.id}): {window.window_type} window on {wall_side}, {window.width:.1f}m wide, position {window.position_on_wall:.2f} along wall, sill height {window.sill_height:.1f}m")
        windows_text = "\n".join(windows_info) if windows_info else "- No windows"
        
        existing_objects_info = []
        if room.objects:
            for obj in room.objects:
                obj_x_rel = (obj.position.x - room.position.x)
                obj_y_rel = (obj.position.y - room.position.y)
                existing_objects_info.append(f"- {obj.type} (ID: {obj.id}): {obj.description}, positioned at ({obj_x_rel:.1f}m, {obj_y_rel:.1f}m) relative to room corner, dimensions {obj.dimensions.width:.1f}×{obj.dimensions.length:.1f}×{obj.dimensions.height:.1f}m, rotation {obj.rotation.z}°")
        existing_objects_text = "\n".join(existing_objects_info) if existing_objects_info else "- No existing objects"
        # print(f"existing_objects_text: {existing_objects_text}", file=sys.stderr)
        # Create analysis prompt
        analysis_prompt = f"""You are an interior design expert analyzing a request to add exactly ONE object to a room.

ROOM: {room.room_type} ({room.dimensions.width:.1f}m × {room.dimensions.length:.1f}m, {room.dimensions.width * room.dimensions.length:.1f}m²)

DOORS ({len(room.doors)}): {doors_text}
WINDOWS ({len(room.windows)}): {windows_text}
EXISTING OBJECTS ({len(room.objects)}): {existing_objects_text}

USER REQUEST: {condition.strip()}

TASK: Determine exactly ONE object to add based on the user's request, room type, available space, and existing objects.

PLACEMENT TARGET RULES:
- "floor": placing on the ground/floor (includes against walls, in corners, or anywhere on floor surface)
- "wall": ONLY for objects that are attached/mounted directly on the wall (like wall shelves, paintings, wall-mounted TVs)
- "object_id": placing onto/on top of a specific existing object

EXAMPLES:
- "add table against the wall" → "floor" (table sits on floor, positioned near wall)
- "add sofa to corner" → "floor" (sofa sits on floor in corner)
- "add picture on wall" → "wall" (picture is mounted on wall)
- "add lamp on table" → use table's object_id (lamp goes on top of table)
- "add picture on wall region that on top of the bed" → "wall" (picture is mounted on wall region that on top of the bed)

OUTPUT FORMAT:
```json
{{
    "success": true,
    "object_type": "precise_descriptive_name",
    "object_description": "Visual appearance only: materials, shape, style, color",
    "size_estimate": [length_cm, width_cm, height_cm],
    "placement_location": "floor" or "wall" or "existing_object_id from EXISTING OBJECTS IN ROOM",
    "placement_guidance": "Spatial instructions from user condition: wall references, alignments, distance relationships, priorities",
    "justification": "Why this object was chosen"
}}
```

NAMING REQUIREMENTS:
- Use descriptive names: "wooden_office_chair", "glass_coffee_table", "ceramic_table_lamp"
- Description: Physical characteristics only, no function/purpose
- Must work for 3D generation: "A model of {{object_type.replace('_', ' ')}}, {{object_description}}"

Analyze the condition now:"""

        # Call Claude API
        response = call_vlm(
            vlm_type="qwen",
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            temperature=1.0,
            messages=[
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ]
        )
        
        response_text = response.content[0].text.strip()
        
        # Parse JSON response
        try:
            # Extract JSON content using robust parsing
            json_content = extract_json_from_response(response_text)
            if not json_content:
                raise ValueError("Could not extract JSON content from Claude response")
            
            analysis_result = json.loads(json_content)
            
            # Validate required fields
            required_fields = ["object_type", "object_description", "size_estimate", "placement_location", "placement_guidance"]
            for field in required_fields:
                if field not in analysis_result:
                    return {"success": False, "error": f"Missing required field: {field}"}
            
            # Validate size_estimate format
            if not isinstance(analysis_result["size_estimate"], list) or len(analysis_result["size_estimate"]) != 3:
                analysis_result["size_estimate"] = [100, 50, 75]  # default size
            
            # Ensure all size values are positive
            try:
                analysis_result["size_estimate"] = [max(1, int(float(x))) for x in analysis_result["size_estimate"]]
            except (ValueError, TypeError):
                analysis_result["size_estimate"] = [100, 50, 75]
            
            # Add success flag if not present
            analysis_result["success"] = True

            # Replace spaces with underscores in object type
            analysis_result["object_type"] = analysis_result["object_type"].replace(" ", "_")
            
            # Add Claude API interaction data for debugging/logging
            analysis_result["claude_prompt"] = analysis_prompt
            analysis_result["claude_response"] = response_text
            
            return analysis_result
            
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Failed to parse Claude response as JSON: {e}"}
    
    except Exception as e:
        return {"success": False, "error": f"Error in analyze_single_object_from_condition: {e}"}


async def create_single_object(object_analysis: Dict[str, Any], room: Room, current_layout: FloorPlan) -> Tuple[Optional[Object], Dict[str, Any]]:
    """
    Create a single object based on the analysis result.
    
    Args:
        object_analysis: Result from analyze_single_object_from_condition
        room: Target room
        current_layout: Current floor plan layout
        
    Returns:
        Tuple of (created_object, creation_info)
    """
    creation_info = {"success": False, "error": None, "object_created": None}
    
    try:
        # Extract object information from analysis
        object_type = object_analysis["object_type"]
        object_description = object_analysis["object_description"]
        size_estimate = object_analysis["size_estimate"]
        placement_location = object_analysis["placement_location"]
        placement_guidance = object_analysis["placement_guidance"]
        # print(f"placement_location: {placement_location}", file=sys.stderr)
        
        # Create object info dict for object selection (similar to place_objects_in_room)
        object_info_dict = {
            object_type: {
                "description": object_description,
                "location": placement_location,
                "size": size_estimate,  # [length, width, height] in cm
                "quantity": 1,
                "variance_type": "same",
                "place_guidance": placement_guidance
            }
        }
        
        # Use existing select_objects function to get actual object candidates
        selected_objects, updated_recommendation_list = select_objects(
            object_info_dict, room, room.objects, current_layout
        )
        
        if not selected_objects:
            creation_info["error"] = "No suitable object candidates found"
            return None, creation_info
        
        # Take the first (and should be only) selected object
        created_object = selected_objects[0]
        
        creation_info["success"] = True
        creation_info["object_created"] = {
            "id": created_object.id,
            "type": created_object.type,
            "description": created_object.description,
            "dimensions": {
                "width": created_object.dimensions.width,
                "length": created_object.dimensions.length,
                "height": created_object.dimensions.height
            },
            "source": created_object.source,
            "source_id": created_object.source_id,
            "place_guidance": created_object.place_guidance,
        }
        
        return created_object, creation_info
        
    except Exception as e:
        creation_info["error"] = f"Error creating object: {e}"
        return None, creation_info


async def get_placement_location_from_claude_wall(room: Room, current_layout: FloorPlan, object_to_place: Object) -> Dict[str, Any]:
    """
    Placeholder function for wall placement using Claude API.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout
        object_to_place: Object that needs to be placed on wall
        
    Returns:
        Dictionary containing placement information
    """
    # TODO: Implement wall placement logic
    return {
        "success": True,
        "placement_type": "wall",
        "placement_reasoning": "Placeholder wall placement logic - will resolved in step 4",
        "design_strategy": "Wall placement placeholder"
    }


async def get_placement_location_from_claude_object(room: Room, current_layout: FloorPlan, object_to_place: Object, target_object_id: str) -> Dict[str, Any]:
    """
    Placeholder function for object-on-object placement using Claude API.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout
        object_to_place: Object that needs to be placed
        target_object_id: ID of the object to place on
        
    Returns:
        Dictionary containing placement information
    """
    # TODO: Implement object-on-object placement logic
    return {
        "success": True,
        "placement_type": "object",
        "target_object_id": target_object_id,
        "placement_reasoning": f"Placeholder object placement logic - would place on {target_object_id}",
        "design_strategy": "Object-on-object placement placeholder"
    }


def create_visualization_grid_points(room_poly: Polygon, initial_state: Dict[str, Any], grid_size: int = 20) -> List[Tuple[float, float]]:
    """
    Create grid points for visualization purposes. This is a simplified version of the DFS solver's
    grid creation that only creates points for visualization (no complex placement logic).
    
    Args:
        room_poly: Room polygon
        initial_state: Existing objects/doors/windows
        grid_size: Grid spacing in cm
        
    Returns:
        List of (x, y) grid points that are inside the room and not occupied
    """
    try:
        # Get room bounds
        minx, miny, maxx, maxy = room_poly.bounds
        
        # Create grid points
        grid_points = []
        x = minx
        while x <= maxx:
            y = miny
            while y <= maxy:
                point = Point(x, y)
                if room_poly.contains(point):
                    grid_points.append((x, y))
                y += grid_size
            x += grid_size
        
        # Remove points that are occupied by existing objects
        filtered_points = []
        for x, y in grid_points:
            point = Point(x, y)
            is_occupied = False
            
            for obj_id, (center, rotation, vertices, _) in initial_state.items():
                obj_poly = Polygon(vertices)
                if obj_poly.contains(point) or obj_poly.distance(point) < 10:  # 10cm buffer
                    is_occupied = True
                    break
            
            if not is_occupied:
                filtered_points.append((x, y))
        
        return filtered_points
        
    except Exception as e:
        print(f"Warning: Error creating visualization grid points: {e}")
        return []


async def get_placement_location_from_claude_floor(room: Room, current_layout: FloorPlan, object_to_place: Object) -> Dict[str, Any]:
    """
    Use Claude API with visualization to determine the optimal floor placement location for the object.
    This is the full implementation for floor placement.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout
        object_to_place: Object that needs to be placed on floor
        
    Returns:
        Dictionary containing placement information
    """
    api_key = ANTHROPIC_API_KEY
    if not api_key:
        return {"success": False, "error": "ANTHROPIC_API_KEY not available"}
    
    try:
        # Create room polygon and visualization
        max_wall_thickness_cm = 5  # Default thickness
        if room.walls:
            max_wall_thickness_cm = max(wall.thickness * 0.5 * 100 for wall in room.walls)
        
        # Create inner room polygon
        inner_width_cm = (room.dimensions.width * 100) - (2 * max_wall_thickness_cm)
        inner_length_cm = (room.dimensions.length * 100) - (2 * max_wall_thickness_cm)
        
        room_vertices = [
            (max_wall_thickness_cm, max_wall_thickness_cm),
            (max_wall_thickness_cm, max_wall_thickness_cm + inner_length_cm),
            (max_wall_thickness_cm + inner_width_cm, max_wall_thickness_cm + inner_length_cm),
            (max_wall_thickness_cm + inner_width_cm, max_wall_thickness_cm)
        ]
        room_poly = Polygon(room_vertices)
        
        # Get door and window placements
        initial_state = get_door_window_placements(room)
        
        # Add existing objects to initial state
        for obj in room.objects:
            if obj.id == object_to_place.id:
                continue  # Skip the object we're trying to place
                
            obj_x_cm = (obj.position.x - room.position.x) * 100
            obj_y_cm = (obj.position.y - room.position.y) * 100
            obj_width_cm = obj.dimensions.width * 100 + 5
            obj_length_cm = obj.dimensions.length * 100 + 5
            
            # Handle rotation
            object_rotation = obj.rotation.z
            if object_rotation == 0:
                obj_length_x_cm = obj_width_cm
                obj_length_y_cm = obj_length_cm
            elif object_rotation == 90:
                obj_length_x_cm = obj_length_cm
                obj_length_y_cm = obj_width_cm
            elif object_rotation == 180:
                obj_length_x_cm = obj_width_cm
                obj_length_y_cm = obj_length_cm
            elif object_rotation == 270:
                obj_length_x_cm = obj_length_cm
                obj_length_y_cm = obj_width_cm
            else:
                # For non-standard rotations, use original dimensions
                obj_length_x_cm = obj_width_cm
                obj_length_y_cm = obj_length_cm
            
            # Create object vertices
            obj_vertices = [
                (obj_x_cm - obj_length_x_cm/2, obj_y_cm - obj_length_y_cm/2),
                (obj_x_cm + obj_length_x_cm/2, obj_y_cm - obj_length_y_cm/2),
                (obj_x_cm + obj_length_x_cm/2, obj_y_cm + obj_length_y_cm/2),
                (obj_x_cm - obj_length_x_cm/2, obj_y_cm + obj_length_y_cm/2)
            ]
            
            initial_state[f"existing-{obj.id}"] = (
                (obj_x_cm, obj_y_cm),
                obj.rotation.z,
                obj_vertices,
                1
            )
        
        # Generate room visualization using RoomVisualizer
        room_visualization_base64, vis_path = generate_room_visualization_image(room, current_layout)
        
        # Prepare detailed room description
        room_description = get_detailed_room_description_for_placement(room, current_layout, object_to_place, initial_state)
        
        # Create placement prompt using constraint system + suggested position
        placement_prompt = f"""You are an experienced interior designer with expertise in space planning and furniture arrangement.

I am providing you with both a visual representation of the room and detailed text information. I need your help to provide both a suggested placement position and optimal constraints for a single object.

ROOM AND CONTEXT:
{room_description}

OBJECT TO PLACE:
- ID: {object_to_place.id}
- Type: {object_to_place.type}
- Description: {object_to_place.description}
- Dimensions: {object_to_place.dimensions.width*100:.0f} × {object_to_place.dimensions.length*100:.0f} × {object_to_place.dimensions.height*100:.0f} cm
- Placement Guidance: {object_to_place.place_guidance}

VISUAL ANALYSIS:
Please examine the room layout image I've provided. This image shows:
- The room boundaries and walls (rendered background)
- Brown rectangles: Door openings and swing areas that must be kept clear
- Cyan rectangles: Windows that provide natural light
- Colored rectangles with arrows: Existing furniture objects already placed in the room
- White arrows: Show the facing direction of existing furniture (+Y = up, -Y = down, +X = right, -X = left)
- Grid lines: Help with spatial understanding (each grid square represents distance)
- Text labels: Show object IDs for reference in constraints
- Coordinate system: (0,0) is at the bottom-left corner, x is right, y is up

COORDINATE SYSTEM:
- The room layout uses a coordinate system where (0,0) is at the bottom-left corner
- All measurements are in centimeters
- Grid points show possible placement locations

CONSTRAINT SYSTEM:
Available constraints:

1. GLOBAL CONSTRAINT (required):
   - edge: Place close to walls/room perimeter (when user requests wall placement, corner placement, or object needs wall support)
   - middle: Place in central/interior area of room (when user requests central placement, or object needs all-around access)

2. DISTANCE CONSTRAINTS (only if explicitly mentioned in user guidance):
   - near, [object_id]: Place near another object (50cm-150cm distance)
   - far, [object_id]: Place far from another object (150cm+ distance)

3. POSITION CONSTRAINTS (relative to target object's facing direction, NOT the visualization):
   - in front of, [object_id]: Position in front of another object (in the direction it faces)
   - around, [object_id]: Position around another object (typically for chairs around tables)
   - side of, [object_id]: Position to the left or right side of another object
   - left of, [object_id]: Position to the LEFT of another object (relative to its facing direction)
   - right of, [object_id]: Position to the RIGHT of another object (relative to its facing direction)
   
   IMPORTANT: "left", "right", and "front" are relative to the TARGET OBJECT'S facing direction (shown by white arrows), 
   NOT the absolute directions in the visualization. For example, if a chair faces +Y direction and you want to place 
   a table "in front of" it, the table should be placed in the +Y direction from the chair.

4. ALIGNMENT CONSTRAINTS:
   - center aligned, [object_id]: Align centers with another object

5. ROTATION CONSTRAINTS:
   - face to, [object_id]: Orient to face toward another object's center

CONSTRAINT ANALYSIS APPROACH:
Analyze the user's specific request and object placement guidance to determine constraints:

1. SPATIAL KEYWORDS IN USER REQUEST:
   - Look for explicit spatial instructions: "against wall", "corner", "center", "middle", "between objects", etc.
   - Pay attention to relative positioning: "near window", "close to door", "away from bed", etc.
   - IMPORTANT: Only add distance constraints (near/far) when explicitly mentioned in the user's guidance

2. CONTEXTUAL REQUIREMENTS:
   - Consider the specific function described in the user's request
   - Consider interaction patterns mentioned in the guidance
   - Consider the room layout and existing furniture arrangement

3. CONSTRAINT SELECTION:
   - "edge": Choose when user mentions wall placement, corner placement, or when the specific context suggests perimeter positioning
   - "middle": Choose when user mentions central placement, between objects, or when the specific context suggests interior positioning
   - AVOID defaulting based on object type - the same object type can be placed differently based on user intent
   - DO NOT add distance constraints (near/far) unless the user explicitly mentions distance relationships in their condition

TASK:
Based on the visual room layout, object placement guidance, and the constraint analysis above, provide BOTH a suggested position and optimal constraints.

Please respond with a JSON object in this exact format:

```json
{{
    "success": true,
    "suggested_position": {{
        "x": center_x_coordinate_in_cm,
        "y": center_y_coordinate_in_cm
    }},
    "constraints": [
        "constraint_1",
        "constraint_2",
        "constraint_3"
    ],
    "placement_reasoning": "Detailed explanation of why this position and constraints are optimal based on what you see in the image",
    "design_strategy": "Brief explanation of the overall design approach for this object"
}}
```

GUIDELINES:
1. Suggested position should be your best estimate of where the object center should be placed
2. Constraints should support and refine this placement decision
3. Work around existing objects - they cannot be moved
4. Use exact object IDs as shown in the image labels for constraints
5. Consider door swing areas and traffic flow paths
6. ANALYZE THE USER'S SPECIFIC REQUEST to choose constraints - avoid assumptions based on object type alone
7. NOTE: The current system only supports "edge" (near walls) or "middle" (central area) - choose based on the user's specific guidance and spatial requirements

CONSTRAINT FORMAT:
- First constraint must be either "edge" or "middle" (global constraint)
- Additional constraints should follow the format: "constraint_type, object_id" where applicable

EXAMPLES (context-dependent):
- User says "wardrobe against the right wall near window": {{"constraints": ["edge", "near, window-3", "far, bed_abc123"]}}
- User says "coffee table in front of sofa": {{"constraints": ["middle", "in front of, sofa_abc123", "center aligned, sofa_abc123"]}}
- User says "reading chair in the corner": {{"constraints": ["edge", "far, door-1", "near, window-2"]}}
- User says "chair around the dining table": {{"constraints": ["middle", "around, table_xyz789"]}}

SYSTEM LIMITATION NOTE: 
The current system only supports binary "edge" (near walls) vs "middle" (central) positioning. Ideally, a gradient system with options like "edge"/"intermediate"/"middle" would provide more nuanced placement, but you must choose between the two available options based on which better matches the user's specific request.

IMPORTANT: In your placement_reasoning, explain why you chose "edge" vs "middle" based specifically on the user's guidance and spatial requirements, not general object type assumptions.

Analyze the visual layout and provide your placement recommendation:"""

        # Check if visualization was generated successfully
        if not room_visualization_base64:
            return {
                "success": False, 
                "error": "Failed to generate room visualization",
                "debug_info": {
                    "visualization_method": "RoomVisualizer.visualize_2d_render"
                }
            }
        
        image_data = room_visualization_base64

        # Call Claude API with both image and text
        try:
            response = call_vlm(
                vlm_type="qwen",
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                temperature=0.2,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": placement_prompt
                            }
                        ]
                    }
                ]
            )
        except Exception as api_error:
            return {
                "success": False,
                "error": f"Claude API call failed: {api_error}",
                "debug_info": {
                    "api_error_type": type(api_error).__name__,
                    "visualization_path": vis_path,
                    "prompt_length": len(placement_prompt),
                    "image_data_length": len(image_data) if 'image_data' in locals() else 0
                }
            }
        
        response_text = response.content[0].text.strip()
        
        # Debug: Check if response is empty
        if not response_text:
            return {
                "success": False, 
                "error": "Claude returned empty response for placement analysis",
                "debug_info": {
                    "response_length": len(response_text),
                    "visualization_path": vis_path,
                    "prompt_length": len(placement_prompt)
                }
            }
        
        # Parse JSON response
        try:
            # Extract JSON content using robust parsing
            json_content = extract_json_from_response(response_text)
            
            # Check if we found valid JSON content
            if not json_content:
                return {
                    "success": False, 
                    "error": "Could not extract JSON content from Claude response",
                    "debug_info": {
                        "original_response": response.content[0].text,
                        "processed_response_preview": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                        "visualization_path": vis_path
                    }
                }
            
            placement_result = json.loads(json_content)
            
            # Validate required fields
            if "constraints" not in placement_result:
                return {"success": False, "error": "Missing constraints in Claude response"}
            if "suggested_position" not in placement_result:
                return {"success": False, "error": "Missing suggested_position in Claude response"}
            
            constraints = placement_result["constraints"]
            if not isinstance(constraints, list) or len(constraints) == 0:
                return {"success": False, "error": "Constraints must be a non-empty list"}
            
            # Validate that first constraint is a global constraint
            if constraints[0].lower() not in ["edge", "middle"]:
                return {"success": False, "error": "First constraint must be 'edge' or 'middle'"}
            
            # Validate suggested position
            suggested_pos = placement_result["suggested_position"]
            if not isinstance(suggested_pos, dict) or "x" not in suggested_pos or "y" not in suggested_pos:
                return {"success": False, "error": "suggested_position must contain x and y coordinates"}
            
            # Add metadata
            placement_result["visualization_path"] = vis_path
            placement_result["claude_interaction"] = {
                "prompt": placement_prompt,
                "response": response_text
            }
            
            return placement_result
            
        except json.JSONDecodeError as e:
            return {
                "success": False, 
                "error": f"Failed to parse Claude constraint response: {e}",
                "debug_info": {
                    "raw_response": response.content[0].text,
                    "processed_response": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                    "extracted_json_content": json_content if 'json_content' in locals() else "Not extracted",
                    "json_content_length": len(json_content) if 'json_content' in locals() and json_content else 0,
                    "visualization_path": vis_path,
                    "json_error": str(e)
                }
            }
    
    except Exception as e:
        return {"success": False, "error": f"Error in get_placement_location_from_claude_floor: {e}"}


async def get_placement_location_from_claude(room: Room, current_layout: FloorPlan, object_to_place: Object) -> Dict[str, Any]:
    """
    Main dispatcher function that determines placement location type and calls appropriate handler.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout
        object_to_place: Object that needs to be placed
        
    Returns:
        Dictionary containing placement information
    """
    try:
        # Get placement location from object's place_guidance or default to floor
        placement_location = getattr(object_to_place, 'place_id', 'not found')
        place_guidance = getattr(object_to_place, 'place_guidance', '').lower()
        
        # If not found in object, try to infer from place_guidance
        if placement_location == 'not found':
            if 'wall' in place_guidance or 'mount' in place_guidance or 'hang' in place_guidance:
                placement_location = 'wall'
            elif any(obj.id in place_guidance for obj in room.objects):
                # Find which object ID is mentioned
                for obj in room.objects:
                    if obj.id in place_guidance:
                        placement_location = obj.id
                        break
            else:
                placement_location = 'floor'
        
        # Dispatch to appropriate handler
        if placement_location == 'wall':
            result = await get_placement_location_from_claude_wall(room, current_layout, object_to_place)
        elif placement_location == 'floor':
            result = await get_placement_location_from_claude_floor(room, current_layout, object_to_place)
        # elif placement_location in [obj.id for obj in room.objects]:
        else:
            # Default to object placement
            result = await get_placement_location_from_claude_object(room, current_layout, object_to_place, placement_location)
        # else:
            # result = await get_placement_location_from_claude_floor(room, current_layout, object_to_place)
        
        # Add placement location info to result
        if result.get("success"):
            result["placement_location"] = placement_location
        
        return result
        
    except Exception as e:
        return {"success": False, "error": f"Error in get_placement_location_from_claude: {e}"}


def generate_room_visualization_image(room: Room, current_layout: FloorPlan) -> str:
    """
    Generate a room visualization image using RoomVisualizer.visualize_2d_render and return as base64 string.
    Returns base64 encoded PNG image, or None if visualization fails.
    """
    try:
        # Create room visualizer
        visualizer = RoomVisualizer(room, current_layout)
        
        # Create a temporary file path for the image
        temp_image_path = f"/tmp/room_viz_{room.id}_{int(time.time())}.png"

        vis_path = f"{SERVER_ROOT_DIR}/vis/room_addition_viz_{room.id}_{int(time.time())}.png"
        
        # Generate the visualization and save to file
        result_path = visualizer.visualize_2d_render(save_path=temp_image_path, show=False)
        # copy the image to the vis directory
        shutil.copy(temp_image_path, vis_path)
        
        if result_path and os.path.exists(temp_image_path):
            # Read the image file and convert to base64
            with open(temp_image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Clean up the temporary file
            os.remove(temp_image_path)
            
            return base64_image, vis_path
        else:
            print("Failed to generate room visualization")
            return None, None
            
    except Exception as e:
        print(f"Error generating room visualization: {str(e)}")
        return None, None


def create_room_visualization(room_poly: Polygon, grid_points: List[Tuple[float, float]], 
                            initial_state: Dict[str, Any], object_to_place: Object, room_id: str) -> str:
    """
    Create a visualization of the room layout for Claude to understand the placement context.
    
    Args:
        room_poly: Room polygon
        grid_points: Available grid points for placement
        initial_state: Existing objects, doors, windows
        object_to_place: Object that needs to be placed
        room_id: Room identifier
        
    Returns:
        Path to the saved visualization image
    """
    # Create vis directory if it doesn't exist
    vis_dir = f"{SERVER_ROOT_DIR}/vis"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    plt.rcParams["font.size"] = 10
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw room boundary
    x, y = room_poly.exterior.xy
    ax.plot(x, y, "-", label="Room Boundary", color="black", linewidth=3)
    
    # Draw grid points
    if grid_points:
        grid_x = [point[0] for point in grid_points]
        grid_y = [point[1] for point in grid_points]
        ax.scatter(grid_x, grid_y, s=8, color="lightgray", alpha=0.6, label="Available Placement Points")
    
    # Draw existing objects, doors, windows
    colors = ['brown', 'cyan', 'red', 'blue', 'green', 'orange', 'purple', 'pink']
    color_idx = 0
    
    for object_id, (center, rotation, vertices, _) in initial_state.items():
        center_x, center_y = center
        
        # Create polygon for the object
        obj_poly = Polygon(vertices)
        x_coords, y_coords = obj_poly.exterior.xy
        
        if object_id.startswith('door-'):
            ax.plot(x_coords, y_coords, "-", linewidth=3, color="brown")
            ax.fill(x_coords, y_coords, color="brown", alpha=0.5)
            ax.text(center_x, center_y, object_id, fontsize=8, ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        elif object_id.startswith('window-'):
            ax.plot(x_coords, y_coords, "-", linewidth=2, color="cyan")
            ax.fill(x_coords, y_coords, color="cyan", alpha=0.5)
            ax.text(center_x, center_y, object_id, fontsize=8, ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        else:
            # Existing furniture
            current_color = colors[color_idx % len(colors)]
            ax.plot(x_coords, y_coords, "-", linewidth=2, color=current_color)
            ax.fill(x_coords, y_coords, color=current_color, alpha=0.4)
            
            # Label with object type
            ax.text(center_x, center_y, object_id, fontsize=8, ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
            
            # Show orientation with arrow
            arrow_length = 20
            if rotation == 0:
                ax.arrow(center_x, center_y, 0, arrow_length, head_width=6, 
                        head_length=4, fc=current_color, ec=current_color, alpha=0.7)
            elif rotation == 90:
                ax.arrow(center_x, center_y, -arrow_length, 0, head_width=6, 
                        head_length=4, fc=current_color, ec=current_color, alpha=0.7)
            elif rotation == 180:
                ax.arrow(center_x, center_y, 0, -arrow_length, head_width=6, 
                        head_length=4, fc=current_color, ec=current_color, alpha=0.7)
            elif rotation == 270:
                ax.arrow(center_x, center_y, arrow_length, 0, head_width=6, 
                        head_length=4, fc=current_color, ec=current_color, alpha=0.7)
            
            color_idx += 1
    
    # Add title and labels
    ax.set_title(f"Room Layout - Placing {object_to_place.type}\nRoom: {room_id}", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel("X Position (cm)", fontsize=12)
    ax.set_ylabel("Y Position (cm)", fontsize=12)
    
    # Set equal aspect ratio and grid
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Create filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{vis_dir}/placement_{room_id}_{object_to_place.type}_{timestamp}.png"
    
    # Save figure
    plt.savefig(filename, bbox_inches="tight", dpi=150, facecolor='white')
    plt.close(fig)
    
    return filename


def parse_claude_constraints(claude_constraints: List[str], object_id: str, existing_objects: List[Object] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse Claude's constraint strings into the format expected by the DFS solver.
    
    Args:
        claude_constraints: List of constraint strings from Claude
        object_id: ID of the object being placed
        existing_objects: List of existing objects in the room
        
    Returns:
        Dictionary mapping object_id to list of constraint dictionaries
    """
    parsed_constraints = {object_id: []}
    
    # Create mapping of existing object IDs for reference validation
    if existing_objects is None:
        existing_objects = []
    existing_object_ids = {f"existing-{obj.id}" for obj in existing_objects}
    
    for constraint_str in claude_constraints:
        constraint_str = constraint_str.strip()
        
        if constraint_str.lower() in ["edge", "middle"]:
            # Global constraint
            parsed_constraints[object_id].append({
                "type": "global",
                "constraint": constraint_str.lower(),
                "target": None
            })
        
        elif "," in constraint_str:
            # Constraint with object reference
            parts = [part.strip() for part in constraint_str.split(",", 1)]
            if len(parts) == 2:
                constraint_type, target_object_id = parts
                
                # Map constraint types to solver categories
                constraint_mapping = {
                    "near": ("distance", "near"),
                    "far": ("distance", "far"),
                    "in front of": ("relative", "in front of"),
                    "behind": ("relative", "behind"),
                    "left of": ("relative", "left of"),
                    "right of": ("relative", "right of"),
                    "side of": ("relative", "side of"),
                    "face to": ("direction", "face to"),
                    "center aligned": ("alignment", "center aligned")
                }
                
                if constraint_type.lower() in constraint_mapping:
                    solver_type, solver_constraint = constraint_mapping[constraint_type.lower()]
                    
                    # Validate target object exists
                    if target_object_id in existing_object_ids or \
                        (not target_object_id.startswith("existing-") and f"existing-{target_object_id}" in existing_object_ids):
                        parsed_constraints[object_id].append({
                            "type": solver_type,
                            "constraint": solver_constraint,
                            "target": target_object_id if target_object_id.startswith("existing-") else f"existing-{target_object_id}"
                        })
                    # else:
                    #     # Handle special cases like "face to, window"
                    #     if constraint_type.lower() in ["face to", "near", "far"] and target_object_id.lower() in ["window", "door"]:
                    #         parsed_constraints[object_id].append({
                    #             "type": solver_type,
                    #             "constraint": solver_constraint,
                    #             "target": target_object_id.lower()
                    #         })
                    #     else:
                    #         # Skip invalid object references
                    #         print(f"Warning: Skipping constraint '{constraint_str}' - target object '{target_object_id}' not found")
                else:
                    print(f"Warning: Skipping unrecognized constraint type: '{constraint_type}'")
        else:
            # Single word constraints that might be rotation or special constraints
            if constraint_str.lower() in ["face to window", "face to door"]:
                parsed_constraints[object_id].append({
                    "type": "direction",
                    "constraint": "face to",
                    "target": constraint_str.split()[-1].lower()
                })
            else:
                print(f"Warning: Skipping unrecognized constraint format: '{constraint_str}'")
    
    # Ensure at least one global constraint exists
    has_global = any(c["type"] == "global" for c in parsed_constraints[object_id])
    if not has_global:
        # Default to edge placement
        parsed_constraints[object_id].insert(0, {"type": "global", "constraint": "edge", "target": None})
    
    return parsed_constraints


class AdditionFloorSolver:
    """
    Custom floor solver for single object placement that combines Claude's suggested position 
    with constraint-based ranking to find optimal placement.
    """
    
    def __init__(self, grid_size=20, constraint_bouns=0.2, position_tolerance=100):
        self.grid_size = grid_size
        self.constraint_bouns = constraint_bouns
        self.position_tolerance = position_tolerance  # cm tolerance around suggested position
        
        # Constraint type weights for scoring
        pass  # No func_dict needed - using direct method calls

        self.constraint_type2weight = {
            "relative": 0.5,
            "direction": 0.5,
            "alignment": 0.5,
            "distance": 1.8,
        }

        self.edge_bouns = 0.0
    
    def get_best_placement(self, room_poly, object_dim, suggested_position, constraints, initial_state):
        """
        Find the best placement by filtering grid points near suggested position 
        and ranking by constraints.
        
        Args:
            room_poly: Room polygon
            object_dim: Object dimensions (width_cm, length_cm)
            suggested_position: Claude's suggested position {"x": x_cm, "y": y_cm}
            constraints: List of constraint dictionaries
            initial_state: Existing objects/doors/windows
            
        Returns:
            Best placement tuple: (center, rotation, vertices, score) or None
        """
        try:
            # Step 1: Create all possible grid points
            grid_points = self.create_grids(room_poly)
            grid_points = self.remove_points(grid_points, initial_state)
            
            if not grid_points:
                return None
            
            # Step 2: Generate all possible solutions (positions + rotations)
            all_solutions = self.get_all_solutions(room_poly, grid_points, object_dim)
            if not all_solutions:
                return None
            
            # Step 3: Filter by collisions and wall facing
            solutions = self.filter_collision(initial_state, all_solutions)
            solutions = self.filter_facing_wall(room_poly, solutions, object_dim)
            
            if not solutions:
                return None
            
            # Step 4: Filter solutions near Claude's suggested position
            target_x = float(suggested_position["x"])
            target_y = float(suggested_position["y"])
            filtered_solutions = self.filter_by_suggested_position(
                solutions, target_x, target_y, self.position_tolerance
            )
            
            # If no solutions near suggested position, use all solutions as fallback
            if not filtered_solutions:
                print(f"Warning: No solutions found near suggested position ({target_x}, {target_y}). Using all valid solutions.")
                filtered_solutions = solutions
            
            # Step 5: Apply global constraints (edge/middle)
            candidate_solutions = self.apply_global_constraints(
                room_poly, filtered_solutions, object_dim, constraints
            )
            
            if not candidate_solutions:
                return None
            
            # Step 6: Rank solutions by constraints
            best_solution = self.rank_solutions_by_constraints(
                candidate_solutions, constraints, initial_state
            )
            
            return best_solution
            
        except Exception as e:
            print(f"Error in get_best_placement: {e}")
            return None
    
    def create_grids(self, room_poly):
        """Create grid points within the room polygon."""
        min_x, min_y, max_x, max_y = room_poly.bounds
        grid_points = []
        for x in range(int(min_x), int(max_x), self.grid_size):
            for y in range(int(min_y), int(max_y), self.grid_size):
                point = Point(x, y)
                if room_poly.contains(point):
                    grid_points.append((x, y))
        return grid_points
    
    def remove_points(self, grid_points, objects_dict):
        """Remove grid points that are occupied by existing objects."""
        if not objects_dict:
            return grid_points
            
        # Create an r-tree index
        idx = index.Index()
        
        # Populate the index with bounding boxes of the objects
        for i, (_, _, obj, _) in enumerate(objects_dict.values()):
            idx.insert(i, Polygon(obj).bounds)
        
        # Create Shapely Polygon objects only once
        polygons = [Polygon(obj) for _, _, obj, _ in objects_dict.values()]
        
        valid_points = []
        for point in grid_points:
            p = Point(point)
            # Get a list of potential candidates
            candidates = [polygons[i] for i in idx.intersection(p.bounds)]
            # Check if point is in any of the candidate polygons
            if not any(candidate.contains(p) for candidate in candidates):
                valid_points.append(point)
        
        return valid_points
    
    def get_all_solutions(self, room_poly, grid_points, object_dim):
        """Generate all possible object placements (position + rotation)."""
        obj_length, obj_width = object_dim
        obj_half_length, obj_half_width = obj_length / 2, obj_width / 2

        rotation_adjustments = {
            0: ((-obj_half_length, -obj_half_width), (obj_half_length, obj_half_width)),
            90: ((-obj_half_width, -obj_half_length), (obj_half_width, obj_half_length)),
            180: ((-obj_half_length, -obj_half_width), (obj_half_length, obj_half_width)),
            270: ((-obj_half_width, -obj_half_length), (obj_half_width, obj_half_length)),
        }

        solutions = []
        for rotation in [0, 90, 180, 270]:
            for point in grid_points:
                center_x, center_y = point
                lower_left_adjustment, upper_right_adjustment = rotation_adjustments[rotation]
                lower_left = (
                    center_x + lower_left_adjustment[0],
                    center_y + lower_left_adjustment[1],
                )
                upper_right = (
                    center_x + upper_right_adjustment[0],
                    center_y + upper_right_adjustment[1],
                )
                obj_box = box(*lower_left, *upper_right)

                if room_poly.contains(obj_box):
                    solutions.append([point, rotation, tuple(obj_box.exterior.coords[:]), 1.0])

        return solutions
    
    def filter_collision(self, objects_dict, solutions):
        """Filter out solutions that collide with existing objects."""
        if not objects_dict:
            return solutions
            
        valid_solutions = []
        object_polygons = [
            Polygon(obj_coords) for _, _, obj_coords, _ in objects_dict.values()
        ]
        for solution in solutions:
            sol_obj_coords = solution[2]
            sol_obj = Polygon(sol_obj_coords)
            if not any(sol_obj.intersects(obj) for obj in object_polygons):
                valid_solutions.append(solution)
        return valid_solutions
    
    def filter_facing_wall(self, room_poly, solutions, obj_dim):
        """Filter out solutions where front of object is too close to wall."""
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2

        front_center_adjustments = {
            0: (0, obj_half_width),
            90: (-obj_half_width, 0),
            180: (0, -obj_half_width),
            270: (obj_half_width, 0),
        }

        valid_solutions = []
        for solution in solutions:
            center_x, center_y = solution[0]
            rotation = solution[1]

            front_center_adjustment = front_center_adjustments[rotation]
            front_center_x, front_center_y = (
                center_x + front_center_adjustment[0],
                center_y + front_center_adjustment[1],
            )

            front_center_distance = room_poly.boundary.distance(
                Point(front_center_x, front_center_y)
            )

            if front_center_distance >= 10:  # 10cm minimum distance from wall
                valid_solutions.append(solution)

        return valid_solutions
    
    def filter_by_suggested_position(self, solutions, target_x, target_y, tolerance):
        """Filter solutions to those near Claude's suggested position."""
        filtered_solutions = []
        for solution in solutions:
            center_x, center_y = solution[0]
            distance = math.sqrt((center_x - target_x)**2 + (center_y - target_y)**2)
            if distance <= tolerance:
                # Add bonus score based on proximity to suggested position
                proximity_score = max(0, 1.0 - distance / tolerance)
                solution[3] += proximity_score * 0.5  # Boost score for closer solutions
                filtered_solutions.append(solution)
        return filtered_solutions
    
    def apply_global_constraints(self, room_poly, solutions, obj_dim, constraints):
        """Apply global constraints (edge/middle) to filter solutions."""
        # Find global constraint - constraints is a list of dicts with 'type' and 'constraint' keys
        global_constraint = None
        for constraint in constraints:
            if constraint.get("type") == "global" and constraint.get("constraint") in ["edge", "middle"]:
                global_constraint = constraint
                break
        
        # Default to edge if no global constraint found
        if not global_constraint:
            global_constraint = {"type": "global", "constraint": "edge"}
        
        if global_constraint["constraint"] == "edge":
            # Apply edge constraint
            edge_solutions = self.place_edge(room_poly, copy.deepcopy(solutions), obj_dim)
            return edge_solutions if edge_solutions else solutions
        else:
            # Middle constraint - return solutions as-is (they're already not at edges)
            return solutions
    
    def place_edge(self, room_poly, solutions, obj_dim):
        """Apply edge placement constraint."""
        valid_solutions = []
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2

        back_center_adjustments = {
            0: (0, -obj_half_width),
            90: (obj_half_width, 0),
            180: (0, obj_half_width),
            270: (-obj_half_width, 0),
        }

        for solution in solutions:
            center_x, center_y = solution[0]
            rotation = solution[1]

            back_center_adjustment = back_center_adjustments[rotation]
            back_center_x, back_center_y = (
                center_x + back_center_adjustment[0],
                center_y + back_center_adjustment[1],
            )

            back_center_distance = room_poly.boundary.distance(
                Point(back_center_x, back_center_y)
            )
            center_distance = room_poly.boundary.distance(Point(center_x, center_y))

            if (
                back_center_distance <= self.grid_size
                and back_center_distance < center_distance
            ):
                solution[3] += self.constraint_bouns

                # Move the object to the edge
                center2back_vector = np.array(
                    [back_center_x - center_x, back_center_y - center_y]
                )
                if np.linalg.norm(center2back_vector) > 0:
                    center2back_vector /= np.linalg.norm(center2back_vector)
                    offset = center2back_vector * back_center_distance
                    solution[0] = (center_x + offset[0], center_y + offset[1])
                    solution[2] = tuple([
                        (coord[0] + offset[0], coord[1] + offset[1])
                        for coord in solution[2]
                    ])
                valid_solutions.append(solution)

        return valid_solutions
    
    def rank_solutions_by_constraints(self, candidate_solutions, constraints, initial_state):
        """Rank solutions by how well they satisfy constraints."""
        if not candidate_solutions:
            return None
        
        # Initialize scores
        placement2score = {tuple(solution[:3]): solution[3] for solution in candidate_solutions}
        
        # Apply each constraint
        for constraint in constraints:
            if constraint.get("type") == "global":
                continue  # Already handled in global constraints
            
            constraint_type = constraint.get("type")
            constraint_value = constraint.get("constraint")
            target_id = constraint.get("target")
            
            if not constraint_type or not constraint_value:
                continue
            
            # Skip constraints that reference objects not in initial_state
            if target_id and target_id not in initial_state:
                continue
            
            # Apply constraint-specific scoring
            if constraint_type == "distance" and constraint_value in ["near", "far"]:
                self.apply_distance_constraint(
                    candidate_solutions, constraint_value, initial_state[target_id], placement2score
                )
            elif constraint_type == "relative" and constraint_value in ["in front of", "behind", "left of", "right of", "side of"]:
                self.apply_relative_constraint(
                    candidate_solutions, constraint_value, initial_state[target_id], placement2score
                )
            elif constraint_type == "direction" and constraint_value == "face to":
                self.apply_face_constraint(
                    candidate_solutions, initial_state[target_id], placement2score
                )
            elif constraint_type == "alignment" and constraint_value == "center aligned":
                self.apply_alignment_constraint(
                    candidate_solutions, initial_state[target_id], placement2score
                )
        
        # Find the best solution
        if not placement2score:
            return candidate_solutions[0] if candidate_solutions else None
        
        best_placement = max(placement2score, key=placement2score.get)
        best_solution = next(
            (sol for sol in candidate_solutions if tuple(sol[:3]) == best_placement), 
            None
        )
        
        if best_solution:
            best_solution[3] = placement2score[best_placement]
        
        return best_solution
    
    def apply_distance_constraint(self, solutions, distance_type, target_object, placement2score):
        """Apply distance constraints (near/far)."""
        target_coords = target_object[2]
        target_poly = Polygon(target_coords)
        
        distances = []
        for solution in solutions:
            sol_coords = solution[2]
            sol_poly = Polygon(sol_coords)
            distance = target_poly.distance(sol_poly)
            distances.append(distance)
        
        if not distances:
            return
        
        min_distance = min(distances)
        max_distance = max(distances)
        
        if distance_type == "near":
            if min_distance < 80:
                points = [(min_distance, 1), (80, 0), (max_distance, 0)]
            else:
                points = [(min_distance, 0), (max_distance, 0)]
        elif distance_type == "far":
            points = [(min_distance, 0), (max_distance, 1)]
        else:
            return
        
        if len(points) > 1:
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            f = interp1d(x, y, kind="linear", fill_value="extrapolate")
            
            for i, solution in enumerate(solutions):
                distance = distances[i]
                constraint_score = float(f(distance)) * self.constraint_type2weight["distance"]
                placement_key = tuple(solution[:3])
                if placement_key in placement2score:
                    placement2score[placement_key] += constraint_score
    
    def apply_relative_constraint(self, solutions, place_type, target_object, placement2score):
        """Apply relative position constraints."""
        _, target_rotation, target_coords, _ = target_object
        target_polygon = Polygon(target_coords)
        min_x, min_y, max_x, max_y = target_polygon.bounds
        mean_x = (min_x + max_x) / 2
        mean_y = (min_y + max_y) / 2

        comparison_dict = {
            "left of": {
                0: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y,
                90: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x,
                180: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y,
                270: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x,
            },
            "right of": {
                0: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y,
                90: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x,
                180: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y,
                270: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x,
            },
            "in front of": {
                0: lambda sol_center: sol_center[1] > max_y and mean_x - self.grid_size < sol_center[0] < mean_x + self.grid_size,
                90: lambda sol_center: sol_center[0] > max_x and mean_y - self.grid_size < sol_center[1] < mean_y + self.grid_size,
                180: lambda sol_center: sol_center[1] < min_y and mean_x - self.grid_size < sol_center[0] < mean_x + self.grid_size,
                270: lambda sol_center: sol_center[0] < min_x and mean_y - self.grid_size < sol_center[1] < mean_y + self.grid_size,
            },
            "behind": {
                0: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x,
                90: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y,
                180: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x,
                270: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y,
            },
            "side of": {
                0: lambda sol_center: min_y <= sol_center[1] <= max_y,
                90: lambda sol_center: min_x <= sol_center[0] <= max_x,
                180: lambda sol_center: min_y <= sol_center[1] <= max_y,
                270: lambda sol_center: min_x <= sol_center[0] <= max_x,
            },
        }

        compare_func = comparison_dict.get(place_type, {}).get(target_rotation)
        if not compare_func:
            return

        for solution in solutions:
            sol_center = solution[0]
            if compare_func(sol_center):
                placement_key = tuple(solution[:3])
                if placement_key in placement2score:
                    placement2score[placement_key] += self.constraint_bouns * self.constraint_type2weight["relative"]
    
    def apply_face_constraint(self, solutions, target_object, placement2score):
        """Apply face-to constraint."""
        unit_vectors = {
            0: np.array([0.0, 1.0]),   # Facing +Y
            90: np.array([-1.0, 0.0]), # Facing -X
            180: np.array([0.0, -1.0]), # Facing -Y
            270: np.array([1.0, 0.0]),  # Facing +X
        }

        target_coords = target_object[2]
        target_poly = Polygon(target_coords)

        for solution in solutions:
            sol_center = solution[0]
            sol_rotation = solution[1]

            # Define an arbitrarily large point in the direction of the solution's rotation
            far_point = (
                sol_center[0] + 1e6 * unit_vectors[sol_rotation][0],
                sol_center[1] + 1e6 * unit_vectors[sol_rotation][1]
            )

            # Create a half-line from the solution's center to the far point
            half_line = LineString([sol_center, far_point])

            # Check if the half-line intersects with the target polygon
            if half_line.intersects(target_poly):
                placement_key = tuple(solution[:3])
                if placement_key in placement2score:
                    placement2score[placement_key] += self.constraint_bouns * self.constraint_type2weight["direction"]
    
    def apply_alignment_constraint(self, solutions, target_object, placement2score):
        """Apply center alignment constraint."""
        target_center = target_object[0]
        eps = 5  # 5cm tolerance
        
        for solution in solutions:
            sol_center = solution[0]
            if (
                abs(sol_center[0] - target_center[0]) < eps
                or abs(sol_center[1] - target_center[1]) < eps
            ):
                placement_key = tuple(solution[:3])
                if placement_key in placement2score:
                    placement2score[placement_key] += self.constraint_bouns * self.constraint_type2weight["alignment"]
    
    def visualize_single_object_placement(self, room_poly, grid_points, initial_state, object_to_place, best_solution, suggested_position, room_id="unknown"):
        """
        Visualize single object placement showing room, grid points, existing objects, suggested position and final placement.
        """
        try:
            # Create vis directory if it doesn't exist
            vis_dir = f"{SERVER_ROOT_DIR}/vis"
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
                
            plt.rcParams["font.size"] = 12

            # create a new figure
            fig, ax = plt.subplots(figsize=(14, 12))

            # draw the room
            x, y = room_poly.exterior.xy
            ax.plot(x, y, "-", label="Room Boundary", color="black", linewidth=3)

            # draw the grid points
            if grid_points:
                grid_x = [point[0] for point in grid_points]
                grid_y = [point[1] for point in grid_points]
                ax.scatter(grid_x, grid_y, s=8, color="lightgray", alpha=0.6, label="Available Grid Points")

            # Color map for different object types
            colors = ['brown', 'cyan', 'red', 'blue', 'green', 'orange', 'purple', 'pink']
            color_idx = 0

            # draw the initial state (existing objects, doors, windows)
            for object_id, solution in initial_state.items():
                center, rotation, box_coords = solution[:3]
                center_x, center_y = center

                # create a polygon for the object
                obj_poly = Polygon(box_coords)
                x_coords, y_coords = obj_poly.exterior.xy
                
                if object_id.startswith('door-'):
                    ax.plot(x_coords, y_coords, "-", linewidth=3, color="brown", alpha=0.8)
                    ax.fill(x_coords, y_coords, color="brown", alpha=0.5)
                    ax.text(center_x, center_y, object_id, fontsize=8, ha='center', va='center', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                elif object_id.startswith('window-'):
                    ax.plot(x_coords, y_coords, "-", linewidth=2, color="cyan", alpha=0.8)
                    ax.fill(x_coords, y_coords, color="cyan", alpha=0.5)
                    ax.text(center_x, center_y, object_id, fontsize=8, ha='center', va='center', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                elif object_id.startswith('existing-'):
                    # Existing furniture objects
                    current_color = colors[color_idx % len(colors)]
                    ax.plot(x_coords, y_coords, "-", linewidth=2, color=current_color, alpha=0.8)
                    ax.fill(x_coords, y_coords, color=current_color, alpha=0.4)
                    
                    # Label with object ID (remove existing- prefix)
                    display_id = object_id.replace("existing-", "")
                    ax.text(center_x, center_y, display_id, fontsize=8, ha='center', va='center', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
                    
                    # Show orientation with arrow
                    arrow_length = 20
                    if rotation == 0:
                        ax.arrow(center_x, center_y, 0, arrow_length, head_width=6, 
                                head_length=4, fc=current_color, ec=current_color, alpha=0.7)
                    elif rotation == 90:
                        ax.arrow(center_x, center_y, -arrow_length, 0, head_width=6, 
                                head_length=4, fc=current_color, ec=current_color, alpha=0.7)
                    elif rotation == 180:
                        ax.arrow(center_x, center_y, 0, -arrow_length, head_width=6, 
                                head_length=4, fc=current_color, ec=current_color, alpha=0.7)
                    elif rotation == 270:
                        ax.arrow(center_x, center_y, arrow_length, 0, head_width=6, 
                                head_length=4, fc=current_color, ec=current_color, alpha=0.7)
                    
                    color_idx += 1

            # Draw Claude's suggested position
            if suggested_position:
                suggested_x = suggested_position["x"]
                suggested_y = suggested_position["y"]
                ax.scatter([suggested_x], [suggested_y], c='magenta', s=150, marker='*', 
                          label='Claude Suggested Position', alpha=0.8, edgecolors='black', linewidth=2)
                
                # Add tolerance circle around suggested position
                tolerance_circle = plt.Circle((suggested_x, suggested_y), self.position_tolerance, 
                                            color='magenta', fill=False, linestyle='--', alpha=0.5, linewidth=2)
                ax.add_patch(tolerance_circle)

            # Draw the final placement if solution found
            if best_solution:
                center, rotation, polygon_coords, score = best_solution
                center_x, center_y = center

                # create a polygon for the placed object
                obj_poly = Polygon(polygon_coords)
                x_coords, y_coords = obj_poly.exterior.xy
                
                # Use a distinctive color for the new object
                ax.plot(x_coords, y_coords, "-", linewidth=4, color="gold", alpha=0.9)
                ax.fill(x_coords, y_coords, color="gold", alpha=0.6)

                # Add object label with score
                ax.text(center_x, center_y, f'{object_to_place.type}\n{object_to_place.id}\nScore: {score:.2f}', 
                       fontsize=10, ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.9))

                # Show object orientation with arrow
                arrow_length = 30
                if rotation == 0:
                    ax.arrow(center_x, center_y, 0, arrow_length, head_width=8, 
                            head_length=6, fc="darkgoldenrod", ec="darkgoldenrod", linewidth=2)
                elif rotation == 90:
                    ax.arrow(center_x, center_y, -arrow_length, 0, head_width=8, 
                            head_length=6, fc="darkgoldenrod", ec="darkgoldenrod", linewidth=2)
                elif rotation == 180:
                    ax.arrow(center_x, center_y, 0, -arrow_length, head_width=8, 
                            head_length=6, fc="darkgoldenrod", ec="darkgoldenrod", linewidth=2)
                elif rotation == 270:
                    ax.arrow(center_x, center_y, arrow_length, 0, head_width=8, 
                            head_length=6, fc="darkgoldenrod", ec="darkgoldenrod", linewidth=2)

            # Add title and labels
            title = f"Single Object Placement: {object_to_place.type} (ID: {object_to_place.id})"
            if best_solution:
                title += f"\nRoom: {room_id} | Placement: SUCCESS"
            else:
                title += f"\nRoom: {room_id} | Placement: FAILED"
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("X Position (cm)", fontsize=12)
            ax.set_ylabel("Y Position (cm)", fontsize=12)
            
            # axis formatting
            ax.set_aspect("equal", "box")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='upper right')
            
            # Add summary information
            summary_text = f"""Object Details:
Type: {object_to_place.type}
Dimensions: {object_to_place.dimensions.width*100:.0f} × {object_to_place.dimensions.length*100:.0f} × {object_to_place.dimensions.height*100:.0f} cm
Grid Size: {self.grid_size} cm
Position Tolerance: {self.position_tolerance} cm"""
            
            if best_solution:
                summary_text += f"""
Final Position: ({center[0]:.0f}, {center[1]:.0f}) cm
Final Rotation: {rotation}°
Final Score: {score:.3f}"""
            
            # Add summary as text box
            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                   verticalalignment='top')
            
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{vis_dir}/single_object_placement_{room_id}_{object_to_place.id}_{timestamp}.png"
            
            # Save the figure
            plt.savefig(filename, bbox_inches="tight", dpi=150, facecolor='white')
            print(f"Single object placement visualization saved: {filename}")
            
            # Close the figure to free memory
            plt.close(fig)
            
            return filename
            
        except Exception as e:
            print(f"Error creating single object placement visualization: {e}")
            # Don't let visualization errors break the placement process
            return None
    
    # Other constraint methods (place_relative, place_distance, etc.) would be similar
    # but adapted for the single-object placement context


def calculate_object_bounding_box(obj: Object, room: Room) -> str:
    """
    Calculate the bounding box of an object considering its rotation.
    Returns formatted string with x-y range relative to room coordinates.
    """
    # Convert object position to room-relative coordinates in cm
    obj_x_cm = (obj.position.x - room.position.x) * 100
    obj_y_cm = (obj.position.y - room.position.y) * 100
    obj_width_cm = obj.dimensions.width * 100
    obj_length_cm = obj.dimensions.length * 100
    
    # Handle rotation - adjust dimensions based on rotation angle
    object_rotation = obj.rotation.z
    if object_rotation == 0:
        obj_length_x_cm = obj_width_cm
        obj_length_y_cm = obj_length_cm
    elif object_rotation == 90:
        obj_length_x_cm = obj_length_cm
        obj_length_y_cm = obj_width_cm
    elif object_rotation == 180:
        obj_length_x_cm = obj_width_cm
        obj_length_y_cm = obj_length_cm
    elif object_rotation == 270:
        obj_length_x_cm = obj_length_cm
        obj_length_y_cm = obj_width_cm
    else:
        # For non-standard rotations, use original dimensions
        obj_length_x_cm = obj_width_cm
        obj_length_y_cm = obj_length_cm
    
    # Calculate bounding box edges
    x_min = obj_x_cm - obj_length_x_cm/2
    x_max = obj_x_cm + obj_length_x_cm/2
    y_min = obj_y_cm - obj_length_y_cm/2
    y_max = obj_y_cm + obj_length_y_cm/2
    
    return f"{x_min:.0f}-{x_max:.0f} x {y_min:.0f}-{y_max:.0f} cm"


def get_object_facing_direction(obj: Object) -> str:
    """
    Get the facing direction of an object based on its rotation.
    Returns the direction as +x, -x, +y, or -y.
    """
    # Normalize rotation to 0-360 range
    rotation = obj.rotation.z % 360
    
    # Map rotation to facing direction
    # Based on the unit vectors used in the DFS solver place_face_to function
    if rotation == 0:
        return "+y"
    elif rotation == 90:
        return "-x"
    elif rotation == 180:
        return "-y"
    elif rotation == 270:
        return "+x"
    else:
        # For non-standard rotations, find the closest standard direction
        if 0 <= rotation < 45 or 315 <= rotation < 360:
            return "+y"
        elif 45 <= rotation < 135:
            return "-x"
        elif 135 <= rotation < 225:
            return "-y"
        else:  # 225 <= rotation < 315
            return "+x"


def get_detailed_room_description_for_placement(room: Room, current_layout: FloorPlan, 
                                              object_to_place: Object, initial_state: Dict[str, Any]) -> str:
    """
    Generate a detailed room description for placement context.
    """
    description_parts = []
    
    # Basic room info
    description_parts.append(f"Room Type: {room.room_type}")
    description_parts.append(f"Dimensions: {room.dimensions.width*100:.0f} × {room.dimensions.length*100:.0f} cm")
    description_parts.append(f"Area: {room.dimensions.width * room.dimensions.length:.1f} m²")
    description_parts.append(f"Ceiling Height: {room.ceiling_height*100:.0f} cm")
    description_parts.append(f"Floor Material: {room.floor_material}")
    
    # Doors and windows
    if room.doors:
        description_parts.append(f"\nDoors ({len(room.doors)}):")
        for i, door in enumerate(room.doors):
            pos_cm = door.position_on_wall * room.dimensions.width * 100
            description_parts.append(f"  - Door {i+1}: {door.width*100:.0f}cm wide at {pos_cm:.0f}cm from left wall")
    
    if room.windows:
        description_parts.append(f"\nWindows ({len(room.windows)}):")
        for i, window in enumerate(room.windows):
            pos_cm = window.position_on_wall * room.dimensions.width * 100
            description_parts.append(f"  - Window {i+1}: {window.width*100:.0f}cm wide at {pos_cm:.0f}cm from left wall")
    
    # Existing objects
    existing_furniture = [obj for obj in room.objects if obj.id != object_to_place.id]
    if existing_furniture:
        description_parts.append(f"\nExisting Objects ({len(existing_furniture)}):")
        for obj in existing_furniture:
            pos_x_cm = (obj.position.x - room.position.x) * 100
            pos_y_cm = (obj.position.y - room.position.y) * 100
            dims_cm = f"{obj.dimensions.width*100:.0f}×{obj.dimensions.length*100:.0f}×{obj.dimensions.height*100:.0f}"
            obj_bbox = calculate_object_bounding_box(obj, room)
            obj_facing = get_object_facing_direction(obj)
            description_parts.append(f"  - ID: {obj.id} | Type: {obj.type} | Size: {dims_cm}cm | Position: ({pos_x_cm:.0f}, {pos_y_cm:.0f})")
            description_parts.append(f"    Bounding Box: {obj_bbox}")
            description_parts.append(f"    Facing Direction: {obj_facing}")
            if hasattr(obj, 'description') and obj.description:
                description_parts.append(f"    Description: {obj.description}")
    else:
        description_parts.append("\nExisting Objects: None")
    
    return "\n".join(description_parts)


async def add_object_wall(room: Room, current_layout: FloorPlan, object_to_place: Object, 
                         claude_placement_result: Dict[str, Any]) -> Tuple[List[Object], FloorPlan, Dict[str, Any]]:
    """
    Add a single object to walls using geometric placement and optional Claude API constraints.
    Adapted from place_wall_objects for single object placement.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout  
        object_to_place: Object to place on wall
        claude_placement_result: Result from get_placement_location_from_claude (not used currently)
        
    Returns:
        Tuple of (updated_room_objects, updated_layout, placement_info)
    """

    
    placement_info = {
        "success": False,
        "error": None,
        "object_placed": None,
        "placement_result": None
    }
    
    try:
        # Get all placed objects from current layout (equivalent to placed_objects parameter in place_wall_objects)
        placed_objects = []
        for layout_room in current_layout.rooms:
            placed_objects.extend(layout_room.objects)
        
        # Step 1: Create wall coordinate systems and sample grid points
        wall_systems = create_wall_coordinate_systems(room)
        wall_grids = create_wall_grid_points(wall_systems, grid_density=20)
        
        # Step 2: Calculate impossible placement regions for this specific wall object
        impossible_regions = calculate_impossible_wall_regions(room, placed_objects, wall_systems, object_to_place)
        
        # Step 3: Filter valid placement points for this object
        valid_points = filter_valid_wall_points(object_to_place, wall_grids, impossible_regions, wall_systems)
        
        if not valid_points:
            placement_info["error"] = "No valid placement points found on any wall"
            return room.objects, current_layout, placement_info
        
        # Step 4: Score and select best placement
        best_placement = select_best_wall_placement(object_to_place, valid_points, placed_objects, wall_systems)
        
        # Visualize wall placement process for this object
        visualize_wall_placement(object_to_place, wall_systems, wall_grids, impossible_regions, valid_points, best_placement, room.id)
        
        if best_placement:
            # print(f"best_placement: {best_placement}")
            # Create placed object with 3D position and proper rotation
            # Adjust object position to account for wall mounting offset
            adjusted_placement = adjust_wall_object_position(best_placement, object_to_place, wall_systems)
            placed_obj = create_wall_placed_object(object_to_place, adjusted_placement, room)
            
            # Update room objects - replace existing object if it exists, otherwise add new one
            updated_room_objects = [obj for obj in room.objects if obj.id != object_to_place.id]
            updated_room_objects.append(placed_obj)
            
            # Update the room in the layout
            for layout_room in current_layout.rooms:
                if layout_room.id == room.id:
                    layout_room.objects = updated_room_objects
                    break
            
            placement_info["success"] = True
            placement_info["object_placed"] = {
                "id": placed_obj.id,
                "type": placed_obj.type,
                "wall_id": best_placement["wall_id"],
                "position": {
                    "x": placed_obj.position.x,
                    "y": placed_obj.position.y,
                    "z": placed_obj.position.z
                },
                "rotation": {
                    "x": placed_obj.rotation.x,
                    "y": placed_obj.rotation.y,
                    "z": placed_obj.rotation.z
                }
            }
            placement_info["placement_result"] = {
                "wall_id": best_placement["wall_id"],
                "position_3d": best_placement["position_3d"],
                "rotation": best_placement["rotation"],
                "valid_points_found": len(valid_points)
            }
            
            return updated_room_objects, current_layout, placement_info
            
        elif best_placement is None:
            # Check if this was due to VLM constraint failure
            placement_info["error"] = "Failed to generate VLM constraints for wall object placement"
            return room.objects, current_layout, placement_info
        else:
            placement_info["error"] = "Failed to find suitable wall placement"
            return room.objects, current_layout, placement_info
            
    except Exception as e:
        placement_info["error"] = f"Exception during wall object placement: {str(e)}"
        print(f"Error in add_object_wall: {str(e)}", file=sys.stderr)
        return room.objects, current_layout, placement_info


async def add_object_on_object(room: Room, current_layout: FloorPlan, object_to_place: Object, 
                               claude_placement_result: Dict[str, Any], target_object_id: str) -> Tuple[List[Object], FloorPlan, Dict[str, Any]]:
    """
    Placeholder function for adding objects on top of other objects.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout  
        object_to_place: Object to place
        claude_placement_result: Result from get_placement_location_from_claude
        target_object_id: ID of object to place on
        
    Returns:
        Tuple of (updated_room_objects, updated_layout, placement_info)
    """
    # TODO: Implement object-on-object placement logic
    from objects.object_on_top_placement import (
        get_random_placements_on_target_object,
        filter_placements_by_physics_critic
    )

    placements = get_random_placements_on_target_object(current_layout, room, target_object_id, object_to_place, sample_count=150)
    print("placements: ", len(placements))
    safe_placements = filter_placements_by_physics_critic(current_layout, room, object_to_place, placements)
    print("safe_placements: ", len(safe_placements))

    if len(safe_placements) > 0:
        position_placed = safe_placements[0]["position"]
        rotation_placed = safe_placements[0]["rotation"]

        placement_info = {
            "success": True,
            "object_placed": {
                "id": object_to_place.id,
                "type": object_to_place.type,
                "position": {
                    "x": position_placed["x"],
                    "y": position_placed["y"],
                    "z": position_placed["z"]
                },
                "rotation": {
                    "x": rotation_placed["x"],
                    "y": rotation_placed["y"],
                    "z": rotation_placed["z"]
                },
                "score": 1.0,
                "placement_type": "object",
                "target_object_id": target_object_id
            },
            "solver_result": {
                "successfully placed": True,
            }
        }

        object_to_place.position = Point3D(
            x=position_placed["x"],
            y=position_placed["y"],
            z=position_placed["z"]
        )
        object_to_place.rotation = Euler(
            x=rotation_placed["x"] * 180 / np.pi,
            y=rotation_placed["y"] * 180 / np.pi,
            z=rotation_placed["z"] * 180 / np.pi
        )
        
        # For now, just add object to room without actual placement logic
        updated_room_objects = room.objects + [object_to_place]
        
        # Update the room in the layout
        for layout_room in current_layout.rooms:
            if layout_room.id == room.id:
                layout_room.objects = updated_room_objects
                break
    else:
        placement_info = {
            "success": False,
            "solver_result": {
                "successfully placed": False,
            }
        }
        updated_room_objects = room.objects

    
    
    return updated_room_objects, current_layout, placement_info


async def add_object_floor(room: Room, current_layout: FloorPlan, object_to_place: Object, 
                          claude_placement_result: Dict[str, Any]) -> Tuple[List[Object], FloorPlan, Dict[str, Any]]:
    """
    Add the object to the room floor using AdditionFloorSolver based on Claude's suggested position and constraints.
    This is the full implementation for floor placement.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout  
        object_to_place: Object to place on floor
        claude_placement_result: Result from get_placement_location_from_claude (contains suggested_position and constraints)
        
    Returns:
        Tuple of (updated_room_objects, updated_layout, placement_info)
    """
    placement_info = {
        "success": False,
        "error": None,
        "object_placed": None,
        "claude_recommendation": claude_placement_result,
        "solver_result": None
    }
    
    try:
        if not claude_placement_result.get("success"):
            placement_info["error"] = f"Claude constraint analysis failed: {claude_placement_result.get('error', 'Unknown error')}"
            return room.objects, current_layout, placement_info
        
        # Extract suggested position and constraints from Claude's response
        suggested_position = claude_placement_result.get("suggested_position", {"x": 0, "y": 0})
        claude_constraints = claude_placement_result.get("constraints", ["edge"])
        placement_info["claude_constraints"] = claude_constraints
        placement_info["suggested_position"] = suggested_position
        
        # Parse constraints into solver format
        try:
            parsed_constraints = parse_claude_constraints(claude_constraints, object_to_place.id, room.objects)
            print(f"parsed_constraints: {parsed_constraints}", file=sys.stderr)
            placement_info["parsed_constraints"] = parsed_constraints
        except Exception as e:
            placement_info["error"] = f"Failed to parse constraints: {e}"
            return room.objects, current_layout, placement_info
        
        # Prepare room geometry (convert to centimeters)
        max_wall_thickness_cm = 5
        if room.walls:
            max_wall_thickness_cm = max(wall.thickness * 0.5 * 100 for wall in room.walls)
        
        inner_width_cm = (room.dimensions.width * 100) - (2 * max_wall_thickness_cm)
        inner_length_cm = (room.dimensions.length * 100) - (2 * max_wall_thickness_cm)
        
        room_vertices = [
            (max_wall_thickness_cm, max_wall_thickness_cm),
            (max_wall_thickness_cm, max_wall_thickness_cm + inner_length_cm),
            (max_wall_thickness_cm + inner_width_cm, max_wall_thickness_cm + inner_length_cm),
            (max_wall_thickness_cm + inner_width_cm, max_wall_thickness_cm)
        ]
        room_poly = Polygon(room_vertices)
        
        # Get initial state (doors, windows, existing objects) in cm coordinates
        initial_state = get_door_window_placements(room)
        
        # Add existing objects (excluding the one we're placing)
        for obj in room.objects:
            if obj.id == object_to_place.id:
                continue

            if obj.place_id == "wall":
                wall_object_z = obj.position.z 
                if wall_object_z > object_to_place.dimensions.height:
                    continue

            elif obj.place_id != 'floor':
                continue
                
            obj_x_cm = (obj.position.x - room.position.x) * 100
            obj_y_cm = (obj.position.y - room.position.y) * 100
            obj_width_cm = obj.dimensions.width * 100
            obj_length_cm = obj.dimensions.length * 100
            
            # Handle rotation
            object_rotation = int(obj.rotation.z) % 360
            if object_rotation == 0:
                obj_length_x_cm = obj_width_cm
                obj_length_y_cm = obj_length_cm
            elif object_rotation == 90:
                obj_length_x_cm = obj_length_cm
                obj_length_y_cm = obj_width_cm
            elif object_rotation == 180:
                obj_length_x_cm = obj_width_cm
                obj_length_y_cm = obj_length_cm
            elif object_rotation == 270:
                obj_length_x_cm = obj_length_cm
                obj_length_y_cm = obj_width_cm
            else:
                obj_length_x_cm = max(obj_width_cm, obj_length_cm)
                obj_length_y_cm = max(obj_length_cm, obj_width_cm)
            
            obj_vertices = [
                (obj_x_cm - obj_length_x_cm/2, obj_y_cm - obj_length_y_cm/2),
                (obj_x_cm + obj_length_x_cm/2, obj_y_cm - obj_length_y_cm/2),
                (obj_x_cm + obj_length_x_cm/2, obj_y_cm + obj_length_y_cm/2),
                (obj_x_cm - obj_length_x_cm/2, obj_y_cm + obj_length_y_cm/2),
                (obj_x_cm - obj_length_x_cm/2, obj_y_cm - obj_length_y_cm/2)
            ]
            
            initial_state[f"existing-{obj.id}"] = [
                (obj_x_cm, obj_y_cm),
                object_rotation,
                obj_vertices,
                1.0
            ]
        
        # Set up custom AdditionFloorSolver
        solver = AdditionFloorSolver(grid_size=20, constraint_bouns=0.2, position_tolerance=100)
        
        # Object dimensions in cm
        object_dim = (object_to_place.dimensions.width * 100, object_to_place.dimensions.length * 100)
        
        # Use AdditionFloorSolver to find best placement
        best_solution = solver.get_best_placement(
            room_poly=room_poly,
            object_dim=object_dim,
            suggested_position=suggested_position,
            constraints=parsed_constraints.get(object_to_place.id, []),
            initial_state=initial_state
        )

        # Visualize the placement process for debugging
        try:
            grid_points = solver.create_grids(room_poly)
            grid_points = solver.remove_points(grid_points, initial_state)
            vis_filename = solver.visualize_single_object_placement(
                room_poly=room_poly,
                grid_points=grid_points,
                initial_state=initial_state,
                object_to_place=object_to_place,
                best_solution=best_solution,
                suggested_position=suggested_position,
                room_id=room.id
            )
            placement_info["visualization_path"] = vis_filename
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            # Don't let visualization errors break the placement process
            pass
        
        placement_info["solver_result"] = {
            "solution_found": best_solution is not None,
            "best_solution": best_solution,
            "constraints_used": parsed_constraints,
            "constraints_input": parsed_constraints.get(object_to_place.id, []),
            "used_suggested_position": True
        }
        
        if not best_solution:
            placement_info["error"] = "AdditionFloorSolver could not find valid placement for object"
            return room.objects, current_layout, placement_info
        
        # Extract placement from solution
        center, rotation, polygon_coords, score = best_solution
        
        # Convert back to Object format (cm to meters, adjust for room position)
        object_to_place.position = Point3D(
            x=room.position.x + center[0] / 100, 
            y=room.position.y + center[1] / 100, 
            z=0.0
        )
        object_to_place.rotation = Euler(x=0, z=rotation, y=0)
        
        # Update room objects
        updated_room_objects = [obj for obj in room.objects if obj.id != object_to_place.id]
        updated_room_objects.append(object_to_place)
        
        # Update the room in the layout
        for layout_room in current_layout.rooms:
            if layout_room.id == room.id:
                layout_room.objects = updated_room_objects
                break
        
        placement_info["success"] = True
        placement_info["object_placed"] = {
            "id": object_to_place.id,
            "type": object_to_place.type,
            "position": {
                "x": object_to_place.position.x,
                "y": object_to_place.position.y,
                "z": object_to_place.position.z
            },
            "rotation": rotation,
            "score": score,
            "polygon_coords_cm": polygon_coords
        }
        
        return updated_room_objects, current_layout, placement_info
        
    except Exception as e:
        placement_info["error"] = f"Error in add_object_floor: {e}"
        return room.objects, current_layout, placement_info


async def add_object(room: Room, current_layout: FloorPlan, object_to_place: Object, 
                    claude_placement_result: Dict[str, Any]) -> Tuple[List[Object], FloorPlan, Dict[str, Any]]:
    """
    Main dispatcher function for adding objects based on their placement location.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout  
        object_to_place: Object to place
        claude_placement_result: Result from get_placement_location_from_claude
        
    Returns:
        Tuple of (updated_room_objects, updated_layout, placement_info)
    """
    try:
        # Determine placement location type
        placement_location = claude_placement_result.get("placement_location", "floor")
        
        # If not specified, try to infer from analysis result
        if placement_location == "floor" and "object_analysis" in claude_placement_result:
            placement_location = claude_placement_result["object_analysis"].get("placement_location", "floor")
        
        # Dispatch to appropriate handler
        if placement_location == "wall":
            return await add_object_wall(room, current_layout, object_to_place, claude_placement_result)
        elif placement_location in [obj.id for obj in room.objects]:
            return await add_object_on_object(room, current_layout, object_to_place, claude_placement_result, placement_location)
        else: # Default to floor placement
            return await add_object_floor(room, current_layout, object_to_place, claude_placement_result)
        
    except Exception as e:
        placement_info = {
            "success": False,
            "error": f"Error in add_object dispatcher: {e}",
            "claude_recommendation": claude_placement_result
        }
        return room.objects, current_layout, placement_info 