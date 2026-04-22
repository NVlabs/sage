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
from models import Object, Room, FloorPlan, Point3D, Euler
from typing import List, Dict, Any, Tuple
import json
from key import ANTHROPIC_API_KEY
import anthropic
from objects.get_objects import get_object_mesh
from vlm import call_vlm
from utils import extract_json_from_response
def place_objects(selected_objects: List[Object], room: Room, current_layout: FloorPlan) -> Tuple[List[Object], FloorPlan, Dict[str, Any]]:
    """
    Place selected objects in a room using Claude API for intelligent placement.
    
    Args:
        selected_objects: List of objects to place
        room: Target room for placement
        current_layout: Current floor plan layout
        
    Returns:
        Tuple of (placed_objects, updated_layout, claude_interactions)
    """
    claude_interactions = {
        "floor_placement": None,
        "wall_placement": None,
        "total_api_calls": 0,
        "placement_method": []
    }
    
    if not selected_objects:
        return selected_objects, current_layout, claude_interactions
    
    # Sort objects by placement location
    floor_objects = [obj for obj in selected_objects if obj.place_id == "floor"]
    wall_objects = [obj for obj in selected_objects if obj.place_id == "wall"]
    
    placed_objects = []
    
    # Place floor objects first
    if floor_objects:
        placed_floor_objects, floor_interaction = place_floor_objects(floor_objects, room, current_layout)
        placed_objects.extend(placed_floor_objects)
        claude_interactions["floor_placement"] = floor_interaction
        if floor_interaction and floor_interaction.get("api_called"):
            claude_interactions["total_api_calls"] += 1
            claude_interactions["placement_method"].append("claude_api_floor")
        else:
            claude_interactions["placement_method"].append("fallback_floor")
    
    # Place wall objects
    if wall_objects:
        placed_wall_objects, wall_interaction = place_wall_objects(wall_objects, room, current_layout)
        placed_objects.extend(placed_wall_objects)
        claude_interactions["wall_placement"] = wall_interaction
        if wall_interaction and wall_interaction.get("api_called"):
            claude_interactions["total_api_calls"] += 1
            claude_interactions["placement_method"].append("claude_api_wall")
        else:
            claude_interactions["placement_method"].append("fallback_wall")
    
    # Update the room's objects in the current layout
    # Find the room in the layout and update its objects
    for layout_room in current_layout.rooms:
        if layout_room.id == room.id:
            # Add the newly placed objects to the room's objects list
            layout_room.objects.extend(placed_objects)
            break
    
    return placed_objects, current_layout, claude_interactions


def place_floor_objects(floor_objects: List[Object], room: Room, current_layout: FloorPlan) -> Tuple[List[Object], Dict[str, Any]]:
    """
    Place floor objects using Claude API for intelligent positioning.
    """
    interaction_info = {
        "api_called": False,
        "prompt": None,
        "response": None,
        "placement_method": "fallback",
        "error": None,
        "objects_count": len(floor_objects)
    }
    
    if not floor_objects:
        return [], interaction_info
    
    try:
        # Check API key
        api_key = ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        # Prepare room information for Claude
        room_info = prepare_room_info_for_claude(room)
        
        # Prepare object information
        objects_info = []
        for obj in floor_objects:
            obj_mesh = get_object_mesh(obj.source, obj.source_id, current_layout.id)
            mesh_bounds = None
            if obj_mesh is not None:
                bounds = obj_mesh.bounds
                mesh_bounds = {
                    "min": bounds[0].tolist(),
                    "max": bounds[1].tolist(),
                    "dimensions": (bounds[1] - bounds[0]).tolist()
                }
            
            objects_info.append({
                "id": obj.id,
                "description": obj.description,
                "dimensions": {
                    "width": obj.dimensions.width,
                    "length": obj.dimensions.length,
                    "height": obj.dimensions.height
                },
                "mesh_bounds": mesh_bounds,
                "source": obj.source,
                "source_id": obj.source_id
            })
        
        # Create prompt for Claude
        prompt = create_floor_placement_prompt(room_info, objects_info)
        interaction_info["prompt"] = prompt
        
        # Call Claude API with thinking enabled
        response = call_vlm(
            vlm_type="qwen",
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
            # thinking={"type": "enabled", "budget_tokens": 2048}
        )
        
        interaction_info["api_called"] = True
        # Handle thinking response format
        if hasattr(response, 'thinking') and response.thinking:
            interaction_info["thinking"] = response.thinking
            interaction_info["response"] = response.content[0].text
        else:
            interaction_info["response"] = response.content[0].text
        interaction_info["placement_method"] = "claude_api"
        
        # Parse Claude's response
        placement_plan = parse_claude_placement_response(response.content[0].text)
        
        # Apply placement plan to objects
        placed_objects = apply_floor_placement_plan(floor_objects, placement_plan, room)
        
        return placed_objects, interaction_info
        
    except Exception as e:
        error_msg = f"Error in floor object placement: {e}"
        print(error_msg)
        interaction_info["error"] = error_msg
        interaction_info["placement_method"] = "fallback"
        
        # Fallback to simple placement
        placed_objects = apply_simple_floor_placement(floor_objects, room)
        return placed_objects, interaction_info


def place_wall_objects(wall_objects: List[Object], room: Room, current_layout: FloorPlan) -> Tuple[List[Object], Dict[str, Any]]:
    """
    Place wall objects using Claude API for intelligent positioning.
    """
    interaction_info = {
        "api_called": False,
        "prompt": None,
        "response": None,
        "placement_method": "fallback",
        "error": None,
        "objects_count": len(wall_objects)
    }
    
    if not wall_objects:
        return [], interaction_info
    
    try:
        # Check API key
        api_key = ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        # Prepare room information for Claude
        room_info = prepare_room_info_for_claude(room)
        
        # Prepare object information
        objects_info = []
        for obj in wall_objects:
            obj_mesh = get_object_mesh(obj.source, obj.source_id, current_layout.id)
            mesh_bounds = None
            if obj_mesh is not None:
                bounds = obj_mesh.bounds
                mesh_bounds = {
                    "min": bounds[0].tolist(),
                    "max": bounds[1].tolist(),
                    "dimensions": (bounds[1] - bounds[0]).tolist()
                }
            
            objects_info.append({
                "id": obj.id,
                "description": obj.description,
                "dimensions": {
                    "width": obj.dimensions.width,
                    "length": obj.dimensions.length,
                    "height": obj.dimensions.height
                },
                "mesh_bounds": mesh_bounds,
                "source": obj.source,
                "source_id": obj.source_id
            })
        
        # Create prompt for Claude
        prompt = create_wall_placement_prompt(room_info, objects_info)
        interaction_info["prompt"] = prompt
        
        # Call Claude API with thinking enabled
        response = call_vlm(
            vlm_type="qwen",
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
            # thinking={"type": "enabled", "budget_tokens": 2048}
        )
        
        interaction_info["api_called"] = True
        # Handle thinking response format
        if hasattr(response, 'thinking') and response.thinking:
            interaction_info["thinking"] = response.thinking
            interaction_info["response"] = response.content[0].text
        else:
            interaction_info["response"] = response.content[0].text
        interaction_info["placement_method"] = "claude_api"
        
        # Parse Claude's response
        placement_plan = parse_claude_placement_response(response.content[0].text)
        
        # Apply placement plan to objects
        placed_objects = apply_wall_placement_plan(wall_objects, placement_plan, room)
        
        return placed_objects, interaction_info
        
    except Exception as e:
        error_msg = f"Error in wall object placement: {e}"
        print(error_msg)
        interaction_info["error"] = error_msg
        interaction_info["placement_method"] = "fallback"
        
        # Fallback to simple placement
        placed_objects = apply_simple_wall_placement(wall_objects, room)
        return placed_objects, interaction_info


def prepare_room_info_for_claude(room: Room) -> Dict[str, Any]:
    """
    Prepare room information in a format suitable for Claude.
    """
    return {
        "id": room.id,
        "type": room.room_type,
        "dimensions": {
            "width": room.dimensions.width,
            "length": room.dimensions.length,
            "height": room.dimensions.height
        },
        "position": {
            "x": room.position.x,
            "y": room.position.y,
            "z": room.position.z
        },
        "doors": [
            {
                "id": door.id,
                "width": door.width,
                "height": door.height,
                "position_on_wall": door.position_on_wall,
                "wall_id": door.wall_id,
                "door_type": door.door_type
            } for door in room.doors
        ],
        "windows": [
            {
                "id": window.id,
                "width": window.width,
                "height": window.height,
                "position_on_wall": window.position_on_wall,
                "wall_id": window.wall_id,
                "sill_height": window.sill_height,
                "window_type": window.window_type
            } for window in room.windows
        ],
        "walls": [
            {
                "id": wall.id,
                "start_point": {
                    "x": wall.start_point.x,
                    "y": wall.start_point.y,
                    "z": wall.start_point.z
                },
                "end_point": {
                    "x": wall.end_point.x,
                    "y": wall.end_point.y,
                    "z": wall.end_point.z
                },
                "height": wall.height
            } for wall in room.walls
        ],
        "existing_objects": [
            {
                "id": obj.id,
                "description": obj.description,
                "position": {
                    "x": obj.position.x,
                    "y": obj.position.y,
                    "z": obj.position.z
                },
                "dimensions": {
                    "width": obj.dimensions.width,
                    "length": obj.dimensions.length,
                    "height": obj.dimensions.height
                },
                "place_id": obj.place_id
            } for obj in room.objects
        ]
    }


def create_floor_placement_prompt(room_info: Dict[str, Any], objects_info: List[Dict[str, Any]]) -> str:
    """
    Create a prompt for Claude to plan floor object placement.
    """
    # Calculate room bounds in world coordinates
    room_x_min = room_info['position']['x']
    room_y_min = room_info['position']['y']
    room_z_min = room_info['position']['z']
    room_x_max = room_x_min + room_info['dimensions']['width']
    room_y_max = room_y_min + room_info['dimensions']['length']
    room_z_max = room_z_min + room_info['dimensions']['height']
    
    # Create detailed bounding box information for each object
    detailed_objects_info = []
    for obj in objects_info:
        obj_info = f"""
Object: {obj['id']}
- Description: {obj['description']}
- Declared Dimensions: {obj['dimensions']['width']:.2f}m(W) × {obj['dimensions']['length']:.2f}m(L) × {obj['dimensions']['height']:.2f}m(H)"""
        
        if obj.get('mesh_bounds'):
            mesh_dims = obj['mesh_bounds']['dimensions']
            obj_info += f"""
- Actual Mesh Bounds: {mesh_dims[0]:.2f}m(W) × {mesh_dims[1]:.2f}m(L) × {mesh_dims[2]:.2f}m(H)
- Mesh Min Point: ({obj['mesh_bounds']['min'][0]:.2f}, {obj['mesh_bounds']['min'][1]:.2f}, {obj['mesh_bounds']['min'][2]:.2f})
- Mesh Max Point: ({obj['mesh_bounds']['max'][0]:.2f}, {obj['mesh_bounds']['max'][1]:.2f}, {obj['mesh_bounds']['max'][2]:.2f})
- Use MESH BOUNDS for collision detection and spacing calculations"""
        else:
            obj_info += f"""
- Mesh Bounds: Not available, use declared dimensions
- Use DECLARED DIMENSIONS for collision detection and spacing calculations"""
        
        detailed_objects_info.append(obj_info)

    prompt = f"""You are an expert interior designer with precise spatial reasoning skills. I need you to plan the placement of floor objects in a {room_info['type']}.

ROOM ANALYSIS:
- Type: {room_info['type']}
- Dimensions: {room_info['dimensions']['width']:.2f}m × {room_info['dimensions']['length']:.2f}m × {room_info['dimensions']['height']:.2f}m
- World Position: ({room_info['position']['x']:.2f}, {room_info['position']['y']:.2f}, {room_info['position']['z']:.2f})
- Available Floor Area: {room_x_min:.2f}m to {room_x_max:.2f}m (X) × {room_y_min:.2f}m to {room_y_max:.2f}m (Y)

WALLS (World Coordinates):
{chr(10).join([f"  - {wall['id']}: from ({wall['start_point']['x']:.2f}, {wall['start_point']['y']:.2f}) to ({wall['end_point']['x']:.2f}, {wall['end_point']['y']:.2f})" for wall in room_info['walls']])}

DOORS & CLEARANCE ZONES:
{chr(10).join([f"  - Door {door['id']}: {door['width']:.2f}m wide on {door['wall_id']} at position {door['position_on_wall']:.2f}m - KEEP 1.5m CLEARANCE" for door in room_info['doors']]) if room_info['doors'] else "  - No doors"}

WINDOWS:
{chr(10).join([f"  - Window {window['id']}: {window['width']:.2f}m wide on {window['wall_id']} at {window['position_on_wall']:.2f}m, sill {window['sill_height']:.2f}m" for window in room_info['windows']]) if room_info['windows'] else "  - No windows"}

EXISTING OBJECTS (OCCUPIED SPACE):
{chr(10).join([f"  - {obj['id']}: {obj['description']} at ({obj['position']['x']:.2f}, {obj['position']['y']:.2f}) occupies {obj['dimensions']['width']:.2f}×{obj['dimensions']['length']:.2f}m" for obj in room_info['existing_objects']]) if room_info['existing_objects'] else "  - No existing objects"}

OBJECTS TO PLACE (with precise dimensions):
{chr(10).join(detailed_objects_info)}

PLACEMENT METHODOLOGY:
Think step-by-step and place objects one by one using this iterative approach:

1. ANALYZE ROOM LAYOUT: Identify traffic patterns, functional zones, and spatial relationships
2. PRIORITIZE OBJECTS: Order by importance and size (largest/most important first)
3. FOR EACH OBJECT:
   a) Calculate exact bounding box using mesh bounds (if available) or declared dimensions
   b) Identify suitable placement zones considering function and aesthetics
   c) Check collision with ALL previously placed objects and existing objects
   d) Ensure minimum clearances: 0.8m from walls, 1.5m from doors, 0.6m between objects
   e) Verify the position is within room bounds
   f) Calculate optimal rotation for functionality and space efficiency

COLLISION DETECTION RULES:
- Two objects collide if their bounding boxes overlap
- Object A at (x1,y1) with dimensions (w1,l1) occupies space from (x1-w1/2, y1-l1/2) to (x1+w1/2, y1+l1/2)
- Object B at (x2,y2) with dimensions (w2,l2) occupies space from (x2-w2/2, y2-l2/2) to (x2+w2/2, y2+l2/2)
- NO OVERLAP allowed between any objects

COORDINATE SYSTEM:
- World coordinates: Room spans ({room_x_min:.2f}, {room_y_min:.2f}) to ({room_x_max:.2f}, {room_y_max:.2f})
- Object positions are CENTER POINTS in world coordinates
- Floor level: z = {room_z_min:.2f}
- All positions must be within room bounds with object dimensions considered

Please think through the placement step by step, then provide your final JSON response:

{{
    "placement_plan": [
        {{
            "object_id": "object_id_here",
            "position": {{
                "x": precise_x_coordinate,
                "y": precise_y_coordinate,
                "z": {room_z_min:.2f}
            }},
            "rotation": {{
                "x": 0.0,
                "y": 0.0,
                "z": rotation_angle_in_radians
            }},
            "reasoning": "Detailed explanation of placement decision, collision checks, and spatial considerations"
        }}
    ]
}}

CRITICAL VALIDATION:
- Verify ALL coordinates are within [{room_x_min:.2f}, {room_y_min:.2f}] to [{room_x_max:.2f}, {room_y_max:.2f}]
- Ensure NO collisions between any objects (including existing ones)
- Maintain proper clearances and functional relationships
- Double-check that object centers plus half-dimensions don't exceed room boundaries
"""
    return prompt


def create_wall_placement_prompt(room_info: Dict[str, Any], objects_info: List[Dict[str, Any]]) -> str:
    """
    Create a prompt for Claude to plan wall object placement.
    """
    # Calculate room bounds in world coordinates
    room_x_min = room_info['position']['x']
    room_y_min = room_info['position']['y']
    room_z_min = room_info['position']['z']
    room_x_max = room_x_min + room_info['dimensions']['width']
    room_y_max = room_y_min + room_info['dimensions']['length']
    room_z_max = room_z_min + room_info['dimensions']['height']
    
    # Default wall mounting height (world coordinates)
    wall_mount_z = room_z_min + 1.5
    
    # Create detailed bounding box information for each object
    detailed_objects_info = []
    for obj in objects_info:
        obj_info = f"""
Object: {obj['id']}
- Description: {obj['description']}
- Declared Dimensions: {obj['dimensions']['width']:.2f}m(W) × {obj['dimensions']['length']:.2f}m(L) × {obj['dimensions']['height']:.2f}m(H)"""
        
        if obj.get('mesh_bounds'):
            mesh_dims = obj['mesh_bounds']['dimensions']
            obj_info += f"""
- Actual Mesh Bounds: {mesh_dims[0]:.2f}m(W) × {mesh_dims[1]:.2f}m(L) × {mesh_dims[2]:.2f}m(H)
- Mesh Min Point: ({obj['mesh_bounds']['min'][0]:.2f}, {obj['mesh_bounds']['min'][1]:.2f}, {obj['mesh_bounds']['min'][2]:.2f})
- Mesh Max Point: ({obj['mesh_bounds']['max'][0]:.2f}, {obj['mesh_bounds']['max'][1]:.2f}, {obj['mesh_bounds']['max'][2]:.2f})
- Use MESH BOUNDS for collision detection and spacing calculations"""
        else:
            obj_info += f"""
- Mesh Bounds: Not available, use declared dimensions
- Use DECLARED DIMENSIONS for collision detection and spacing calculations"""
        
        detailed_objects_info.append(obj_info)

    # Create detailed wall analysis
    wall_analysis = []
    for wall in room_info['walls']:
        wall_length = ((wall['end_point']['x'] - wall['start_point']['x'])**2 + 
                      (wall['end_point']['y'] - wall['start_point']['y'])**2)**0.5
        
        # Find doors and windows on this wall
        wall_doors = [door for door in room_info['doors'] if door['wall_id'] == wall['id']]
        wall_windows = [window for window in room_info['windows'] if window['wall_id'] == wall['id']]
        
        wall_info = f"""
Wall {wall['id']}:
- Start: ({wall['start_point']['x']:.2f}, {wall['start_point']['y']:.2f})
- End: ({wall['end_point']['x']:.2f}, {wall['end_point']['y']:.2f})
- Length: {wall_length:.2f}m
- Height: {wall['height']:.2f}m"""
        
        if wall_doors:
            wall_info += f"""
- Doors: {', '.join([f"{door['id']} at {door['position_on_wall']:.2f}m ({door['width']:.2f}m wide)" for door in wall_doors])}"""
        
        if wall_windows:
            wall_info += f"""
- Windows: {', '.join([f"{window['id']} at {window['position_on_wall']:.2f}m ({window['width']:.2f}m wide, sill {window['sill_height']:.2f}m)" for window in wall_windows])}"""
        
        if not wall_doors and not wall_windows:
            wall_info += f"""
- Clear wall space available for mounting"""
        
        wall_analysis.append(wall_info)

    prompt = f"""You are an expert interior designer with precise spatial reasoning skills. I need you to plan the placement of wall-mounted objects in a {room_info['type']}.

ROOM ANALYSIS:
- Type: {room_info['type']}
- Dimensions: {room_info['dimensions']['width']:.2f}m × {room_info['dimensions']['length']:.2f}m × {room_info['dimensions']['height']:.2f}m
- World Position: ({room_info['position']['x']:.2f}, {room_info['position']['y']:.2f}, {room_info['position']['z']:.2f})
- Wall mounting height range: {room_z_min + 1.0:.2f}m to {room_z_min + 2.5:.2f}m (typical range)

DETAILED WALL ANALYSIS:
{chr(10).join(wall_analysis)}

EXISTING OBJECTS (may affect wall mounting):
{chr(10).join([f"  - {obj['id']}: {obj['description']} at ({obj['position']['x']:.2f}, {obj['position']['y']:.2f}, {obj['position']['z']:.2f}) - size {obj['dimensions']['width']:.2f}×{obj['dimensions']['length']:.2f}×{obj['dimensions']['height']:.2f}m" for obj in room_info['existing_objects']]) if room_info['existing_objects'] else "  - No existing objects"}

OBJECTS TO MOUNT (with precise dimensions):
{chr(10).join(detailed_objects_info)}

WALL MOUNTING METHODOLOGY:
Think step-by-step and mount objects one by one using this iterative approach:

1. ANALYZE WALL SUITABILITY: Evaluate each wall for mounting potential
   - Consider wall length, door/window positions, and existing objects
   - Identify clear mounting zones on each wall
   - Consider viewing angles and room function

2. PRIORITIZE OBJECTS: Order by importance and mounting requirements
   - Larger objects first (need more wall space)
   - Functional objects near related furniture
   - Decorative objects in visible locations

3. FOR EACH OBJECT:
   a) Calculate object dimensions using mesh bounds (if available) or declared dimensions
   b) Select appropriate wall based on function and aesthetics
   c) Find optimal position along the wall avoiding doors/windows
   d) Calculate mounting height based on object type and room function
   e) Check clearance from existing wall-mounted objects
   f) Verify position is within room bounds and wall boundaries
   g) Calculate rotation for proper wall alignment

WALL MOUNTING RULES:
- Objects must be positioned ON or VERY CLOSE to wall surfaces
- Maintain 0.5m minimum clearance from doors and windows
- Maintain 0.3m minimum spacing between wall-mounted objects
- Mounting heights: Artwork 1.4-1.7m, Shelves 1.5-2.0m, Mirrors 1.2-1.8m
- Position coordinates are object CENTER POINTS in world coordinates

COLLISION DETECTION FOR WALL OBJECTS:
- Check overlap with other wall objects on the same wall
- Consider 3D bounding boxes for proper clearance
- Account for object depth when checking furniture clearance

COORDINATE SYSTEM:
- World coordinates: Room spans ({room_x_min:.2f}, {room_y_min:.2f}, {room_z_min:.2f}) to ({room_x_max:.2f}, {room_y_max:.2f}, {room_z_max:.2f})
- Object positions are CENTER POINTS in world coordinates
- Wall mounting typically at z = {wall_mount_z:.2f} ± 0.5m
- All positions must be within room bounds

Please think through each wall mounting decision step by step, then provide your final JSON response:

```json
{{
    "placement_plan": [
        {{
            "object_id": "object_id_here",
            "position": {{
                "x": precise_x_coordinate,
                "y": precise_y_coordinate,
                "z": precise_z_coordinate_for_mounting_height
            }},
            "rotation": {{
                "x": 0.0,
                "y": 0.0,
                "z": rotation_angle_for_wall_alignment
            }},
            "wall_id": "specific_wall_id",
            "reasoning": "Detailed explanation of wall selection, positioning, height choice, and collision avoidance"
        }}
    ]
}}
```

CRITICAL VALIDATION:
- Verify ALL coordinates are within room bounds [{room_x_min:.2f}, {room_y_min:.2f}, {room_z_min:.2f}] to [{room_x_max:.2f}, {room_y_max:.2f}, {room_z_max:.2f}]
- Ensure objects are positioned appropriately on their selected walls
- Maintain proper clearances from doors, windows, and other wall objects
- Verify mounting heights are appropriate for object type and room function
- Double-check wall_id corresponds to an actual wall in the room
"""
    return prompt


def parse_claude_placement_response(response_text: str) -> Dict[str, Any]:
    """
    Parse Claude's placement response into a usable format.
    """
    # try:
    #     # Handle markdown code blocks if present
    #     json_text = response_text.strip()
    #     if json_text.startswith('```json') or json_text.startswith('```JSON'):
    #         lines = json_text.split('\n')
    #         json_text = '\n'.join(line for line in lines[1:-1] if line != '```')
    #     elif json_text.startswith('```') and json_text.endswith('```'):
    #         json_text = json_text[3:-3].strip()
        
    #     placement_data = json.loads(json_text)
    #     return placement_data.get("placement_plan", [])
    
    # except json.JSONDecodeError as e:
    #     print(f"Error parsing Claude response: {e}")
    #     return []

    response_text = extract_json_from_response(response_text)
    if not response_text:
        raise ValueError("Could not extract JSON content from Claude response")
    placement_data = json.loads(response_text)
    return placement_data.get("placement_plan", [])


def apply_floor_placement_plan(objects: List[Object], placement_plan: List[Dict[str, Any]], room: Room) -> List[Object]:
    """
    Apply the placement plan to floor objects.
    """
    placed_objects = []
    plan_dict = {plan["object_id"]: plan for plan in placement_plan}
    
    for obj in objects:
        if obj.id in plan_dict:
            plan = plan_dict[obj.id]
            
            # Update object position and rotation
            obj.position = Point3D(
                x=plan["position"]["x"],
                y=plan["position"]["y"],
                z=plan["position"]["z"]
            )
            obj.rotation = Euler(
                x=plan["rotation"]["x"],
                y=plan["rotation"]["y"],
                z=plan["rotation"]["z"]
            )
            # Update place_id to room_id for floor objects
            obj.place_id = room.id
        else:
            # Apply simple fallback placement
            obj = apply_simple_floor_placement([obj], room)[0]
        
        placed_objects.append(obj)
    
    return placed_objects


def apply_wall_placement_plan(objects: List[Object], placement_plan: List[Dict[str, Any]], room: Room) -> List[Object]:
    """
    Apply the placement plan to wall objects.
    """
    placed_objects = []
    plan_dict = {plan["object_id"]: plan for plan in placement_plan}
    
    for obj in objects:
        if obj.id in plan_dict:
            plan = plan_dict[obj.id]
            
            # Update object position and rotation
            obj.position = Point3D(
                x=plan["position"]["x"],
                y=plan["position"]["y"],
                z=plan["position"]["z"]
            )
            obj.rotation = Euler(
                x=plan["rotation"]["x"],
                y=plan["rotation"]["y"],
                z=plan["rotation"]["z"]
            )
            # Update place_id to wall_id for wall objects
            wall_id = plan.get("wall_id")
            if wall_id and any(wall.id == wall_id for wall in room.walls):
                obj.place_id = wall_id
            else:
                # Fallback to first wall if specified wall not found
                obj.place_id = room.walls[0].id if room.walls else room.id
        else:
            # Apply simple fallback placement
            obj = apply_simple_wall_placement([obj], room)[0]
        
        placed_objects.append(obj)
    
    return placed_objects


def apply_simple_floor_placement(objects: List[Object], room: Room) -> List[Object]:
    """
    Simple fallback placement for floor objects using world coordinates.
    """
    placed_objects = []
    spacing = 1.0  # 1 meter spacing
    
    # Start positioning from room's world position with spacing
    current_x = room.position.x + spacing
    current_y = room.position.y + spacing
    floor_z = room.position.z  # Floor level in world coordinates
    
    for obj in objects:
        # Ensure object fits in room (check against room bounds in world coordinates)
        room_x_max = room.position.x + room.dimensions.width
        if current_x + obj.dimensions.width > room_x_max - spacing:
            current_x = room.position.x + spacing
            current_y += spacing + obj.dimensions.length
        
        # Set position in world coordinates
        obj.position = Point3D(x=current_x, y=current_y, z=floor_z)
        obj.rotation = Euler(x=0.0, y=0.0, z=0.0)
        obj.place_id = room.id
        
        current_x += obj.dimensions.width + spacing
        placed_objects.append(obj)
    
    return placed_objects


def apply_simple_wall_placement(objects: List[Object], room: Room) -> List[Object]:
    """
    Simple fallback placement for wall objects using world coordinates.
    """
    placed_objects = []
    wall_mount_height = 1.5  # Default wall mounting height above floor
    
    for i, obj in enumerate(objects):
        # Choose wall (cycle through available walls)
        wall_idx = i % len(room.walls) if room.walls else 0
        wall = room.walls[wall_idx] if room.walls else None
        
        if wall:
            # Position on wall center (walls already have world coordinates)
            wall_center_x = (wall.start_point.x + wall.end_point.x) / 2
            wall_center_y = (wall.start_point.y + wall.end_point.y) / 2
            wall_mount_z = room.position.z + wall_mount_height  # World coordinate Z
            
            obj.position = Point3D(x=wall_center_x, y=wall_center_y, z=wall_mount_z)
            obj.rotation = Euler(x=0.0, y=0.0, z=0.0)
            obj.place_id = wall.id
        else:
            # Fallback to room center if no walls (using world coordinates)
            obj.position = Point3D(
                x=room.position.x + room.dimensions.width / 2,
                y=room.position.y + room.dimensions.length / 2,
                z=room.position.z + wall_mount_height
            )
            obj.rotation = Euler(x=0.0, y=0.0, z=0.0)
            obj.place_id = room.id
        
        placed_objects.append(obj)
    
    return placed_objects