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
import json
import anthropic
from typing import Dict, Any, List
from key import ANTHROPIC_API_KEY
from vlm import call_vlm
from validation import validate_room_layout, validate_llm_response_structure, validate_room_only_layout
import base64
import os
import time
import sys
from constants import SERVER_ROOT_DIR
from utils import extract_json_from_response
from datetime import datetime
import asyncio
import networkx as nx
import random
import numpy as np
from room_solver import RectangleContactSolver, RectangleContactRelaxationSolver, RectangleSpec, RectangleLayout

def convert_vertices_to_position_dimensions(layout_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert vertices format back to position + dimensions format.
    
    Input format:
    {
        "rooms": [
            {
                "room_type": "living room",
                "vertices": [{"x": 0, "y": 0}, {"x": 0, "y": 6}, {"x": 4.5, "y": 6}, {"x": 4.5, "y": 0}],
                "height": 2.7
            }
        ]
    }
    
    Output format:
    {
        "rooms": [
            {
                "room_type": "living room", 
                "dimensions": {"width": 4.5, "length": 6, "height": 2.7},
                "position": {"x": 0, "y": 0, "z": 0}
            }
        ]
    }
    """
    if "rooms" not in layout_data:
        return layout_data
    
    converted_rooms = []
    for room in layout_data["rooms"]:
        if "vertices" in room and "height" in room:
            # Extract vertices
            vertices = room["vertices"]
            if len(vertices) != 4:
                raise ValueError(f"Room {room.get('room_type', 'unknown')} must have exactly 4 vertices")
            
            # Calculate bounding box
            x_coords = [v["x"] for v in vertices]
            y_coords = [v["y"] for v in vertices]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Calculate dimensions
            width = max_x - min_x
            length = max_y - min_y
            height = room["height"]
            
            # Create converted room
            converted_room = {
                "room_type": room["room_type"],
                "dimensions": {
                    "width": width,
                    "length": length, 
                    "height": height
                },
                "position": {
                    "x": min_x,
                    "y": min_y,
                    "z": 0
                }
            }
            converted_rooms.append(converted_room)
        else:
            # If room doesn't have vertices format, keep as is
            converted_rooms.append(room)
    
    # Return layout with converted rooms
    result = layout_data.copy()
    result["rooms"] = converted_rooms
    return result

async def call_llm_for_layout(input_text: str) -> Dict[str, Any]:
    """Call Claude Sonnet 4 to generate room layout from text description."""
    
    
    # Construct the prompt for Claude
    prompt = f"""You are an expert architect and interior designer. Analyze this description and generate an appropriate layout: "{input_text}"

STEP 1: ANALYZE REQUEST TYPE
First determine if the user is asking for:
- SINGLE ROOM: Only one specific room (e.g., "design a living room", "create a bedroom", "generate a kitchen")
- MULTIPLE ROOMS: A complete layout, house, apartment, or multiple connected rooms

STEP 2: GENERATE APPROPRIATE LAYOUT
For SINGLE ROOM requests:
- Generate just that one room with appropriate dimensions
- Include doors and windows suitable for that room type
- Position the room at coordinates (0, 0, 0)
- Use realistic proportions for the specific room type
- Consider the room's intended function and furniture layout
- Example single room dimensions: living room (4.5×6m), bedroom (3.5×4m), kitchen (3×4m)

For MULTIPLE ROOMS requests:
- Create a functional, well-designed layout that considers:
  - Architectural best practices and building codes
  - Natural light and ventilation
  - Traffic flow and accessibility
  - Room adjacencies and relationships
  - Realistic proportions and dimensions

EXAMPLES OF REQUEST TYPES:
SINGLE ROOM: "design a master bedroom", "create a modern kitchen", "living room with fireplace"
MULTIPLE ROOMS: "2 bedroom apartment", "small house", "office layout", "3 bedroom home"

Respond with ONLY a valid JSON object using this exact structure:

```json
{{
    "building_style": "description of architectural style (e.g., Modern residential, Traditional colonial, etc.)",
    "rooms": [
        {{
            "room_type": "specific room type (e.g., living room, master bedroom, kitchen, etc.)",
            "dimensions": {{"width": float, "length": float, "height": float}},
            "position": {{"x": float, "y": float, "z": 0}},
            "doors": [
                {{
                    "width": float,
                    "height": float,
                    "position_on_wall": float,
                    "wall_side": "north/south/east/west",
                    "door_type": "standard/sliding/french/pocket/etc."
                }}
            ],
            "windows": [
                {{
                    "width": float,
                    "height": float,
                    "position_on_wall": float,
                    "wall_side": "north/south/east/west",
                    "sill_height": float,
                    "window_type": "standard/bay/casement/picture/sliding/etc."
                }}
            ]
        }}
    ]
}}
```

CRITICAL REQUIREMENTS:
- For SINGLE ROOM: Include exactly 1 room in the "rooms" array
- For MULTIPLE ROOMS: Include all rooms in the "rooms" array with proper layout
- Use metric dimensions (meters)
- Standard ceiling height: 2.7m (can vary for special rooms)
- Door dimensions: height 2.0-2.1m, width 0.7-1.0m
- Window sill heights: 0.9-1.1m typically
- position_on_wall: 0.0-1.0 (fraction along wall length)
- For single rooms: position at (0,0,0) and focus on room functionality
- For multiple rooms: arrange logically without overlaps and ensure connectivity
- Include appropriate doors for room access and flow
- Place windows for natural light and views
- Consider the specific requirements mentioned in the description

Generate a layout that truly reflects the input description with appropriate room types, sizes, and relationships."""

    try:
        # Make API call to Claude
        response = call_vlm(
            vlm_type="qwen",
            model="claude-sonnet-4-20250514",  # Using Claude Sonnet 4 as specified
            max_tokens=3000,
            temperature=0.3,  # Lower temperature for more consistent architectural output
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Extract the response content with error checking
        if not response:
            raise ValueError("No response received from Claude API")
        
        if not hasattr(response, 'content') or not response.content:
            raise ValueError("Claude API response has no content")
        
        if len(response.content) == 0:
            raise ValueError("Claude API response content is empty")
        
        if not hasattr(response.content[0], 'text'):
            raise ValueError("Claude API response content has no text attribute")
        
        response_text = response.content[0].text
        
        if not response_text or not response_text.strip():
            raise ValueError("Claude API returned empty response text")
        
        response_text = response_text.strip()
        
        # Parse JSON response
        # try:
        #     layout_data = json.loads(response_text)
        #     return layout_data
        # except json.JSONDecodeError as e:
        #     # If JSON parsing fails, try to extract JSON from the response
        #     import re
        #     json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        #     if json_match:
        #         layout_data = json.loads(json_match.group())
        #         return layout_data
        #     else:
        #         raise ValueError(f"Could not parse JSON from Claude response: {e}")
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError("Could not extract JSON content from Claude response")
        layout_data = json.loads(response_text)
        return layout_data
            
                
    except anthropic.APIError as e:
        raise ValueError(f"Anthropic API error: {e}")
    except Exception as e:
        raise ValueError(f"Error calling Claude API: {e}")


async def call_llm_for_layout_with_validation(input_text: str, max_attempts: int = 1) -> Dict[str, Any]:
    """
    Call Claude to generate room layout with validation and iterative correction.
    
    Args:
        input_text: Original layout description
        max_attempts: Maximum number of correction attempts
        
    Returns:
        Validated layout data
    """
    
    attempt = 0
    last_response = None
    
    while attempt < max_attempts:
        try:
            if attempt == 0:
                # First attempt - use original prompt
                response = await call_llm_for_layout(input_text)
            else:
                # Correction attempt - use correction prompt
                if last_response is None:
                    # If last response is None, retry with original prompt
                    response = await call_llm_for_layout(input_text)
                else:
                    validation_result = validate_room_layout(last_response["rooms"])
                    correction_prompt = generate_correction_prompt(validation_result, input_text, last_response)
                    
                    # Get API key and call Claude with correction prompt
                    claude_response = call_vlm(
                        vlm_type="qwen",
                        model="claude-sonnet-4-20250514",
                        max_tokens=3000,
                        temperature=0.2,  # Lower temperature for corrections
                        messages=[
                            {
                                "role": "user",
                                "content": correction_prompt
                            }
                        ]
                    )
                    
                    # Extract response with error checking
                    if not claude_response:
                        raise ValueError("No response received from Claude API during correction")
                    
                    if not hasattr(claude_response, 'content') or not claude_response.content:
                        raise ValueError("Claude API correction response has no content")
                    
                    if len(claude_response.content) == 0:
                        raise ValueError("Claude API correction response content is empty")
                    
                    if not hasattr(claude_response.content[0], 'text'):
                        raise ValueError("Claude API correction response content has no text attribute")
                    
                    response_text = claude_response.content[0].text
                    
                    if not response_text or not response_text.strip():
                        raise ValueError("Claude API returned empty correction response text")
                    
                    # try:
                    #     response = json.loads(response_text)
                    # except json.JSONDecodeError:
                    #     import re
                    #     json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    #     if json_match:
                    #         response = json.loads(json_match.group())
                    #     else:
                    #         raise ValueError("Could not parse correction response from Claude")

                    response_text = response_text.strip()
                    response_text = extract_json_from_response(response_text)
                    if not response_text:
                        raise ValueError("Could not extract JSON content from Claude response")
                    response = json.loads(response_text)
            
            # Validate the response
            if "rooms" not in response or not response["rooms"]:
                raise ValueError("Invalid response structure")
            
            validation_result = validate_room_layout(response["rooms"])
            
            if validation_result["valid"]:
                # Layout is valid, return it
                return response
            else:
                # Layout still has issues, try again
                last_response = response
                attempt += 1
                
                if attempt >= max_attempts:
                    # Return the best attempt we have with a warning
                    response["validation_warning"] = {
                        "message": f"Layout still has issues after {max_attempts} attempts",
                        "remaining_issues": validation_result["issues"]
                    }
                    return response
        
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                raise e
            # Try again with a different approach
            last_response = None
    
    # If we get here, all attempts failed
    raise ValueError(f"Failed to generate valid layout after {max_attempts} attempts")


def generate_correction_prompt(validation_result: Dict[str, Any], original_input: str, previous_attempt: Dict[str, Any]) -> str:
    """Generate a correction prompt for Claude based on validation issues."""
    
    issues_text = ""
    
    if validation_result["overlaps"]:
        overlaps_desc = []
        for overlap in validation_result["overlaps"]:
            overlaps_desc.append(f"'{overlap['room1']}' and '{overlap['room2']}' (overlap area: {overlap['overlap_area']:.2f} m²)")
        issues_text += f"\n- OVERLAPPING ROOMS: {', '.join(overlaps_desc)}"
    
    if validation_result["detached_rooms"]:
        issues_text += f"\n- DETACHED ROOMS: {', '.join(validation_result['detached_rooms'])} (these rooms have no shared walls with other rooms)"
    
    prompt = f"""The previous floor plan layout had CRITICAL ARCHITECTURAL ERRORS that must be fixed:
{issues_text}

REQUIREMENTS FOR CORRECTION:
1. NO ROOM OVERLAPS: Rooms must not overlap in space. Each room should have distinct boundaries.
2. CONNECTED LAYOUT: All rooms must be connected through shared walls. No isolated/detached rooms.
3. REALISTIC ADJACENCIES: Rooms should be logically connected (e.g., kitchen near dining, bedrooms in private areas).
4. PROPER SPACING: Leave small gaps (0.1m) between non-adjacent rooms if needed.

Original description: "{original_input}"

Please generate a CORRECTED floor plan that fixes these issues. Ensure:
- Rooms are arranged in a logical, connected layout
- No overlapping areas
- All rooms share walls with at least one other room
- Maintain the same room types and approximate sizes from the original request

Respond with ONLY the JSON layout data in the same format as before."""

    return prompt


async def request_missing_information(original_response: Dict[str, Any], validation_result: Dict[str, Any], input_text: str) -> Dict[str, Any]:
    """
    Ask Claude to provide missing information in the response.
    """
    
    missing_info = ""
    if validation_result["missing_fields"]:
        missing_info += f"\nMISSING FIELDS: {', '.join(validation_result['missing_fields'])}"
    
    if validation_result["invalid_fields"]:
        missing_info += f"\nINVALID FIELDS: {', '.join(validation_result['invalid_fields'])}"
    
    correction_prompt = f"""The previous JSON response was incomplete or had invalid fields:{missing_info}

Please provide a COMPLETE and CORRECTED JSON response for the layout: "{input_text}"

The response must include ALL required fields:
OUTPUT FORMAT:
```json
{{
    "building_style": "architectural style description",
    "rooms": [
        {{
            "room_type": "room name",
            "dimensions": {{"width": float, "length": float, "height": float}},
            "position": {{"x": float, "y": float, "z": float}},
            "doors": [
                {{
                    "width": float,
                    "height": float, 
                    "position_on_wall": float,
                    "wall_side": "north/south/east/west",
                    "door_type": "standard"
                }}
            ],
            "windows": [
                {{
                    "width": float,
                    "height": float,
                    "position_on_wall": float, 
                    "wall_side": "north/south/east/west",
                    "sill_height": float,
                    "window_type": "standard"
                }}
            ]
        }}
    ]
}}
```

CRITICAL: Ensure ALL fields are present and properly formatted. Include at least one door per room and appropriate windows."""

    try:
        response = call_vlm(
            vlm_type="qwen",
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            temperature=0.1,  # Very low temperature for precise completion
            messages=[
                {
                    "role": "user",
                    "content": correction_prompt
                }
            ]
        )
        
        # Extract and validate response
        if not response or not hasattr(response, 'content') or not response.content:
            raise ValueError("No valid response from Claude for missing information request")
        
        response_text = response.content[0].text.strip()
        
        # try:
        #     corrected_response = json.loads(response_text)
        #     return corrected_response
        # except json.JSONDecodeError:
        #     import re
        #     json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        #     if json_match:
        #         corrected_response = json.loads(json_match.group())
        #         return corrected_response
        #     else:
        #         raise ValueError("Could not parse corrected response from Claude")
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError("Could not extract JSON content from Claude response")
        corrected_response = json.loads(response_text)
        return corrected_response
                
    except Exception as e:
        raise ValueError(f"Failed to get missing information from Claude: {str(e)}")


async def test_claude_api_connection() -> Dict[str, Any]:
    """Test the connection to Claude Sonnet 4 API."""
    try:
        
        response = call_vlm(
            vlm_type="qwen",
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": "Respond with just 'Connection successful' if you can read this."
                }
            ]
        )
        
        response_text = response.content[0].text
        
        return {
            "success": True,
            "message": "Claude API connection successful",
            "model": "claude-sonnet-4-20250514",
            "response": response_text.strip(),
            "api_key_status": f"API key found (ending in ...{api_key[-4:]})"
        }
        
    except anthropic.APIError as e:
        return {
            "success": False,
            "error": f"Anthropic API error: {str(e)}",
            "error_type": "api_error"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Connection test failed: {str(e)}",
            "error_type": "general_error"
        }


async def call_llm_for_rooms_only(input_text: str) -> Dict[str, Any]:
    """Call Claude Sonnet 4 to generate room layout WITHOUT doors and windows."""
    
    
    # Construct the prompt for Claude - ROOMS ONLY
    prompt = f"""You are an expert architect and interior designer. Analyze this description and generate an appropriate ROOM LAYOUT (rooms and walls only, NO doors or windows): "{input_text}"

🚨 CRITICAL LAYOUT CONSTRAINTS 🚨
- **ABSOLUTELY NO OVERLAPPING ROOMS**: Room boundaries must NEVER intersect or overlap
- **NO ISOLATED ROOMS**: Every room must be adjacent to at least one other room for connectivity

STEP 1: ANALYZE REQUEST TYPE
First determine if the user is asking for:
- SINGLE ROOM: Only one specific room (e.g., "design a living room", "create a bedroom", "generate a kitchen")
- MULTIPLE ROOMS: A complete layout, house, apartment, or multiple connected rooms

STEP 2: GENERATE ROOM STRUCTURE ONLY
For SINGLE ROOM requests:
- Generate just that one room with appropriate rectangular vertices
- Position the room starting at coordinates (0, 0) as the first vertex
- Use realistic proportions for the specific room type
- Consider the room's intended function and furniture layout
- Example single room vertices: living room [(0,0), (0,6), (4.5,6), (4.5,0)], bedroom [(0,0), (0,4), (3.5,4), (3.5,0)]
- Vertices must form a perfect rectangle with 90-degree corners

For MULTIPLE ROOMS requests:
- **MANDATORY**: Ensure NO ROOM OVERLAPS and NO ISOLATED ROOMS using vertex-based checking
- **VERTEX CONSTRAINT**: When placing each new room, ensure NO vertex of the new room is inside any previously generated room's boundary
- Create a functional, well-designed layout that considers:
  - Architectural best practices and building codes
  - Natural light and ventilation
  - Traffic flow and accessibility
  - Room adjacencies and relationships (every room must touch at least one other room)
  - Realistic proportions using rectangular vertices
  - **VERIFY**: No room vertices are inside other rooms' boundaries
  - **VERIFY**: Every room is positioned adjacent to at least one other room
  - **VERIFY**: All rooms use perfect rectangular vertices with 90-degree corners

CRITICAL: Generate ONLY the room structure (rectangular vertices, height) - NO doors or windows at this stage.

Respond with ONLY a valid JSON object using this exact structure:

OUTPUT FORMAT:
```json
{{
    "building_style": "description of architectural style (e.g., Modern residential, Traditional colonial, etc.)",
    "rooms": [
        {{
            "room_type": "specific room type (e.g., living room, master bedroom, kitchen, etc.)",
            "vertices": [
                {{"x": float, "y": float}},
                {{"x": float, "y": float}},
                {{"x": float, "y": float}},
                {{"x": float, "y": float}}
            ],
            "height": float
        }}
    ]
}}
```

CRITICAL REQUIREMENTS:
- For SINGLE ROOM: Include exactly 1 room in the "rooms" array
- For MULTIPLE ROOMS: Include all rooms in the "rooms" array with proper layout
- Use metric coordinates (meters) for all vertices
- Standard ceiling height: 2.7m (can vary for special rooms)  
- For single rooms: start vertices at (0,0) and focus on room functionality
- For multiple rooms: arrange rectangular vertices logically without overlaps and ensure rooms can be connected later
- **VERTICES FORMAT**: Each room must have exactly 4 vertices forming a perfect rectangle
- **VERTEX ORDERING**: List vertices in clockwise or counter-clockwise order around the rectangle
- **ABSOLUTELY NO OVERLAPPING ROOMS**: Ensure no room vertices are inside other rooms and no boundaries intersect
- **NO ISOLATED ROOMS**: Every room must be positioned so it can be connected to at least one other room via doors (adjacency is essential)
- **VERTEX-BASED CHECKING**: Verify each room's 4 vertices do not fall inside any other room's boundary
- 🚨 DOUBLE-CHECK: Before finalizing, verify NO vertex overlaps and NO isolated rooms 🚨
- DO NOT include doors or windows arrays - these will be added in a separate step
- Focus on creating a solid architectural foundation with proper rectangular room vertices and relationships

Generate a room-only layout that truly reflects the input description with appropriate room types, rectangular vertices, and spatial relationships.

🚨 FINAL REMINDER - THESE ARE MANDATORY 🚨
- **NO OVERLAPPING ROOMS**: Check that no room vertices are inside other rooms and no boundaries intersect
- **NO ISOLATED ROOMS**: Verify every room can connect to at least one other room
- **PERFECT RECTANGLES**: Every room must have exactly 4 vertices forming a rectangle with 90-degree corners"""

    try:
        # Make API call to Claude
        response = call_vlm(
            vlm_type="openai",
            model="claude-sonnet-4-20250514",  # Using Claude Sonnet 4 as specified
            max_tokens=16000,  # Increased significantly for better results
            temperature=1.0,
            thinking=True,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Extract the response content with error checking
        if not response:
            raise ValueError("No response received from Claude API")
        
        if not hasattr(response, 'content') or not response.content:
            raise ValueError("Claude API response has no content")
        
        if len(response.content) == 0:
            raise ValueError("Claude API response content is empty")
        
        response_text = None
        for content in response.content:
            if hasattr(content, 'text'):
                response_text = content.text
                break
        if response_text is None:
            raise ValueError("Claude API response content has no text attribute")
        
        if not response_text or not response_text.strip():
            raise ValueError("Claude API returned empty response text")
        
        # Parse JSON response
        # try:
        #     layout_data = json.loads(response_text)
        #     return layout_data
        # except json.JSONDecodeError as e:
        #     # If JSON parsing fails, try to extract JSON from the response
        #     import re
        #     json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        #     if json_match:
        #         layout_data = json.loads(json_match.group())
        #         return layout_data
        #     else:
        #         raise ValueError(f"Could not parse JSON from Claude response: {e}")

        response_text = extract_json_from_response(response_text)

        if not response_text:
            raise ValueError("Could not extract JSON content from Claude response")
        layout_data = json.loads(response_text)


        # debug: get a visualization of the room layout
        debug_vis_dir = f"{SERVER_ROOT_DIR}/vis"
        os.makedirs(debug_vis_dir, exist_ok=True)
        debug_vis_path = os.path.join(debug_vis_dir, f"room_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        await get_vlm_room_layout_2d_visualization(layout_data, debug_vis_path)
        print(f"🔍 Debug: Room layout visualization saved to {debug_vis_path}", file=sys.stderr)
        
        # Convert vertices format back to position + dimensions format
        layout_data = convert_vertices_to_position_dimensions(layout_data)
        return layout_data
                
    except anthropic.APIError as e:
        raise ValueError(f"Anthropic API error: {e}")
    except Exception as e:
        raise ValueError(f"Error calling Claude API: {e}")

async def get_vlm_room_layout_2d_visualization(layout_data: Dict[str, Any], debug_vis_path: str) -> None:
    """
    Get a 2D visualization of the room layout using matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import Polygon
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization", file=sys.stderr)
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Check if we have rooms
    if "rooms" not in layout_data or not layout_data["rooms"]:
        # Create empty plot
        ax.text(0.5, 0.5, "No rooms found", ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.savefig(debug_vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Colors for different room types
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 
              'lightgray', 'lightcyan', 'wheat', 'lavender', 'mistyrose',
              'peachpuff', 'lightsteelblue', 'thistle', 'honeydew', 'azure']
    
    all_x = []
    all_y = []
    legend_elements = []
    
    # Set to maintain unique room vertices
    room_vertices = set()
    
    # Plot each room
    for i, room in enumerate(layout_data["rooms"]):
        room_type = room.get("room_type", f"Room {i+1}")
        
        # Handle both vertices format and position+dimensions format
        if "vertices" in room:
            # Vertices format (new format from our modified prompt)
            vertices = room["vertices"]
            if len(vertices) != 4:
                print(f"Warning: Room {room_type} has {len(vertices)} vertices, expected 4", file=sys.stderr)
                continue
                
            # Extract coordinates
            x_coords = [v["x"] for v in vertices]
            y_coords = [v["y"] for v in vertices]
            
            # Add vertices to set
            for v in vertices:
                room_vertices.add((v["x"], v["y"]))
            
        elif "position" in room and "dimensions" in room:
            # Position + dimensions format (legacy format)
            pos = room["position"]
            dim = room["dimensions"]
            
            x = pos["x"]
            y = pos["y"]
            width = dim["width"]
            length = dim["length"]
            
            # Create rectangle coordinates (counter-clockwise from bottom-left)
            x_coords = [x, x, x + width, x + width]
            y_coords = [y, y + length, y + length, y]
            
            # Add vertices to set
            room_vertices.add((x, y))
            room_vertices.add((x, y + length))
            room_vertices.add((x + width, y + length))
            room_vertices.add((x + width, y))
            
        else:
            print(f"Warning: Room {room_type} has neither vertices nor position+dimensions", file=sys.stderr)
            continue
        
        # Add to bounds calculation
        all_x.extend(x_coords)
        all_y.extend(y_coords)
        
        # Create polygon coordinates
        polygon_coords = list(zip(x_coords, y_coords))
        
        # Create and add polygon
        color = colors[i % len(colors)]
        polygon = Polygon(polygon_coords, closed=True, facecolor=color, 
                         edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(polygon)
        
        # Add room label at center
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        # Format room type for better display
        display_name = room_type.replace('_', ' ').title()
        
        # Add text with background for better readability
        ax.text(center_x, center_y, display_name, ha='center', va='center', 
                fontsize=9, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add height information if available
        height = room.get("height", "N/A")
        ax.text(center_x, center_y - 0.3, f"H: {height}m", ha='center', va='center', 
                fontsize=7, style='italic', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.6))
        
        # Add to legend
        legend_elements.append(patches.Patch(color=color, label=display_name))
    
    # Plot all room vertices as red points
    if room_vertices:
        vertex_x = [v[0] for v in room_vertices]
        vertex_y = [v[1] for v in room_vertices]
        
        # Plot vertices as red dots
        ax.scatter(vertex_x, vertex_y, color='red', s=50, zorder=10, alpha=0.8, 
                  edgecolors='darkred', linewidth=1, label='Vertices')
        
        # Add coordinate labels for vertices
        for x, y in room_vertices:
            ax.annotate(f'({x:.1f},{y:.1f})', 
                        xy=(x, y), xytext=(3, 3), 
                        textcoords='offset points', 
                        fontsize=7, color='darkred', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor='red'))
        
        # Add vertices to legend
        legend_elements.append(patches.Patch(color='red', label=f'Vertices ({len(room_vertices)})'))
    
    # Set axis properties
    if all_x and all_y:
        margin = max(2.0, (max(all_x) - min(all_x)) * 0.1)  # Dynamic margin
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    else:
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)
    
    # Set equal aspect ratio and styling
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X Coordinate (meters)', fontsize=12)
    ax.set_ylabel('Y Coordinate (meters)', fontsize=12)
    
    # Title with building style
    building_style = layout_data.get("building_style", "Unknown Style")
    ax.set_title(f'Room Layout Visualization\nBuilding Style: {building_style}\nTotal Rooms: {len(layout_data["rooms"])}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add legend if we have rooms
    if legend_elements:
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                 title="Room Types", title_fontsize=12, fontsize=10)
    
    # Add coordinate annotations at corners
    if all_x and all_y:
        # Bottom-left corner
        ax.annotate(f'({min(all_x):.1f}, {min(all_y):.1f})', 
                   xy=(min(all_x), min(all_y)), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Top-right corner  
        ax.annotate(f'({max(all_x):.1f}, {max(all_y):.1f})', 
                   xy=(max(all_x), max(all_y)), xytext=(-5, -5), 
                   textcoords='offset points', fontsize=8, alpha=0.7, 
                   ha='right', va='top')
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.02, 0.02, f'Generated: {timestamp}', fontsize=8, alpha=0.6)
    
    # Save the plot
    plt.tight_layout()
    try:
        plt.savefig(debug_vis_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✅ Room layout visualization saved successfully to {debug_vis_path}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error saving visualization: {e}", file=sys.stderr)
    finally:
        plt.close()

async def call_llm_for_doors_windows(rooms_data: List[Dict[str, Any]], input_text: str) -> Dict[str, Any]:
    """
    Call Claude Sonnet 4 to add doors and windows to an existing room layout using shared walls analysis.
    
    This function:
    1. Analyzes shared walls between rooms and with exterior
    2. Generates a floor plan visualization
    3. Sends both image and data to Claude
    4. Asks Claude for specific door/window placement
    5. Transforms response back to original format
    """
    
    # Initialize debug information collection
    debug_info = {
        "prompts": [],
        "responses": [],
        "steps": []
    }
    
    # Import required functions
    from utils import find_all_shared_walls, generate_floor_plan_visualization, calculate_room_wall_position_from_shared_wall
    
    # Step 1: Find all shared walls
    shared_walls_info = find_all_shared_walls(rooms_data)
    
    # Step 2: Generate floor plan visualization
    try:
        visualization_path = generate_floor_plan_visualization(rooms_data)
        
        # Read the image file
        with open(visualization_path, 'rb') as image_file:
            image_data = image_file.read()
        
        # TODO for debugging, save the image to a file under ./vis/
        debug_vis_dir = f"{SERVER_ROOT_DIR}/vis"
        os.makedirs(debug_vis_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = int(time.time())
        debug_vis_path = os.path.join(debug_vis_dir, f"floor_plan_{timestamp}.png")
        
        # Save a copy for debugging
        with open(debug_vis_path, 'wb') as debug_file:
            debug_file.write(image_data)
        
        print(f"🔍 Debug: Floor plan visualization saved to {debug_vis_path}", file=sys.stderr)
        
        # Clean up the temporary file
        os.unlink(visualization_path)
        
    except Exception as e:
        raise ValueError(f"Failed to generate floor plan visualization: {str(e)}")
    
    # Step 3: Prepare rooms summary for Claude
    rooms_summary = []
    for i, room in enumerate(rooms_data):
        room_summary = {
            "room_index": i,
            "room_type": room["room_type"],
            "dimensions": f"{room['dimensions']['width']}m × {room['dimensions']['length']}m × {room['dimensions']['height']}m",
            "position": f"({room['position']['x']}, {room['position']['y']}, {room['position']['z']})",
            "area": f"{room['dimensions']['width'] * room['dimensions']['length']:.1f} sq m"
        }
        rooms_summary.append(room_summary)
    
    # Step 4: Prepare shared walls summary for Claude
    room_room_walls_summary = []
    for i, wall in enumerate(shared_walls_info["room_room_walls"]):
        wall_summary = {
            "shared_wall_index": i,
            "room1": f"Room {wall['room1']['index']} ({wall['room1']['room_type']})",
            "room2": f"Room {wall['room2']['index']} ({wall['room2']['room_type']})",
            "room1_wall_side": wall["room1_wall"],
            "room2_wall_side": wall["room2_wall"],
            "shared_length": f"{wall['overlap_length']:.2f}m",
            "coordinates": f"({wall['x_start']:.1f}, {wall['y_start']:.1f}) to ({wall['x_end']:.1f}, {wall['y_end']:.1f})"
        }
        room_room_walls_summary.append(wall_summary)
    
    room_exterior_walls_summary = []
    for i, wall in enumerate(shared_walls_info["room_exterior_walls"]):
        wall_summary = {
            "exterior_wall_index": i,
            "room": f"Room {wall['room']['index']} ({wall['room']['room_type']})",
            "wall_side": wall["wall_side"],
            "wall_length": f"{wall['overlap_length']:.2f}m",
            "coordinates": f"({wall['x_start']:.1f}, {wall['y_start']:.1f}) to ({wall['x_end']:.1f}, {wall['y_end']:.1f})"
        }
        room_exterior_walls_summary.append(wall_summary)
    
    # Step 4.5: Traffic Flow Analysis using Claude
    debug_info["steps"].append("Step 4.5: Generating traffic flow analysis")
    
    traffic_flow_prompt = f"""You are an expert architect analyzing a floor plan to determine optimal traffic flow patterns for door placement.

I'm providing you with:
1. A floor plan visualization showing the room layout
2. Detailed room information
3. Shared wall analysis between rooms
4. Exterior wall segments

ORIGINAL REQUEST: "{input_text}"

ROOM LAYOUT:
{json.dumps(rooms_summary, indent=2)}

SHARED WALLS BETWEEN ROOMS ({len(room_room_walls_summary)} total):
{json.dumps(room_room_walls_summary, indent=2)}

EXTERIOR WALLS ({len(room_exterior_walls_summary)} total):
{json.dumps(room_exterior_walls_summary, indent=2)}

TASK: Analyze the floor plan and determine the optimal traffic flow strategy for this layout.

Please provide:

1. ENTRY STRATEGY:
   - Which room should have the main entrance door?
   - Which exterior wall segment is best for the entry door?
   - Reasoning for entry door placement

2. TRAFFIC FLOW GRAPH:
   - How should people move through the space from the entry?
   - Which rooms need direct connections to which other rooms?
   - What is the circulation hierarchy (main paths vs secondary paths)?
   - Which shared walls should have connecting doors?

3. CONNECTIVITY REQUIREMENTS:
   - List all room-to-room connections needed for good traffic flow
   - Prioritize connections (essential vs optional)
   - Consider both functional needs and user convenience

FORMAT your response as JSON:

```json
{{
    "entry_strategy": {{
        "entry_room_index": int,
        "preferred_exterior_wall_index": int,
        "reasoning": "Why this entry location is optimal"
    }},
    "traffic_flow": {{
        "main_circulation_path": ["room_index_sequence"],
        "secondary_paths": [["room_index_sequence"]],
        "circulation_reasoning": "Explanation of flow strategy"
    }},
    "required_connections": [
        {{
            "shared_wall_index": int,
            "priority": "essential/important/optional",
            "reasoning": "Why this connection is needed"
        }}
    ]
}}
```

Focus on creating an efficient, logical flow that serves the original request: "{input_text}"
"""

    try:
        debug_info["prompts"].append({
            "step": "4.5_traffic_flow",
            "prompt": traffic_flow_prompt
        })
        
        # Make API call to Claude for traffic flow analysis
        traffic_response = call_vlm(
            vlm_type="qwen",
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": traffic_flow_prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(image_data).decode('utf-8')
                            }
                        }
                    ]
                }
            ]
        )
        
        traffic_response_text = traffic_response.content[0].text.strip()
        debug_info["responses"].append({
            "step": "4.5_traffic_flow",
            "response": traffic_response_text
        })
        
        # Parse traffic flow response
        try:
            # if '```json' in traffic_response_text:
            #     json_start = traffic_response_text.find('```json') + 7
            #     json_end = traffic_response_text.find('```', json_start)
            #     if json_end == -1:
            #         json_end = len(traffic_response_text)
            #     traffic_flow_data = json.loads(traffic_response_text[json_start:json_end].strip())
            # elif '```' in traffic_response_text:
            #     json_start = traffic_response_text.find('```') + 3
            #     json_end = traffic_response_text.find('```', json_start)
            #     if json_end == -1:
            #         json_end = len(traffic_response_text)
            #     traffic_flow_data = json.loads(traffic_response_text[json_start:json_end].strip())
            # else:
            #     # Try to find JSON object in the response
            #     start_idx = traffic_response_text.find('{')
            #     end_idx = traffic_response_text.rfind('}')
            #     if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            #         traffic_flow_data = json.loads(traffic_response_text[start_idx:end_idx + 1])
            #     else:
            #         # Fallback to empty traffic flow data
            #         traffic_flow_data = {}

            traffic_flow_data = extract_json_from_response(traffic_response_text)
            if not traffic_flow_data:
                raise ValueError("Could not extract JSON content from Claude response")
            traffic_flow_data = json.loads(traffic_flow_data)
        except json.JSONDecodeError:
            # Fallback to empty traffic flow data if parsing fails
            traffic_flow_data = {}
        
        debug_info["steps"].append("Step 4.5 completed: Traffic flow analysis generated")
        print(f"🔍 Debug: Traffic flow analysis completed", file=sys.stderr)
        
    except Exception as e:
        # Traffic flow analysis is optional, continue if it fails
        traffic_flow_data = {}
        debug_info["steps"].append(f"Step 4.5 failed: {str(e)}")
        print(f"⚠️ Debug: Traffic flow analysis failed: {e}")
    
    # Step 5: Construct the prompt for Claude
    traffic_flow_section = ""
    if traffic_flow_data:
        traffic_flow_section = f"""

TRAFFIC FLOW ANALYSIS:
{json.dumps(traffic_flow_data, indent=2)}

IMPORTANT: Use the traffic flow analysis above to guide your door placement decisions. Pay special attention to:
- The recommended entry strategy
- Required connections for good circulation
- Priority levels of different connections
"""

    prompt = f"""You are an expert architect analyzing a floor plan to add doors and windows. I'm providing you with:
1. A floor plan visualization image
2. Room layout data
3. Shared wall analysis
4. Traffic flow analysis for optimal circulation

ORIGINAL REQUEST: "{input_text}"

ROOM LAYOUT:
{json.dumps(rooms_summary, indent=2)}

SHARED WALLS BETWEEN ROOMS ({len(room_room_walls_summary)} total):
{json.dumps(room_room_walls_summary, indent=2)}

EXTERIOR WALLS ({len(room_exterior_walls_summary)} total):
{json.dumps(room_exterior_walls_summary, indent=2)}{traffic_flow_section}

COORDINATE SYSTEM:
- X-axis: horizontal (left-right)
- Y-axis: vertical (bottom-top)
- Wall sides: north (top), south (bottom), east (right), west (left)

YOUR TASK:
Based on the floor plan image, data, and traffic flow analysis, provide door and window placement recommendations following these rules:

1. ENTRY DOOR (exactly one for the entire floor plan):
   - Must be placed on an exterior wall (room-exterior shared wall)
   - Follow the traffic flow analysis recommendations if available
   - Predict appropriate entry door material/style that matches the building aesthetic
   - Specify: room_index, wall_side, center_position (0.0-1.0), width, height, door_material (brief description)

2. CONNECTING DOORS (for room-room shared walls):
   - Place doors to connect rooms according to traffic flow requirements
   - Prioritize essential connections over optional ones
   - Only place on room-room shared walls (between two rooms)
   - Choose between physical doors or wide openings based on room function:
     * Use physical doors (opening=false) for: bedrooms, bathrooms, private offices, storage rooms
     * Use wide openings (opening=true) for: living room-kitchen, living room-dining, open-plan concepts
   - Predict appropriate door material/style based on room types and design aesthetics
   - Specify: shared_wall_index, center_position_on_shared_wall (0.0-1.0), width, height, opening (true/false), door_material (brief description)

3. WINDOWS (only on exterior walls):
   - Place on room-exterior shared walls only
   - Consider natural light, ventilation, and room function
   - Specify: exterior_wall_index, center_position (0.0-1.0), width, height, sill_height

FORMATTING:
Return your response as a JSON object with this exact structure:

```json
{{
    "entry_door": {{
        "room_index": int,
        "wall_side": "north/south/east/west",
        "center_position": float_0_to_1,
        "width": float_meters,
        "height": float_meters,
        "door_type": "entry/main/front",
        "door_material": "Brief description of door style/material (e.g., solid wood entry door, glass panel door, traditional front door)",
        "reasoning": "Why this location was chosen"
    }},
    "connecting_doors": [
        {{
            "shared_wall_index": int,
            "center_position_on_shared_wall": float_0_to_1,
            "width": float_meters,
            "height": float_meters,
            "door_type": "interior/standard",
            "opening": bool,
            "door_material": "Brief description of door style/material (e.g., wooden panel door, glass door, modern steel door)",
            "reasoning": "Why this connection is needed and why door/opening was chosen"
        }}
    ],
    "windows": [
        {{
            "exterior_wall_index": int,
            "center_position": float_0_to_1,
            "width": float_meters,
            "height": float_meters,
            "sill_height": float_meters,
            "window_type": "standard",
            "window_grid": [nx, ny],
            "window_appearance_description": "string",
            "glass_color": [r, g, b],
            "frame_color": [r, g, b],
            "reasoning": "Why this window placement was chosen"
        }}
    ]
}}
```

IMPORTANT COLOR SELECTION GUIDANCE:
- Analyze the building style and room function before choosing colors
- For TRADITIONAL/CLASSIC buildings: Use warmer, earth-tone colors with natural wood frames
- For MODERN/CONTEMPORARY buildings: Use cooler, neutral colors with metal or painted frames
- For INDUSTRIAL buildings: Use darker, muted colors with metal frames
- Match glass tint to room privacy needs: bedrooms (subtle tints), bathrooms (privacy glass), living areas (clear)
- Ensure frame colors complement the overall architectural aesthetic
- Avoid repetitive color choices - vary colors appropriately across different rooms and window types

CONSTRAINTS:
- Entry door: width 0.8-1.2m, height 2.0-2.1m
- Interior doors: width 0.7-0.9m, height 2.0-2.1m  
- Wide openings: width 1.2-3.0m, height 2.0-2.4m (for open-plan connections)
- Windows: width 0.8-2.0m, height 1.0-1.8m, sill height 0.8-1.2m
- Window grid [nx, ny]: Define window pane layout (nx=horizontal panes, ny=vertical panes), examples:
  * Fixed/Picture: [1, 1] - single large pane
  * Slider: [2, 1] - two horizontal sliding panes
  * Hung: [1, 2] - two vertical stacked panes
  * Bay: [2, 2] to [3, 3] - multiple panes in grid
  * Casement: [1, 1] - single hinged pane
- Window appearance description: Brief description combining frame style and glass type, examples:
  * "classic white painted wooden window frame with frosted glass"
  * "modern black aluminum window frame with translucent glass panels"
  * "classic white window frame with frosted glass panels"
  * "classic natural wood grain window frame with frosted glass panes"
- Window colors: Generate contextually appropriate glass_color and frame_color as RGB values [0-255] based on:
  * BUILDING STYLE: Traditional/classic → warmer tones, Modern → cooler/neutral tones, Industrial → darker frames
  * ROOM FUNCTION: Bedrooms → softer tints, Kitchens → clear/bright, Bathrooms → privacy (frosted/tinted)
  * AESTHETIC HARMONY: Match building style and room character with meaningful color choices
  
  GLASS COLOR EXAMPLES (choose based on context):
  * Clear/Neutral: [245, 247, 250] (modern clear), [252, 251, 244] (warm clear)
  * Subtle Tints: [240, 248, 255] (alice blue), [176, 224, 230] (powder blue), [173, 216, 230] (light blue)
  * Privacy Glass: [199, 227, 225] (frosted blue glass), [184, 227, 233] (frosted powder blue), [192, 230, 235] (frosted light blue)
  * Colored Glass: [135, 206, 250] (sky blue), [166, 227, 224] (ice crystal blue), [200, 233, 238] (pale blue)
  
  FRAME COLOR EXAMPLES (choose based on style):
  * Natural Wood: [49, 36, 26] (dark walnut), [139, 97, 69] (dark oak), [60, 34, 24] (espresso)
  * Painted Wood: [252, 251, 244] (cream white), [64, 89, 71] (forest green), [71, 82, 107] (navy blue)
  * Metal Frames: [54, 69, 79] (charcoal), [191, 191, 191] (brushed aluminum), [115, 107, 97] (bronze)
  * Modern Colors: [89, 97, 107] (graphite), [255, 253, 228] (creamy white), [92, 62, 41] (antique oak)

- Keep doors/windows away from corners (position 0.1-0.9 range)
- Ensure no overlapping doors/windows on the same wall segment
- Consider furniture placement and room circulation

Focus on creating a functional and well-lit layout that serves the original request: "{input_text}"
"""

    debug_info["prompts"].append({
        "step": "5_main_door_window_placement",
        "prompt": prompt
    })

    try:
        # Make API call to Claude with both text and image
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
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(image_data).decode('utf-8')
                            }
                        }
                    ]
                }
            ]
        )
        
        # Extract the response content with error checking
        if not response or not hasattr(response, 'content') or not response.content:
            raise ValueError("No response received from Claude API")
        
        if len(response.content) == 0 or not hasattr(response.content[0], 'text'):
            raise ValueError("Claude API response content is empty or invalid")
        
        response_text = response.content[0].text
        
        debug_info["responses"].append({
            "step": "5_main_door_window_placement",
            "response": response_text
        })
        
        if not response_text or not response_text.strip():
            raise ValueError("Claude API returned empty response text")
        
        # Parse Claude's JSON response
        try:
            # Handle markdown code blocks if present
            # if '```json' in response_text:
            #     json_start = response_text.find('```json') + 7
            #     json_end = response_text.find('```', json_start)
            #     if json_end == -1:
            #         json_end = len(response_text)
            #     claude_response = json.loads(response_text[json_start:json_end].strip())
            # elif '```' in response_text:
            #     json_start = response_text.find('```') + 3
            #     json_end = response_text.find('```', json_start)
            #     if json_end == -1:
            #         json_end = len(response_text)
            #     claude_response = json.loads(response_text[json_start:json_end].strip())
            # else:
            #     # Try to find JSON object in the response
            #     start_idx = response_text.find('{')
            #     end_idx = response_text.rfind('}')
            #     if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            #         claude_response = json.loads(response_text[start_idx:end_idx + 1])
            #     else:
            #         raise ValueError("Could not find JSON object in Claude response")
            claude_response = extract_json_from_response(response_text)
            if not claude_response:
                raise ValueError("Could not extract JSON content from Claude response")
            claude_response = json.loads(claude_response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from Claude response: {e}")
        
        # Step 6: Transform Claude response back to original format
        result = {
            "building_style": "same as before",
            "rooms": []
        }
        
        # Initialize rooms with existing data
        for i, room_data in enumerate(rooms_data):
            room_result = {
                "room_type": room_data["room_type"],
                "dimensions": room_data["dimensions"],
                "position": room_data["position"],
                "doors": [],
                "windows": []
            }
            result["rooms"].append(room_result)
        
        # Add entry door
        if "entry_door" in claude_response and claude_response["entry_door"]:
            entry_door = claude_response["entry_door"]
            room_index = entry_door["room_index"]
            
            if 0 <= room_index < len(result["rooms"]):
                door_info = {
                    "width": entry_door["width"],
                    "height": entry_door["height"],
                    "position_on_wall": entry_door["center_position"],
                    "wall_side": entry_door["wall_side"],
                    "door_type": entry_door.get("door_type", "entry"),
                    "door_material": entry_door.get("door_material", "standard wooden door")
                }
                
                # CRITICAL: Validate both width AND position to ensure door fits within wall boundaries
                original_width = door_info["width"]
                original_position = door_info["position_on_wall"]
                
                # 1. Calculate maximum width constraint based on room dimensions
                if entry_door["wall_side"] in ["north", "south"]:
                    wall_length = rooms_data[room_index]["dimensions"]["width"]
                else:
                    wall_length = rooms_data[room_index]["dimensions"]["length"]
                
                max_width_constraint = wall_length * 0.8
                
                # 2. Calculate position-based width constraint (how much width fits at this position)
                # Door extends from (position - width/2) to (position + width/2)
                # Both bounds must be within [0.1, 0.9] safety range
                
                safe_start = 0.1
                safe_end = 0.9
                max_half_width_from_start = original_position - safe_start  # Space available to the left
                max_half_width_from_end = safe_end - original_position      # Space available to the right
                max_half_width = min(max_half_width_from_start, max_half_width_from_end)
                
                position_based_max_width = max_half_width * 2 * wall_length  # Convert back to meters
                
                # 3. Apply the most restrictive constraint
                final_width = min(original_width, max_width_constraint, max(0.7, position_based_max_width))  # Minimum 0.7m door
                
                # 4. If width was reduced significantly, try to adjust position to allow larger width
                if final_width < original_width * 0.8:  # If width was reduced by more than 20%
                    # Try to find a better position that allows a larger width
                    desired_width = min(original_width, max_width_constraint)
                    required_half_width_ratio = (desired_width / 2) / wall_length
                    
                    # Find the center of the safe zone where this width would fit
                    min_center = safe_start + required_half_width_ratio
                    max_center = safe_end - required_half_width_ratio
                    
                    if min_center <= max_center:
                        # There's a valid range, choose the position closest to original
                        if original_position < min_center:
                            final_position = min_center
                        elif original_position > max_center:
                            final_position = max_center
                        else:
                            final_position = original_position
                        
                        final_width = desired_width
                    else:
                        # No position allows the desired width, keep the reduced width and original position
                        final_position = max(safe_start, min(safe_end, original_position))
                else:
                    # Width is acceptable, just ensure position is in safe range
                    final_position = max(safe_start, min(safe_end, original_position))
                
                # 5. Apply the final values
                door_info["width"] = final_width
                door_info["position_on_wall"] = final_position
                
                result["rooms"][room_index]["doors"].append(door_info)
        
        # Add connecting doors
        if "connecting_doors" in claude_response and claude_response["connecting_doors"]:
            for connecting_door in claude_response["connecting_doors"]:
                shared_wall_index = connecting_door["shared_wall_index"]
                
                if 0 <= shared_wall_index < len(shared_walls_info["room_room_walls"]):
                    shared_wall = shared_walls_info["room_room_walls"][shared_wall_index]
                    
                    # Get the two rooms
                    room1_index = shared_wall["room1"]["index"]
                    room2_index = shared_wall["room2"]["index"]
                    
                    # CRITICAL: Apply clipping/validation to the shared door BEFORE transforming to room coordinates
                    # This ensures both rooms get the same physical door dimensions and position
                    
                    # 1. Validate and clip the door width (same width for both rooms since it's the same physical door)
                    original_door_width = connecting_door["width"]
                    
                    # Calculate maximum allowed width based on both rooms and the shared wall length
                    room1_max_width = rooms_data[room1_index]["dimensions"]["width"] * 0.8 if shared_wall["room1_wall"] in ["north", "south"] else rooms_data[room1_index]["dimensions"]["length"] * 0.8
                    room2_max_width = rooms_data[room2_index]["dimensions"]["width"] * 0.8 if shared_wall["room2_wall"] in ["north", "south"] else rooms_data[room2_index]["dimensions"]["length"] * 0.8
                    shared_wall_max_width = shared_wall["overlap_length"] * 0.8  # Leave some margin on the shared wall
                    
                    # Use the most restrictive constraint
                    final_door_width = min(original_door_width, room1_max_width, room2_max_width, shared_wall_max_width)
                    
                    # 2. Validate and clip the position on shared wall
                    original_shared_position = connecting_door["center_position_on_shared_wall"]
                    
                    # Calculate how much space the door needs on the shared wall
                    door_half_width_ratio = (final_door_width / 2) / shared_wall["overlap_length"]
                    min_safe_position = max(0.1, door_half_width_ratio + 0.05)  # 5% margin
                    max_safe_position = min(0.9, 1.0 - door_half_width_ratio - 0.05)  # 5% margin
                    
                    # Clip the position to safe range
                    final_shared_position = max(min_safe_position, min(max_safe_position, original_shared_position))
                    
                    # 3. Now calculate positions on each room's wall using the SAME validated shared position
                    room1_wall_position = calculate_room_wall_position_from_shared_wall(
                        shared_wall, final_shared_position, rooms_data[room1_index], True
                    )
                    
                    room2_wall_position = calculate_room_wall_position_from_shared_wall(
                        shared_wall, final_shared_position, rooms_data[room2_index], False
                    )
                    
                    # 4. Create identical doors for both rooms (same width, corresponding positions)
                    door1_info = {
                        "width": final_door_width,  # Same width for both
                        "height": connecting_door["height"],  # Same height for both
                        "position_on_wall": room1_wall_position,
                        "wall_side": shared_wall["room1_wall"],
                        "door_type": connecting_door.get("door_type", "interior"),
                        "opening": connecting_door.get("opening", False),  # Parse opening property
                        "door_material": connecting_door.get("door_material", "standard wooden door")
                    }
                    
                    door2_info = {
                        "width": final_door_width,  # Same width for both
                        "height": connecting_door["height"],  # Same height for both
                        "position_on_wall": room2_wall_position,
                        "wall_side": shared_wall["room2_wall"],
                        "door_type": connecting_door.get("door_type", "interior"),
                        "opening": connecting_door.get("opening", False),  # Parse opening property
                        "door_material": connecting_door.get("door_material", "standard wooden door")
                    }
                    
                    # No additional clipping needed since we already validated everything at the shared wall level
                    
                    # Add doors to both rooms
                    result["rooms"][room1_index]["doors"].append(door1_info)
                    result["rooms"][room2_index]["doors"].append(door2_info)
        
        # Add windows
        if "windows" in claude_response and claude_response["windows"]:
            for window in claude_response["windows"]:
                exterior_wall_index = window["exterior_wall_index"]
                
                if 0 <= exterior_wall_index < len(shared_walls_info["room_exterior_walls"]):
                    exterior_wall = shared_walls_info["room_exterior_walls"][exterior_wall_index]
                    room_index = exterior_wall["room"]["index"]
                    
                    window_info = {
                        "width": window["width"],
                        "height": window["height"],
                        "position_on_wall": window["center_position"],
                        "wall_side": exterior_wall["wall_side"],
                        "sill_height": window["sill_height"],
                        "window_type": window.get("window_type", "standard"),
                        "window_grid": window.get("window_grid", [1, 1]),
                        "glass_color": window.get("glass_color", [204, 230, 255]),
                        "frame_color": window.get("frame_color", [77, 77, 77])
                    }
                    
                    # CRITICAL: Apply width clipping considering BOTH room wall dimensions AND exterior wall segment length
                    # This ensures windows don't exceed the available exterior wall segment
                    
                    original_window_width = window_info["width"]
                    original_window_position = window_info["position_on_wall"]
                    
                    # 1. Calculate room-based constraint (based on room's overall wall dimensions)
                    if window_info["wall_side"] in ["north", "south"]:
                        room_max_width = rooms_data[room_index]["dimensions"]["width"] * 0.8
                    else:
                        room_max_width = rooms_data[room_index]["dimensions"]["length"] * 0.8
                    
                    # 2. Calculate exterior wall segment constraint (based on actual available wall segment)
                    exterior_wall_max_width = exterior_wall["overlap_length"] * 0.8  # Leave 20% margin on the segment
                    
                    # 3. Calculate position-based width constraint for the exterior wall segment
                    # Window extends from (position - width/2) to (position + width/2) within the segment
                    # Both bounds must be within [0.1, 0.9] safety range on the segment
                    
                    safe_start = 0.1
                    safe_end = 0.9
                    max_half_width_from_start = original_window_position - safe_start
                    max_half_width_from_end = safe_end - original_window_position
                    max_half_width = min(max_half_width_from_start, max_half_width_from_end)
                    
                    position_based_max_width = max_half_width * 2 * exterior_wall["overlap_length"]  # Convert back to meters
                    
                    # 4. Use the most restrictive constraint
                    max_width_constraint = min(room_max_width, exterior_wall_max_width)
                    final_window_width = min(original_window_width, max_width_constraint, max(0.8, position_based_max_width))  # Minimum 0.8m window
                    
                    # 5. If width was reduced significantly, try to adjust position to allow larger width
                    if final_window_width < original_window_width * 0.8:  # If width was reduced by more than 20%
                        # Try to find a better position that allows a larger width
                        desired_width = min(original_window_width, max_width_constraint)
                        required_half_width_ratio = (desired_width / 2) / exterior_wall["overlap_length"]
                        
                        # Find the center of the safe zone where this width would fit
                        min_center = safe_start + required_half_width_ratio
                        max_center = safe_end - required_half_width_ratio
                        
                        if min_center <= max_center:
                            # There's a valid range, choose the position closest to original
                            if original_window_position < min_center:
                                final_window_position = min_center
                            elif original_window_position > max_center:
                                final_window_position = max_center
                            else:
                                final_window_position = original_window_position
                            
                            final_window_width = desired_width
                        else:
                            # No position allows the desired width, keep the reduced width and clamp position
                            final_window_position = max(safe_start, min(safe_end, original_window_position))
                    else:
                        # Width is acceptable, just ensure position is in safe range
                        final_window_position = max(safe_start, min(safe_end, original_window_position))
                    
                    # 6. Apply the final values
                    window_info["width"] = final_window_width
                    window_info["position_on_wall"] = final_window_position
                    
                    # 7. Check for conflicts with existing doors/windows on the same wall
                    existing_items = []
                    for door in result["rooms"][room_index]["doors"]:
                        if door["wall_side"] == window_info["wall_side"]:
                            existing_items.append({
                                "position": door["position_on_wall"],
                                "width": door["width"],
                                "type": "door"
                            })
                    for existing_window in result["rooms"][room_index]["windows"]:
                        if existing_window["wall_side"] == window_info["wall_side"]:
                            existing_items.append({
                                "position": existing_window["position_on_wall"],
                                "width": existing_window["width"],
                                "type": "window"
                            })
                    
                    # 8. Improved conflict resolution: find the best non-conflicting position
                    if existing_items:
                        wall_length = exterior_wall["overlap_length"]
                        window_width_ratio = window_info["width"] / wall_length
                        window_half_width_ratio = window_width_ratio / 2
                        
                        # Sort existing items by position
                        existing_items.sort(key=lambda x: x["position"])
                        
                        # Try to find a safe position that doesn't conflict
                        safe_position = window_info["position_on_wall"]
                        position_found = False
                        
                        # Create a list of occupied segments
                        occupied_segments = []
                        for item in existing_items:
                            item_width_ratio = item["width"] / wall_length
                            item_half_width_ratio = item_width_ratio / 2
                            margin = 0.05  # 5% margin between items
                            
                            segment_start = max(0.0, item["position"] - item_half_width_ratio - margin)
                            segment_end = min(1.0, item["position"] + item_half_width_ratio + margin)
                            occupied_segments.append((segment_start, segment_end))
                        
                        # Merge overlapping segments
                        if occupied_segments:
                            occupied_segments.sort()
                            merged_segments = [occupied_segments[0]]
                            for current_start, current_end in occupied_segments[1:]:
                                last_start, last_end = merged_segments[-1]
                                if current_start <= last_end:
                                    # Overlapping segments, merge them
                                    merged_segments[-1] = (last_start, max(last_end, current_end))
                                else:
                                    # Non-overlapping segment
                                    merged_segments.append((current_start, current_end))
                        
                        # Find a free space for the window
                        min_required_space = window_width_ratio + 0.1  # Window + 10% margin
                        
                        # Check if current position is still valid
                        current_start = safe_position - window_half_width_ratio
                        current_end = safe_position + window_half_width_ratio
                        
                        conflicts_with_current = any(
                            not (current_end <= seg_start or current_start >= seg_end)
                            for seg_start, seg_end in merged_segments
                        )
                        
                        if not conflicts_with_current and 0.1 <= current_start and current_end <= 0.9:
                            # Current position is fine
                            position_found = True
                        else:
                            # Find alternative position
                            # Try gaps between occupied segments
                            for i in range(len(merged_segments) + 1):
                                if i == 0:
                                    # Before first segment
                                    gap_start = 0.1
                                    gap_end = merged_segments[0][0] if merged_segments else 0.9
                                elif i == len(merged_segments):
                                    # After last segment
                                    gap_start = merged_segments[-1][1]
                                    gap_end = 0.9
                                else:
                                    # Between segments
                                    gap_start = merged_segments[i-1][1]
                                    gap_end = merged_segments[i][0]
                                
                                gap_size = gap_end - gap_start
                                if gap_size >= min_required_space:
                                    # Found a suitable gap
                                    safe_position = gap_start + gap_size / 2
                                    safe_position = max(0.1, min(0.9, safe_position))
                                    position_found = True
                                    break
                        
                        # If no good position found, skip this window
                        if not position_found:
                            continue  # Skip adding this window
                        
                        window_info["position_on_wall"] = safe_position
                    
                    # 9. CRITICAL: Validate window height against room/wall height
                    # Ensure that sill_height + window_height does not exceed the room height
                    room_height = rooms_data[room_index]["dimensions"]["height"]
                    original_sill_height = window_info["sill_height"]
                    original_window_height = window_info["height"]
                    total_window_height = original_sill_height + original_window_height
                    
                    if total_window_height > room_height:
                        # Window is too tall for the room, need to scale down proportionally
                        # Leave a small margin (10cm) from the ceiling
                        max_total_height = room_height - 0.1  # 10cm margin from ceiling
                        
                        # Calculate scaling ratio
                        height_ratio = max_total_height / total_window_height
                        
                        # Apply scaling to both sill_height and window height
                        window_info["sill_height"] = original_sill_height * height_ratio
                        window_info["height"] = original_window_height * height_ratio
                        
                        # Ensure minimum sill height (at least 0.5m from floor for safety)
                        min_sill_height = 0.5
                        if window_info["sill_height"] < min_sill_height:
                            window_info["sill_height"] = min_sill_height
                            # Recalculate window height with the new sill height
                            remaining_height = max_total_height - min_sill_height
                            if remaining_height > 0.3:  # Minimum 30cm window height
                                window_info["height"] = remaining_height
                            else:
                                # Room is too short for a proper window, skip this window
                                continue
                    
                    result["rooms"][room_index]["windows"].append(window_info)
        
        # Add debug information to the result
        result["debug_info"] = debug_info
        result["debug_info"]["steps"].append("Step 6 completed: All doors and windows processed")
        
        return result
        
    except anthropic.APIError as e:
        raise ValueError(f"Anthropic API error: {e}")
    except Exception as e:
        raise ValueError(f"Error calling Claude API: {e}")


async def call_llm_for_rooms_with_validation(input_text: str, max_attempts: int = 3) -> Dict[str, Any]:
    """
    Call Claude to generate room layout (rooms only) with validation and iterative correction.
    
    Args:
        input_text: Original layout description
        max_attempts: Maximum number of correction attempts
        
    Returns:
        Validated room-only layout data
    """
    
    attempt = 0
    last_response = None
    
    while attempt < max_attempts:
        try:
            if attempt == 0:
                # First attempt - use rooms-only prompt
                response = await call_llm_for_rooms_only_solver(input_text)
            else:
                # Correction attempt - use correction prompt
                if last_response is None:
                    # If last response is None, retry with original prompt
                    response = await call_llm_for_rooms_only_solver(input_text)
                else:
                    # Import validation function for rooms-only
                    validation_result = validate_room_only_layout(last_response["rooms"])
                    correction_prompt = generate_room_correction_prompt(validation_result, input_text, last_response)
                    
                    claude_response = call_vlm(
                        vlm_type="qwen",
                        model="claude-sonnet-4-20250514",
                        max_tokens=3000,
                        temperature=0.2,  # Lower temperature for corrections
                        messages=[
                            {
                                "role": "user",
                                "content": correction_prompt
                            }
                        ]
                    )
                    
                    # Extract and parse Claude's response
                    if not claude_response or not hasattr(claude_response, 'content') or not claude_response.content:
                        raise ValueError("No response received from Claude API during correction")
                    
                    response_text = claude_response.content[0].text
                    
                    # try:
                    #     response = json.loads(response_text)
                    # except json.JSONDecodeError:
                    #     import re
                    #     json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    #     if json_match:
                    #         response = json.loads(json_match.group())
                    #     else:
                    #         raise ValueError("Could not parse correction response from Claude")
                    response_text = extract_json_from_response(response_text)
                    if not response_text:
                        raise ValueError("Could not extract JSON content from Claude response")
                    response = json.loads(response_text)
            
            # Validate the response (rooms only)
            if "rooms" not in response or not response["rooms"]:
                raise ValueError("Invalid response structure")
            
            # Import validation function for rooms-only
            validation_result = validate_room_only_layout(response["rooms"])
            
            if validation_result["valid"]:
                # Layout is valid, return it
                return response
            else:
                # Layout still has issues, try again
                last_response = response
                attempt += 1
                
                if attempt >= max_attempts:
                    # Return the best attempt we have with a warning
                    response["validation_warning"] = {
                        "message": f"Room layout still has issues after {max_attempts} attempts",
                        "remaining_issues": validation_result["issues"]
                    }
                    return response
        
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                raise e
            # Try again with a different approach
            last_response = None
    
    # If we get here, all attempts failed
    raise ValueError(f"Failed to generate valid room layout after {max_attempts} attempts")


def generate_room_correction_prompt(validation_result: Dict[str, Any], original_input: str, previous_attempt: Dict[str, Any]) -> str:
    """Generate a correction prompt for room-only layout issues."""
    
    issues = validation_result.get("issues", [])
    overlaps = validation_result.get("overlaps", [])
    detached_rooms = validation_result.get("detached_rooms", [])
    
    prompt = f"""The previous room layout attempt has architectural issues that need correction.

ORIGINAL REQUEST: "{original_input}"

PREVIOUS ATTEMPT:
{json.dumps(previous_attempt, indent=2)}

ISSUES FOUND:
{json.dumps(issues, indent=2)}

SPECIFIC PROBLEMS:
"""
    
    if overlaps:
        prompt += f"\nROOM OVERLAPS: {overlaps}"
        prompt += "\n- Move rooms so they don't occupy the same space"
        prompt += "\n- Ensure clear separation between room boundaries"
    
    if detached_rooms:
        prompt += f"\nDETACHED ROOMS: {detached_rooms}"
        prompt += "\n- Arrange rooms so they share walls (adjacent positioning)"
        prompt += "\n- Ensure rooms can be connected with doors later"
    
    prompt += """

Please generate a CORRECTED room-only layout that:
1. Fixes all the issues listed above
2. Maintains the original intent from the user request
3. Uses appropriate room sizes and relationships
4. Ensures no room overlaps
5. Arranges rooms to share walls where appropriate

Return the corrected layout using the SAME JSON structure (rooms only, no doors/windows):

{
    "building_style": "...",
    "rooms": [
        {
            "room_type": "...",
            "dimensions": {"width": float, "length": float, "height": float},
            "position": {"x": float, "y": float, "z": 0}
        }
    ]
}"""
    
    return prompt 

async def incremental_floor_plan_additions(existing_layout_data: Dict[str, Any], rejected_rooms: List[str], input_text: str) -> Dict[str, Any]:
    """
    Use Claude API to add previously rejected rooms to an existing layout.
    
    Args:
        existing_layout_data: Dict containing successfully added rooms with their vertices
        rejected_rooms: List of room names that were previously rejected
        
    Returns:
        Dict containing the complete layout with existing + newly added rooms
    """
    
    
    # Generate visualization of existing layout
    debug_vis_dir = f"{SERVER_ROOT_DIR}/vis"
    os.makedirs(debug_vis_dir, exist_ok=True)
    existing_vis_path = os.path.join(debug_vis_dir, f"existing_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    await get_vlm_room_layout_2d_visualization(existing_layout_data, existing_vis_path)
    
    # Prepare existing rooms text description
    existing_rooms_text = ""
    if "rooms" in existing_layout_data and existing_layout_data["rooms"]:
        existing_rooms_text = "EXISTING ROOMS IN LAYOUT:\n"
        for i, room in enumerate(existing_layout_data["rooms"]):
            room_type = room.get("room_type", f"Room_{i+1}")
            vertices = room["vertices"]
            vertices_str = ", ".join([f"({v['x']:.1f},{v['y']:.1f})" for v in vertices])
            
            existing_rooms_text += f"- {room_type}: vertices [{vertices_str}], height {room.get('height', 2.7)}m\n"
    else:
        existing_rooms_text = "NO EXISTING ROOMS (starting fresh layout)\n"
    
    # Prepare rejected rooms list
    rejected_rooms_text = "ROOMS TO ADD:\n"
    if rejected_rooms:
        for room_name in rejected_rooms:
            rejected_rooms_text += f"- {room_name}\n"
    else:
        return existing_layout_data  # Nothing to add
    
    # Create prompt for Claude
    prompt = f"""You are an expert architect and interior designer. I have an EXISTING ROOM LAYOUT and need you to add SPECIFIC ADDITIONAL ROOMS without overlapping or creating isolated rooms.

ORIGINAL USER REQUEST:
"{input_text.strip()}"

CONTEXT:
This existing layout was generated as part of fulfilling the above user request. However, some rooms from the original request could not be added due to collision or isolation issues. Your task is to add these remaining rooms while respecting the existing layout and maintaining the original design intent.

STEP 1: IDENTIFY ROOMS TO ADD
{rejected_rooms_text}

STEP 2: ANALYZE ROOM REQUIREMENTS
Before placing these rooms, carefully analyze the ORIGINAL USER REQUEST above to understand:
- What are the specific functional requirements for each room you need to add?
- What are the size expectations (compact, medium, large, spacious) for each room type?
- What are the connectivity and adjacency requirements mentioned in the original request?
- How should these rooms relate to the existing rooms in terms of circulation and privacy?
- What are the architectural style and design principles that should guide the placement?

STEP 3: PLACEMENT CONSTRAINTS
🚨 CRITICAL LAYOUT CONSTRAINTS 🚨
- **ABSOLUTELY NO OVERLAPPING ROOMS**: New room vertices must NEVER intersect or overlap with ALL existing rooms
- **NO ISOLATED ROOMS**: Every new room must be adjacent to at least one existing room OR another new room for connectivity
- **PERFECT RECTANGLES**: Every new room must have exactly 4 vertices forming a rectangle with 90-degree corners
- **FLEXIBLE COORDINATE SPACE**: Rooms can be placed at ANY coordinate location including NEGATIVE coordinates (x < 0, y < 0) if needed for optimal layout design

{existing_rooms_text}

LAYOUT REQUIREMENTS:
- **FULFILL ORIGINAL REQUEST**: Ensure the added rooms serve the functions and relationships described in the original user request
- **VERTEX-BASED PLACEMENT**: Position new rooms so NO vertex falls inside ALL existing room boundaries
- **ADJACENCY REQUIRED**: Each new room must touch at least one existing room or another new room (sharing edges/corners)
- **RECTANGULAR ROOMS ONLY**: All rooms must be perfect rectangles with 90-degree corners
- **AVOID OVERLAPS**: Carefully check that new room boundaries don't intersect ALL existing rooms
- **MAINTAIN CONNECTIVITY**: Ensure all new rooms can be connected via doors to the existing layout
- **FUNCTIONAL RELATIONSHIPS**: Consider the room relationships and circulation patterns mentioned in the original request
- **COORDINATE FREEDOM**: Use ANY coordinate values including negative numbers (x < 0, y < 0) as needed - rooms are NOT required to be above (0,0)

BUILDING STYLE: {existing_layout_data.get('building_style', 'Modern functional architecture')}

STEP 4: ROOM PLACEMENT TASK
Following the analysis from Steps 1-3 above, add the specified rooms to the existing layout using rectangular vertices. Your placement must:

**FULFILL ORIGINAL REQUIREMENTS**:
- Implement the specific functional requirements you identified for each room type
- Use the appropriate room sizes (compact/medium/large/spacious) as indicated in the original request
- Respect the connectivity and adjacency relationships described in the original user request
- Support the circulation patterns and privacy levels mentioned in the original design

**MAINTAIN DESIGN COHERENCE**:
- Follow the architectural style and design principles from the original request
- Ensure proper spatial relationships and functional adjacencies
- Create logical circulation flow that integrates with existing rooms
- Position rooms to support the overall design concept and usage patterns

**TECHNICAL CONSTRAINTS**:
- No overlaps with existing rooms and ensure proper connectivity to avoid isolation
- Use perfect rectangular vertices for all new rooms
- Feel free to use negative coordinates (x < 0, y < 0) if they create better spatial arrangements

Respond with your thinking process and then a valid JSON object for the NEW ROOMS using this exact structure:

```json
{{
    "building_style": "description of architectural style",
    "rooms": [
        {{
            "room_type": "specific room type from the rooms to add list",
            "vertices": [
                {{"x": float, "y": float}},
                {{"x": float, "y": float}},
                {{"x": float, "y": float}},
                {{"x": float, "y": float}}
            ],
            "height": float
        }}
    ]
}}
```

CRITICAL VERIFICATION STEPS:
1. **VERIFY ANALYSIS**: Confirm you have identified the specific requirements for each room from the original user request
2. **VERIFY REQUIREMENTS FULFILLMENT**: Ensure each new room meets the functional, size, and adjacency requirements from the original request
3. **CHECK OVERLAPS**: Verify no new room vertex is inside any existing room boundary
4. **CHECK ADJACENCY**: Verify each new room touches at least one existing room or another new room  
5. **CHECK RECTANGLES**: Verify all rooms have exactly 4 vertices forming perfect rectangles
6. **CHECK CONNECTIVITY**: Ensure new rooms support the circulation patterns described in the original request

🚨 FINAL REMINDER - THESE ARE MANDATORY 🚨
- **NO OVERLAPPING ROOMS**: Check that no new room vertices are inside existing rooms and no boundaries intersect
- **NO ISOLATED ROOMS**: Verify every new room can connect to existing layout
- **PERFECT RECTANGLES**: Every room must have exactly 4 vertices forming a rectangle with 90-degree corners
- **COORDINATE FLEXIBILITY**: Remember you can use ANY coordinates including negative values - optimal placement is more important than staying above (0,0)

Generate ONLY the new rooms that need to be added - do not repeat existing rooms."""

    try:
        # Read and encode the visualization image
        with open(existing_vis_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Make API call to Claude with image and text
        response = call_vlm(
            vlm_type="qwen",
            model="claude-sonnet-4-20250514",
            # max_tokens=10000,
            # temperature=1.0,
            # thinking={
            #     "type": "enabled",
            #     "budget_tokens": 6000
            # },
            max_tokens=4000,
            temperature=0.05,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        }
                    ]
                }
            ]
        )
        
        # Extract response content
        if not response or not hasattr(response, 'content') or not response.content:
            raise ValueError("No response received from Claude API")
        
        response_text = None
        for content in response.content:
            if hasattr(content, 'text'):
                response_text = content.text
                break
        
        if not response_text:
            raise ValueError("Claude API response content has no text")
        
        # Parse JSON response
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError("Could not extract JSON content from Claude response")
        
        new_rooms_data = json.loads(response_text)
        
        # Combine existing rooms with newly added rooms
        combined_layout = {
            "building_style": existing_layout_data.get("building_style", new_rooms_data.get("building_style", "Combined Layout")),
            "rooms": []
        }
        
        # Add existing rooms
        if "rooms" in existing_layout_data:
            combined_layout["rooms"].extend(existing_layout_data["rooms"])
        
        # Add new rooms
        if "rooms" in new_rooms_data:
            combined_layout["rooms"].extend(new_rooms_data["rooms"])
        
        print(f"✅ Successfully added {len(new_rooms_data.get('rooms', []))} new rooms to existing layout", file=sys.stderr)
        print(f"📊 Total rooms in combined layout: {len(combined_layout['rooms'])}", file=sys.stderr)
        
        return combined_layout
        
    except Exception as e:
        print(f"❌ Error in incremental floor plan additions: {e}", file=sys.stderr)
        # Return original layout if addition fails
        return existing_layout_data
    finally:
        # Clean up temporary visualization file
        try:
            if os.path.exists(existing_vis_path):
                os.remove(existing_vis_path)
        except:
            pass

async def test_incremental_room_collisions_isolations(layout_data, stop_on_first_rejection=False):
    """
    Test layout for room collisions and isolations by incrementally adding rooms.
    
    Args:
        layout_data: Dict containing 'rooms' list with vertex-based room definitions
        stop_on_first_rejection: If True, stop processing immediately when first problematic room is found
                               and mark all remaining rooms as rejected. If False, continue processing all rooms.
        
    Returns:
        Dict containing:
        - 'added_rooms': List of rooms that were successfully added
        - 'rejected_rooms': List of room names that were rejected due to collision/isolation
        - 'rejection_reason': Reason for stopping ('collision', 'isolation', or 'completed')
        - 'rejected_room_name': Name of the first rejected room (if any)
    """
    
    def get_room_bounds(room):
        """Extract min/max coordinates from room vertices."""
        if "vertices" in room:
            vertices = room["vertices"]
            x_coords = [v["x"] for v in vertices]
            y_coords = [v["y"] for v in vertices]
        elif "position" in room and "dimensions" in room:
            pos = room["position"]
            dim = room["dimensions"]
            x_coords = [pos["x"], pos["x"] + dim["width"]]
            y_coords = [pos["y"], pos["y"] + dim["length"]]
        else:
            raise ValueError(f"Room {room.get('room_type', 'unknown')} has invalid format")
            
        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }
    
    def rectangles_overlap(bounds1, bounds2):
        """Check if two rectangles overlap (have interior intersection)."""
        return (bounds1['min_x'] < bounds2['max_x'] and 
                bounds1['max_x'] > bounds2['min_x'] and
                bounds1['min_y'] < bounds2['max_y'] and 
                bounds1['max_y'] > bounds2['min_y'])
    
    def rectangles_touch(bounds1, bounds2):
        """Check if two rectangles touch (share edge or corner)."""
        # First check if they're completely separate
        if (bounds1['max_x'] < bounds2['min_x'] or bounds2['max_x'] < bounds1['min_x'] or
            bounds1['max_y'] < bounds2['min_y'] or bounds2['max_y'] < bounds1['min_y']):
            return False
            
        # They touch if they don't overlap but their bounding boxes intersect
        # This includes sharing edges or corners
        return True
    
    def room_is_isolated(new_room_bounds, existing_room_bounds_list):
        """Check if a room is isolated (doesn't touch any existing room)."""
        if not existing_room_bounds_list:  # First room is never isolated
            return False
            
        for existing_bounds in existing_room_bounds_list:
            if rectangles_touch(new_room_bounds, existing_bounds):
                return False
        return True
    
    def room_has_collision(new_room_bounds, existing_room_bounds_list):
        """Check if a room collides (overlaps) with any existing room."""
        for existing_bounds in existing_room_bounds_list:
            if rectangles_overlap(new_room_bounds, existing_bounds):
                return True
        return False
    
    # Initialize results
    added_rooms = []
    rejected_rooms = []
    rejected_room_details = []  # Track specific rooms and their issues
    existing_room_bounds = []
    first_rejection_reason = "completed"
    first_rejected_room_name = None
    
    if "rooms" not in layout_data or not layout_data["rooms"]:
        return {
            'added_rooms': [],
            'rejected_rooms': [],
            'rejection_reason': 'no_rooms',
            'rejected_room_name': None
        }
    
    # Process each room incrementally - continue even when problems are found
    for i, room in enumerate(layout_data["rooms"]):
        room_name = room.get("room_type", f"Room_{i+1}")
        
        try:
            # Get bounds for this room
            room_bounds = get_room_bounds(room)
            room_has_problem = False
            problem_reason = ""
            
            # Check for collision with existing (successfully added) rooms
            if room_has_collision(room_bounds, existing_room_bounds):
                room_has_problem = True
                problem_reason = "collision"
                if not first_rejected_room_name:  # Record first problem for summary
                    first_rejection_reason = "collision"
                    first_rejected_room_name = room_name
            
            # Check for isolation (only after first room and only if no collision)
            elif room_is_isolated(room_bounds, existing_room_bounds):
                room_has_problem = True
                problem_reason = "isolation"
                if not first_rejected_room_name:  # Record first problem for summary
                    first_rejection_reason = "isolation"
                    first_rejected_room_name = room_name
            
            # Handle the room based on whether it has problems
            if room_has_problem:
                # Mark this specific room as rejected
                rejected_rooms.append(room_name)
                rejected_room_details.append({
                    'room_name': room_name,
                    'reason': problem_reason,
                    'index': i
                })
                
                if stop_on_first_rejection:
                    # Stop immediately and mark all remaining rooms as rejected
                    print(f"🛑 Room '{room_name}' rejected due to {problem_reason}, stopping and rejecting all remaining rooms...", file=sys.stderr)
                    
                    # Add all remaining unprocessed rooms to rejected list
                    for j in range(i + 1, len(layout_data["rooms"])):
                        remaining_room = layout_data["rooms"][j]
                        remaining_room_name = remaining_room.get("room_type", f"Room_{j+1}")
                        rejected_rooms.append(remaining_room_name)
                        rejected_room_details.append({
                            'room_name': remaining_room_name,
                            'reason': 'unprocessed_due_to_early_stop',
                            'index': j
                        })
                    
                    # Break out of the processing loop
                    break
                else:
                    # Continue processing with existing behavior
                    print(f"⚠️  Room '{room_name}' rejected due to {problem_reason}, continuing with next room...", file=sys.stderr)
            else:
                # Room passed all checks - add it to successful rooms
                added_rooms.append(room)
                existing_room_bounds.append(room_bounds)
            
        except Exception as e:
            # Handle errors
            problem_reason = f"error: {str(e)}"
            rejected_rooms.append(room_name)
            rejected_room_details.append({
                'room_name': room_name,
                'reason': problem_reason,
                'index': i
            })
            if not first_rejected_room_name:  # Record first problem for summary
                first_rejection_reason = problem_reason
                first_rejected_room_name = room_name
            
            if stop_on_first_rejection:
                # Stop immediately and mark all remaining rooms as rejected
                print(f"🛑 Room '{room_name}' rejected due to error: {e}, stopping and rejecting all remaining rooms...", file=sys.stderr)
                
                # Add all remaining unprocessed rooms to rejected list
                for j in range(i + 1, len(layout_data["rooms"])):
                    remaining_room = layout_data["rooms"][j]
                    remaining_room_name = remaining_room.get("room_type", f"Room_{j+1}")
                    rejected_rooms.append(remaining_room_name)
                    rejected_room_details.append({
                        'room_name': remaining_room_name,
                        'reason': 'unprocessed_due_to_early_stop',
                        'index': j
                    })
                
                # Break out of the processing loop
                break
            else:
                # Continue processing with existing behavior
                print(f"❌ Room '{room_name}' rejected due to error: {e}, continuing with next room...", file=sys.stderr)
    
    return {
        'added_rooms': added_rooms,
        'rejected_rooms': rejected_rooms,
        'rejected_room_details': rejected_room_details,
        'rejection_reason': first_rejection_reason,
        'rejected_room_name': first_rejected_room_name,
        'total_rooms': len(layout_data["rooms"]),
        'added_count': len(added_rooms),
        'rejected_count': len(rejected_rooms)
    }

async def call_llm_for_rooms_only_solver(input_text: str) -> Dict[str, Any]:
    """
    Call Claude Sonnet 4 to generate room shapes and adjacency graph, then use RectangleContactSolver 
    to generate the final layout. Same input/output format as call_llm_for_rooms_only.
    """
    
    # Construct the prompt for Claude to generate shapes and adjacency graph
    prompt = f"""You are an expert architect and interior designer. Analyze this description and generate room specifications and adjacency relationships: "{input_text}"

🚨 CRITICAL REQUIREMENTS 🚨
- Generate ONLY room shapes (rectangles) and adjacency connections
- NO overlapping rooms, NO isolated rooms
- Every room must connect to at least one other room

STEP 1: ANALYZE REQUEST TYPE
First determine if the user is asking for:
- SINGLE ROOM: Only one specific room (e.g., "design a living room", "create a bedroom", "generate a kitchen")
- MULTIPLE ROOMS: A complete layout, house, apartment, or multiple connected rooms

STEP 2: GENERATE ROOM SPECIFICATIONS AND ADJACENCY GRAPH
For SINGLE ROOM requests:
- Generate just that one room with appropriate dimensions
- Use realistic proportions for the specific room type
- No adjacency graph needed (single room)

For MULTIPLE ROOMS requests:
- Create functional room specifications with realistic dimensions
- Generate adjacency relationships ensuring connectivity
- Consider architectural best practices and room relationships
- Ensure every room is connected to at least one other room

Respond with ONLY a valid JSON object using this exact structure:

```json
{{
    "building_style": "description of architectural style (e.g., Modern residential, Traditional colonial, etc.)",
    "shapes": [
        {{
            "room_type": "specific room type (e.g., living room, master bedroom, kitchen, etc.)",
            "width": float,
            "height": float
        }}
    ],
    "graph": {{
        "room_type_1": ["room_type_2", "room_type_3"],
        "room_type_2": ["room_type_1"],
        "room_type_3": ["room_type_1"]
    }}
}}
```

CRITICAL REQUIREMENTS:
- For SINGLE ROOM: Include exactly 1 room in "shapes" array, empty graph object {{}}
- For MULTIPLE ROOMS: Include all rooms in "shapes" array with proper adjacency in "graph"
- Use metric units (meters) for all dimensions
- Standard room dimensions: living room (4-6m x 5-7m), bedroom (3-4m x 4-5m), kitchen (3-4m x 3-5m), etc.
- Graph keys must exactly match room_type values in shapes array
- Ensure graph represents a connected structure (no isolated rooms)
- Room dimensions should be realistic for their intended function

Generate room specifications and adjacency graph that truly reflects the input description with appropriate room types, dimensions, and spatial relationships.

🚨 FINAL REMINDER 🚨
- **CONNECTED GRAPH**: Every room must be reachable from every other room
- **MATCHING NAMES**: Graph keys must exactly match room_type values in shapes
- **REALISTIC DIMENSIONS**: Use appropriate sizes for each room type"""

    try:
        # Make API call to Claude
        response = call_vlm(
            vlm_type="openai",
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            temperature=1.0,
            thinking=True,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Extract the response content with error checking
        if not response:
            raise ValueError("No response received from Claude API")
        
        if not hasattr(response, 'content') or not response.content:
            raise ValueError("Claude API response has no content")
        
        if len(response.content) == 0:
            raise ValueError("Claude API response content is empty")
        
        response_text = None
        for content in response.content:
            if hasattr(content, 'text'):
                response_text = content.text
                break
        if response_text is None:
            raise ValueError("Claude API response content has no text attribute")
        
        if not response_text or not response_text.strip():
            raise ValueError("Claude API returned empty response text")
        
        # Parse JSON response
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError("Could not extract JSON content from Claude response")
        
        solver_data = json.loads(response_text)
        
        # Validate the solver data structure
        if "shapes" not in solver_data or "graph" not in solver_data:
            raise ValueError("Invalid solver data: missing 'shapes' or 'graph' fields")
        
        shapes = solver_data["shapes"]
        graph_dict = solver_data["graph"]
        
        if not shapes:
            raise ValueError("No room shapes provided")
        
        # Handle single room case
        if len(shapes) == 1:
            room = shapes[0]
            # Create simple layout for single room
            layout_data = {
                "building_style": solver_data.get("building_style", "Modern residential"),
                "rooms": [
                    {
                        "room_type": room["room_type"],
                        "dimensions": {
                            "width": room["width"],
                            "length": room["height"],  # Note: height in shapes becomes length in output
                            "height": 2.7  # Standard ceiling height
                        },
                        "position": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0
                        }
                    }
                ]
            }
            return layout_data
        
        # For multiple rooms, use RectangleContactSolver
        
        # Create networkx graph from adjacency dict
        G = nx.Graph()
        
        # Add nodes (room indices)
        room_type_to_index = {room["room_type"]: i for i, room in enumerate(shapes)}
        index_to_room_type = {i: room["room_type"] for i, room in enumerate(shapes)}
        
        for i in range(len(shapes)):
            G.add_node(i)
        
        # Add edges from adjacency dict
        for room_type, adjacent_types in graph_dict.items():
            if room_type not in room_type_to_index:
                continue
            room_idx = room_type_to_index[room_type]
            for adj_type in adjacent_types:
                if adj_type in room_type_to_index:
                    adj_idx = room_type_to_index[adj_type]
                    G.add_edge(room_idx, adj_idx)
        
        # Validate graph connectivity
        if not nx.is_connected(G):
            raise ValueError("Generated graph is not connected - all rooms must be reachable")
        
        # Create rectangle specifications
        rectangles = {}
        for i, room in enumerate(shapes):
            rectangles[i] = RectangleSpec(w=room["width"], h=room["height"])
        
        # Solve using RectangleContactSolver
        solver = RectangleContactRelaxationSolver(G, rectangles)
        solved_graph, layout = solver.solve(max_attempts=50)
        
        # Validate the solution
        # if not solver._validate_solution(solved_graph, layout):
        #     raise ValueError("RectangleContactSolver failed to find a valid solution")
        
        # Convert solver output to the expected format
        converted_rooms = []
        for i, rect_layout in layout.items():
            room_type = index_to_room_type[i]
            converted_room = {
                "room_type": room_type,
                "dimensions": {
                    "width": rect_layout.w,
                    "length": rect_layout.h,
                    "height": 2.7  # Standard ceiling height
                },
                "position": {
                    "x": rect_layout.x - rect_layout.w / 2,  # Convert from center to corner
                    "y": rect_layout.y - rect_layout.h / 2,  # Convert from center to corner
                    "z": 0.0
                }
            }
            converted_rooms.append(converted_room)
        
        # Create final layout data in the expected format
        layout_data = {
            "building_style": solver_data.get("building_style", "Modern residential"),
            "rooms": converted_rooms
        }
        
        # Debug: create visualization if needed
        debug_vis_dir = f"{SERVER_ROOT_DIR}/vis"
        os.makedirs(debug_vis_dir, exist_ok=True)
        debug_vis_path = os.path.join(debug_vis_dir, f"solver_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        await get_vlm_room_layout_2d_visualization(layout_data, debug_vis_path)
        print(f"🔍 Debug: Solver layout visualization saved to {debug_vis_path}", file=sys.stderr)
        
        return layout_data
                
    except anthropic.APIError as e:
        raise ValueError(f"Anthropic API error: {e}")
    except Exception as e:
        raise ValueError(f"Error in solver-based room generation: {e}")

async def get_doors_windows_layout_visualization(layout_result: Dict[str, Any], save_path: str, mst, entry_room_index: int) -> None:
    """
    Create a comprehensive visualization of the room layout with doors, windows, and MST connectivity.
    Similar style to get_vlm_room_layout_2d_visualization but with door/window details.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available, skipping doors/windows visualization", file=sys.stderr)
        return
    
    # Create figure with subplots for main layout and legend
    fig = plt.figure(figsize=(16, 12))
    ax_main = plt.subplot(1, 2, (1, 1))
    ax_legend = plt.subplot(1, 2, 2)
    
    # Check if we have rooms
    if "rooms" not in layout_result or not layout_result["rooms"]:
        ax_main.text(0.5, 0.5, "No rooms found", ha='center', va='center', transform=ax_main.transAxes, fontsize=16)
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Colors for different room types - more distinct and vibrant
    room_colors = ['#FFB6C1', '#98FB98', '#FFB347', '#87CEEB', '#DDA0DD', 
                   '#F0E68C', '#FF6347', '#40E0D0', '#FFE4B5', '#98FB98',
                   '#FFA07A', '#20B2AA', '#F5DEB3', '#FFB6C1', '#87CEFA',
                   '#DDA0DD', '#F0E68C', '#FF7F50', '#40E0D0', '#FFFFE0']
    
    # Special colors for doors and windows
    door_color = '#8B4513'  # Brown for doors
    window_color = '#87CEEB'  # Sky blue for windows
    entry_door_color = '#FF4500'  # Orange red for entry door
    mst_edge_color = '#FF6347'  # Tomato for MST connections
    
    all_x = []
    all_y = []
    room_centers = {}
    legend_elements = []
    
    # Plot each room
    rooms_data = layout_result["rooms"]
    for i, room in enumerate(rooms_data):
        room_type = room.get("room_type", f"Room {i+1}")
        
        # Use position + dimensions format
        pos = room["position"]
        dim = room["dimensions"]
        
        x = pos["x"]
        y = pos["y"]
        width = dim["width"]
        length = dim["length"]
        
        # Create rectangle coordinates (counter-clockwise from bottom-left)
        x_coords = [x, x, x + width, x + width]
        y_coords = [y, y + length, y + length, y]
        
        # Store room center for MST visualization
        center_x = x + width / 2
        center_y = y + length / 2
        room_centers[i] = (center_x, center_y)
        
        # Add to bounds calculation
        all_x.extend(x_coords)
        all_y.extend(y_coords)
        
        # Create and add room polygon
        polygon_coords = list(zip(x_coords, y_coords))
        color = room_colors[i % len(room_colors)]
        
        # Highlight entry room
        edge_color = 'red' if i == entry_room_index else 'black'
        edge_width = 3 if i == entry_room_index else 2
        
        polygon = Polygon(polygon_coords, closed=True, facecolor=color, 
                         edgecolor=edge_color, linewidth=edge_width, alpha=0.8)
        ax_main.add_patch(polygon)
        
        # Add room label at center
        display_name = room_type.replace('_', ' ').title()
        entry_marker = " ⭐ [ENTRY]" if i == entry_room_index else ""
        
        ax_main.text(center_x, center_y, f"{display_name}{entry_marker}", ha='center', va='center', 
                    fontsize=9, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, 
                             edgecolor=edge_color, linewidth=1))
        
        # Add room index for reference
        ax_main.text(x + 0.1, y + length - 0.1, f"#{i}", ha='left', va='top', 
                    fontsize=8, fontweight='bold', color='darkblue',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
        
        # Add to legend
        legend_label = f"{display_name} ({i})"
        if i == entry_room_index:
            legend_label += " ⭐"
        legend_elements.append(patches.Patch(color=color, label=legend_label))
    
    # Draw MST connections
    if mst and mst.edges():
        for edge in mst.edges():
            room1_idx, room2_idx = edge
            if room1_idx in room_centers and room2_idx in room_centers:
                x1, y1 = room_centers[room1_idx]
                x2, y2 = room_centers[room2_idx]
                
                # Draw MST edge as thick red dashed line
                ax_main.plot([x1, x2], [y1, y2], color=mst_edge_color, linewidth=3, 
                           linestyle='--', alpha=0.8, zorder=5)
                
                # Add MST label at midpoint
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                # ax_main.text(mid_x, mid_y, 'MST', ha='center', va='center', 
                #            fontsize=7, fontweight='bold', color='white',
                #            bbox=dict(boxstyle="round,pad=0.2", facecolor=mst_edge_color, alpha=0.9))
    
    # Draw doors and windows
    door_count = 0
    window_count = 0
    
    for i, room in enumerate(rooms_data):
        pos = room["position"]
        dim = room["dimensions"]
        
        x = pos["x"]
        y = pos["y"]
        width = dim["width"]
        length = dim["length"]
        
        # Draw doors
        if "doors" in room:
            room_color = room_colors[i % len(room_colors)]
            for door in room["doors"]:
                door_count += 1
                wall_side = door["wall_side"]
                position = door["position_on_wall"]
                door_width = door["width"]
                door_height = door.get("height", 2.0)
                door_type = door.get("door_type", "interior")
                
                # Calculate door position based on wall side
                # Draw door WITHIN the room rectangle, using room color
                door_thickness = 0.15  # Thickness of door representation
                
                if wall_side == "north":  # Top wall
                    door_x = x + position * width - door_width / 2
                    door_y = y + length - door_thickness  # Inside the room
                    door_rect_width = door_width
                    door_rect_height = door_thickness
                    # Door center for arrow
                    door_center_x = door_x + door_width / 2
                    door_center_y = door_y + door_thickness / 2
                    # Arrow direction (pointing into room center)
                    arrow_dx = 0
                    arrow_dy = -0.3  # Point toward room center
                elif wall_side == "south":  # Bottom wall
                    door_x = x + position * width - door_width / 2
                    door_y = y  # Inside the room
                    door_rect_width = door_width
                    door_rect_height = door_thickness
                    # Door center for arrow
                    door_center_x = door_x + door_width / 2
                    door_center_y = door_y + door_thickness / 2
                    # Arrow direction (pointing into room center)
                    arrow_dx = 0
                    arrow_dy = 0.3  # Point toward room center
                elif wall_side == "east":  # Right wall
                    door_x = x + width - door_thickness  # Inside the room
                    door_y = y + position * length - door_width / 2
                    door_rect_width = door_thickness
                    door_rect_height = door_width
                    # Door center for arrow
                    door_center_x = door_x + door_thickness / 2
                    door_center_y = door_y + door_width / 2
                    # Arrow direction (pointing into room center)
                    arrow_dx = -0.3  # Point toward room center
                    arrow_dy = 0
                else:  # west - Left wall
                    door_x = x  # Inside the room
                    door_y = y + position * length - door_width / 2
                    door_rect_width = door_thickness
                    door_rect_height = door_width
                    # Door center for arrow
                    door_center_x = door_x + door_thickness / 2
                    door_center_y = door_y + door_width / 2
                    # Arrow direction (pointing into room center)
                    arrow_dx = 0.3  # Point toward room center
                    arrow_dy = 0
                
                # Draw door rectangle using room color but darker
                door_color_final = room_color if door_type not in ["entry", "main", "front"] else entry_door_color
                
                door_rect = Rectangle((door_x, door_y), door_rect_width, door_rect_height,
                                    facecolor=door_color_final, edgecolor='black', linewidth=2, alpha=0.9)
                ax_main.add_patch(door_rect)
                
                # Draw direction arrow showing door direction
                ax_main.arrow(door_center_x, door_center_y, arrow_dx, arrow_dy,
                             head_width=0.08, head_length=0.08, fc='black', ec='black', linewidth=2)
                
                # Add door label
                label_x = door_center_x
                label_y = door_center_y
                door_symbol = "🚪E" if door_type in ["entry", "main", "front"] else "🚪"
                ax_main.text(label_x, label_y, door_symbol, ha='center', va='center', 
                           fontsize=8, fontweight='bold', 
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
        
        # Draw windows
        if "windows" in room:
            for window in room["windows"]:
                window_count += 1
                wall_side = window["wall_side"]
                position = window["position_on_wall"]
                window_width = window["width"]
                window_height = window.get("height", 1.5)
                
                # Calculate window position based on wall side
                if wall_side == "north":  # Top wall
                    win_x = x + position * width - window_width / 2
                    win_y = y + length - 0.15
                    win_rect_width = window_width
                    win_rect_height = 0.3
                elif wall_side == "south":  # Bottom wall
                    win_x = x + position * width - window_width / 2
                    win_y = y - 0.15
                    win_rect_width = window_width
                    win_rect_height = 0.3
                elif wall_side == "east":  # Right wall
                    win_x = x + width - 0.15
                    win_y = y + position * length - window_width / 2
                    win_rect_width = 0.3
                    win_rect_height = window_width
                else:  # west - Left wall
                    win_x = x - 0.15
                    win_y = y + position * length - window_width / 2
                    win_rect_width = 0.3
                    win_rect_height = window_width
                
                # Draw window rectangle
                window_rect = Rectangle((win_x, win_y), win_rect_width, win_rect_height,
                                      facecolor=window_color, edgecolor='navy', linewidth=1, alpha=0.9)
                ax_main.add_patch(window_rect)
                
                # Add window label
                label_x = win_x + win_rect_width / 2
                label_y = win_y + win_rect_height / 2
                ax_main.text(label_x, label_y, "🪟", ha='center', va='center', 
                           fontsize=6, fontweight='bold')
    
    # Set axis properties for main plot
    if all_x and all_y:
        margin = max(1.0, (max(all_x) - min(all_x)) * 0.15)
        ax_main.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax_main.set_ylim(min(all_y) - margin, max(all_y) + margin)
    else:
        ax_main.set_xlim(-1, 10)
        ax_main.set_ylim(-1, 10)
    
    # Style main plot
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.set_xlabel('X Coordinate (meters)', fontsize=12)
    ax_main.set_ylabel('Y Coordinate (meters)', fontsize=12)
    ax_main.set_title(f'Graph-Based Layout: Doors & Windows\nEntry Room: #{entry_room_index} | Doors: {door_count} | Windows: {window_count}', 
                     fontsize=14, fontweight='bold', pad=20)
    
    # Add door and window legend elements
    legend_elements.append(patches.Patch(color=entry_door_color, label='🚪 Entry Door'))
    legend_elements.append(patches.Patch(color='gray', label='🚪 Interior Door (room color)'))
    legend_elements.append(patches.Patch(color=window_color, label='🪟 Window'))
    legend_elements.append(patches.Patch(color=mst_edge_color, label='MST Connection'))
    legend_elements.append(patches.Patch(color='black', label='➡️ Door Direction'))
    
    # Create comprehensive legend in the right subplot
    ax_legend.axis('off')
    
    # Split legend into sections
    room_legend_elements = [elem for elem in legend_elements if '🚪' not in elem.get_label() and '🪟' not in elem.get_label() and 'MST' not in elem.get_label()]
    feature_legend_elements = [elem for elem in legend_elements if '🚪' in elem.get_label() or '🪟' in elem.get_label() or 'MST' in elem.get_label()]
    
    # Rooms section
    if room_legend_elements:
        legend1 = ax_legend.legend(handles=room_legend_elements, loc='upper left', 
                                  title="🏠 Rooms", title_fontsize=12, fontsize=10)
        legend1.get_frame().set_facecolor('lightblue')
        legend1.get_frame().set_alpha(0.8)
    
    # Features section  
    if feature_legend_elements:
        legend2 = ax_legend.legend(handles=feature_legend_elements, loc='center left',
                                  title="🔧 Features", title_fontsize=12, fontsize=10)
        legend2.get_frame().set_facecolor('lightgreen')
        legend2.get_frame().set_alpha(0.8)
    
    # Add statistics box
    stats_text = f"""📊 Layout Statistics:
• Total Rooms: {len(rooms_data)}
• Entry Room: #{entry_room_index}
• Total Doors: {door_count}
• Total Windows: {window_count}
• MST Edges: {len(mst.edges()) if mst else 0}
• Graph-based Algorithm: ✅

🎨 Visualization Features:
• Doors shown within rooms
• Door directions with arrows
• Distinct room colors
• Entry room highlighted"""
    
    ax_legend.text(0.1, 0.3, stats_text, fontsize=10, 
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9),
                  verticalalignment='top')
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.02, 0.02, f'Generated: {timestamp}', fontsize=8, alpha=0.6)
    
    # Save the plot
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✅ Doors & windows layout visualization saved successfully to {save_path}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error saving doors/windows visualization: {e}", file=sys.stderr)
    finally:
        plt.close()

async def call_llm_for_doors_windows_graph_based(rooms_data: List[Dict[str, Any]], input_text: str) -> Dict[str, Any]:
    """
    Graph-based door and window placement using minimum spanning tree approach.
    
    This function:
    1. Finds all shared walls (for connected room doors) and exterior walls (for windows and entry door)
    2. Builds a graph where each vertex is a room, edges exist for rooms with shared walls
    3. Uses LLM to identify the entry room from input text and room types
    4. Calculates MST with entry room as root node
    5. Selects door/window placements based on MST and rules
    6. Uses LLM to generate detailed specifications
    7. Returns in same format as original function
    
    Args:
        rooms_data: List of room data dictionaries
        input_text: Original user request text
        
    Returns:
        Dictionary with same format as call_llm_for_doors_windows
    """
    
    # Initialize debug information collection
    debug_info = {
        "prompts": [],
        "responses": [],
        "steps": []
    }
    
    # Import required functions
    from utils import find_all_shared_walls, generate_floor_plan_visualization, calculate_room_wall_position_from_shared_wall
    
    # Step 1: Find all shared walls and exterior walls
    debug_info["steps"].append("Step 1: Finding shared walls and exterior walls")
    shared_walls_info = find_all_shared_walls(rooms_data)
    
    # Step 2: Build graph where vertices are rooms and edges exist for rooms with shared walls
    debug_info["steps"].append("Step 2: Building room connectivity graph")
    room_graph = nx.Graph()
    
    # Add all rooms as vertices
    for i, room in enumerate(rooms_data):
        room_graph.add_node(i, room_type=room["room_type"])
    
    # Add edges for rooms with shared walls
    for shared_wall in shared_walls_info["room_room_walls"]:
        room1_idx = shared_wall["room1"]["index"]
        room2_idx = shared_wall["room2"]["index"]
        # Edge weight could be shared wall length (shorter walls = higher weight for MST)
        weight = 1.0 / max(shared_wall["overlap_length"], 0.1)  # Inverse length as weight
        room_graph.add_edge(room1_idx, room2_idx, weight=weight, shared_wall_info=shared_wall)
    
    # Step 3: Identify entry room (skip LLM call for single room)
    debug_info["steps"].append("Step 3: Identifying entry room")
    
    if len(rooms_data) == 1:
        # Single room: obvious choice
        entry_room_index = 0
        debug_info["steps"].append("Step 3: Single room - entry room is room 0")
        print(f"🏠 Single room layout: entry room automatically set to room 0", file=sys.stderr)
    else:
        # Multiple rooms: use LLM to identify entry room
        debug_info["steps"].append("Step 3: Multiple rooms - using LLM to identify entry room")
        
        # Prepare room types summary for LLM
        room_types_summary = []
        for i, room in enumerate(rooms_data):
            room_types_summary.append({
                "room_index": i,
                "room_type": room["room_type"],
                "area": f"{room['dimensions']['width'] * room['dimensions']['length']:.1f} sq m"
            })
        
        entry_room_prompt = f"""You are an expert architect analyzing a floor plan to determine the entry room.

ORIGINAL REQUEST: "{input_text}"

AVAILABLE ROOMS:
{json.dumps(room_types_summary, indent=2)}

TASK: Based on the original request and the room types, determine which room should have the main entrance door.

Consider:
- Typical entry patterns (living room, foyer, hallway for residential; lobby, reception for commercial)
- The original request context and building type
- Logical flow for visitors entering the space

Return your response as JSON:
```json
{{
    "entry_room_index": int,
    "reasoning": "Brief explanation of why this room should be the entry point"
}}
```
"""

        try:
            debug_info["prompts"].append({
                "step": "3_entry_room_identification",
                "prompt": entry_room_prompt
            })
            
            entry_response = call_vlm(
                vlm_type="qwen",
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.2,
                messages=[
                    {
                        "role": "user",
                        "content": entry_room_prompt
                    }
                ]
            )
            
            entry_response_text = entry_response.content[0].text.strip()
            debug_info["responses"].append({
                "step": "3_entry_room_identification",
                "response": entry_response_text
            })
            
            # Parse entry room response
            entry_data = extract_json_from_response(entry_response_text)
            if not entry_data:
                raise ValueError("Could not extract JSON content from Claude response")
            entry_data = json.loads(entry_data)
            
            entry_room_index = entry_data["entry_room_index"]
            
            # Validate entry room index
            if not (0 <= entry_room_index < len(rooms_data)):
                entry_room_index = 0  # Fallback to first room
                
        except Exception as e:
            # Fallback: use the first room or largest room as entry
            entry_room_index = 0
            if len(rooms_data) > 1:
                # Choose largest room as entry point fallback
                largest_area = 0
                for i, room in enumerate(rooms_data):
                    area = room['dimensions']['width'] * room['dimensions']['length']
                    if area > largest_area:
                        largest_area = area
                        entry_room_index = i
            debug_info["steps"].append(f"Step 3 failed: {str(e)}, using fallback entry room {entry_room_index}")
    
    # Step 4: Calculate MST with entry room as root
    debug_info["steps"].append("Step 4: Calculating minimum spanning tree")
    
    # Check if graph is connected, if not, we'll handle disconnected components
    if nx.is_connected(room_graph):
        # Calculate MST
        mst = nx.minimum_spanning_tree(room_graph, weight='weight')
    else:
        # Handle disconnected graph - create MST for each component and add arbitrary connections
        components = list(nx.connected_components(room_graph))
        mst = nx.Graph()
        
        # Add MST for each component
        for component in components:
            subgraph = room_graph.subgraph(component)
            if len(component) > 1:
                component_mst = nx.minimum_spanning_tree(subgraph, weight='weight')
                mst = nx.union(mst, component_mst)
            else:
                mst.add_node(list(component)[0])
        
        # Connect components - add edges between closest rooms in different components
        component_list = [list(comp) for comp in components]
        for i in range(len(component_list) - 1):
            # Find shortest connection between component i and component i+1
            min_distance = float('inf')
            best_edge = None
            
            for room1 in component_list[i]:
                for room2 in component_list[i + 1]:
                    # Calculate distance between room centers
                    pos1 = rooms_data[room1]["position"]
                    pos2 = rooms_data[room2]["position"]
                    distance = ((pos1["x"] - pos2["x"]) ** 2 + (pos1["y"] - pos2["y"]) ** 2) ** 0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_edge = (room1, room2)
            
            if best_edge:
                mst.add_edge(best_edge[0], best_edge[1], weight=min_distance, artificial=True)
    
    # Step 5: Select door and window placements based on MST and rules
    debug_info["steps"].append("Step 5: Selecting door and window placements")
    
    # Handle single room case
    if len(rooms_data) == 1:
        debug_info["steps"].append("Step 5: Single room detected - no connecting doors needed")
        print(f"🏠 Single room layout detected - skipping connecting door placement", file=sys.stderr)
    
    # Identify leaf nodes (rooms that need windows)
    # Note: In single room case, degree is 0, not 1, so we handle it separately
    if len(rooms_data) == 1:
        # Single room is effectively a leaf node for window placement
        leaf_nodes = [0]
    else:
        leaf_nodes = [node for node in mst.nodes() if mst.degree(node) == 1]
    
    # Ensure entry room has at least one window if it's not a leaf
    if entry_room_index not in leaf_nodes and mst.degree(entry_room_index) > 1:
        leaf_nodes.append(entry_room_index)
    
    # Select MST edges for doors (these are required connections)
    required_doors = []
    for edge in mst.edges():
        room1, room2 = edge
        # Find the shared wall for this connection
        for i, shared_wall in enumerate(shared_walls_info["room_room_walls"]):
            if ((shared_wall["room1"]["index"] == room1 and shared_wall["room2"]["index"] == room2) or 
                (shared_wall["room1"]["index"] == room2 and shared_wall["room2"]["index"] == room1)):
                required_doors.append({
                    "shared_wall_index": i,
                    "wall_size": shared_wall["overlap_length"],
                    "adjacent_room_types": [shared_wall["room1"]["room_type"], shared_wall["room2"]["room_type"]]
                })
                break
    
    # Randomly select additional doors on non-MST shared walls (max 30% of remaining walls)
    remaining_shared_walls = []
    for i, shared_wall in enumerate(shared_walls_info["room_room_walls"]):
        room1 = shared_wall["room1"]["index"]
        room2 = shared_wall["room2"]["index"]
        # Check if this wall is already in MST
        is_mst_edge = any(
            ((door["shared_wall_index"] == i) for door in required_doors)
        )
        if not is_mst_edge:
            remaining_shared_walls.append({
                "shared_wall_index": i,
                "wall_size": shared_wall["overlap_length"],
                "adjacent_room_types": [shared_wall["room1"]["room_type"], shared_wall["room2"]["room_type"]]
            })
    
    # Randomly select up to 30% of remaining walls for additional doors
    # Handle single room case: if no remaining shared walls, don't add doors
    additional_doors = []
    if remaining_shared_walls:
        max_additional_doors = max(1, int(len(remaining_shared_walls) * 0.3))
        additional_doors = random.sample(remaining_shared_walls, min(max_additional_doors, len(remaining_shared_walls)))
    
    all_selected_doors = required_doors + additional_doors
    
    # Find entry door location FIRST (before window selection to avoid conflicts)
    entry_room_exterior_walls = [
        i for i, wall in enumerate(shared_walls_info["room_exterior_walls"])
        if wall["room"]["index"] == entry_room_index
    ]
    
    if entry_room_exterior_walls:
        # Select the best exterior wall for entry door
        entry_door_wall_idx = min(entry_room_exterior_walls,
                                 key=lambda i: shared_walls_info["room_exterior_walls"][i]["overlap_length"])
    else:
        # No exterior wall on entry room, pick the shortest exterior wall and reassign entry room
        if shared_walls_info["room_exterior_walls"]:
            entry_door_wall_idx = 0
            entry_room_index = shared_walls_info["room_exterior_walls"][entry_door_wall_idx]["room"]["index"]
        else:
            raise ValueError("No exterior walls found in the layout")
    
    entry_door_info = {
        "room_index": entry_room_index,
        "wall_index": entry_door_wall_idx
    }
    
    # Select windows for leaf nodes and randomly for other exterior walls
    # IMPORTANT: Exclude the wall that has the entry door to avoid conflicts
    required_windows = []
    
    # Ensure windows for leaf nodes
    for room_idx in leaf_nodes:
        room_exterior_walls = [
            i for i, wall in enumerate(shared_walls_info["room_exterior_walls"])
            if wall["room"]["index"] == room_idx and i != entry_door_wall_idx  # Exclude entry door wall
        ]
        if room_exterior_walls:
            # Select the largest exterior wall for this room (that doesn't have entry door)
            best_wall_idx = max(room_exterior_walls, 
                               key=lambda i: shared_walls_info["room_exterior_walls"][i]["overlap_length"])
            required_windows.append({
                "exterior_wall_index": best_wall_idx,
                "wall_size": shared_walls_info["room_exterior_walls"][best_wall_idx]["overlap_length"],
                "wall_room_type": shared_walls_info["room_exterior_walls"][best_wall_idx]["room"]["room_type"]
            })
    
    # Randomly select additional windows (max 40% of remaining exterior walls)
    # IMPORTANT: Exclude walls that already have windows OR the entry door
    remaining_exterior_walls = []
    for i, wall in enumerate(shared_walls_info["room_exterior_walls"]):
        is_already_selected = any(w["exterior_wall_index"] == i for w in required_windows)
        is_entry_door_wall = (i == entry_door_wall_idx)  # Exclude entry door wall
        if not is_already_selected and not is_entry_door_wall:
            remaining_exterior_walls.append({
                "exterior_wall_index": i,
                "wall_size": wall["overlap_length"],
                "wall_room_type": wall["room"]["room_type"]
            })
    
    # Randomly select additional windows (handle empty case like doors)
    additional_windows = []
    if remaining_exterior_walls:
        max_additional_windows = max(1, int(len(remaining_exterior_walls) * 0.4))
        additional_windows = random.sample(remaining_exterior_walls, min(max_additional_windows, len(remaining_exterior_walls)))
    
    all_selected_windows = required_windows + additional_windows

    print(f"🚪 Selected connecting doors: {len(all_selected_doors)}")
    print(f"🪟 Selected windows: {len(all_selected_windows)}")
    print(f"🚪 Entry door wall index: {entry_door_wall_idx} (excluded from window placement)")
    if len(rooms_data) == 1:
        print(f"🏠 Single room layout: only entry door and windows will be placed")
    
    # Step 6: Prompt LLM for detailed door and window specifications
    debug_info["steps"].append("Step 6: Getting detailed door and window specifications from LLM")
    
    # Generate floor plan visualization for LLM
    try:
        visualization_path = generate_floor_plan_visualization(rooms_data)
        
        # Read the image file
        with open(visualization_path, 'rb') as image_file:
            image_data = image_file.read()
        
        # Save debug copy
        debug_vis_dir = f"{SERVER_ROOT_DIR}/vis"
        os.makedirs(debug_vis_dir, exist_ok=True)
        timestamp = int(time.time())
        debug_vis_path = os.path.join(debug_vis_dir, f"floor_plan_graph_{timestamp}.png")
        
        with open(debug_vis_path, 'wb') as debug_file:
            debug_file.write(image_data)
        
        print(f"🔍 Debug: Graph-based floor plan visualization saved to {debug_vis_path}", file=sys.stderr)
        
        # Clean up the temporary file
        os.unlink(visualization_path)
        
    except Exception as e:
        raise ValueError(f"Failed to generate floor plan visualization: {str(e)}")
    
    # Prepare summary for detailed specification prompt
    connecting_doors_section = ""
    if len(all_selected_doors) > 0:
        connecting_doors_section = f"""
CONNECTING DOORS ({len(all_selected_doors)} total):
{json.dumps(all_selected_doors, indent=2)}"""
    else:
        connecting_doors_section = """
CONNECTING DOORS: None (single room layout - no connecting doors needed)"""

    detailed_prompt = f"""You are an expert architect providing detailed door and window specifications for a floor plan.

ORIGINAL REQUEST: "{input_text}"

SELECTED DOOR AND WINDOW PLACEMENTS:

ENTRY DOOR:
- Room Index: {entry_door_info["room_index"]} ({rooms_data[entry_door_info["room_index"]]["room_type"]})
- Exterior Wall Index: {entry_door_info["wall_index"]}
{connecting_doors_section}

WINDOWS ({len(all_selected_windows)} total):
{json.dumps(all_selected_windows, indent=2)}

ROOM LAYOUT:
{json.dumps([{
    "room_index": i,
    "room_type": room["room_type"],
    "dimensions": f"{room['dimensions']['width']}m × {room['dimensions']['length']}m × {room['dimensions']['height']}m"
} for i, room in enumerate(rooms_data)], indent=2)}

TASK: Provide detailed specifications for entry door{", each selected connecting door" if len(all_selected_doors) > 0 else ""} and window. Return EXACTLY this JSON format:

```json
{{
    "entry_door": {{
        "exterior_wall_index": {entry_door_wall_idx},
        "center_position": float_0_to_1 (Position along the exterior wall segment (0.0 = start, 1.0 = end), choose float number between 0.20 to 0.80 randomly),
        "width": float_meters,
        "height": float_meters,
        "door_type": "entry/main/front",
        "door_material": "Brief description of door style/material",
        "reasoning": "Why this location was chosen"
    }},{f'''
    "connecting_doors": [
        {{
            "shared_wall_index": int,
            "center_position_on_shared_wall": float_0_to_1 (Position along the shared wall segment (0.0 = start, 1.0 = end), choose float number between 0.20 to 0.80 randomly),
            "width": float_meters,
            "height": float_meters,
            "door_type": "interior/standard",
            "opening": bool,
            "door_material": "Brief description of door style/material",
            "reasoning": "Why this connection is needed"
        }}
    ],''' if len(all_selected_doors) > 0 else '''
    "connecting_doors": [],'''}
    "windows": [
        {{
            "exterior_wall_index": int,
            "center_position": float_0_to_1 (Position along the exterior wall segment (0.0 = start, 1.0 = end), choose float number between 0.20 to 0.80 randomly),
            "width": float_meters,
            "height": float_meters,
            "sill_height": float_meters,
            "window_type": "standard",
            "window_grid": [nx, ny],
            "window_appearance_description": "Detailed description combining frame style and glass type, including the color style of the frame and glass.",
            "glass_color": [r, g, b],
            "frame_color": [r, g, b],
            "reasoning": "Why this window placement was chosen"
        }}
    ]
}}
```

CONSTRAINTS:
- Entry door: width 0.8-1.2m, height 2.0-2.1m
{f"- Interior doors: width 0.7-0.9m, height 2.0-2.1m" if len(all_selected_doors) > 0 else ""}
{f"- Wide openings: width 1.2-3.0m, height 2.0-2.4m" if len(all_selected_doors) > 0 else ""}
- Windows: width 0.8-3.0m, height 0.8-2.0m, sill_height 0.3-1.2m
- Window grid: [nx, ny]  where nx=horizontal grid, ny=vertical grid
Examples:

"fixed": (1, 1),
"slider": (2, 1),  # 2 horizontal panes, 1 vertical
"hung": (1, 2),    # 1 horizontal pane, 2 vertical
"bay_small": (2, 2),
"bay_medium": (3, 2),
"bay_large": (3, 3),

- Window colors: Generate DISTINCT and CONTRASTING glass_color and frame_color as RGB values [0-255].
CRITICAL: Ensure high visual contrast between glass and frame colors - they must be easily distinguishable.
Colors should be consistent with the window appearance description.

  GLASS COLOR EXAMPLES (choose ONE type, ensure contrast with frame):
  * Clear Glass: [245, 250, 255] (bright clear), [250, 248, 240] (warm clear), [248, 252, 255] (cool clear)
  * Tinted Glass: [200, 230, 255] (light blue tint), [230, 245, 220] (light green tint), [255, 245, 220] (warm amber)
  * Privacy Glass: [220, 220, 220] (frosted white), [200, 210, 225] (frosted blue), [210, 200, 190] (frosted bronze)
  * Colored Glass: [150, 200, 255] (sky blue), [200, 255, 200] (mint green), [255, 220, 180] (warm amber)
  
  FRAME COLOR EXAMPLES (choose to CONTRAST with glass color):
  * Dark Frames: [40, 30, 20] (dark brown), [25, 25, 25] (charcoal black), [60, 40, 30] (dark walnut)
  * Light Frames: [250, 245, 235] (cream white), [240, 240, 240] (light gray), [245, 240, 225] (off-white)
  * Colored Frames: [120, 80, 60] (medium brown), [80, 100, 120] (blue-gray), [100, 120, 80] (sage green)
  * Metal Frames: [180, 180, 180] (brushed aluminum), [160, 140, 120] (bronze), [100, 100, 100] (gunmetal)

  CONTRAST GUIDELINES:
  * Light glass (245+) → Use dark frame (60 or below)
  * Dark/colored glass (200 or below) → Use light frame (200+) or contrasting color
  * Ensure at least 100+ difference in brightness between glass and frame RGB averages
- Position values are 0.0-1.0 (center of wall)
{f"- Use opening=true for wide openings, opening=false for doors with frames" if len(all_selected_doors) > 0 else ""}
"""

    try:
        debug_info["prompts"].append({
            "step": "6_detailed_specifications",
            "prompt": detailed_prompt
        })
        
        # Make API call to Claude for detailed specifications
        response = call_vlm(
            vlm_type="openai",
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
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
        
        # Extract the response content with error checking
        if not response or not hasattr(response, 'content') or not response.content:
            raise ValueError("No response received from Claude API")
        
        if len(response.content) == 0 or not hasattr(response.content[0], 'text'):
            raise ValueError("Claude API response content is empty or invalid")
        
        response_text = response.content[0].text
        
        debug_info["responses"].append({
            "step": "6_detailed_specifications",
            "response": response_text
        })
        
        if not response_text or not response_text.strip():
            raise ValueError("Claude API returned empty response text")
        
        # Parse Claude's JSON response
        try:
            claude_response_json = extract_json_from_response(response_text)
            if not claude_response_json:
                raise ValueError("Could not extract JSON content from Claude response")
            claude_response = json.loads(claude_response_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Claude's JSON response: {e}")
        
        debug_info["steps"].append("Step 6 completed: Detailed specifications received")
        
    except Exception as e:
        raise ValueError(f"Error getting detailed specifications: {e}")
    
    # Step 7: Format output to match original function format
    debug_info["steps"].append("Step 7: Formatting output to match original function interface")
    
    # Initialize result structure matching the original function
    result = {
        "rooms": []
    }
    
    # Initialize each room with empty doors and windows
    for i, room_data in enumerate(rooms_data):
        room_result = {
            "room_type": room_data["room_type"],
            "dimensions": room_data["dimensions"],
            "position": room_data["position"],
            "doors": [],
            "windows": []
        }
        result["rooms"].append(room_result)
    
    # Filter and correct LLM outputs to ensure consistency with selected doors/windows
    debug_info["steps"].append("Step 7.1: Filtering and correcting LLM outputs")
    
    # 1. Check and correct entry door
    entry_door_corrected = False
    if "entry_door" not in claude_response or not claude_response["entry_door"]:
        entry_door_corrected = True
        debug_info["steps"].append("Step 7.1a: Entry door missing from LLM output, creating one")
        print(f"Step 7.1a: Entry door missing from LLM output, creating one", file=sys.stderr)
    else:
        entry_door = claude_response["entry_door"]
        # Check if entry door is on the correct exterior wall
        entry_wall_idx = entry_door.get("exterior_wall_index", -1)
        
        # Verify this corresponds to our selected entry door wall
        if entry_wall_idx != entry_door_wall_idx:
            entry_door_corrected = True
            debug_info["steps"].append(f"Step 7.1a: Entry door misplaced (wall {entry_wall_idx}), correcting to wall {entry_door_wall_idx}")
            print(f"Step 7.1a: Entry door misplaced (wall {entry_wall_idx}), correcting to wall {entry_door_wall_idx}", file=sys.stderr)
    if entry_door_corrected:
        # Create entry door on our selected wall
        claude_response["entry_door"] = {
            "exterior_wall_index": entry_door_wall_idx,
            "center_position": float(np.random.uniform(0.2, 0.8)),  # Center of wall
            "width": 1.0,  # Standard entry door width
            "height": 2.1,  # Standard entry door height
            "door_type": "entry",
            "door_material": "solid wood entry door",
            "reasoning": "Auto-generated to match selected entry door location"
        }
        debug_info["steps"].append(f"Step 7.1a: Created entry door on wall {entry_door_wall_idx}")
        print(f"Step 7.1a: Created entry door on wall {entry_door_wall_idx}", file=sys.stderr)
    # 2. Check and correct connecting doors
    connecting_doors_corrected = 0
    if "connecting_doors" not in claude_response:
        claude_response["connecting_doors"] = []
    
    # Get LLM's connecting doors and check if they match our selections
    llm_door_walls = set()
    if claude_response["connecting_doors"]:
        for door in claude_response["connecting_doors"]:
            wall_idx = door.get("shared_wall_index", -1)
            if 0 <= wall_idx < len(shared_walls_info["room_room_walls"]):
                llm_door_walls.add(wall_idx)
    
    # Get our selected door walls
    selected_door_walls = set(door["shared_wall_index"] for door in all_selected_doors)
    
    # Find missing doors and add them
    missing_doors = selected_door_walls - llm_door_walls
    for wall_idx in missing_doors:
        if wall_idx < len(shared_walls_info["room_room_walls"]):
            shared_wall = shared_walls_info["room_room_walls"][wall_idx]
            claude_response["connecting_doors"].append({
                "shared_wall_index": wall_idx,
                "center_position_on_shared_wall": 0.5,  # Center of shared wall
                "width": 0.9,  # Standard interior door width
                "height": 2.1,  # Standard door height
                "door_type": "interior",
                "opening": False,  # Physical door
                "door_material": "wooden panel door",
                "reasoning": "Auto-generated to match MST connectivity requirements"
            })
            connecting_doors_corrected += 1
    
    if connecting_doors_corrected > 0:
        debug_info["steps"].append(f"Step 7.1b: Added {connecting_doors_corrected} missing connecting doors")
        print(f"Step 7.1b: Added {connecting_doors_corrected} missing connecting doors", file=sys.stderr)
    # Remove doors that are not in our selected list
    original_doors = claude_response["connecting_doors"][:]
    claude_response["connecting_doors"] = [
        door for door in original_doors 
        if door.get("shared_wall_index", -1) in selected_door_walls
    ]
    removed_doors = len(original_doors) - len(claude_response["connecting_doors"])
    if removed_doors > 0:
        debug_info["steps"].append(f"Step 7.1b: Removed {removed_doors} doors not in selected walls")
        print(f"Step 7.1b: Removed {removed_doors} doors not in selected walls", file=sys.stderr)
    # 3. Check and correct windows
    windows_corrected = 0
    if "windows" not in claude_response:
        claude_response["windows"] = []
    
    # Remove windows that are on invalid walls (entry door wall or non-exterior walls)
    original_windows = claude_response["windows"][:]
    claude_response["windows"] = []
    
    for window in original_windows:
        wall_idx = window.get("exterior_wall_index", -1)
        
        # Check if wall index is valid
        if not (0 <= wall_idx < len(shared_walls_info["room_exterior_walls"])):
            debug_info["steps"].append(f"Step 7.1c: Removed window on invalid wall index {wall_idx}")
            print(f"Step 7.1c: Removed window on invalid wall index {wall_idx}", file=sys.stderr)
            continue
        
        # Check if wall is the entry door wall
        if wall_idx == entry_door_wall_idx:
            debug_info["steps"].append(f"Step 7.1c: Removed window on entry door wall {wall_idx}")
            print(f"Step 7.1c: Removed window on entry door wall {wall_idx}", file=sys.stderr)
            continue
        
        # Window is valid, keep it
        claude_response["windows"].append(window)
    
    removed_windows = len(original_windows) - len(claude_response["windows"])
    if removed_windows > 0:
        debug_info["steps"].append(f"Step 7.1c: Removed {removed_windows} invalid windows")
        print(f"Step 7.1c: Removed {removed_windows} invalid windows", file=sys.stderr)
    
    # 4. Remove duplicate doors on same shared wall (keep only first occurrence)
    if claude_response["connecting_doors"]:
        seen_door_walls = set()
        unique_doors = []
        duplicates_removed = 0
        
        for door in claude_response["connecting_doors"]:
            wall_idx = door.get("shared_wall_index", -1)
            if wall_idx not in seen_door_walls:
                seen_door_walls.add(wall_idx)
                unique_doors.append(door)
            else:
                duplicates_removed += 1
        
        claude_response["connecting_doors"] = unique_doors
        if duplicates_removed > 0:
            debug_info["steps"].append(f"Step 7.1d: Removed {duplicates_removed} duplicate doors on same shared walls")
            print(f"Step 7.1d: Removed {duplicates_removed} duplicate doors on same shared walls", file=sys.stderr)
    # 5. Remove duplicate windows on same exterior wall (keep only first occurrence)
    if claude_response["windows"]:
        seen_window_walls = set()
        unique_windows = []
        duplicates_removed = 0
        
        for window in claude_response["windows"]:
            wall_idx = window.get("exterior_wall_index", -1)
            if wall_idx not in seen_window_walls:
                seen_window_walls.add(wall_idx)
                unique_windows.append(window)
            else:
                duplicates_removed += 1
        
        claude_response["windows"] = unique_windows
        if duplicates_removed > 0:
            debug_info["steps"].append(f"Step 7.1e: Removed {duplicates_removed} duplicate windows on same exterior walls")
            print(f"Step 7.1e: Removed {duplicates_removed} duplicate windows on same exterior walls", file=sys.stderr) 
    debug_info["steps"].append(f"Step 7.1: Filtering completed - Entry door: {'corrected' if entry_door_corrected else 'valid'}, Doors: {len(claude_response['connecting_doors'])}, Windows: {len(claude_response['windows'])}")
    print(f"Step 7.1: Filtering completed - Entry door: {'corrected' if entry_door_corrected else 'valid'}, Doors: {len(claude_response['connecting_doors'])}, Windows: {len(claude_response['windows'])}", file=sys.stderr)
    # Add entry door
    if "entry_door" in claude_response and claude_response["entry_door"]:
        entry_door = claude_response["entry_door"]
        exterior_wall_index = entry_door["exterior_wall_index"]
        
        if 0 <= exterior_wall_index < len(shared_walls_info["room_exterior_walls"]):
            exterior_wall = shared_walls_info["room_exterior_walls"][exterior_wall_index]
            room_index = exterior_wall["room"]["index"]
            door_info = {
                "width": entry_door["width"],
                "height": entry_door["height"],
                "position_on_wall": float(np.random.uniform(0.2, 0.8)),
                "wall_side": exterior_wall["wall_side"],
                "door_type": entry_door.get("door_type", "entry"),
                "opening": False,  # Entry doors are always physical doors
                "door_material": entry_door.get("door_material", "solid wood entry door")
            }
            
            # CRITICAL: Validate both width AND position to ensure door fits within wall boundaries
            original_width = door_info["width"]
            original_position = door_info["position_on_wall"]
            
            # 1. Calculate actual wall segment length from exterior wall coordinates
            if exterior_wall["direction"] == "x":
                # Wall runs along X-axis, length is x_end - x_start
                wall_segment_length = exterior_wall["x_end"] - exterior_wall["x_start"]
            else:
                # Wall runs along Y-axis, length is y_end - y_start
                wall_segment_length = exterior_wall["y_end"] - exterior_wall["y_start"]
            
            max_width_constraint = wall_segment_length * 0.8
            
            # 2. Calculate position-based width constraint (how much width fits at this position)
            # Door extends from (position - width/2) to (position + width/2)
            # Both bounds must be within [0.1, 0.9] safety range relative to the wall segment
            
            safe_start = 0.1
            safe_end = 0.9
            max_half_width_from_start = original_position - safe_start  # Space available to the left
            max_half_width_from_end = safe_end - original_position      # Space available to the right
            max_half_width = min(max_half_width_from_start, max_half_width_from_end)
            
            position_based_max_width = max_half_width * 2 * wall_segment_length  # Convert back to meters
            
            # 3. Apply the most restrictive constraint
            final_width = min(original_width, max_width_constraint, max(0.7, position_based_max_width))  # Minimum 0.7m door
            
            # 4. If width was reduced significantly, try to adjust position to allow larger width
            if final_width < original_width * 0.8:  # If width was reduced by more than 20%
                # Try to find a better position that allows a larger width
                desired_width = min(original_width, max_width_constraint)
                required_half_width_ratio = (desired_width / 2) / wall_segment_length
                
                # Find the center of the safe zone where this width would fit
                min_center = safe_start + required_half_width_ratio
                max_center = safe_end - required_half_width_ratio
                
                if min_center <= max_center:
                    # There's a valid range, choose the position closest to original
                    if original_position < min_center:
                        final_position = min_center
                    elif original_position > max_center:
                        final_position = max_center
                    else:
                        final_position = original_position
                    
                    final_width = desired_width
                else:
                    # No position allows the desired width, keep the reduced width and original position
                    final_position = max(safe_start, min(safe_end, original_position))
            else:
                # Width is acceptable, just ensure position is in safe range
                final_position = max(safe_start, min(safe_end, original_position))
            
            # 5. CRITICAL: Convert segment-relative position to wall-relative position
            # The entry door position is relative to the exterior wall segment, but we need
            # to store it as relative to the full room wall for consistency with connecting doors
            
            # Calculate the absolute position of the door center within the exterior wall segment
            if exterior_wall["direction"] == "x":
                # X-direction wall segment
                segment_start = exterior_wall["x_start"]
                segment_length = exterior_wall["x_end"] - exterior_wall["x_start"]
                door_absolute_position = segment_start + (final_position * segment_length)
                
                # Convert to position relative to the full room wall
                room_wall_start = rooms_data[room_index]["position"]["x"] if exterior_wall["wall_side"] == "west" else rooms_data[room_index]["position"]["x"]
                if exterior_wall["wall_side"] == "north" or exterior_wall["wall_side"] == "south":
                    room_wall_start = rooms_data[room_index]["position"]["x"]
                    room_wall_length = rooms_data[room_index]["dimensions"]["width"]
                else:
                    room_wall_start = rooms_data[room_index]["position"]["y"] 
                    room_wall_length = rooms_data[room_index]["dimensions"]["length"]
                
                wall_relative_position = (door_absolute_position - room_wall_start) / room_wall_length
            else:
                # Y-direction wall segment  
                segment_start = exterior_wall["y_start"]
                segment_length = exterior_wall["y_end"] - exterior_wall["y_start"]
                door_absolute_position = segment_start + (final_position * segment_length)
                
                # Convert to position relative to the full room wall
                if exterior_wall["wall_side"] == "east" or exterior_wall["wall_side"] == "west":
                    room_wall_start = rooms_data[room_index]["position"]["y"]
                    room_wall_length = rooms_data[room_index]["dimensions"]["length"]
                else:
                    room_wall_start = rooms_data[room_index]["position"]["x"]
                    room_wall_length = rooms_data[room_index]["dimensions"]["width"]
                
                wall_relative_position = (door_absolute_position - room_wall_start) / room_wall_length
            
            # 6. Apply the final values with wall-relative position
            door_info["width"] = final_width
            door_info["position_on_wall"] = wall_relative_position
            
            # 7. Add debugging to track door placement
            print(f"🚪 Adding ENTRY door to Room {room_index} ({rooms_data[room_index]['room_type']}) {door_info['wall_side']} wall:", file=sys.stderr)
            print(f"   Exterior segment: Y[{exterior_wall['y_start']}, {exterior_wall['y_end']}] (length: {wall_segment_length})", file=sys.stderr)
            print(f"   Segment-relative position: {final_position:.3f} → Absolute position: {door_absolute_position:.3f}", file=sys.stderr)
            print(f"   Wall-relative position: {wall_relative_position:.3f}", file=sys.stderr)
            
            result["rooms"][room_index]["doors"].append(door_info)
    
    # Add connecting doors (using the same logic as original function)
    if "connecting_doors" in claude_response and claude_response["connecting_doors"]:
        for connecting_door in claude_response["connecting_doors"]:
            shared_wall_index = connecting_door["shared_wall_index"]
            
            if 0 <= shared_wall_index < len(shared_walls_info["room_room_walls"]):
                shared_wall = shared_walls_info["room_room_walls"][shared_wall_index]
                room1_index = shared_wall["room1"]["index"]
                room2_index = shared_wall["room2"]["index"]
                
                # Apply same validation and clipping logic as original function
                original_door_width = connecting_door["width"]
                
                # Calculate maximum allowed width
                room1_max_width = rooms_data[room1_index]["dimensions"]["width"] * 0.8 if shared_wall["room1_wall"] in ["north", "south"] else rooms_data[room1_index]["dimensions"]["length"] * 0.8
                room2_max_width = rooms_data[room2_index]["dimensions"]["width"] * 0.8 if shared_wall["room2_wall"] in ["north", "south"] else rooms_data[room2_index]["dimensions"]["length"] * 0.8
                shared_wall_max_width = shared_wall["overlap_length"] * 0.8
                
                final_door_width = min(original_door_width, room1_max_width, room2_max_width, shared_wall_max_width)
                
                # Validate and clip position
                original_shared_position = connecting_door["center_position_on_shared_wall"]
                door_half_width_ratio = (final_door_width / 2) / shared_wall["overlap_length"]
                min_safe_position = max(0.1, door_half_width_ratio + 0.05)
                max_safe_position = min(0.9, 1.0 - door_half_width_ratio - 0.05)
                final_shared_position = max(min_safe_position, min(max_safe_position, original_shared_position))
                
                # Calculate positions on each room's wall
                room1_wall_position = calculate_room_wall_position_from_shared_wall(
                    shared_wall, final_shared_position, rooms_data[room1_index], True
                )
                
                room2_wall_position = calculate_room_wall_position_from_shared_wall(
                    shared_wall, final_shared_position, rooms_data[room2_index], False
                )
                
                # Create doors for both rooms
                if final_door_width < 0.4:
                    continue

                door1_info = {
                    "width": final_door_width,
                    "height": connecting_door["height"],
                    "position_on_wall": room1_wall_position,
                    "wall_side": shared_wall["room1_wall"],
                    "door_type": connecting_door.get("door_type", "interior"),
                    "opening": connecting_door.get("opening", False),
                    "door_material": connecting_door.get("door_material", "standard wooden door")
                }
                
                door2_info = {
                    "width": final_door_width,
                    "height": connecting_door["height"],
                    "position_on_wall": room2_wall_position,
                    "wall_side": shared_wall["room2_wall"],
                    "door_type": connecting_door.get("door_type", "interior"),
                    "opening": connecting_door.get("opening", False),
                    "door_material": connecting_door.get("door_material", "standard wooden door")
                }
                
                # Add debugging for connecting doors
                print(f"🔗 Adding CONNECTING door between Room {room1_index} ({rooms_data[room1_index]['room_type']}) and Room {room2_index} ({rooms_data[room2_index]['room_type']}):", file=sys.stderr)
                print(f"   Shared segment: Y[{shared_wall['y_start']}, {shared_wall['y_end']}] (length: {shared_wall['overlap_length']})", file=sys.stderr)
                print(f"   Segment-relative position: {final_shared_position:.3f}", file=sys.stderr)
                print(f"   Room1 {shared_wall['room1_wall']} wall position: {room1_wall_position:.3f}", file=sys.stderr)
                print(f"   Room2 {shared_wall['room2_wall']} wall position: {room2_wall_position:.3f}", file=sys.stderr)

                result["rooms"][room1_index]["doors"].append(door1_info)
                result["rooms"][room2_index]["doors"].append(door2_info)
    
    # Add windows (using same logic as original function)
    if "windows" in claude_response and claude_response["windows"]:
        for window in claude_response["windows"]:
            exterior_wall_index = window["exterior_wall_index"]
            
            if 0 <= exterior_wall_index < len(shared_walls_info["room_exterior_walls"]):
                exterior_wall = shared_walls_info["room_exterior_walls"][exterior_wall_index]
                room_index = exterior_wall["room"]["index"]
                
                window_info = {
                    "width": window["width"],
                    "height": window["height"],
                    "position_on_wall": window["center_position"],
                    "wall_side": exterior_wall["wall_side"],
                    "sill_height": window["sill_height"],
                    "window_type": window.get("window_type", "standard"),
                    "window_grid": window.get("window_grid", [1, 1]),
                    "glass_color": window.get("glass_color", [204, 230, 255]),
                    "frame_color": window.get("frame_color", [77, 77, 77]),
                    "window_appearance_description": window.get("window_appearance_description", "standard window")
                }
                
                # CRITICAL: Apply width clipping considering BOTH room wall dimensions AND exterior wall segment length
                # This ensures windows don't exceed the available exterior wall segment
                
                original_window_width = window_info["width"]
                original_window_position = window_info["position_on_wall"]
                
                # 1. Calculate room-based constraint (based on room's overall wall dimensions)
                if window_info["wall_side"] in ["north", "south"]:
                    room_max_width = rooms_data[room_index]["dimensions"]["width"] * 0.8
                else:
                    room_max_width = rooms_data[room_index]["dimensions"]["length"] * 0.8
                
                # 2. Calculate exterior wall segment constraint (based on actual available wall segment)
                exterior_wall_max_width = exterior_wall["overlap_length"] * 0.8  # Leave 20% margin on the segment
                
                # 3. Calculate position-based width constraint for the exterior wall segment
                # Window extends from (position - width/2) to (position + width/2) within the segment
                # Both bounds must be within [0.1, 0.9] safety range on the segment
                
                safe_start = 0.1
                safe_end = 0.9
                max_half_width_from_start = original_window_position - safe_start
                max_half_width_from_end = safe_end - original_window_position
                max_half_width = min(max_half_width_from_start, max_half_width_from_end)
                
                position_based_max_width = max_half_width * 2 * exterior_wall["overlap_length"]  # Convert back to meters
                
                # 4. Use the most restrictive constraint
                max_width_constraint = min(room_max_width, exterior_wall_max_width)
                final_window_width = min(original_window_width, max_width_constraint, max(0.8, position_based_max_width))  # Minimum 0.8m window
                
                # 5. If width was reduced significantly, try to adjust position to allow larger width
                if final_window_width < original_window_width * 0.8:  # If width was reduced by more than 20%
                    # Try to find a better position that allows a larger width
                    desired_width = min(original_window_width, max_width_constraint)
                    required_half_width_ratio = (desired_width / 2) / exterior_wall["overlap_length"]
                    
                    # Find the center of the safe zone where this width would fit
                    min_center = safe_start + required_half_width_ratio
                    max_center = safe_end - required_half_width_ratio
                    
                    if min_center <= max_center:
                        # There's a valid range, choose the position closest to original
                        if original_window_position < min_center:
                            final_window_position = min_center
                        elif original_window_position > max_center:
                            final_window_position = max_center
                        else:
                            final_window_position = original_window_position
                        
                        final_window_width = desired_width
                    else:
                        # No position allows the desired width, keep the reduced width and clamp position
                        final_window_position = max(safe_start, min(safe_end, original_window_position))
                else:
                    # Width is acceptable, just ensure position is in safe range
                    final_window_position = max(safe_start, min(safe_end, original_window_position))
                
                # 6. CRITICAL: Convert segment-relative position to wall-relative position
                # The window position is relative to the exterior wall segment, but we need
                # to store it as relative to the full room wall for consistency with doors
                
                # Calculate the absolute position of the window center within the exterior wall segment
                if exterior_wall["direction"] == "x":
                    # X-direction wall segment
                    segment_start = exterior_wall["x_start"]
                    segment_length = exterior_wall["x_end"] - exterior_wall["x_start"]
                    window_absolute_position = segment_start + (final_window_position * segment_length)
                    
                    # Convert to position relative to the full room wall
                    if exterior_wall["wall_side"] == "north" or exterior_wall["wall_side"] == "south":
                        room_wall_start = rooms_data[room_index]["position"]["x"]
                        room_wall_length = rooms_data[room_index]["dimensions"]["width"]
                    else:
                        room_wall_start = rooms_data[room_index]["position"]["y"] 
                        room_wall_length = rooms_data[room_index]["dimensions"]["length"]
                    
                    wall_relative_position = (window_absolute_position - room_wall_start) / room_wall_length
                else:
                    # Y-direction wall segment  
                    segment_start = exterior_wall["y_start"]
                    segment_length = exterior_wall["y_end"] - exterior_wall["y_start"]
                    window_absolute_position = segment_start + (final_window_position * segment_length)
                    
                    # Convert to position relative to the full room wall
                    if exterior_wall["wall_side"] == "east" or exterior_wall["wall_side"] == "west":
                        room_wall_start = rooms_data[room_index]["position"]["y"]
                        room_wall_length = rooms_data[room_index]["dimensions"]["length"]
                    else:
                        room_wall_start = rooms_data[room_index]["position"]["x"]
                        room_wall_length = rooms_data[room_index]["dimensions"]["width"]
                    
                    wall_relative_position = (window_absolute_position - room_wall_start) / room_wall_length
                
                # 7. Apply the final values with wall-relative position
                window_info["width"] = final_window_width
                window_info["position_on_wall"] = wall_relative_position
                
                # 8. Add debugging to track window placement
                print(f"🪟 Adding WINDOW to Room {room_index} ({rooms_data[room_index]['room_type']}) {window_info['wall_side']} wall:", file=sys.stderr)
                print(f"   Exterior segment: Y[{exterior_wall['y_start']}, {exterior_wall['y_end']}] (length: {segment_length})", file=sys.stderr)
                print(f"   Segment-relative position: {final_window_position:.3f} → Absolute position: {window_absolute_position:.3f}", file=sys.stderr)
                print(f"   Wall-relative position: {wall_relative_position:.3f}", file=sys.stderr)
                
                # 9. Check for conflicts with existing doors/windows on the same wall
                existing_items = []
                for door in result["rooms"][room_index]["doors"]:
                    if door["wall_side"] == window_info["wall_side"]:
                        existing_items.append({
                            "position": door["position_on_wall"],
                            "width": door["width"],
                            "type": "door"
                        })
                for existing_window in result["rooms"][room_index]["windows"]:
                    if existing_window["wall_side"] == window_info["wall_side"]:
                        existing_items.append({
                            "position": existing_window["position_on_wall"],
                            "width": existing_window["width"],
                            "type": "window"
                        })
                
                # 8. Improved conflict resolution: find the best non-conflicting position
                if existing_items:
                    wall_length = exterior_wall["overlap_length"]
                    window_width_ratio = window_info["width"] / wall_length
                    window_half_width_ratio = window_width_ratio / 2
                    
                    # Sort existing items by position
                    existing_items.sort(key=lambda x: x["position"])
                    
                    # Try to find a safe position that doesn't conflict
                    safe_position = window_info["position_on_wall"]
                    position_found = False
                    
                    # Create a list of occupied segments
                    occupied_segments = []
                    for item in existing_items:
                        item_width_ratio = item["width"] / wall_length
                        item_half_width_ratio = item_width_ratio / 2
                        margin = 0.05  # 5% margin between items
                        
                        segment_start = max(0.0, item["position"] - item_half_width_ratio - margin)
                        segment_end = min(1.0, item["position"] + item_half_width_ratio + margin)
                        occupied_segments.append((segment_start, segment_end))
                    
                    # Merge overlapping segments
                    if occupied_segments:
                        occupied_segments.sort()
                        merged_segments = [occupied_segments[0]]
                        for current_start, current_end in occupied_segments[1:]:
                            last_start, last_end = merged_segments[-1]
                            if current_start <= last_end:
                                # Overlapping segments, merge them
                                merged_segments[-1] = (last_start, max(last_end, current_end))
                            else:
                                # Non-overlapping segment
                                merged_segments.append((current_start, current_end))
                    
                    # Find a free space for the window
                    min_required_space = window_width_ratio + 0.1  # Window + 10% margin
                    
                    # Check if current position is still valid
                    current_start = safe_position - window_half_width_ratio
                    current_end = safe_position + window_half_width_ratio
                    
                    conflicts_with_current = any(
                        not (current_end <= seg_start or current_start >= seg_end)
                        for seg_start, seg_end in merged_segments
                    )
                    
                    if not conflicts_with_current and 0.1 <= current_start and current_end <= 0.9:
                        # Current position is fine
                        position_found = True
                    else:
                        # Find alternative position
                        # Try gaps between occupied segments
                        for i in range(len(merged_segments) + 1):
                            if i == 0:
                                # Before first segment
                                gap_start = 0.1
                                gap_end = merged_segments[0][0] if merged_segments else 0.9
                            elif i == len(merged_segments):
                                # After last segment
                                gap_start = merged_segments[-1][1]
                                gap_end = 0.9
                            else:
                                # Between segments
                                gap_start = merged_segments[i-1][1]
                                gap_end = merged_segments[i][0]
                            
                            gap_size = gap_end - gap_start
                            if gap_size >= min_required_space:
                                # Found a suitable gap
                                safe_position = gap_start + gap_size / 2
                                safe_position = max(0.1, min(0.9, safe_position))
                                position_found = True
                                break
                    
                    # If no good position found, skip this window
                    if not position_found:
                        continue  # Skip adding this window
                    
                    window_info["position_on_wall"] = safe_position
                
                # 9. CRITICAL: Validate window height against room/wall height
                # Ensure that sill_height + window_height does not exceed the room height
                room_height = rooms_data[room_index]["dimensions"]["height"]
                original_sill_height = window_info["sill_height"]
                original_window_height = window_info["height"]
                total_window_height = original_sill_height + original_window_height
                
                if total_window_height > room_height:
                    # Window is too tall for the room, need to scale down proportionally
                    # Leave a small margin (10cm) from the ceiling
                    max_total_height = room_height - 0.1  # 10cm margin from ceiling
                    
                    # Calculate scaling ratio
                    height_ratio = max_total_height / total_window_height
                    
                    # Apply scaling to both sill_height and window height
                    window_info["sill_height"] = original_sill_height * height_ratio
                    window_info["height"] = original_window_height * height_ratio
                    
                    # Ensure minimum sill height (at least 0.5m from floor for safety)
                    min_sill_height = 0.5
                    if window_info["sill_height"] < min_sill_height:
                        window_info["sill_height"] = min_sill_height
                        # Recalculate window height with the new sill height
                        remaining_height = max_total_height - min_sill_height
                        if remaining_height > 0.3:  # Minimum 30cm window height
                            window_info["height"] = remaining_height
                        else:
                            # Room is too short for a proper window, skip this window
                            continue
                
                result["rooms"][room_index]["windows"].append(window_info)
    
    # Add debug information to result
    result["debug_info"] = debug_info
    result["debug_info"]["steps"].append("Step 7 completed: Output formatted successfully")
    
    debug_info["steps"].append("Graph-based door and window placement completed successfully")
    
    # Create comprehensive visualization with doors and windows (similar to solver visualization)
    debug_vis_dir = f"{SERVER_ROOT_DIR}/vis"
    os.makedirs(debug_vis_dir, exist_ok=True)
    graph_vis_path = os.path.join(debug_vis_dir, f"graph_doors_windows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    await get_doors_windows_layout_visualization(result, graph_vis_path, mst, entry_room_index)
    print(f"🔍 Debug: Graph-based doors & windows visualization saved to {graph_vis_path}", file=sys.stderr)
    
    return result


if __name__ == "__main__":
    layout_data = json.loads("""
{
    "building_style": "Modern functional student apartment with linear circulation",
    "rooms": [
        {
            "room_type": "entry foyer",
            "vertices": [
                {"x": 0, "y": 0},
                {"x": 0, "y": 3},
                {"x": 2, "y": 3},
                {"x": 2, "y": 0}
            ],
            "height": 2.7
        },
        {
            "room_type": "main hallway room",
            "vertices": [
                {"x": 2, "y": 1},
                {"x": 2, "y": 2.5},
                {"x": 13, "y": 2.5},
                {"x": 13, "y": 1}
            ],
            "height": 2.7
        },
        {
            "room_type": "bedroom 3",
            "vertices": [
                {"x": 2, "y": 2.5},
                {"x": 2, "y": 6.5},
                {"x": 5.5, "y": 6.5},
                {"x": 5.5, "y": 2.5}
            ],
            "height": 2.7
        },
        {
            "room_type": "bedroom 2",
            "vertices": [
                {"x": 5.5, "y": 2.5},
                {"x": 5.5, "y": 6.5},
                {"x": 9, "y": 6.5},
                {"x": 9, "y": 2.5}
            ],
            "height": 2.7
        },
        {
            "room_type": "bedroom 1",
            "vertices": [
                {"x": 9, "y": 2.5},
                {"x": 9, "y": 6.5},
                {"x": 12.5, "y": 6.5},
                {"x": 12.5, "y": 2.5}
            ],
            "height": 2.7
        },
        {
            "room_type": "master bathroom",
            "vertices": [
                {"x": 12.5, "y": 1},
                {"x": 12.5, "y": 3.5},
                {"x": 15.5, "y": 3.5},
                {"x": 15.5, "y": 1}
            ],
            "height": 2.7
        },
        {
            "room_type": "living room",
            "vertices": [
                {"x": 2, "y": -3},
                {"x": 2, "y": 1},
                {"x": 9, "y": 1},
                {"x": 9, "y": -3}
            ],
            "height": 2.7
        },
        {
            "room_type": "secondary hallway room",
            "vertices": [
                {"x": 9, "y": 0},
                {"x": 9, "y": 2},
                {"x": 13, "y": 2},
                {"x": 13, "y": 0}
            ],
            "height": 2.7
        },
        {
            "room_type": "kitchen",
            "vertices": [
                {"x": 13, "y": 0},
                {"x": 13, "y": 4},
                {"x": 16, "y": 4},
                {"x": 16, "y": 0}
            ],
            "height": 2.7
        },
        {
            "room_type": "dining area",
            "vertices": [
                {"x": 13, "y": 4},
                {"x": 13, "y": 7},
                {"x": 16, "y": 7},
                {"x": 16, "y": 4}
            ],
            "height": 2.7
        },
        {
            "room_type": "powder room",
            "vertices": [
                {"x": 9, "y": -1},
                {"x": 9, "y": 1},
                {"x": 11, "y": 1},
                {"x": 11, "y": -1}
            ],
            "height": 2.7
        }
    ]
}
""")
    input_text = """
# DETAILED ARCHITECTURAL SPECIFICATION: 3-BEDROOM STUDENT APARTMENT

## FLOOR PLAN STRUCTURE & SPATIAL ORGANIZATION

Design a contemporary 3-bedroom, 2-bathroom student apartment with a **linear corridor circulation strategy** and dedicated entry planning. The floor plan consists of **multiple rectangular rooms** connected through a central hallway spine that ensures privacy while maintaining excellent connectivity between shared spaces.

**OVERALL LAYOUT PATTERN**: Linear arrangement with a central rectangular hallway room serving as the primary circulation spine, flanked by private bedrooms on one side and shared living spaces clustered on the other side. This creates clear separation between private study/sleeping areas and social/functional zones while maintaining efficient access to all spaces.

**BUILDING STYLE**: Modern functional design emphasizing practicality, natural light distribution, and flexible living arrangements suitable for undergraduate lifestyle patterns.

## ROOM CONNECTIVITY & CIRCULATION

**ENTRANCE STRATEGY**: The main entrance (standard width front door) opens into a **dedicated rectangular entry foyer room** (compact size) that serves as a transition zone and distribution point. This entry foyer connects directly to the main rectangular hallway room, creating a proper arrival sequence that prevents direct visual access into private areas while establishing clear circulation hierarchy.

**HALLWAY ROOM SYSTEM**: 
- **Main Hallway Room** (medium-length rectangular corridor): Positioned centrally, running lengthwise through the apartment. This rectangular hallway room connects the entry foyer to all major spaces and serves as the primary circulation spine.
- **Secondary Hallway Room** (compact rectangular connector): A shorter rectangular hallway room that branches from the main hallway to provide access to the kitchen and dining area, creating separation between circulation and food preparation activities.

**CIRCULATION HIERARCHY**:
- **Primary Route**: Entry foyer → Main hallway room → Individual bedrooms and shared bathroom
- **Secondary Route**: Main hallway room → Secondary hallway room → Kitchen and dining area
- **Tertiary Route**: Living room connects directly to main hallway room and secondary hallway room for flexible social circulation

## DETAILED ROOM SPECIFICATIONS

### ENTRY SPACES
**Entry Foyer Room** (compact rectangular room): Positioned at the apartment entrance, provides transition space with coat storage capability. Connects to main hallway room via standard interior door.

**Main Hallway Room** (medium-length rectangular corridor): Central circulation spine running lengthwise through the apartment. Connects entry foyer, all three bedrooms, shared bathroom, and living room. Positioned to provide efficient access while maintaining privacy separation.

**Secondary Hallway Room** (compact rectangular connector): Branches from main hallway room to access kitchen and dining areas. Creates functional separation between circulation and food service areas.

### PRIVATE SPACES
**Bedroom 1** (medium-sized rectangular room): Positioned at the far end of main hallway room for maximum privacy. Sized appropriately for single student occupancy with study area capability. Connected to main hallway room via standard interior door.

**Bedroom 2** (medium-sized rectangular room): Located mid-way along main hallway room, adjacent to Bedroom 1. Mirror layout to Bedroom 1 for consistency. Connected to main hallway room via standard interior door.

**Bedroom 3** (medium-sized rectangular room): Positioned closest to entry foyer along main hallway room. Slightly larger than other bedrooms to accommodate potential common storage. Connected to main hallway room via standard interior door.

### BATHROOM FACILITIES
**Master Bathroom** (medium-sized rectangular room): Full bathroom accessible from main hallway room, positioned centrally between bedrooms for equal access. Includes full bathing facilities and serves as primary bathroom for all residents.

**Powder Room** (compact rectangular room): Half-bath accessible from secondary hallway room near kitchen and dining areas. Provides convenient access during social gatherings and meal preparation without accessing private corridor areas.

### SHARED LIVING SPACES
**Living Room** (large rectangular room): Spacious social hub positioned at the intersection of main hallway room and secondary hallway room. Central location enables easy access from all bedrooms while serving as primary gathering space. Connected via wide openings to both hallway rooms for flexible circulation.

**Kitchen** (medium-sized rectangular galley room): Functional cooking space accessible from secondary hallway room. Positioned for efficient service to dining area while maintaining separation from main circulation. Connected to secondary hallway room via standard door and to dining area via wide opening.

**Dining Area** (medium-sized rectangular room): Dedicated eating space connected to kitchen via wide opening and to secondary hallway room via standard door. Positioned to enable social dining while maintaining connection to living areas.

## DOOR AND WINDOW PLACEMENT STRATEGY

### DOOR SPECIFICATIONS
- **Main Entrance**: Standard width exterior door opening into entry foyer room
- **Hallway Connections**: Standard interior doors connecting entry foyer to main hallway room, and main hallway room to secondary hallway room
- **Bedroom Access**: Each bedroom connected to main hallway room via standard interior doors positioned for privacy
- **Bathroom Access**: Master bathroom accessed from main hallway room; powder room accessed from secondary hallway room
- **Living Area Connections**: Living room connected via wide openings to both hallway rooms for flexible circulation
- **Kitchen/Dining**: Kitchen connected to secondary hallway room via standard door; kitchen to dining area via wide opening; dining area to secondary hallway room via standard door

### WINDOW STRATEGY
- **Each Bedroom**: One large window for natural light and ventilation, positioned for study area illumination
- **Living Room**: Multiple large windows on primary facade for maximum natural light and social atmosphere
- **Kitchen**: Medium window over work area for task lighting and ventilation
- **Dining Area**: Medium window for pleasant meal atmosphere and natural light
- **Bathrooms**: Small ventilation windows in master bathroom; powder room may utilize mechanical ventilation
- **Hallway Rooms**: Main hallway room includes one medium window at far end for natural light penetration; secondary hallway room relies on borrowed light from connected spaces

## ARCHITECTURAL CONTEXT & FUNCTIONALITY

**FUNCTIONAL REQUIREMENTS**: Layout supports independent student lifestyles while encouraging appropriate social interaction. Private bedrooms enable individual study and rest, while clustered shared spaces promote community building. Kitchen and dining areas sized for group meal preparation and shared dining experiences.

**PRIVACY CONSIDERATIONS**: Hallway room system creates buffer zones between private bedrooms and shared spaces. Entry foyer prevents direct visual access into living areas from entrance. Bedroom positioning along main hallway room ensures equal privacy levels for all residents.

**DAILY USAGE PATTERNS**: Morning routines supported by efficient bathroom access from bedroom corridor. Study periods accommodated by private bedroom spaces with good natural light. Social activities centered in living room with easy kitchen access for entertaining. Meal preparation and dining areas positioned for both individual and group use.

**SERVICE ACCESS**: Kitchen positioned for efficient grocery delivery and waste management. Utility connections centralized near kitchen and bathroom areas. Storage opportunities integrated throughout hallway rooms and entry foyer for shared items and personal belongings.

This design creates an efficient, livable environment that balances the need for personal privacy with opportunities for social interaction, while the rectangular hallway room system ensures logical circulation and clear spatial organization suitable for undergraduate student living.
"""
    async def run_tests():
        stop_on_first_rejection = True
        # Test collision and isolation detection with continue processing mode
        print("🔍 Testing incremental room collision and isolation detection (continue processing mode)...")
        results = await test_incremental_room_collisions_isolations(layout_data, stop_on_first_rejection=stop_on_first_rejection)
        
        print(f"\n📊 Test Results (Continue Processing Mode):")
        print(f"   Total rooms: {results['total_rooms']}")
        print(f"   Added rooms: {results['added_count']}")
        print(f"   Rejected rooms: {results['rejected_count']}")
        print(f"   Rejection reason: {results['rejection_reason']}")
        if results['rejected_room_name']:
            print(f"   First rejected room: {results['rejected_room_name']}")

        results = results
        
        print(f"\n✅ Successfully added rooms:")
        for room in results['added_rooms']:
            print(f"   - {room['room_type']}")
        
        if results['rejected_rooms']:
            print(f"\n❌ Initially rejected rooms:")
            for room_name in results['rejected_rooms']:
                print(f"   - {room_name}")
            
            # Show detailed rejection reasons if available
            if 'rejected_room_details' in results and results['rejected_room_details']:
                print(f"\n📋 Detailed rejection reasons:")
                for detail in results['rejected_room_details']:
                    print(f"   - {detail['room_name']}: {detail['reason']} (position {detail['index']+1})")
        
        # Generate visualization of original layout
        debug_vis_path = f"{SERVER_ROOT_DIR}/vis/original_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        await get_vlm_room_layout_2d_visualization(layout_data, debug_vis_path)
        
        # Generate visualization of just the added rooms if some were rejected
        if results['rejected_rooms']:
            added_layout = {
                "building_style": layout_data["building_style"],
                "rooms": results['added_rooms']
            }
            added_vis_path = f"{SERVER_ROOT_DIR}/vis/added_rooms_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            await get_vlm_room_layout_2d_visualization(added_layout, added_vis_path)
            print(f"🔍 Added rooms only visualization saved to {added_vis_path}")
            
            # Iteratively try to use Claude API to add the rejected rooms
            print(f"\n🤖 Attempting to use Claude API to add rejected rooms iteratively...")
            
            current_layout = added_layout
            remaining_rejected_rooms = results['rejected_rooms'].copy()
            max_iterations = 5  # Prevent infinite loops
            iteration = 1
            
            while remaining_rejected_rooms and iteration <= max_iterations:
                print(f"\n🔄 Iteration {iteration}: Attempting to add {len(remaining_rejected_rooms)} remaining rooms...")
                print(f"   Rooms to add: {', '.join(remaining_rejected_rooms)}")
                
                try:
                    # Try to add the remaining rejected rooms
                    combined_layout = await incremental_floor_plan_additions(current_layout, remaining_rejected_rooms, input_text)
                    
                    # Test the new combined layout for collisions/isolations
                    print(f"🔍 Testing iteration {iteration} layout for issues...")
                    iteration_results = await test_incremental_room_collisions_isolations(combined_layout, stop_on_first_rejection=stop_on_first_rejection)
                    
                    print(f"\n📊 Iteration {iteration} Results:")
                    print(f"   Total rooms: {iteration_results['total_rooms']}")
                    print(f"   Added rooms: {iteration_results['added_count']}")
                    print(f"   Rejected rooms: {iteration_results['rejected_count']}")
                    print(f"   Rejection reason: {iteration_results['rejection_reason']}")
                    
                    # Check if we made progress
                    if iteration_results['rejected_count'] == 0:
                        # All rooms successfully added!
                        print("🎉 SUCCESS: All rooms successfully added with no collisions or isolations!")
                        current_layout = combined_layout
                        remaining_rejected_rooms = []
                        break
                        
                    elif iteration_results['rejected_count'] < len(remaining_rejected_rooms):
                        # Made progress - some rooms were added
                        progress = len(remaining_rejected_rooms) - iteration_results['rejected_count']
                        print(f"✅ Progress made: {progress} room(s) successfully added this iteration")
                        
                        # Update for next iteration
                        current_layout = {
                            "building_style": combined_layout["building_style"],
                            "rooms": iteration_results['added_rooms']
                        }
                        remaining_rejected_rooms = iteration_results['rejected_rooms']
                    
                    
                    iteration += 1
                        
                except Exception as e:
                    print(f"❌ Error in iteration {iteration}: {e}")
                    break
            
        else:
            print("🎉 All rooms were successfully added on first attempt!")
    
    asyncio.run(run_tests())