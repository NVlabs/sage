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
from nvdiffrast_rendering.mesh import build_mesh_dict
import numpy as np
from nvdiffrast_rendering.camera import get_intrinsic, get_camera_perspective_projection_matrix, get_mvp_matrix, build_camera_matrix
import torch
from nvdiffrast_rendering.render import rasterize_mesh_with_uv
from nvdiffrast_rendering.context import get_glctx
import tempfile
from PIL import Image
import json
import uuid
import os
import sys
import re
from key import ANTHROPIC_API_KEY
from constants import SERVER_ROOT_DIR
from vlm import call_vlm
from utils import extract_json_from_response
def get_front_right_up_pose(vertices):
    # get the bounding box of the vertices
    min_x = vertices[:, 0].min()
    max_x = vertices[:, 0].max()
    min_y = vertices[:, 1].min()
    max_y = vertices[:, 1].max()
    min_z = vertices[:, 2].min()
    max_z = vertices[:, 2].max()
    
    # get the center of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    center = np.array([center_x, center_y, center_z])
    
    # get the scale of the bounding box (max of x, y, z)
    scale = max(max_x - min_x, max_y - min_y, max_z - min_z)

    # theta = 45.0 * np.pi / 180.0
    theta = 135.0 * np.pi / 180.0
    phi = 60.0 * np.pi / 180.0

    radius = scale * 1.5

    eye = center + radius * np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
    at = center
    up = np.array([0, 0, 1])

    camera_matrix = build_camera_matrix(
        torch.from_numpy(eye).float(),
        torch.from_numpy(at).float(),
        torch.from_numpy(up).float()
    )

    return camera_matrix

def get_up_pose_given_direction(vertices, direction=90.0):
    # get the bounding box of the vertices
    min_x = vertices[:, 0].min()
    max_x = vertices[:, 0].max()
    min_y = vertices[:, 1].min()
    max_y = vertices[:, 1].max()
    min_z = vertices[:, 2].min()
    max_z = vertices[:, 2].max()
    
    # get the center of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    center = np.array([center_x, center_y, center_z])
    
    # get the scale of the bounding box (max of x, y, z)
    scale = max(max_x - min_x, max_y - min_y, max_z - min_z)

    theta = direction * np.pi / 180.0
    phi = 60.0 * np.pi / 180.0

    radius = scale * 1.5

    eye = center + radius * np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
    at = center
    up = np.array([0, 0, 1])

    camera_matrix = build_camera_matrix(
        torch.from_numpy(eye).float(),
        torch.from_numpy(at).float(),
        torch.from_numpy(up).float()
    )

    return camera_matrix

def estimate_front_from_mesh(mesh_dict):
    import base64
    import trimesh

    directions = [90.0, 180.0, 270.0, 0.0]
    
    vertices = mesh_dict["mesh"].vertices
    triangles = mesh_dict["mesh"].faces
    vts = mesh_dict["tex_coords"]["vts"]
    fts = mesh_dict["tex_coords"]["fts"]
    texture_map = mesh_dict["texture"]

    mesh_dict_for_rendering = build_mesh_dict(vertices, triangles, vts, fts, texture_map)
    glctx = get_glctx()
    intrinsic = get_intrinsic(60, 1024, 1024)

    temp_file_paths = []

    for direction in directions:

        camera_matrix = get_up_pose_given_direction(vertices, direction)

        projection_matrix = get_camera_perspective_projection_matrix(intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], 1024, 1024, 0.001, 10.0)

        mvp_matrix = get_mvp_matrix(camera_matrix, projection_matrix).to("cuda")
        valid, triangle_id, depth, rgb = rasterize_mesh_with_uv(mesh_dict_for_rendering, mvp_matrix, glctx, (1024, 1024))

        rgb = rgb.cpu().numpy().clip(0, 1)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file.name
        Image.fromarray((rgb * 255).astype(np.uint8)).save(temp_file.name)
        temp_file_paths.append(temp_file.name)

    # Get category from mesh_dict if available
    category = "object"
    if "object_attributes" in mesh_dict and "name" in mesh_dict["object_attributes"]:
        category = mesh_dict["object_attributes"]["name"]
    
    # Prepare prompt for VLM
    prompt_text = (
        f"You have been given 4 images of an {category}, each taken from a different angle. Your task is to identify the image that shows the 'front view' of the object. The front view refers to the perspective where the object's main face or most important features are most clearly visible, typically from the viewer's point of view.\n\n"
        + "Please keep the following in mind:\n"
        + "1. The front view is often characterized by the most significant or most visible face of the object.\n"
        + "2. For objects like cabinets, the front view is typically where the doors and drawers are visible. For chairs, the front view may show the seat and backrest. For other objects, consider the main or most notable side visible from the viewer's point of view.\n"
        + "3. The front view is usually the view where the object faces the camera directly or is oriented in such a way that the most prominent features (such as a face, label, or handle) are visible.\n"
        + "4. If the images are taken at different angles (front, right, left, back), choose the image where the object faces you (from the viewer's perspective).\n"
        + "5. The indices of the images start from 0.\n\n"
        + "Please provide your response in the following JSON format:\n\n"
        + "```json\n"
        + "{\n"
        + '    "front_index": <index number 0-3>,\n'
        + '    "reasoning": "<brief explanation of why this view is the front>"\n'
        + "}\n"
        + "```\n\n"
        + "Respond with only the JSON object, no additional text."
    )
    
    # Build content with text and images
    content = [{"type": "text", "text": prompt_text}]
    
    # Add all 4 images to the content
    for temp_file_path in temp_file_paths:
        with open(temp_file_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_data
            }
        })
    
    # Call VLM to estimate the front direction
    try:
        response = call_vlm(
            vlm_type="qwen",
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        
        # Extract response text
        if not response or not hasattr(response, 'content') or not response.content:
            raise ValueError("No response received from VLM")
        
        response_text = None
        for content_item in response.content:
            if hasattr(content_item, 'text'):
                response_text = content_item.text
                break
        
        if not response_text:
            raise ValueError("VLM response content has no text")
        
        # Parse JSON response
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError("Could not extract JSON content from VLM response")
        
        result = json.loads(response_text)
        front_idx = int(result.get("front_index", 0))
        reasoning = result.get("reasoning", "No reasoning provided")
        
        print(f"VLM identified front view at index {front_idx}")
        print(f"Reasoning: {reasoning}")
        
    except Exception as e:
        print(f"Error calling VLM for front estimation: {e}")
        print("Defaulting to front_idx = 0")
        front_idx = 0
    
    # Delete the temp files
    for temp_file_path in temp_file_paths:
        try:
            os.remove(temp_file_path)
        except Exception as e:
            print(f"Warning: Could not delete temp file {temp_file_path}: {e}")
    
    # Rotate the mesh_dict to the front direction along z axis
    rotate_angles = [0.0, -90.0, -180.0, -270.0]
    # if front idx=0, then rotate 0 degrees
    # if front idx=1, then rotate 90 degrees
    # if front idx=2, then rotate 180 degrees
    # if front idx=3, then rotate 270 degrees
    
    if 0 <= front_idx < len(rotate_angles):
        rotation_angle = rotate_angles[front_idx]
        
        if rotation_angle != 0.0:
            print(f"Rotating mesh by {rotation_angle} degrees around Z-axis to align front view")
            
            # Create rotation matrix for Z-axis rotation
            rotation_radians = np.radians(rotation_angle)
            rotation_matrix = trimesh.transformations.rotation_matrix(
                rotation_radians, 
                [0, 0, 1]  # Z-axis
            )
            
            # Apply rotation to the mesh
            mesh = mesh_dict["mesh"]
            mesh.apply_transform(rotation_matrix)
            mesh_dict["mesh"] = mesh

    return mesh_dict

def infer_attributes_from_claude(mesh_dict, caption=None):

    vertices = mesh_dict["mesh"].vertices
    triangles = mesh_dict["mesh"].faces
    vts = mesh_dict["tex_coords"]["vts"]
    fts = mesh_dict["tex_coords"]["fts"]
    texture_map = mesh_dict["texture"]

    height_retrieved = float(vertices[:, 2].max() - vertices[:, 2].min())

    mesh_dict_for_rendering = build_mesh_dict(vertices, triangles, vts, fts, texture_map)

    camera_matrix = get_front_right_up_pose(vertices)

    intrinsic = get_intrinsic(60, 1024, 1024)
    projection_matrix = get_camera_perspective_projection_matrix(intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], 1024, 1024, 0.001, 100.0)

    mvp_matrix = get_mvp_matrix(camera_matrix, projection_matrix).to("cuda")

    glctx = get_glctx()

    valid, triangle_id, depth, rgb = rasterize_mesh_with_uv(mesh_dict_for_rendering, mvp_matrix, glctx, (1024, 1024))
    rgb = rgb.cpu().numpy().clip(0, 1)
    
    # Save the rgb image into a temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.name
    Image.fromarray((rgb * 255).astype(np.uint8)).save(temp_file.name)
    
    # Prepare the prompt with caption consideration
    caption_text = caption if caption else "No caption provided"

    generated_height = float(vertices[:, 2].max() - vertices[:, 2].min())
    generated_width = float(vertices[:, 0].max() - vertices[:, 0].min())
    generated_length = float(vertices[:, 1].max() - vertices[:, 1].min())

    # Determine the relative ordering of dimensions
    dimensions = [
        ("width", generated_width),
        ("length", generated_length),
        ("height", generated_height)
    ]
    dimensions_sorted = sorted(dimensions, key=lambda x: x[1], reverse=True)
    dimension_ordering = " > ".join([d[0] for d in dimensions_sorted])
    
    prompt = f"""
You are a professional 3D artist and object analyst who can provide accurate assessments of 3D models. 

GIVEN CAPTION: {caption_text}

TASK: 
You are provided with an image of a 3D object rendered from the upper right view.
Analyze the 3D object in the image and provide a comprehensive assessment. 
Pay special attention to whether the object semantically aligns with the given caption.

HEIGHT DEFINITION:
- The height of the object is defined as the distance of the highest point to the lowest point of the object in the image.

ANALYSIS REQUIREMENTS:
- For semantic_alignment: Compare the core object type from the given caption with what you actually see. Set to true if they represent the same type of object (e.g., "desk" and "table" are both tables, "chair" and "armchair" are both chairs), set to false if they are fundamentally different object types (e.g., "desk" vs "chair", "table" vs "lamp"). Focus on the primary object category, not specific variations or detailed descriptions.
- For height estimation, you need to take the object placement spatial orientation into account. Different spatial orientation has different height value as the height definition suggests.
- Be specific and accurate in your descriptions

PBR MATERIAL PARAMETERS:
- Estimate the PBR (Physically Based Rendering) material properties of the object:
  * Metallic: 0.0 for non-metallic materials (wood, plastic, fabric, ceramic) to 1.0 for pure metallic materials (polished metal, chrome, brass)
  * Roughness: 0.0 for perfectly smooth/glossy surfaces (mirror, polished metal) to 1.0 for very rough/matte surfaces (concrete, rough wood, fabric)
- Consider the material and surface finish you observe in the image

RELATIVE SIZE INFORMATION:
- The relative ordering of the object's dimensions (from larger to smaller) is: {dimension_ordering}
- Your estimated width, length, and height should follow the same relative ordering.
- This means your estimates should satisfy: {dimension_ordering}

Please provide your analysis in the following structured JSON format:

```json
{{
    "long_caption": "A detailed description of what you see in the image",
    "short_caption": "A brief, concise description of the object",
    "given_caption": "{caption_text}",
    "semantic_alignment": true or false (whether the given caption describes the object you see),
    "name": "specific object name",
    "explanation": "One paragraph explaining your reasoning for the width, length, height and weight estimates. For height estimation, follow the height definition above to explain. Explain each dimension and weight estimation with your reasoning.",
    "width": "width in meters (only numerical value, no unit)",
    "length": "length in meters (only numerical value, no unit)",
    "height": "height in meters (only numerical value, no unit)",
    "dimension_ordering_check": "Verify that your estimates follow the required ordering: {dimension_ordering}",
    "weight": "weight in kilograms (only numerical value, no unit)",
    "scale_unit": "meter",
    "weight_unit": "kilogram",
    "pbr_parameters": {{
        "explanation": "Detailed explanation of your reasoning for the PBR parameters",
        "metallic": "metallic value from 0.0 to 1.0 (only numerical value)",
        "roughness": "roughness value from 0.0 to 1.0 (only numerical value)"
    }}
}}
```

Please respond with only the JSON object, no additional text.
"""

    try:
        # Convert image to base64 for API call
        import base64
        with open(temp_file.name, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        num_retries = 0

        while num_retries < 3:
        
            # Import anthropic here to avoid import errors if not available
            # Call Claude API with both image and text
            response = call_vlm(
                vlm_type="qwen",
                model="Qwen/Qwen2.5-VL-72B-Instruct",
                max_tokens=8000,
                temperature=0.5,
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
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
        
            response_text = response.content[0].text.strip()

            response_text = extract_json_from_response(response_text)
            if response_text:
                break 

            num_retries += 1
            if num_retries >= 3:
                raise ValueError("Failed to extract JSON content from Claude response")

        result = json.loads(response_text)
        # print(f"Claude response: {result}", file=sys.stderr)
        
        # Validate required fields
        required_fields = ["long_caption", "short_caption", "given_caption", "semantic_alignment", 
                            "name", "height", "weight", "scale_unit", "weight_unit", "explanation", "pbr_parameters"]
        
        for field in required_fields:
            if field not in result:
                print(f"Warning: Missing field {field} in Claude response", file=sys.stderr)
                result[field] = "" if field != "pbr_parameters" else {}
        
        # Ensure numeric fields are properly formatted
        try:
            result["height"] = float(result["height"])
        except (ValueError, TypeError):
            result["height"] = 1.0

        try:
            result["width"] = float(result["width"])
        except (ValueError, TypeError):
            result["width"] = 1.0

        try:
            result["length"] = float(result["length"])
        except (ValueError, TypeError):
            result["length"] = 1.0
            
        try:
            result["weight"] = float(result["weight"])
        except (ValueError, TypeError):
            result["weight"] = 1.0
            
        # Ensure semantic_alignment is boolean
        if isinstance(result["semantic_alignment"], str):
            result["semantic_alignment"] = result["semantic_alignment"].lower() == "true"
        
        # Validate and format pbr_parameters
        if "pbr_parameters" not in result or not isinstance(result["pbr_parameters"], dict):
            result["pbr_parameters"] = {}
        
        # Ensure metallic value is present and valid
        try:
            metallic = float(result["pbr_parameters"].get("metallic", 0.0))
            result["pbr_parameters"]["metallic"] = max(0.0, min(1.0, metallic))
        except (ValueError, TypeError):
            result["pbr_parameters"]["metallic"] = 0.0
        
        # Ensure roughness value is present and valid
        try:
            roughness = float(result["pbr_parameters"].get("roughness", 0.5))
            result["pbr_parameters"]["roughness"] = max(0.0, min(1.0, roughness))
        except (ValueError, TypeError):
            result["pbr_parameters"]["roughness"] = 0.5
        
        # # Debug: save the image with attributes
        # debug_filename = f"{SERVER_ROOT_DIR}/vis/claude_attribute_debug_{result.get('name', 'unknown').replace(' ', '_')}_{result.get('height', 1.0):.2f}_{height_retrieved:.2f}_{str(uuid.uuid4())[:8]}_semantic_alignment_{result['semantic_alignment']}.png"
        # Image.fromarray((rgb * 255).astype(np.uint8)).save(debug_filename)
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"Error parsing Claude response as JSON: {e}", file=sys.stderr)
        return {}
    
    except Exception as e:
        print(f"Error calling Claude API: {e}", file=sys.stderr)
        return {}
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)