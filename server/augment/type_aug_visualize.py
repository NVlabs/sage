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
import argparse
import json
import sys
import os
import numpy as np
from PIL import Image
import tempfile
import uuid
from typing import Dict, List, Set, Tuple, Any
import torch
import importlib.util

# Add the server directory to the Python path
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from constants import RESULTS_DIR, SERVER_ROOT_DIR
from models import FloorPlan, Room, Object

utils_spec = importlib.util.spec_from_file_location("server_utils", os.path.join(SERVER_ROOT_DIR, "utils.py"))
server_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(server_utils)

# Import the specific functions from server utils
dict_to_floor_plan = server_utils.dict_to_floor_plan
# from tex_utils import get_textured_object_mesh_from_object, dict_to_layout
tex_utils_spec = importlib.util.spec_from_file_location("server_tex_utils", os.path.join(SERVER_ROOT_DIR, "tex_utils.py"))
server_tex_utils = importlib.util.module_from_spec(tex_utils_spec)
tex_utils_spec.loader.exec_module(server_tex_utils)

# Import the specific functions from server utils
get_textured_object_mesh_from_object = server_tex_utils.get_textured_object_mesh_from_object
export_layout_to_mesh_dict_list_tree_search_with_object_id = server_tex_utils.export_layout_to_mesh_dict_list_tree_search_with_object_id
export_layout_to_mesh_dict_object_id = server_tex_utils.export_layout_to_mesh_dict_object_id
get_textured_object_mesh = server_tex_utils.get_textured_object_mesh

from nvdiffrast_rendering.mesh import build_mesh_dict
from nvdiffrast_rendering.camera import get_intrinsic, get_camera_perspective_projection_matrix, get_mvp_matrix, build_camera_matrix
from nvdiffrast_rendering.render import rasterize_mesh_with_uv
from nvdiffrast_rendering.context import get_glctx
from utils import get_layout_from_scene_save_dir


def get_front_right_up_pose(vertices):
    """Get camera pose for rendering objects - same as in object_attribute_inference.py"""
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


def load_candidate_layout(candidate_file_path: str) -> FloorPlan:
    """Load a candidate layout from JSON file"""
    with open(candidate_file_path, 'r') as f:
        layout_data = json.load(f)["layout"]
    return dict_to_floor_plan(layout_data)





def render_object(layout: FloorPlan, obj: Object, output_path: str) -> bool:
    """
    Render a single object using the same approach as object_attribute_inference.py
    
    Args:
        layout: The layout containing the object
        obj: The object to render
        output_path: Path to save the rendered image
        
    Returns:
        True if rendering succeeded, False otherwise
    """
    try:
        # Get textured mesh data
        mesh_info_dict = get_textured_object_mesh_from_object(layout, obj)
        
        if mesh_info_dict is None or mesh_info_dict["mesh"] is None:
            print(f"Warning: Could not load mesh for object {obj.id} (source: {obj.source}, source_id: {obj.source_id})")
            return False
        
        vertices = mesh_info_dict["mesh"].vertices
        triangles = mesh_info_dict["mesh"].faces
        
        # Load texture information - structure matches object_attribute_inference.py
        texture_info = mesh_info_dict["texture"]
        vts = texture_info["vts"]
        fts = texture_info["fts"]
        
        # Load texture map
        texture_map_path = texture_info["texture_map_path"]
        if not os.path.exists(texture_map_path):
            print(f"Warning: Texture map not found: {texture_map_path}")
            return False
            
        texture_map = np.array(Image.open(texture_map_path)) / 255.0
        
        # Create mesh_dict structure similar to object_attribute_inference.py
        mesh_dict = {
            "mesh": mesh_info_dict["mesh"],
            "tex_coords": {
                "vts": vts,
                "fts": fts
            },
            "texture": texture_map
        }
        
        # Build mesh dictionary for rendering using the same approach as object_attribute_inference.py
        mesh_dict_for_rendering = build_mesh_dict(vertices, triangles, vts, fts, texture_map)
        
        # Set up camera
        camera_matrix = get_front_right_up_pose(vertices)
        
        # Set up projection
        intrinsic = get_intrinsic(60, 1024, 1024)
        projection_matrix = get_camera_perspective_projection_matrix(
            intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], 
            1024, 1024, 0.001, 100.0
        )
        
        mvp_matrix = get_mvp_matrix(camera_matrix, projection_matrix).to("cuda")
        
        # Get rendering context
        glctx = get_glctx()
        
        # Render
        valid, triangle_id, depth, rgb = rasterize_mesh_with_uv(
            mesh_dict_for_rendering, mvp_matrix, glctx, (1024, 1024)
        )
        rgb = rgb.cpu().numpy().clip(0, 1)
        
        # Save the rendered image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(output_path)
        
        return True
        
    except Exception as e:
        print(f"Error rendering object {obj.id}: {e}")
        return False


def collect_all_unique_objects_with_mapping(layout_id: str, aug_name: str) -> Dict[str, Any]:
    """
    Collect all unique objects from all type augmentation candidates with their original object mapping
    
    Args:
        layout_id: Original layout ID
        aug_name: Augmentation name
        
    Returns:
        Dictionary containing:
        - unique_objects: Set of (source, source_id) tuples
        - object_to_render: Dict mapping (source, source_id) to (layout, obj)
        - original_object_mapping: Dict mapping original_object_id to list of (source, source_id)
        - object_groups: Dict grouping objects by type
    """
    # Load metadata
    metadata_path = os.path.join(RESULTS_DIR, layout_id, aug_name, f"{aug_name}_type_candidates_metadata.json")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    unique_objects = set()
    object_to_render = {}
    original_object_id = metadata.get('object_id', 'unknown')
    original_object_mapping = {original_object_id: []}
    object_groups = {}  # Group by object type
    
    print(f"Processing {metadata['total_candidates']} candidates...")
    
    # Process each candidate layout
    processed_candidates = 0
    for candidate in metadata['candidates']:
        candidate_file = candidate['candidate_file']
        candidate_file_path = os.path.join(RESULTS_DIR, layout_id, aug_name, candidate_file)
        
        if not os.path.exists(candidate_file_path):
            print(f"Warning: Candidate file not found: {candidate_file_path}")
            continue
        
        # Load the candidate layout
        try:
            layout = load_candidate_layout(candidate_file_path)
            processed_candidates += 1
            
            # Extract all objects from all rooms
            for room in layout.rooms:
                for obj in room.objects:
                    obj_key = (obj.source, obj.source_id)
                    unique_objects.add(obj_key)
                    
                    # Store object for rendering (keep first instance found)
                    if obj_key not in object_to_render:
                        object_to_render[obj_key] = (layout, obj)
                        
                        # Map to original object
                        original_object_mapping[original_object_id].append(obj_key)
                        
                        # Group by object type
                        obj_type = obj.type
                        if obj_type not in object_groups:
                            object_groups[obj_type] = []
                        object_groups[obj_type].append(obj_key)
                    
        except Exception as e:
            print(f"Error loading candidate {candidate_file}: {e}")
            continue
    
    print(f"Processed {processed_candidates} candidate layouts")
    print(f"Found {len(unique_objects)} unique objects")
    print(f"Object type distribution: {[(k, len(v)) for k, v in object_groups.items()]}")
    
    return {
        'unique_objects': unique_objects,
        'object_to_render': object_to_render,
        'original_object_mapping': original_object_mapping,
        'object_groups': object_groups,
        'original_object_id': original_object_id
    }


def create_composite_image(image_paths: List[str], output_path: str, images_per_row: int = 4) -> bool:
    """
    Create a composite image from multiple individual images arranged in a grid
    
    Args:
        image_paths: List of paths to individual images
        output_path: Path to save the composite image
        images_per_row: Number of images per row in the composite
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image
        import math
        
        # Load all images
        images = []
        for path in image_paths:
            if os.path.exists(path):
                img = Image.open(path)
                images.append(img)
            else:
                print(f"Warning: Image not found: {path}")
        
        if not images:
            print("No valid images to composite")
            return False
        
        # Calculate grid dimensions
        num_images = len(images)
        rows = math.ceil(num_images / images_per_row)
        cols = min(num_images, images_per_row)
        
        # Assume all images are the same size (1024x1024 from rendering)
        img_width, img_height = images[0].size
        
        # Create composite image
        composite_width = cols * img_width
        composite_height = rows * img_height
        composite = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))
        
        # Place images in grid
        for i, img in enumerate(images):
            row = i // images_per_row
            col = i % images_per_row
            x = col * img_width
            y = row * img_height
            composite.paste(img, (x, y))
        
        # Save composite
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        composite.save(output_path)
        return True
        
    except Exception as e:
        print(f"Error creating composite image: {e}")
        return False


def visualize_type_augmented_objects(layout_id: str, aug_name: str, output_dir: str = None):
    """
    Visualize all unique objects from type augmentation candidates with grouping by original object
    
    Args:
        layout_id: Original layout ID
        aug_name: Augmentation name
        output_dir: Output directory for rendered images (optional)
    """
    if output_dir is None:
        output_dir = os.path.join(SERVER_ROOT_DIR, "vis", "type_aug", layout_id, aug_name)
    
    print(f"Starting visualization for layout {layout_id}, augmentation {aug_name}")
    print(f"Output directory: {output_dir}")
    
    # Collect all unique objects with their mappings
    try:
        collection_result = collect_all_unique_objects_with_mapping(layout_id, aug_name)
        unique_objects = collection_result['unique_objects']
        object_to_render = collection_result['object_to_render']
        object_groups = collection_result['object_groups']
        original_object_id = collection_result['original_object_id']
    except Exception as e:
        print(f"Error collecting unique objects: {e}")
        return
    
    if not unique_objects:
        print("No unique objects found!")
        return
    
    print(f"Rendering {len(object_to_render)} unique objects...")
    
    # Render each unique object
    rendered_count = 0
    failed_count = 0
    rendered_images = {}  # Map object_key to image path
    
    for (source, source_id), (layout, obj) in object_to_render.items():
        # Create output filename
        safe_source = source.replace("/", "_").replace("\\", "_")
        safe_source_id = source_id.replace("/", "_").replace("\\", "_")
        output_filename = f"{safe_source}_{safe_source_id}_{obj.type.replace(' ', '_')}.png"
        output_path = os.path.join(output_dir, "individual", output_filename)
        
        print(f"Rendering object: {obj.type} (source: {source}, source_id: {source_id})")
        
        if render_object(layout, obj, output_path):
            rendered_count += 1
            rendered_images[(source, source_id)] = output_path
            print(f"  ✓ Saved to: {output_path}")
        else:
            failed_count += 1
            print(f"  ✗ Failed to render")
    
    print(f"\nCreating composite images grouped by object type...")
    
    # Create composite images for each object type
    composite_count = 0
    for obj_type, obj_keys in object_groups.items():
        if len(obj_keys) > 1:  # Only create composite if there are multiple objects of this type
            # Collect image paths for this type
            type_image_paths = []
            for obj_key in obj_keys:
                if obj_key in rendered_images:
                    type_image_paths.append(rendered_images[obj_key])
            
            if type_image_paths:
                composite_filename = f"composite_{obj_type.replace(' ', '_')}_variants.png"
                composite_path = os.path.join(output_dir, "composites", composite_filename)
                
                if create_composite_image(type_image_paths, composite_path):
                    composite_count += 1
                    print(f"  ✓ Created composite for {obj_type}: {composite_path} ({len(type_image_paths)} variants)")
                else:
                    print(f"  ✗ Failed to create composite for {obj_type}")
    
    # Create overall composite of all objects
    all_image_paths = list(rendered_images.values())
    if all_image_paths:
        overall_composite_path = os.path.join(output_dir, "all_objects_composite.png")
        if create_composite_image(all_image_paths, overall_composite_path, images_per_row=4):
            print(f"  ✓ Created overall composite: {overall_composite_path}")
        else:
            print(f"  ✗ Failed to create overall composite")
    
    print(f"\nVisualization complete!")
    print(f"Successfully rendered: {rendered_count} individual objects")
    print(f"Failed to render: {failed_count} objects")
    print(f"Created: {composite_count} type-specific composite images")
    print(f"Output directory: {output_dir}")
    
    # Create summary file
    summary = {
        "layout_id": layout_id,
        "aug_name": aug_name,
        "original_object_id": original_object_id,
        "total_unique_objects": len(unique_objects),
        "successfully_rendered": rendered_count,
        "failed_to_render": failed_count,
        "composites_created": composite_count,
        "output_directory": output_dir,
        "object_groups": {k: len(v) for k, v in object_groups.items()},
        "rendered_objects": []
    }
    
    # Add details about rendered objects
    for (source, source_id), (layout, obj) in object_to_render.items():
        safe_source = source.replace("/", "_").replace("\\", "_")
        safe_source_id = source_id.replace("/", "_").replace("\\", "_")
        output_filename = f"{safe_source}_{safe_source_id}_{obj.type.replace(' ', '_')}.png"
        
        summary["rendered_objects"].append({
            "source": source,
            "source_id": source_id,
            "object_type": obj.type,
            "object_description": obj.description,
            "output_filename": output_filename,
            "rendered_successfully": (source, source_id) in rendered_images
        })
    
    summary_path = os.path.join(output_dir, "visualization_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize type-augmented objects")
    parser.add_argument("--layout_id", type=str, required=True, help="Original layout ID")
    parser.add_argument("--aug_name", type=str, required=True, help="Augmentation name")
    parser.add_argument("--output_dir", type=str, help="Output directory for rendered images")
    
    args = parser.parse_args()
    
    visualize_type_augmented_objects(args.layout_id, args.aug_name, args.output_dir)
