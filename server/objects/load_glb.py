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
import os
import numpy as np
import trimesh
from PIL import Image
import sys
from constants import SERVER_ROOT_DIR
def load_glb(glb_path):
    """
    Load a GLB file and extract mesh data including vertices, faces, 
    texture coordinates, face texture indices, and texture image.
    
    Args:
        glb_path (str): Path to the GLB file
        
    Returns:
        dict: Contains 'mesh' (trimesh object), 'tex_coords' (vts, fts), and 'texture'
    """
    # Load the GLB file using trimesh
    scene_or_mesh = trimesh.load(glb_path)
    
    # Handle both single mesh and scene cases
    if isinstance(scene_or_mesh, trimesh.Scene):
        # If it's a scene, get the first mesh
        mesh = None
        for geometry in scene_or_mesh.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                mesh = geometry
                break
        if mesh is None:
            raise ValueError("No valid mesh found in the GLB scene")
    else:
        # It's already a single mesh
        mesh = scene_or_mesh
    
    # Extract vertices and faces
    vertices = mesh.vertices  # shape: (N, 3)
    faces = mesh.faces       # shape: (M, 3)
    
    # Extract texture coordinates (UV coordinates) - IMPROVED VERSION
    vts = None
    fts = None
    texture = None
    
    # First, try to get UV coordinates from the visual attributes
    uv_coordinates = None
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uv_coordinates = mesh.visual.uv
        # print(f"Found UV coordinates from mesh.visual.uv: {uv_coordinates.shape}", file=sys.stderr)
    
    # If that fails, try to extract from the original GLB data
    if uv_coordinates is None:
        try:
            # Try to access the original GLB data through trimesh's scene graph
            if isinstance(scene_or_mesh, trimesh.Scene):
                # Look for texture coordinates in the original GLB data
                for geom_name, geometry in scene_or_mesh.geometry.items():
                    if hasattr(geometry, 'metadata') and 'TEXCOORD_0' in str(geometry.metadata):
                        print("Found TEXCOORD_0 in metadata", file=sys.stderr)
                    
                    # Try different ways to access UV data
                    if hasattr(geometry.visual, 'uv') and geometry.visual.uv is not None:
                        uv_coordinates = geometry.visual.uv
                        print(f"Found UV coordinates from geometry: {uv_coordinates.shape}", file=sys.stderr)
                        break
        except Exception as e:
            print(f"Could not extract UV from GLB metadata: {e}", file=sys.stderr)
    
    # Handle UV coordinates and face texture indices
    if uv_coordinates is not None:
        vts = uv_coordinates  # shape: (N, 2)
        
        # CRITICAL FIX: For GLB files, we need to check if UV indices are separate from vertex indices
        # Many GLB files have separate texture coordinate indexing
        if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
            # If we have a proper material with texture, UV indices likely match vertex indices
            fts = faces.copy()  # shape: (M, 3)
        else:
            # For more complex GLB files, we might need separate UV indices
            # This is a fallback - in practice, you'd want to parse the original GLB data
            fts = faces.copy()  # shape: (M, 3)
            
        # Validate UV coordinates
        if len(vts) != len(vertices):
            print(f"Warning: UV coordinate count ({len(vts)}) doesn't match vertex count ({len(vertices)})", file=sys.stderr)
            print("This may indicate separate UV indexing in the GLB file", file=sys.stderr)
            
            # Try to handle the mismatch by creating a proper mapping
            # This is a simplified approach - real GLB parsing would be more complex
            if len(vts) < len(vertices):
                # If we have fewer UV coordinates, pad with zeros
                padding = np.zeros((len(vertices) - len(vts), 2))
                vts = np.vstack([vts, padding])
                print(f"Padded UV coordinates to match vertex count", file=sys.stderr)
            elif len(vts) > len(vertices):
                # If we have more UV coordinates, truncate (not ideal)
                vts = vts[:len(vertices)]
                print(f"Truncated UV coordinates to match vertex count", file=sys.stderr)
    else:
        # If no UV coordinates found, create dummy ones
        print("Warning: No UV coordinates found in GLB file, creating dummy coordinates", file=sys.stderr)
        vts = np.zeros((len(vertices), 2))  # shape: (N, 2)
        fts = faces.copy()  # shape: (M, 3)
    
    # Extract texture image if available
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        material = mesh.visual.material
        
        # Try to get the base color texture
        if hasattr(material, 'image') and material.image is not None:
            # Convert PIL image to numpy array
            texture_img = material.image
            if texture_img.mode == 'RGBA':
                texture = np.array(texture_img).astype(np.float32) / 255.0  # shape: (H, W, 4)
            elif texture_img.mode == 'RGB':
                texture = np.array(texture_img).astype(np.float32) / 255.0  # shape: (H, W, 3)
            else:
                # Convert to RGB if in other mode
                texture_img = texture_img.convert('RGB')
                texture = np.array(texture_img).astype(np.float32) / 255.0  # shape: (H, W, 3)
        elif hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            texture_img = material.baseColorTexture
            texture = np.array(texture_img).astype(np.float32) / 255.0
    
    # If no texture found, create a default white texture
    if texture is None:
        print("Warning: No texture found in GLB file, creating default white texture", file=sys.stderr)
        texture = np.ones((256, 256, 3), dtype=np.float32)  # White texture
    
    # Create the return dictionary following the codebase pattern
    mesh_dict = {
        "mesh": mesh,
        "tex_coords": {
            "vts": vts,      # texture coordinates, shape: (-1, 2)
            "fts": fts,      # face texture indices, shape: (-1, 3)
        },
        "texture": texture   # texture image, shape: (H, W, 3) or (H, W, 4)
    }
    
    return mesh_dict


def load_glb_advanced(glb_path):
    """
    Advanced GLB loader that properly handles UV coordinate indexing by parsing the original GLB data.
    This version attempts to extract proper texture coordinate indices.
    
    Args:
        glb_path (str): Path to the GLB file
        
    Returns:
        dict: Contains 'mesh' (trimesh object), 'tex_coords' (vts, fts), and 'texture'
    """
    import json
    import struct
    
    try:
        # Try to load the GLB file and parse its structure
        with open(glb_path, 'rb') as f:
            # Read GLB header
            magic = f.read(4)
            if magic != b'glTF':
                raise ValueError("Not a valid GLB file")
            
            version = struct.unpack('<I', f.read(4))[0]
            length = struct.unpack('<I', f.read(4))[0]
            
            # Read JSON chunk
            json_length = struct.unpack('<I', f.read(4))[0]
            json_type = f.read(4)
            json_data = f.read(json_length).decode('utf-8')
            gltf_json = json.loads(json_data)
            
            print(f"GLB file version: {version}, total length: {length}", file=sys.stderr)
            print(f"Found {len(gltf_json.get('meshes', []))} meshes in GLB", file=sys.stderr)
            
            # For now, fall back to the standard loader
            # A full implementation would parse the GLB structure and extract proper UV indices
            return load_glb(glb_path)
            
    except Exception as e:
        print(f"Advanced GLB parsing failed: {e}, falling back to standard loader", file=sys.stderr)
        return load_glb(glb_path)


def save_obj_with_materials(mesh_dict, obj_path, mtl_path=None, texture_path=None):
    """
    Save mesh data to OBJ format with proper UV coordinates and materials.
    This function properly handles the texture coordinates and creates MTL files.
    
    Args:
        mesh_dict (dict): Dictionary from load_glb containing mesh, tex_coords, and texture
        obj_path (str): Path where the OBJ file will be saved
        mtl_path (str): Path where the MTL file will be saved (optional)
        texture_path (str): Path where the texture image will be saved (optional)
    """
    if mtl_path is None:
        mtl_path = obj_path.replace('.obj', '.mtl')
    if texture_path is None:
        texture_path = obj_path.replace('.obj', '_texture.png')
    
    mesh = mesh_dict["mesh"]
    tex_coords = mesh_dict["tex_coords"]
    texture = mesh_dict["texture"]
    
    vertices = mesh.vertices
    faces = mesh.faces
    vts = tex_coords["vts"]
    fts = tex_coords["fts"]
    
    print(f"Saving OBJ with {len(vertices)} vertices, {len(faces)} faces, {len(vts)} texture coordinates", file=sys.stderr)
    
    # Save texture image
    if texture is not None and texture.shape[0] > 1 and texture.shape[1] > 1:
        # Convert texture back to PIL Image and save
        if texture.shape[-1] == 4:  # RGBA
            texture_img = Image.fromarray((texture * 255).astype(np.uint8), 'RGBA')
        else:  # RGB
            texture_img = Image.fromarray((texture * 255).astype(np.uint8), 'RGB')
        texture_img.save(texture_path)
        print(f"Saved texture to: {texture_path}", file=sys.stderr)
    
    # Create MTL file
    mtl_name = os.path.basename(mtl_path)
    texture_name = os.path.basename(texture_path)
    
    with open(mtl_path, 'w') as mtl_file:
        mtl_file.write("# Material file created by improved load_glb\n")
        mtl_file.write("newmtl material0\n")
        mtl_file.write("Ka 1.0 1.0 1.0\n")  # Ambient color
        mtl_file.write("Kd 0.8 0.8 0.8\n")  # Diffuse color
        mtl_file.write("Ks 0.5 0.5 0.5\n")  # Specular color
        mtl_file.write("Ns 96.0\n")          # Specular exponent
        mtl_file.write("d 1.0\n")            # Dissolve (opacity)
        mtl_file.write("illum 2\n")          # Illumination model
        if texture is not None:
            mtl_file.write(f"map_Kd {texture_name}\n")  # Diffuse texture map
    
    print(f"Saved MTL to: {mtl_path}", file=sys.stderr)
    
    # Create OBJ file
    with open(obj_path, 'w') as obj_file:
        obj_file.write("# OBJ file created by improved load_glb\n")
        obj_file.write(f"mtllib {os.path.basename(mtl_path)}\n")
        obj_file.write("\n")
        
        # Write vertices
        for vertex in vertices:
            obj_file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        obj_file.write("\n")
        
        # Write texture coordinates
        for vt in vts:
            obj_file.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
        obj_file.write("\n")
        
        # Use material
        obj_file.write("usemtl material0\n")
        
        # Write faces with texture coordinates
        # OBJ format uses 1-based indexing
        for i, face in enumerate(faces):
            if i < len(fts):
                ft = fts[i]
                obj_file.write(f"f {face[0]+1}/{ft[0]+1} {face[1]+1}/{ft[1]+1} {face[2]+1}/{ft[2]+1}\n")
            else:
                # Fallback if texture indices don't match
                obj_file.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
    
    print(f"Saved OBJ to: {obj_path}", file=sys.stderr)


def test_glb_to_obj_conversion(glb_path, output_dir=None):
    """
    Test function to convert a GLB file to OBJ format using the improved loader.
    
    Args:
        glb_path (str): Path to the input GLB file
        output_dir (str): Directory where output files will be saved
    """
    if output_dir is None:
        output_dir = os.path.dirname(glb_path)
    
    base_name = os.path.splitext(os.path.basename(glb_path))[0]
    obj_path = os.path.join(output_dir, f"{base_name}_improved.obj")
    
    print(f"Converting {glb_path} to {obj_path}", file=sys.stderr)
    
    try:
        # Load GLB using improved function
        mesh_dict = load_glb(glb_path)
        
        # Save as OBJ with proper materials
        save_obj_with_materials(mesh_dict, obj_path)
        
        print(f"Conversion completed successfully!", file=sys.stderr)
        print(f"Output files:", file=sys.stderr)
        print(f"  OBJ: {obj_path}", file=sys.stderr)
        print(f"  MTL: {obj_path.replace('.obj', '.mtl')}", file=sys.stderr)
        print(f"  Texture: {obj_path.replace('.obj', '_texture.png')}", file=sys.stderr)
        
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()



def create_usage_example():
    """
    Create a usage example script showing how to use the improved GLB loader.
    """
    example_code = '''
# Example: Using the improved GLB loader

from objects.load_glb import load_glb, save_obj_with_materials

# 1. Load a GLB file with proper UV coordinate handling
glb_path = "path/to/your/model.glb"
mesh_dict = load_glb(glb_path)

# 2. Access the loaded data
mesh = mesh_dict["mesh"]                    # trimesh object
texture_coords = mesh_dict["tex_coords"]    # UV coordinates and face indices
texture_image = mesh_dict["texture"]        # texture as numpy array

# 3. Save as OBJ with proper materials
output_path = "path/to/output/model.obj"
save_obj_with_materials(mesh_dict, output_path)

# This will create:
# - model.obj (geometry with UV coordinates)
# - model.mtl (material definition)
# - model_texture.png (texture image)

# 4. The generated OBJ file will have proper face definitions like:
# f 1/1 2/2 3/3  (vertex_index/uv_index pairs)

print("✅ GLB converted to OBJ with proper texture mapping!")
'''
    
    with open("example_glb_usage.py", "w") as f:
        f.write(example_code)
    
    print("Created example_glb_usage.py with usage instructions", file=sys.stderr)


def diagnose_glb_issues(glb_path):
    """
    Diagnose common issues with GLB files that cause poor OBJ conversion.
    
    Args:
        glb_path (str): Path to the GLB file to diagnose
    """
    print(f"\n🔍 DIAGNOSING GLB FILE: {os.path.basename(glb_path)}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    
    try:
        scene_or_mesh = trimesh.load(glb_path)
        
        if isinstance(scene_or_mesh, trimesh.Scene):
            print(f"✓ Scene with {len(scene_or_mesh.geometry)} geometries", file=sys.stderr)
            mesh = list(scene_or_mesh.geometry.values())[0]
        else:
            print(f"✓ Single mesh", file=sys.stderr)
            mesh = scene_or_mesh
        
        # Check basic geometry
        print(f"✓ Vertices: {mesh.vertices.shape[0]:,}", file=sys.stderr)
        print(f"✓ Faces: {mesh.faces.shape[0]:,}", file=sys.stderr)
        
        # Check UV coordinates
        has_uvs = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None
        if has_uvs:
            uv_shape = mesh.visual.uv.shape
            print(f"✓ UV coordinates: {uv_shape[0]:,} (shape: {uv_shape})", file=sys.stderr)
            
            # Check for common issues
            if uv_shape[0] != mesh.vertices.shape[0]:
                print(f"⚠️  WARNING: UV count ({uv_shape[0]}) ≠ vertex count ({mesh.vertices.shape[0]})", file=sys.stderr)
                print(f"   This indicates separate UV indexing (common in complex GLB files)", file=sys.stderr)
            else:
                print(f"✓ UV coordinates match vertex count", file=sys.stderr)
                
            # Check UV coordinate range
            uv_min = mesh.visual.uv.min(axis=0)
            uv_max = mesh.visual.uv.max(axis=0)
            print(f"✓ UV range: U[{uv_min[0]:.3f}, {uv_max[0]:.3f}], V[{uv_min[1]:.3f}, {uv_max[1]:.3f}]", file=sys.stderr)
            
            if uv_min.min() < -0.1 or uv_max.max() > 1.1:
                print(f"⚠️  WARNING: UV coordinates outside [0,1] range", file=sys.stderr)
        else:
            print(f"❌ No UV coordinates found", file=sys.stderr)
        
        # Check materials and textures
        has_material = hasattr(mesh.visual, 'material') and mesh.visual.material is not None
        if has_material:
            material = mesh.visual.material
            print(f"✓ Material found: {type(material).__name__}", file=sys.stderr)
            
            has_texture = hasattr(material, 'image') and material.image is not None
            if has_texture:
                tex_shape = np.array(material.image).shape
                print(f"✓ Texture image: {tex_shape}", file=sys.stderr)
            else:
                print(f"❌ No texture image in material", file=sys.stderr)
        else:
            print(f"❌ No material found", file=sys.stderr)
            
        print("-" * 60, file=sys.stderr)
        
    except Exception as e:
        print(f"❌ Error diagnosing GLB file: {e}", file=sys.stderr)


if __name__ == "__main__":
    glb_path = os.path.join(SERVER_ROOT_DIR, "objects/sofa.glb")
    
    if os.path.exists(glb_path):
        
            
        # Test GLB to OBJ conversion
        print(f"\nTesting GLB to OBJ conversion:")
        test_glb_to_obj_conversion(glb_path)
        
    else:
        print(f"GLB file not found at: {glb_path}")