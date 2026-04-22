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
import numpy as np
import base64
import json
from PIL import Image
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive, Attributes, 
    Buffer, BufferView, Accessor, 
    Image as GLTFImage, Texture, Sampler, Material, PbrMetallicRoughness,
    FLOAT, UNSIGNED_INT, SCALAR, VEC2, VEC3, ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
)

# Global variable to store the current GLB scene
_current_scene = None

def create_glb_scene():
    """
    Create a new GLB scene with basic structure.
    
    Returns:
        GLTF2: A new GLTF2 object with basic scene structure
    """
    global _current_scene
    
    # Create a new GLTF2 object
    gltf = GLTF2()
    
    # Set up basic asset info
    gltf.asset = {"version": "2.0"}
    
    # Initialize empty lists for all components
    gltf.scenes = []
    gltf.nodes = []
    gltf.meshes = []
    gltf.materials = []
    gltf.textures = []
    gltf.images = []
    gltf.samplers = []
    gltf.buffers = []
    gltf.bufferViews = []
    gltf.accessors = []
    
    # Create a default scene
    scene = Scene(nodes=[])
    gltf.scenes.append(scene)
    gltf.scene = 0  # Set the default scene
    
    _current_scene = gltf
    return gltf

def add_textured_mesh_to_glb_scene(textured_mesh_dict, scene=None, material_name="Material", mesh_name="Mesh", preserve_coordinate_system=True):
    """
    Add a textured mesh to the GLB scene.
    
    Args:
        textured_mesh_dict: {
            'vertices': numpy array of shape (n, 3)
            'faces': numpy array of shape (m, 3)
            'vts': numpy array of shape (n', 2)
            'fts': numpy array of shape (m, 3)
            'texture_image': numpy array of shape (h, w, 3), np.uint8, RGB
        }
        scene: GLTF2 object to add mesh to. If None, uses the global current scene.
        material_name: Name for the material
        mesh_name: Name for the mesh
        preserve_coordinate_system: If True, preserves original coordinate system (Z-up).
                                   If False, converts to glTF standard (Y-up).
    
    Returns:
        int: Index of the created mesh in the scene
    """
    global _current_scene
    
    if scene is None:
        scene = _current_scene
    
    if scene is None:
        raise ValueError("No scene available. Call create_glb_scene() first.")
    
    vertices = textured_mesh_dict['vertices']
    faces = textured_mesh_dict['faces']
    vts = textured_mesh_dict['vts']
    vts[:, 1] = 1.0 - vts[:, 1]
    fts = textured_mesh_dict['fts']
    texture_image = textured_mesh_dict['texture_image']

    metallic_factor = textured_mesh_dict.get('metallic_factor', 0.03)
    roughness_factor = textured_mesh_dict.get('roughness_factor', 1.0)
    
    # Ensure data types are correct
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.uint32)
    vts = vts.astype(np.float32)
    fts = fts.astype(np.uint32)
    
    # Preserve original coordinate system if requested
    if preserve_coordinate_system:
        # Keep vertices as-is (preserve Z-up coordinate system)
        # Note: This preserves the original coordinate system instead of converting to glTF's Y-up standard
        vertices_transformed = vertices.copy()
        vertices_transformed[:, [1, 2]] = vertices[:, [2, 1]]  # Swap Y and Z
        vertices_transformed[:, 2] = -vertices_transformed[:, 2]  # Flip new Z to maintain handedness
        vertices = vertices_transformed
    else:
        # Convert to glTF standard Y-up coordinate system (Z-up -> Y-up)
        pass
    
    # Ensure texture image is in the right format
    if texture_image.dtype != np.uint8:
        texture_image = (texture_image * 255).astype(np.uint8)
    
    # Validate that face indices are valid for vertices
    if faces.max() >= len(vertices):
        raise ValueError(f"Face indices exceed vertex count: max face index {faces.max()}, vertex count {len(vertices)}")
    
    # Validate that texture face indices are valid for texture coordinates
    if fts.max() >= len(vts):
        raise ValueError(f"Texture face indices exceed texture coordinate count: max fts index {fts.max()}, vts count {len(vts)}")
    
    # For GLB export, we need to create a unified mesh where each vertex has both position and texture coordinates
    # This means we need to "expand" the vertex data to match the texture coordinate indexing
    
    # Create expanded vertex array that matches texture coordinate indices
    # Use the texture face indices (fts) to determine the correspondence
    expanded_vertices = []
    expanded_uvs = []
    new_faces = []
    
    vertex_map = {}  # Map (vertex_idx, uv_idx) -> new_vertex_idx
    next_vertex_idx = 0
    
    for face_idx in range(len(faces)):
        face = faces[face_idx]
        tex_face = fts[face_idx]
        new_face = []
        
        for i in range(3):  # Triangle vertices
            vertex_idx = face[i]
            uv_idx = tex_face[i]
            
            # Create a unique key for this vertex-uv combination
            key = (vertex_idx, uv_idx)
            
            if key not in vertex_map:
                # Add new expanded vertex
                expanded_vertices.append(vertices[vertex_idx])
                expanded_uvs.append(vts[uv_idx])
                vertex_map[key] = next_vertex_idx
                next_vertex_idx += 1
            
            new_face.append(vertex_map[key])
        
        new_faces.append(new_face)
    
    # Convert to numpy arrays
    expanded_vertices = np.array(expanded_vertices, dtype=np.float32)
    expanded_uvs = np.array(expanded_uvs, dtype=np.float32)
    new_faces = np.array(new_faces, dtype=np.uint32)
    
    # Now use the expanded data for GLB export
    vertices = expanded_vertices
    vts = expanded_uvs
    faces = new_faces
    
    # Create buffer data
    vertex_data = vertices.tobytes()
    texcoord_data = vts.tobytes()
    indices_data = faces.flatten().tobytes()
    
    # Calculate buffer sizes
    vertex_size = len(vertex_data)
    texcoord_size = len(texcoord_data)
    indices_size = len(indices_data)
    
    # Align to 4-byte boundaries
    def align_to_4(size):
        return (size + 3) & ~3
    
    vertex_aligned = align_to_4(vertex_size)
    texcoord_aligned = align_to_4(texcoord_size)
    
    # Create combined buffer
    buffer_data = bytearray()
    buffer_data.extend(vertex_data)
    buffer_data.extend(b'\x00' * (vertex_aligned - vertex_size))  # Padding
    
    texcoord_offset = len(buffer_data)
    buffer_data.extend(texcoord_data)
    buffer_data.extend(b'\x00' * (texcoord_aligned - texcoord_size))  # Padding
    
    indices_offset = len(buffer_data)
    buffer_data.extend(indices_data)
    
    # Create buffer
    buffer = Buffer(byteLength=len(buffer_data))
    buffer_index = len(scene.buffers)
    scene.buffers.append(buffer)
    
    # Create buffer views
    vertex_buffer_view = BufferView(
        buffer=buffer_index,
        byteOffset=0,
        byteLength=vertex_size,
        target=ARRAY_BUFFER
    )
    vertex_buffer_view_index = len(scene.bufferViews)
    scene.bufferViews.append(vertex_buffer_view)
    
    texcoord_buffer_view = BufferView(
        buffer=buffer_index,
        byteOffset=texcoord_offset,
        byteLength=texcoord_size,
        target=ARRAY_BUFFER
    )
    texcoord_buffer_view_index = len(scene.bufferViews)
    scene.bufferViews.append(texcoord_buffer_view)
    
    indices_buffer_view = BufferView(
        buffer=buffer_index,
        byteOffset=indices_offset,
        byteLength=indices_size,
        target=ELEMENT_ARRAY_BUFFER
    )
    indices_buffer_view_index = len(scene.bufferViews)
    scene.bufferViews.append(indices_buffer_view)
    
    # Create accessors
    vertex_accessor = Accessor(
        bufferView=vertex_buffer_view_index,
        componentType=FLOAT,
        count=len(vertices),
        type=VEC3,
        min=vertices.min(axis=0).tolist(),
        max=vertices.max(axis=0).tolist()
    )
    vertex_accessor_index = len(scene.accessors)
    scene.accessors.append(vertex_accessor)
    
    texcoord_accessor = Accessor(
        bufferView=texcoord_buffer_view_index,
        componentType=FLOAT,
        count=len(vts),
        type=VEC2,
        min=vts.min(axis=0).tolist(),
        max=vts.max(axis=0).tolist()
    )
    texcoord_accessor_index = len(scene.accessors)
    scene.accessors.append(texcoord_accessor)
    
    indices_accessor = Accessor(
        bufferView=indices_buffer_view_index,
        componentType=UNSIGNED_INT,
        count=len(faces.flatten()),
        type=SCALAR
    )
    indices_accessor_index = len(scene.accessors)
    scene.accessors.append(indices_accessor)
    
    # Create texture
    # Convert texture image to PIL Image
    from io import BytesIO
    pil_image = Image.fromarray(texture_image, 'RGB')
    buffer_io = BytesIO()
    pil_image.save(buffer_io, format='PNG')
    image_data = buffer_io.getvalue()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    image_uri = f"data:image/png;base64,{image_base64}"
    
    # Create image
    gltf_image = GLTFImage(uri=image_uri)
    image_index = len(scene.images)
    scene.images.append(gltf_image)
    
    # Create sampler
    sampler = Sampler()
    sampler_index = len(scene.samplers)
    scene.samplers.append(sampler)
    
    # Create texture
    texture = Texture(source=image_index, sampler=sampler_index)
    texture_index = len(scene.textures)
    scene.textures.append(texture)
    
    # Create material
    pbr_metallic_roughness = PbrMetallicRoughness(
        baseColorTexture={"index": texture_index},
        metallicFactor=metallic_factor,
        roughnessFactor=roughness_factor
    )
    material = Material(
        name=material_name,
        pbrMetallicRoughness=pbr_metallic_roughness
    )
    material_index = len(scene.materials)
    scene.materials.append(material)
    
    # Create primitive
    primitive = Primitive(
        attributes=Attributes(
            POSITION=vertex_accessor_index,
            TEXCOORD_0=texcoord_accessor_index
        ),
        indices=indices_accessor_index,
        material=material_index
    )
    
    # Create mesh
    mesh = Mesh(name=mesh_name, primitives=[primitive])
    mesh_index = len(scene.meshes)
    scene.meshes.append(mesh)
    
    # Create node
    node = Node(mesh=mesh_index)
    node_index = len(scene.nodes)
    scene.nodes.append(node)
    
    # Add node to the scene
    scene.scenes[0].nodes.append(node_index)
    
    # Store buffer data for later saving
    if not hasattr(scene, '_buffer_data'):
        scene._buffer_data = {}
    scene._buffer_data[buffer_index] = buffer_data
    
    return mesh_index

def save_glb_scene(save_path, scene=None):
    """
    Save the GLB scene to a file.
    
    Args:
        save_path: Path where to save the GLB file
        scene: GLTF2 object to save. If None, uses the global current scene.
    """
    global _current_scene
    
    if scene is None:
        scene = _current_scene
    
    if scene is None:
        raise ValueError("No scene available. Call create_glb_scene() first.")
    
    # Consolidate all buffer data into a single buffer for GLB format
    if hasattr(scene, '_buffer_data') and scene._buffer_data:
        # Calculate total size and create unified buffer
        total_size = 0
        buffer_info = []
        
        for i, buffer_data in scene._buffer_data.items():
            if i < len(scene.buffers):
                # Align to 4-byte boundaries
                aligned_size = (len(buffer_data) + 3) & ~3
                buffer_info.append((i, total_size, len(buffer_data), aligned_size, buffer_data))
                total_size += aligned_size
        
        # Create unified buffer
        unified_buffer = bytearray(total_size)
        
        # Copy buffer data and update buffer views
        for buffer_idx, offset, original_size, aligned_size, buffer_data in buffer_info:
            # Copy data to unified buffer
            unified_buffer[offset:offset + original_size] = buffer_data
            # Pad with zeros if needed
            if aligned_size > original_size:
                unified_buffer[offset + original_size:offset + aligned_size] = b'\x00' * (aligned_size - original_size)
            
            # Update buffer views that reference this buffer
            for bv in scene.bufferViews:
                if bv.buffer == buffer_idx:
                    bv.byteOffset += offset
                    bv.buffer = 0  # All buffers now reference the unified buffer
        
        # Replace all buffers with a single unified buffer
        scene.buffers = [Buffer(byteLength=total_size)]
        
        # Set the unified buffer data
        scene.set_binary_blob(unified_buffer)
    
    # Save the file
    scene.save(save_path)

def save_glb_from_mesh_dict(mesh_dict, save_path):
    """
    save a glb file from a mesh dict
    mesh_dict: {
        'vertices': numpy array of shape (n, 3)
        'faces': numpy array of shape (m, 3)
        'vts': numpy array of shape (n', 2)
        'fts': numpy array of shape (m, 3)
        'texture_image': numpy array of shape (h, w, 3), np.uint8, RGB
    }
    save_path: path to save the glb file
    """
    scene = create_glb_scene()
    add_textured_mesh_to_glb_scene(mesh_dict, scene=scene)
    save_glb_scene(save_path, scene=scene)

def load_glb_to_mesh_dict(glb_path):
    """
    load a glb file to a mesh dict
    glb_path: path to the glb file
    return: mesh dict: {
        'vertices': numpy array of shape (n, 3)
        'faces': numpy array of shape (m, 3)
        'vts': numpy array of shape (n', 2)
        'fts': numpy array of shape (m, 3)
        'texture_image': numpy array of shape (h, w, 3), np.uint8, RGB
    }
    """
    from io import BytesIO
    
    # Load the GLB file
    gltf = GLTF2.load(glb_path)
    binary_blob = gltf.binary_blob()
    
    # Get the first mesh (assuming single mesh saved by save_glb_from_mesh_dict)
    mesh = gltf.meshes[0]
    primitive = mesh.primitives[0]
    
    # Extract vertices from POSITION accessor
    position_accessor = gltf.accessors[primitive.attributes.POSITION]
    position_buffer_view = gltf.bufferViews[position_accessor.bufferView]
    position_offset = position_buffer_view.byteOffset + (position_accessor.byteOffset or 0)
    position_data = binary_blob[position_offset:position_offset + position_buffer_view.byteLength]
    vertices = np.frombuffer(position_data, dtype=np.float32).reshape(-1, 3).copy()
    
    # Reverse coordinate transformation (was: swap Y/Z, then flip Z)
    # To reverse: flip Z, then swap Y/Z back
    vertices[:, 2] = -vertices[:, 2]  # Flip Z back
    vertices[:, [1, 2]] = vertices[:, [2, 1]]  # Swap Y and Z back
    
    # Extract texture coordinates from TEXCOORD_0 accessor
    texcoord_accessor = gltf.accessors[primitive.attributes.TEXCOORD_0]
    texcoord_buffer_view = gltf.bufferViews[texcoord_accessor.bufferView]
    texcoord_offset = texcoord_buffer_view.byteOffset + (texcoord_accessor.byteOffset or 0)
    texcoord_data = binary_blob[texcoord_offset:texcoord_offset + texcoord_buffer_view.byteLength]
    vts = np.frombuffer(texcoord_data, dtype=np.float32).reshape(-1, 2).copy()
    
    # Reverse UV flip (was: vts[:, 1] = 1.0 - vts[:, 1])
    vts[:, 1] = 1.0 - vts[:, 1]
    
    # Extract face indices from indices accessor
    indices_accessor = gltf.accessors[primitive.indices]
    indices_buffer_view = gltf.bufferViews[indices_accessor.bufferView]
    indices_offset = indices_buffer_view.byteOffset + (indices_accessor.byteOffset or 0)
    indices_data = binary_blob[indices_offset:indices_offset + indices_buffer_view.byteLength]
    faces = np.frombuffer(indices_data, dtype=np.uint32).reshape(-1, 3).copy()
    
    # Since save_glb_from_mesh_dict expands vertices to match UVs 1:1,
    # faces and fts are the same
    fts = faces.copy()
    
    # Extract texture image
    material = gltf.materials[primitive.material]
    texture_index = material.pbrMetallicRoughness.baseColorTexture['index']
    texture = gltf.textures[texture_index]
    image = gltf.images[texture.source]
    
    if image.uri and image.uri.startswith('data:'):
        # Base64-encoded image in URI
        # Format: data:image/png;base64,<base64_data>
        base64_data = image.uri.split(',', 1)[1]
        image_bytes = base64.b64decode(base64_data)
    elif image.bufferView is not None:
        # Image stored in buffer view
        image_buffer_view = gltf.bufferViews[image.bufferView]
        image_offset = image_buffer_view.byteOffset
        image_bytes = binary_blob[image_offset:image_offset + image_buffer_view.byteLength]
    else:
        raise ValueError("Could not find texture image data")
    
    # Decode image
    pil_image = Image.open(BytesIO(image_bytes))
    texture_image = np.array(pil_image.convert('RGB'), dtype=np.uint8)
    
    return {
        'vertices': vertices,
        'faces': faces,
        'vts': vts,
        'fts': fts,
        'texture_image': texture_image
    }
    