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
import shutil
import glob
from constants import RESULTS_DIR
import argparse
import json
from tex_utils import export_layout_all_objects_mesh_dict_list_no_object_transform
from models import FloorPlan
from utils import dict_to_floor_plan
from PIL import Image
import numpy as np
import pdb


def save_ply_from_mesh_dict(mesh_dict, save_path):
    """
    Save mesh with texture coordinates to PLY file using plyfile package.
    Preserves original topology without modifying vertices or faces.
    
    Args:
        mesh_dict: Dictionary containing:
            - 'vertices': Nx3 array of vertex positions
            - 'faces': Mx3 array of face vertex indices
            - 'vts': Kx2 array of texture coordinates
            - 'fts': Mx3 array of face texture coordinate indices
        save_path: Path to save the PLY file
    """
    from plyfile import PlyData, PlyElement
    
    vertices = np.array(mesh_dict['vertices'])
    faces = np.array(mesh_dict['faces'])
    vts = np.array(mesh_dict['vts'])
    fts = np.array(mesh_dict['fts'])
    
    # Create vertex data (positions only)
    vertex_data = np.zeros(len(vertices), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4')
    ])
    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]
    
    # Create texture coordinate data as separate element
    texcoord_data = np.zeros(len(vts), dtype=[
        ('s', 'f4'), ('t', 'f4')
    ])
    texcoord_data['s'] = vts[:, 0]
    texcoord_data['t'] = vts[:, 1]
    
    # Create face data with both vertex_indices and texcoord_indices
    face_data = np.zeros(len(faces), dtype=[
        ('vertex_indices', 'i4', (3,)),
        ('texcoord_indices', 'i4', (3,))
    ])
    face_data['vertex_indices'] = faces
    face_data['texcoord_indices'] = fts
    
    # Create PlyElements and write PLY file
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    texcoord_element = PlyElement.describe(texcoord_data, 'texcoord')
    face_element = PlyElement.describe(face_data, 'face')
    ply_data = PlyData([vertex_element, texcoord_element, face_element], text=False)
    ply_data.write(save_path)

def load_ply_to_mesh_dict(ply_path):
    """
    Load PLY file to mesh dict with texture coordinates.
    Loads the original topology without any modifications.
    
    Args:
        ply_path: Path to the PLY file
        
    Returns:
        mesh_dict: Dictionary containing:
            - 'vertices': Nx3 array of vertex positions
            - 'faces': Mx3 array of face vertex indices
            - 'vts': Kx2 array of texture coordinates
            - 'fts': Mx3 array of face texture coordinate indices
    """
    from plyfile import PlyData
    
    ply_data = PlyData.read(ply_path)
    
    # Extract vertex data
    vertex_data = ply_data['vertex']
    vertices = np.column_stack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ])
    
    # Extract texture coordinates from separate element
    texcoord_data = ply_data['texcoord']
    vts = np.column_stack([
        texcoord_data['s'],
        texcoord_data['t']
    ])
    
    # Extract face data with both vertex_indices and texcoord_indices
    face_data = ply_data['face']
    faces = np.vstack(face_data['vertex_indices'])
    fts = np.vstack(face_data['texcoord_indices'])
    
    return {
        'vertices': vertices,
        'faces': faces,
        'vts': vts,
        'fts': fts,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_id", type=str, required=True, help="Layout ID to pack")
    parser.add_argument("--upload_name", type=str, required=True, help="Layout ID to pack")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to validate PLY files")
    args = parser.parse_args()

    layout_id = args.layout_id
    upload_name = args.upload_name
    debug = args.debug
    layout_dir = os.path.join(RESULTS_DIR, layout_id)

    # Create a log JSON file to store the status in layout_dir
    log_file_path = os.path.join(layout_dir, f"{layout_id}_pack_log.json")
    pack_log = {
        "layout_id": layout_id,
        "upload_name": upload_name,
        "status": {
            "tmp_upload_dir_packing": False,
            "zip_file_creation": False,
            "tmp_upload_dir_cleaning": False,
        },
        "zip_file_path": None,
    }
    
    # Save initial log state
    with open(log_file_path, "w") as f:
        json.dump(pack_log, f, indent=4)
    print(f"Created pack log: {log_file_path}")

    layout_dict = json.load(open(os.path.join(layout_dir, f"{layout_id}.json"), "r"))
    layout = dict_to_floor_plan(layout_dict)

    # Create temporary directory for upload
    tmp_upload_dir = os.path.join(layout_dir, f"{layout_id}_hf_upload_tmp")
    os.makedirs(tmp_upload_dir, exist_ok=True)
    
    # Create subdirectories
    objects_dir = os.path.join(tmp_upload_dir, "objects")
    materials_dir = os.path.join(tmp_upload_dir, "materials")
    os.makedirs(objects_dir, exist_ok=True)
    os.makedirs(materials_dir, exist_ok=True)

    object_ply_paths = []
    object_texture_paths = []
    mesh_info_dict = export_layout_all_objects_mesh_dict_list_no_object_transform(layout)

    for object_source_path in mesh_info_dict.keys():
        object_source = object_source_path.split("/")[0]
        object_source_id = object_source_path.split("/")[1]
        object_ply_save_path = os.path.join(objects_dir, f"{object_source_id}.ply")
        save_ply_from_mesh_dict({
            'vertices': mesh_info_dict[object_source_path]["mesh"].vertices,
            'faces': mesh_info_dict[object_source_path]["mesh"].faces,
            'vts': mesh_info_dict[object_source_path]["texture"]["vts"],
            'fts': mesh_info_dict[object_source_path]["texture"]["fts"],
        }, object_ply_save_path)
        if debug:
            # validate the ply file by loading it and compare with the original mesh dict
            loaded_mesh_dict = load_ply_to_mesh_dict(object_ply_save_path)
            original_vertices = np.array(mesh_info_dict[object_source_path]["mesh"].vertices)
            original_faces = np.array(mesh_info_dict[object_source_path]["mesh"].faces)
            original_vts = np.array(mesh_info_dict[object_source_path]["texture"]["vts"])
            original_fts = np.array(mesh_info_dict[object_source_path]["texture"]["fts"])
            
            assert np.allclose(loaded_mesh_dict['vertices'], original_vertices), \
                f"Vertices mismatch for {object_source_id}"
            assert np.array_equal(loaded_mesh_dict['faces'], original_faces), \
                f"Faces mismatch for {object_source_id}"
            assert np.allclose(loaded_mesh_dict['vts'], original_vts), \
                f"Texture coordinates mismatch for {object_source_id}"
            assert np.array_equal(loaded_mesh_dict['fts'], original_fts), \
                f"Face texture indices mismatch for {object_source_id}"
            print(f"Validated: {object_source_id}.ply")
        object_ply_paths.append(object_ply_save_path)
        
        # Copy texture file to objects directory
        object_texture_src = os.path.join(layout_dir, object_source, f"{object_source_id}_texture.png")
        object_texture_dst = os.path.join(objects_dir, f"{object_source_id}_texture.png")
        shutil.copy2(object_texture_src, object_texture_dst)
        object_texture_paths.append(object_texture_dst)

    material_names = []

    for room_i, room in enumerate(layout_dict["rooms"]):
        for object_i, object in enumerate(room["objects"]):
            object_source = object["source"]
            object_source_id = object["source_id"]
            object_source_path = f"{object_source}/{object_source_id}"
            metallic_factor = mesh_info_dict[object_source_path]["texture"]["pbr_parameters"]["metallic"]
            roughness_factor = mesh_info_dict[object_source_path]["texture"]["pbr_parameters"]["roughness"]
            layout_dict["rooms"][room_i]["objects"][object_i]["pbr_parameters"] = {
                "metallic": metallic_factor,
                "roughness": roughness_factor,
            }
        for wall in room["walls"]:
            material_names.append(wall["material"])
        material_names.append(room["floor_material"])
        for door in room["doors"]:
            material_names.append(door["door_material"])

    material_names = list(set(material_names))
    material_paths = []
    
    for material_name in material_names:
        material_paths.extend(glob.glob(os.path.join(layout_dir, "materials", f"*{material_name}*")))
    material_paths = list(set(material_paths))

    # Copy material files to materials directory
    for file_path in material_paths:
        dst_path = os.path.join(materials_dir, os.path.basename(file_path))
        shutil.copy2(file_path, dst_path)
        print(f"Copied material: {os.path.basename(file_path)}")

    # Save layout JSON to upload directory
    layout_json_path = os.path.join(tmp_upload_dir, f"{layout_id}.json")
    with open(layout_json_path, "w") as f:
        json.dump(layout_dict, f, indent=4)
    print(f"Created: {layout_id}.json")

    render_script_path = "render_preview.sh"
    render_script_args = f"{layout_dir}/{layout_id}.json"
    render_cmd = f"bash {render_script_path} {render_script_args}"
    os.system(render_cmd)

    preview_dir = os.path.join(layout_dir, "preview")
    # Copy preview directory to upload folder
    if os.path.exists(preview_dir):
        preview_dst = os.path.join(tmp_upload_dir, "preview")
        # remove the preview directory if it exists
        if os.path.exists(preview_dst):
            shutil.rmtree(preview_dst)
        shutil.copytree(preview_dir, preview_dst)
        print(f"Copied preview directory: {len(os.listdir(preview_dst))} files")
    else:
        print(f"Warning: Preview directory not found: {preview_dir}")

    print("Directory to zip: ", tmp_upload_dir)

    # Update log: packing complete
    pack_log["status"]["tmp_upload_dir_packing"] = True
    with open(log_file_path, "w") as f:
        json.dump(pack_log, f, indent=4)
    print("Pack log updated: tmp_upload_dir_packing = True")

    # Zip the tmp_upload_dir and save to a zip file
    zip_file_path = os.path.join(layout_dir, f"{upload_name}.zip")
    shutil.make_archive(
        base_name=os.path.join(layout_dir, upload_name),  # Path without .zip extension
        format='zip',
        root_dir=tmp_upload_dir,  # Root directory to archive
    )
    print(f"Created zip file: {zip_file_path}")

    # Update log: zip creation complete
    pack_log["status"]["zip_file_creation"] = True
    pack_log["zip_file_path"] = zip_file_path
    with open(log_file_path, "w") as f:
        json.dump(pack_log, f, indent=4)
    print("Pack log updated: zip_file_creation = True")

    # Clean up temporary directory
    shutil.rmtree(tmp_upload_dir)
    print(f"Cleaned up temporary directory: {tmp_upload_dir}")

    # Update log: cleaning complete
    pack_log["status"]["tmp_upload_dir_cleaning"] = True
    with open(log_file_path, "w") as f:
        json.dump(pack_log, f, indent=4)
    print("Pack log updated: tmp_upload_dir_cleaning = True")
    print(f"All steps completed. Final log saved to: {log_file_path}")
