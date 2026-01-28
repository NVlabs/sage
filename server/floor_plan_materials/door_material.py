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
import compress_json
import compress_pickle
import torch
from PIL import Image
from tqdm import tqdm

ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2023_09_23")

OBJATHOR_ASSETS_BASE_DIR = os.environ.get(
    "OBJATHOR_ASSETS_BASE_DIR", os.path.expanduser(f"~/.objathor-assets")
)

HOLODECK_BASE_DATA_DIR = os.path.join(OBJATHOR_ASSETS_BASE_DIR, "holodeck", ASSETS_VERSION)


class DoorMaterialSelector:
    def __init__(self, clip_model, clip_preprocess, clip_tokenizer):

        door_data_all = compress_json.load(
            os.path.join(HOLODECK_BASE_DATA_DIR, "doors/door-database.json")
        )
        door_data_single_doorway_ids = [key for key, value in door_data_all.items() if value["size"] == "single" and value["type"] == "doorway"]
        door_data_single_doorframe_ids = [key for key, value in door_data_all.items() if value["size"] == "single" and value["type"] == "doorframe"]

        self.door_data_single_doorway_ids = sorted(door_data_single_doorway_ids)
        self.door_data_single_doorframe_ids = sorted(door_data_single_doorframe_ids)

        self.door_ids = self.get_doors()
        self.create_door_textures()

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer

        self.load_features()


    def get_doors(self):
        door_ids = []
        door_image_paths = []
        for doorway_id, doorframe_id in zip(self.door_data_single_doorway_ids, self.door_data_single_doorframe_ids):
            
            door_image_save_path = os.path.join(HOLODECK_BASE_DATA_DIR, "doors/images", f"{doorway_id.replace('Doorway_', 'Door_')}.png")
            if not os.path.exists(door_image_save_path):
                doorway_image_path = os.path.join(HOLODECK_BASE_DATA_DIR, "doors/images", f"{doorway_id}.png")
                doorframe_image_path = os.path.join(HOLODECK_BASE_DATA_DIR, "doors/images", f"{doorframe_id}.png")
                doorway_image = np.array(Image.open(doorway_image_path)).astype(np.float32) / 255.0
                doorframe_image = np.array(Image.open(doorframe_image_path)).astype(np.float32) / 255.0

                doorway_image[doorframe_image[:, :, -1] > 0] = 0
                Image.fromarray((doorway_image * 255).astype(np.uint8)).save(door_image_save_path)

            door_ids.append(doorway_id.replace('Doorway_', 'Door_'))
            door_image_paths.append(door_image_save_path)

        return door_ids

    def create_door_textures(self):
        for door_id, door_frame_id in zip(self.door_ids, self.door_data_single_doorframe_ids):
            texture_image_save_path = os.path.join(HOLODECK_BASE_DATA_DIR, "doors/textures", f"{door_id}_texture.png")
            tex_coords_save_path = os.path.join(HOLODECK_BASE_DATA_DIR, "doors/textures", f"{door_id}_tex_coords.pkl")

            if not os.path.exists(texture_image_save_path) or not os.path.exists(tex_coords_save_path):
                door_image_path = os.path.join(HOLODECK_BASE_DATA_DIR, "doors/images", f"{door_id}.png")
                door_frame_image_path = os.path.join(HOLODECK_BASE_DATA_DIR, "doors/images", f"{door_frame_id}.png")
                door_image = np.array(Image.open(door_image_path)).astype(np.float32) / 255.0
                door_frame_image = np.array(Image.open(door_frame_image_path)).astype(np.float32) / 255.0
                door_edge_rgb_value = door_frame_image[door_frame_image[:, :, -1] > 0].reshape(-1, 4)[:, :3].mean(axis=0).reshape(3)

                # Create texture image and texture coordinates
                import trimesh
                
                # Get door image dimensions and extract the rectangular door area using alpha channel
                door_height, door_width = door_image.shape[:2]
                door_alpha = door_image[:, :, -1]  # Alpha channel
                
                # Find the bounding box of the door using the alpha channel
                # Get rows and columns where alpha > 0
                rows = np.any(door_alpha > 0, axis=1)
                cols = np.any(door_alpha > 0, axis=0)
                
                # Find the bounds of the door rectangle
                if np.any(rows) and np.any(cols):
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]
                    
                    # Extract the door rectangle
                    door_rect = door_image[y_min:y_max+1, x_min:x_max+1, :3]  # RGB only
                    door_rect_height, door_rect_width = door_rect.shape[:2]
                else:
                    # Fallback: use the entire image
                    door_rect = door_image[:, :, :3]
                    door_rect_height, door_rect_width = door_height, door_width
                    y_min, y_max = 0, door_height - 1
                    x_min, x_max = 0, door_width - 1
                
                # Create a texture image that combines the door face and edge colors
                # Layout: [door_face | edge_color_strip]
                edge_strip_width = 32  # pixels for the edge color strip
                total_width = door_rect_width + edge_strip_width
                
                # Create the texture image
                door_texture_image = np.zeros((door_rect_height, total_width, 3), dtype=np.float32)
                
                # Place the door rectangle on the left side
                door_texture_image[:, :door_rect_width, :] = door_rect
                
                # Fill the edge strip with the edge color
                door_texture_image[:, door_rect_width:, :] = door_edge_rgb_value
                
                # Create texture coordinates for a box mesh
                # Create a temporary box to understand the face structure
                temp_box = trimesh.creation.box([1, 1, 1])  # Unit box for reference
                
                # Get face normals
                face_normals = temp_box.face_normals
                
                # Create unique texture coordinates for each face
                unique_vts = []
                unique_fts = []
                
                for face_idx, (face, normal) in enumerate(zip(temp_box.faces, face_normals)):
                    # Get vertices for this face
                    face_vertices = temp_box.vertices[face]
                    
                    # Determine texture coordinates based on face orientation
                    if abs(normal[1]) > 0.9:  # Front or back face (Y-axis - thickness direction)
                        # Map to door image area
                        # For door faces, we use X (width) and Z (height) coordinates
                        # Get the face bounds in local coordinates
                        min_x, max_x = face_vertices[:, 0].min(), face_vertices[:, 0].max()
                        min_z, max_z = face_vertices[:, 2].min(), face_vertices[:, 2].max()
                        
                        # Map face vertices to door rectangle UV coordinates
                        # X direction (width) maps to U, Z direction (height) maps to V
                        tex_coords = np.array([
                            [(face_vertices[i, 0] - min_x) / (max_x - min_x) * door_rect_width / total_width,
                             (face_vertices[i, 2] - min_z) / (max_z - min_z)]
                            for i in range(3)
                        ])
                    else:  # Side faces (X, Z axis normals)
                        # Map to edge color area (center of the edge strip)
                        edge_center_u = (door_rect_width + edge_strip_width/2) / total_width
                        tex_coords = np.array([
                            [edge_center_u, 0.5],
                            [edge_center_u, 0.5],
                            [edge_center_u, 0.5]
                        ])
                    
                    # Add to unique arrays
                    start_idx = len(unique_vts)
                    unique_vts.extend(tex_coords)
                    unique_fts.append([start_idx, start_idx + 1, start_idx + 2])
                
                # Convert to numpy arrays
                vts = np.array(unique_vts, dtype=np.float32)
                fts = np.array(unique_fts, dtype=np.int32)

                # save the texture image and tex coords
                tex_coords_dict = {
                    "vts": vts, # np array of shape (N, 2)
                    "fts": fts, # np array of shape (M, 3)
                }
                compress_pickle.dump(tex_coords_dict, tex_coords_save_path)
                Image.fromarray((door_texture_image * 255).astype(np.uint8)).save(texture_image_save_path)





    def load_features(self):
        try:
            self.door_feature_clip = compress_pickle.load(
                os.path.join(HOLODECK_BASE_DATA_DIR, "doors/single_door_feature_clip.pkl")
            )
        except:
            print("Precompute image features for doors...")
            self.door_feature_clip = []
            for door_id in tqdm(self.door_ids):
                image = self.clip_preprocess(
                    Image.open(
                        os.path.join(
                            HOLODECK_BASE_DATA_DIR, f"doors/images/{door_id}.png"
                        )
                    )
                ).unsqueeze(0)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                self.door_feature_clip.append(image_features)
            self.door_feature_clip = torch.vstack(self.door_feature_clip)
            compress_pickle.dump(
                self.door_feature_clip,
                os.path.join(HOLODECK_BASE_DATA_DIR, "doors/single_door_feature_clip.pkl"),
            )

    def select_door(self, query):
        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(
                self.clip_tokenizer([query])
            )
            query_feature_clip /= query_feature_clip.norm(dim=-1, keepdim=True)

        clip_similarity = query_feature_clip @ self.door_feature_clip.T
        sorted_indices = torch.argsort(clip_similarity, descending=True)[0]

        return sorted_indices[0]
    
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from foundation_models import get_clip_models
    clip_model, clip_preprocess, clip_tokenizer = get_clip_models()
    door_material_selector = DoorMaterialSelector(clip_model, clip_preprocess, clip_tokenizer)