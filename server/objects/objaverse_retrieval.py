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
from typing import Dict, Any

import compress_json
import compress_pickle
import numpy as np
import torch
import torch.nn.functional as F
import gzip
import pickle
import trimesh
from scipy.spatial.transform import Rotation as R
from PIL import Image
from objects.object_attribute_inference import (
    infer_attributes_from_claude
)

def load_pkl_gz(file_path):
    """Load a .pkl.gz file."""
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

def extract_vertices(vertices_data):
    """Extract vertices into a NumPy array from the given data format."""
    return np.array([[v['x'], v['y'], v['z']] for v in vertices_data])

def extract_vts(vts_data):
    """Extract vertices into a NumPy array from the given data format."""
    return np.array([[v['x'], v['y']] for v in vts_data])

def create_faces(triangles_data):
    """Create faces (triangles) from the given indices."""
    return triangles_data.reshape(-1, 3)

ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2023_09_23")

OBJATHOR_ASSETS_BASE_DIR = os.environ.get(
    "OBJATHOR_ASSETS_BASE_DIR", os.path.expanduser(f"~/.objathor-assets")
)

OBJATHOR_VERSIONED_DIR = os.path.join(OBJATHOR_ASSETS_BASE_DIR, ASSETS_VERSION)

OBJATHOR_FEATURES_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "features")
OBJATHOR_ANNOTATIONS_PATH = os.path.join(OBJATHOR_VERSIONED_DIR, "annotations.json.gz")
OBJATHOR_ASSETS_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "assets")


def get_asset_metadata(obj_data: Dict[str, Any]):
    if "assetMetadata" in obj_data:
        return obj_data["assetMetadata"]
    elif "thor_metadata" in obj_data:
        return obj_data["thor_metadata"]["assetMetadata"]
    else:
        raise ValueError("Can not find assetMetadata in obj_data")


def get_bbox_dims(obj_data: Dict[str, Any]):
    am = get_asset_metadata(obj_data)

    bbox_info = am["boundingBox"]

    if "x" in bbox_info:
        return bbox_info

    if "size" in bbox_info:
        return bbox_info["size"]

    mins = bbox_info["min"]
    maxs = bbox_info["max"]

    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}


class ObjathorRetriever:
    def __init__(
        self,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
        sbert_model,
        retrieval_threshold,
    ):
        
        objathor_annotations = compress_json.load(OBJATHOR_ANNOTATIONS_PATH)
        self.database = {**objathor_annotations}
        
        objathor_clip_features_dict = compress_pickle.load(
            os.path.join(OBJATHOR_FEATURES_DIR, f"clip_features.pkl")
        )  # clip features
        objathor_sbert_features_dict = compress_pickle.load(
            os.path.join(OBJATHOR_FEATURES_DIR, f"sbert_features.pkl")
        )  # sbert features
        assert (
            objathor_clip_features_dict["uids"] == objathor_sbert_features_dict["uids"]
        )

        objathor_uids = objathor_clip_features_dict["uids"]
        objathor_clip_features = objathor_clip_features_dict["img_features"].astype(
            np.float32
        )
        objathor_sbert_features = objathor_sbert_features_dict["text_features"].astype(
            np.float32
        )
        
        self.clip_features = torch.from_numpy(
            objathor_clip_features
        )
        self.clip_features = F.normalize(self.clip_features, p=2, dim=-1)

        self.sbert_features = torch.from_numpy(
            objathor_sbert_features
        )
        
        self.asset_ids = objathor_uids
            
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.sbert_model = sbert_model

        self.retrieval_threshold = retrieval_threshold

        self.use_text = True

    def retrieve(self, queries, threshold=28, max_num_candidates=10):
        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(
                self.clip_tokenizer(queries)
            )

            query_feature_clip = F.normalize(query_feature_clip, p=2, dim=-1)

        clip_similarities = 100 * torch.einsum(
            "ij, lkj -> ilk", query_feature_clip, self.clip_features
        )
        clip_similarities = torch.max(clip_similarities, dim=-1).values

        query_feature_sbert = self.sbert_model.encode(
            queries, convert_to_tensor=True, show_progress_bar=False
        )
        sbert_similarities = query_feature_sbert @ self.sbert_features.T

        if self.use_text:
            similarities = clip_similarities + sbert_similarities
        else:
            similarities = clip_similarities

        threshold_indices = torch.where(clip_similarities > threshold)

        unsorted_results = []
        for query_index, asset_index in zip(*threshold_indices):
            score = similarities[query_index, asset_index].item()
            unsorted_results.append((self.asset_ids[asset_index], score))

        # Sorting the results in descending order by score
        results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)

        return results[:max_num_candidates]

    def compute_size_difference(self, target_size, candidates):
        candidate_sizes = []
        for uid, _ in candidates:
            size = get_bbox_dims(self.database[uid])
            size_list = [size["x"] * 100, size["y"] * 100, size["z"] * 100]
            size_list.sort()
            candidate_sizes.append(size_list)

        candidate_sizes = torch.tensor(candidate_sizes)

        target_size_list = list(target_size)
        target_size_list.sort()
        target_size = torch.tensor(target_size_list)

        size_difference = abs(candidate_sizes - target_size).mean(axis=1) / 100
        size_difference = size_difference.tolist()

        candidates_with_size_difference = []
        for i, (uid, score) in enumerate(candidates):
            candidates_with_size_difference.append(
                (uid, score - size_difference[i] * 10)
            )

        # sort the candidates by score
        candidates_with_size_difference = sorted(
            candidates_with_size_difference, key=lambda x: x[1], reverse=True
        )

        return candidates_with_size_difference


    def load_object(self, asset_id, infer_attributes=True, caption=None):
        

        asset_path = os.path.join(OBJATHOR_ASSETS_DIR, asset_id, asset_id + ".pkl.gz")
        data = load_pkl_gz(asset_path)

        # Extracting vertices and triangles (faces) from the data
        vertices = np.array(data['vertices'])
        triangles = np.array(data['triangles'])
        vts = np.array(data['uvs'])

        vertices = extract_vertices(vertices)
        triangles = create_faces(triangles)

        vts = extract_vts(vts)
        fts = triangles

        albedo_name = os.path.basename(data['albedoTexturePath'])
        input_file_dir = os.path.dirname(asset_path)
        albedo_path = os.path.join(input_file_dir, albedo_name)
        albedo = np.array(Image.open(albedo_path)).astype(np.float32) / 255.0

        rotation_matrix = R.from_euler('y', data['yRotOffset'], degrees=True).as_matrix()[:3, :3]
        vertices = vertices @ rotation_matrix.T

        transformed_vertices = vertices.copy()
        transformed_vertices[:, 1] = vertices[:, 2]   # new y = old z
        transformed_vertices[:, 0] = -vertices[:, 0]   # new -x = old x
        transformed_vertices[:, 2] = vertices[:, 1]   # new z = old y

        transformed_vertices[:, 0] = transformed_vertices[:, 0] - 0.5 * (transformed_vertices[:, 0].max() + transformed_vertices[:, 0].min())
        transformed_vertices[:, 1] = transformed_vertices[:, 1] - 0.5 * (transformed_vertices[:, 1].max() + transformed_vertices[:, 1].min())
        transformed_vertices[:, 2] = transformed_vertices[:, 2] - transformed_vertices[:, 2].min()
        # transformed_vertices[:, 2] = transformed_vertices[:, 2] + 0.001


        # Create and return trimesh object
        mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=triangles)
        
        mesh_dict = {
            "mesh": mesh,
            "tex_coords": {
                "vts": vts,
                "fts": fts.copy(),
            },
            "texture": albedo
        }
    
        if infer_attributes:

            # scale_factor = infer_scale_from_reason1(mesh_dict, caption=caption)
            object_attributes = infer_attributes_from_claude(mesh_dict, caption=caption)

            mesh_dict["object_attributes"] = object_attributes

            scale_factor = object_attributes["height"] / transformed_vertices[:, 2].max()

            mesh_dict["mesh"].vertices = mesh_dict["mesh"].vertices * scale_factor

        mesh_dict["mesh"].vertices[:, 2] = mesh_dict["mesh"].vertices[:, 2] + 0.001

        return mesh_dict

        
        






        