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
import sys 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR, PARTNET_ROOT_DIR
import os
import sapien
import numpy as np
from PIL import Image
from tqdm import tqdm
from vlm import call_vlm
from foundation_models import get_clip_models
import base64
from utils import extract_json_from_response
import json
import torch
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
except ImportError:
    print("UMAP not installed, will use t-SNE instead")
    umap = None
partnet_root_dir = PARTNET_ROOT_DIR

if __name__ == "__main__":
    partnet_data_dir = os.path.join(partnet_root_dir, "dataset")
    partnet_render_dir = os.path.join(partnet_root_dir, "render")
    partnet_desc_dir = os.path.join(partnet_root_dir, "desc")
    
    with open(os.path.join(partnet_root_dir, "clip_features.pkl"), "rb") as f:
        data = pickle.load(f)
    clip_features: np.ndarray = data["image_features"]
    valid_ids: np.ndarray = data["valid_ids"]

    print(clip_features.shape)
    print(valid_ids.shape)

    valid_ids = [str(id) for id in valid_ids]

    object_cats = []

    for id in tqdm(valid_ids):
        id_dir = os.path.join(partnet_data_dir, id)
        urdf_path = os.path.join(id_dir, "mobility.urdf")
        meta_path = os.path.join(id_dir, "meta.json")
        save_im_path = os.path.join(partnet_render_dir, f"{id}.png")

        with open(meta_path, "r") as f:
            meta = json.load(f)
        object_cat = meta["model_cat"]

        object_cats.append(object_cat)

    # Find unique object categories and create mapping
    unique_cats = list(set(object_cats))
    print(f"Found {len(unique_cats)} unique object categories: {unique_cats}")
    
    # Create category to index mapping
    cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
    cat_indices = [cat_to_idx[cat] for cat in object_cats]
    cat_indices = np.array(cat_indices)
    
    print(f"CLIP features shape: {clip_features.shape}")
    print(f"Number of categories: {len(unique_cats)}")
    
    # Apply PCA for initial dimensionality reduction
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=50)  # Reduce to 50 dimensions first
    clip_features_pca = pca.fit_transform(clip_features)
    print(f"PCA reduced features shape: {clip_features_pca.shape}")
    print(f"PCA explained variance ratio (first 10 components): {pca.explained_variance_ratio_[:10]}")
    
    # Apply UMAP or t-SNE for 2D visualization
    if umap is not None:
        print("Applying UMAP for 2D visualization...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        features_2d = reducer.fit_transform(clip_features_pca)
        method_name = "UMAP"
    else:
        print("Applying t-SNE for 2D visualization...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_2d = reducer.fit_transform(clip_features_pca)
        method_name = "t-SNE"
    
    print(f"2D features shape: {features_2d.shape}")
    
    # Create visualization with different colors for each category
    plt.figure(figsize=(15, 12))
    
    # Try different distinct color sets
    n_cats = len(unique_cats)
    
    # Option 1: Automatic selection based on number of categories (DEFAULT)
    if n_cats <= 10:
        # Use Set3 for up to 10 categories - very distinct colors
        colors = plt.cm.Set3(np.linspace(0, 1, n_cats))
        color_scheme = "Set3"
    elif n_cats <= 12:
        # Use Paired colormap for up to 12 categories
        colors = plt.cm.Paired(np.linspace(0, 1, n_cats))
        color_scheme = "Paired"
    elif n_cats <= 20:
        # Use tab20 for up to 20 categories
        colors = plt.cm.tab20(np.linspace(0, 1, n_cats))
        color_scheme = "tab20"
    else:
        # For more categories, use seaborn husl palette for maximum distinction
        colors = sns.color_palette("husl", n_cats)
        color_scheme = "husl"
    
    # Alternative color schemes - uncomment one to try:
    # colors = sns.color_palette("Set2", n_cats)  # Soft, pastel colors
    # colors = sns.color_palette("Dark2", n_cats)  # Dark, distinct colors
    # colors = sns.color_palette("bright", n_cats)  # Bright, vivid colors
    # colors = plt.cm.rainbow(np.linspace(0, 1, n_cats))  # Rainbow spectrum
    # colors = plt.cm.viridis(np.linspace(0, 1, n_cats))  # Viridis (good for colorblind)
    # colors = plt.cm.plasma(np.linspace(0, 1, n_cats))  # Plasma colors
    # colors = sns.color_palette("colorblind", n_cats)  # Colorblind-friendly
    
    print(f"Using {color_scheme} color scheme for {n_cats} categories")
    
    for i, cat in enumerate(unique_cats):
        mask = cat_indices == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=cat, alpha=0.7, s=30)
    
    plt.title(f'PartNet Objects - CLIP Features Visualization (PCA + {method_name}) - {color_scheme} colors')
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    
    # Create legend with smaller font and multiple columns
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the visualization
    vis_save_path = os.path.join(partnet_root_dir, f"clip_features_visualization_{method_name.lower()}.png")
    plt.savefig(vis_save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {vis_save_path}")
    

