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
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from foundation_models import get_sbert_model, get_clip_models


class PartNetRetrieval:
    def __init__(self, partnet_root_dir):
        self.partnet_root_dir = partnet_root_dir
        self.partnet_data_dir = os.path.join(partnet_root_dir, "dataset")

        with open(os.path.join(partnet_root_dir, "sbert_features.pkl"), "rb") as f:
            sbert_data = pickle.load(f)

        with open(os.path.join(partnet_root_dir, "clip_features.pkl"), "rb") as f:
            clip_data = pickle.load(f)

        self.sbert_features = sbert_data["text_features"]
        self.clip_features = clip_data["image_features"]
        self.valid_ids = clip_data["valid_ids"]

        self.sbert_features = torch.from_numpy(self.sbert_features)
        self.clip_features = torch.from_numpy(self.clip_features)


        self.sbert_model = get_sbert_model()
        self.clip_model, self.clip_preprocess, self.clip_tokenizer = get_clip_models()

    def retrieve(self, query_text, topk=10, threshold=0.):
        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(
                self.clip_tokenizer([query_text])
            )
            query_feature_clip = F.normalize(query_feature_clip, p=2, dim=-1)
            clip_similarities = query_feature_clip @ self.clip_features.T
            clip_similarities = clip_similarities.reshape(-1)

            query_feature_sbert = self.sbert_model.encode(
                [query_text], convert_to_tensor=True, show_progress_bar=False
            )
            sbert_similarities = query_feature_sbert @ self.sbert_features.T
            sbert_similarities = sbert_similarities.reshape(-1)

            similarities = clip_similarities + sbert_similarities
            accept = clip_similarities > threshold

            score = similarities[accept]
            asset_ids = self.valid_ids[accept]

            # Sorting the results in descending order by score
            results = sorted(zip(asset_ids, score), key=lambda x: x[1], reverse=True)

            return results[:topk]


if __name__ == "__main__":
    partnet_root_dir = PARTNET_ROOT_DIR
    partnet_data_dir = os.path.join(partnet_root_dir, "dataset")
    partnet_renders_dir = os.path.join(partnet_root_dir, "render")

    retrieval = PartNetRetrieval(partnet_root_dir)
    # results = retrieval.retrieve("A model of laptop computer with keyboard", topk=10, threshold=0.)
    results = retrieval.retrieve("A model of cabinet storage with handles", topk=10, threshold=0.)

    # Load and concatenate all render images
    images = []
    valid_ids = []
    
    for id_tuple in results:
        # Extract ID from tuple (id, score)
        asset_id = id_tuple[0] if isinstance(id_tuple, tuple) else id_tuple
        render_path = os.path.join(partnet_renders_dir, f"{asset_id}.png")
        
        if os.path.exists(render_path):
            try:
                img = Image.open(render_path)
                images.append(img)
                valid_ids.append(asset_id)
                print(f"Loaded image for ID: {asset_id}")
            except Exception as e:
                print(f"Failed to load image for ID {asset_id}: {e}")
        else:
            print(f"Render image not found: {render_path}")
    
    if images:
        # Calculate grid dimensions (prefer square-ish layout)
        num_images = len(images)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        
        # Get image dimensions (assume all images are the same size)
        img_width, img_height = images[0].size
        
        # Create the concatenated image
        concat_width = cols * img_width
        concat_height = rows * img_height
        concat_image = Image.new('RGB', (concat_width, concat_height), color='white')
        
        # Paste images into grid
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            x = col * img_width
            y = row * img_height
            concat_image.paste(img, (x, y))
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), "test")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the concatenated image
        output_path = os.path.join(output_dir, "retrieval_results.png")
        concat_image.save(output_path)
        print(f"Saved concatenated image with {len(images)} images to: {output_path}")
        
        # Print the IDs for reference
        print(f"Image contains renders for IDs: {valid_ids}")
    else:
        print("No valid render images found to concatenate")
