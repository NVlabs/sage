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
from difflib import SequenceMatcher

import compress_json
import compress_pickle
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image

ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2023_09_23")

OBJATHOR_ASSETS_BASE_DIR = os.environ.get(
    "OBJATHOR_ASSETS_BASE_DIR", os.path.expanduser(f"~/.objathor-assets")
)

HOLODECK_BASE_DATA_DIR = os.path.join(OBJATHOR_ASSETS_BASE_DIR, "holodeck", ASSETS_VERSION)


class MaterialSelector:
    def __init__(self, clip_model, clip_preprocess, clip_tokenizer):
        materials = compress_json.load(
            os.path.join(HOLODECK_BASE_DATA_DIR, "materials/material-database.json")
        )
        self.selected_materials = (
            materials["Wall"] + materials["Wood"] + materials["Fabric"]
        )

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer

        self.load_features()

    def load_features(self):
        try:
            self.material_feature_clip = compress_pickle.load(
                os.path.join(
                    HOLODECK_BASE_DATA_DIR, "materials/material_feature_clip.pkl"
                )
            )
        except:
            print("Precompute image features for materials...")
            self.material_feature_clip = []
            for material in self.selected_materials:
                image = self.preprocess(
                    Image.open(
                        os.path.join(
                            HOLODECK_BASE_DATA_DIR, f"materials/images/{material}.png"
                        )
                    )
                ).unsqueeze(0)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                self.material_feature_clip.append(image_features)
            self.material_feature_clip = torch.vstack(self.material_feature_clip)
            compress_pickle.dump(
                self.material_feature_clip,
                os.path.join(
                    HOLODECK_BASE_DATA_DIR, "materials/material_feature_clip.pkl"
                ),
            )

        try:
            self.color_feature_clip = compress_pickle.load(
                os.path.join(HOLODECK_BASE_DATA_DIR, "materials/color_feature_clip.pkl")
            )
        except:
            print("Precompute text features for colors...")
            with torch.no_grad():
                self.color_feature_clip = self.clip_model.encode_text(
                    self.clip_tokenizer(self.colors)
                )
                self.color_feature_clip /= self.color_feature_clip.norm(
                    dim=-1, keepdim=True
                )

            compress_pickle.dump(
                self.color_feature_clip,
                os.path.join(
                    HOLODECK_BASE_DATA_DIR, "materials/color_feature_clip.pkl"
                ),
            )

    def match_material(self, queries, topk=5):
        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(
                self.clip_tokenizer(queries)
            )
            query_feature_clip /= query_feature_clip.norm(dim=-1, keepdim=True)

        clip_similarity = query_feature_clip @ self.material_feature_clip.T
        string_similarity = torch.tensor(
            [
                [
                    self.string_match(query, material)
                    for material in self.selected_materials
                ]
                for query in queries
            ]
        )

        joint_similarity = (
            string_similarity + clip_similarity
        )  # use visual embedding only seems to be better

        results = []
        scores = []
        for sim in joint_similarity:
            indices = torch.argsort(sim, descending=True)[:topk]
            results.append([self.selected_materials[ind] for ind in indices])
            scores.append([sim[ind] for ind in indices])
        return results, scores


    def string_match(self, a, b):
        return SequenceMatcher(None, a, b).ratio()
