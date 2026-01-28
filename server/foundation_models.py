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
from sentence_transformers import SentenceTransformer
import open_clip


def init_clip():
    # initialize CLIP
    print("loading clip model")
    (
        clip_model,
        _,
        clip_preprocess,
    ) = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k", device="cpu"
    )
    print("loaded clip model")
    print("loading clip tokenizer")
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    print("loaded clip tokenizer")
    return clip_model, clip_preprocess, clip_tokenizer

def init_sbert():
    # initialize sentence transformer
    print("loading sbert_model")
    sbert_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    print("loaded sbert_model")
    return sbert_model



clip_model, clip_preprocess, clip_tokenizer = init_clip()
sbert_model = init_sbert()


def get_clip_models():
    global clip_model, clip_preprocess, clip_tokenizer
    return clip_model, clip_preprocess, clip_tokenizer

def get_sbert_model():
    global sbert_model
    return sbert_model