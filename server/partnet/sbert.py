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
sys.path.insert(0, SERVER_ROOT_DIR)
from foundation_models import get_sbert_model
import os
import json 
import numpy as np
import pickle
import torch
from tqdm import tqdm
import torch.nn.functional as F

if __name__ == "__main__":

    sbert_model = get_sbert_model()

    partnet_root_dir = PARTNET_ROOT_DIR
    partnet_desc_dir = os.path.join(partnet_root_dir, "desc")
    partnet_data_dir = os.path.join(partnet_root_dir, "dataset")

    id_list = sorted(os.listdir(partnet_data_dir))

    sbert_features = []
    valid_ids = []
    for id in tqdm(id_list):
        desc_path = os.path.join(partnet_desc_dir, f"{id}.json")
        if not os.path.exists(desc_path):
            continue

        with open(desc_path, "r") as f:
            desc = json.load(f)
        short_caption = desc["short_caption"]

        # Encode short_caption with sbert_model and append to sbert_features
        encoded_feature = sbert_model.encode(
            short_caption, convert_to_tensor=True, show_progress_bar=False
        )
        encoded_feature = encoded_feature.reshape(1, -1)
        encoded_feature = F.normalize(encoded_feature, p=2, dim=-1)
        sbert_features.append(encoded_feature)
        valid_ids.append(id)

    # save sbert_features
    valid_ids = np.array(valid_ids).astype(np.int32)
    sbert_features = torch.cat(sbert_features, dim=0).cpu().numpy()
    print(sbert_features.shape)
    print(valid_ids.shape)
    with open(os.path.join(partnet_root_dir, "sbert_features.pkl"), "wb") as f:
        pickle.dump({
            "text_features": sbert_features,
            "valid_ids": valid_ids,
        }, f)