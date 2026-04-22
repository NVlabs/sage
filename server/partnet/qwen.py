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
import base64
from utils import extract_json_from_response
import json
partnet_root_dir = PARTNET_ROOT_DIR

if __name__ == "__main__":
    partnet_data_dir = os.path.join(partnet_root_dir, "dataset")
    partnet_render_dir = os.path.join(partnet_root_dir, "render")
    partnet_desc_dir = os.path.join(partnet_root_dir, "desc")
    os.makedirs(partnet_render_dir, exist_ok=True)
    os.makedirs(partnet_desc_dir, exist_ok=True)

    id_list = sorted(os.listdir(partnet_data_dir))
    for id in tqdm(id_list):
        id_dir = os.path.join(partnet_data_dir, id)
        urdf_path = os.path.join(id_dir, "mobility.urdf")
        meta_path = os.path.join(id_dir, "meta.json")
        save_im_path = os.path.join(partnet_render_dir, f"{id}.png")
        if not os.path.exists(save_im_path):
            continue

        if os.path.exists(os.path.join(partnet_desc_dir, f"{id}.json")):
            continue

        with open(meta_path, 'r') as f:
            object_cat = json.load(f)["model_cat"]

        type_prompt = f"""
You are a professional 3D artist and object analyst who can provide accurate assessments of 3D models. 

GIVEN TYPE: {object_cat}

TASK: Analyze the 3D object in the rendered image and provide a comprehensive assessment. Pay special attention to whether the object semantically aligns with the given caption.

ANALYSIS REQUIREMENTS:
- For height/weight: Use your knowledge of typical object dimensions and materials to provide realistic estimates
- Be specific and accurate in your descriptions
- If no caption was provided, focus on accurate object identification and measurement

Please provide your analysis in the following structured JSON format:

{{
    "long_caption": "A model of {object_cat}, detailed description physical characteristic and detailed description of what you see in the image, including design, color, style, materials, and finish",
    "short_caption": "A model of {object_cat}, brief description physical characteristic",
    "given_caption": "{object_cat}",
    "name": "specific object name",
    "height": "height in meters (only numerical value, no unit)",
    "weight": "weight in kilograms (only numerical value, no unit)",
    "scale_unit": "meter",
    "weight_unit": "kilogram",
    "explanation": "One paragraph explaining your reasoning for the height and weight estimates, including any assumptions made"
}}

Please respond with only the JSON object, no additional text.
"""

        with open(save_im_path, 'rb') as image_file:
            image_data = image_file.read()

        while True:

            try:
        
                response = call_vlm(
                    vlm_type="qwen",
                    model="claude-sonnet-4-20250514",
                    max_tokens=3000,
                    temperature=0.2,  # Lower temperature for corrections
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": type_prompt
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": base64.b64encode(image_data).decode('utf-8')
                                    }
                                }
                            ]
                        }
                    ]
                )

                response_text = response.content[0].text.strip()

                type_data = extract_json_from_response(response_text)
                type_data = json.loads(type_data)
                break
            
            except Exception as e:
                print(e)
                continue

        

        with open(os.path.join(partnet_desc_dir, f"{id}.json"), "w") as f:
            json.dump(type_data, f, indent=4)
