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
from constants import MATFUSE_ROOT_DIR
import os
# Use importlib.util to import a function named generate_texture_map_from_prompt from the generate.py file in MATFUSE_ROOT_DIR
import importlib.util
import sys
import random
# Create the full path to the generate.py file
generate_py_path = os.path.join(MATFUSE_ROOT_DIR, "generate.py")

# Load the module from the file path
spec = importlib.util.spec_from_file_location("generate_module", generate_py_path)
generate_module = importlib.util.module_from_spec(spec)
sys.modules["generate_module"] = generate_module
spec.loader.exec_module(generate_module)

# Import the specific function
generate_texture_map_from_prompt = generate_module.generate_texture_map_from_prompt
generate_texture_map_from_prompt_and_sketch = generate_module.generate_texture_map_from_prompt_and_sketch
generate_texture_map_from_prompt_and_sketch_and_image = generate_module.generate_texture_map_from_prompt_and_sketch_and_image
generate_texture_map_from_prompt_and_color = generate_module.generate_texture_map_from_prompt_and_color
generate_texture_map_from_prompt_and_color_and_sketch = generate_module.generate_texture_map_from_prompt_and_color_and_sketch
generate_texture_map_from_prompt_and_color_palette = generate_module.generate_texture_map_from_prompt_and_color_palette

def material_generate_from_prompt(prompts):
    results = []
    for prompt in prompts:
        texture_map_pil = generate_texture_map_from_prompt(prompt)

        results.append(texture_map_pil)
    return results

# from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
# from diffusers.utils import load_image
# import numpy as np
# import torch

# import cv2
# from PIL import Image

# # initialize the models and pipeline
# controlnet_conditioning_scale = 0.5  # recommended for good generalization
# controlnet = ControlNetModel.from_pretrained(
#     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
# )
# vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
# pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
# )
# pipe.enable_model_cpu_offload()


# def generate_texture_map_from_prompt_and_sketch_controlnet(prompt, sketch):

#     sketch = Image.fromarray(sketch)

#     # Set a fixed seed for reproducible results
#     generator = torch.Generator().manual_seed(random.randint(0, 1000000))

#     # generate image with stable parameters for architectural materials
#     image = pipe(
#         prompt, 
#         image=sketch,
#         height=512,
#         width=512,
#         num_inference_steps=30,  # More steps for better quality and stability
#         guidance_scale=7.5,  # Higher guidance for better prompt adherence
#         controlnet_conditioning_scale=controlnet_conditioning_scale,
#         control_guidance_start=0.0,  # Apply control from beginning
#         control_guidance_end=0.8,  # Reduce control influence near end for natural results
#         generator=generator,  # Fixed seed for consistency
#         eta=0.0,  # Deterministic sampling for stability
#         negative_prompt="low quality, bad quality, dirty, smudged, stained, grimy, dusty, fingerprints, water spots, streaks, cloudy, foggy, cracked, broken, distorted, warped, blurry, scratched, damaged, weathered, aged, yellowed, tinted, reflective glare, harsh reflections, unrealistic, fantasy elements, cartoon, anime, abstract patterns, decorative ornaments, "
#     ).images[0]

#     return image