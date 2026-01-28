from PIL import Image
import sys
import os
import numpy as np
import uuid
import shutil
import importlib.util
import random
from typing import List
import torch
taming_transformers_dir = "/home/hongchix/main/taming-transformers/"
root_dir_of_matfuse = os.path.dirname(os.path.dirname(__file__))

sys.path.insert(0, taming_transformers_dir)
sys.path.insert(0, os.path.join(root_dir_of_matfuse, "src"))

config_path = os.path.join(root_dir_of_matfuse, "src/configs/diffusion/matfuse-ldm-vq_f8.yaml")
ckpt_path = os.path.join(root_dir_of_matfuse, "ckpts/matfuse-full.ckpt")

# Use importlib to load the inference_helpers module from file path
inference_helpers_path = os.path.join(root_dir_of_matfuse, "src/utils/inference_helpers.py")
spec = importlib.util.spec_from_file_location("inference_helpers", inference_helpers_path)
inference_helpers = importlib.util.module_from_spec(spec)
sys.modules["inference_helpers"] = inference_helpers
spec.loader.exec_module(inference_helpers)

# Import the specific functions
run_generation = inference_helpers.run_generation
get_model = inference_helpers.get_model


matfuse_model = get_model(config_path, ckpt_path)

def pseudo_render_texture_map(albedo, roughness, normal_map, light_dir=np.array([0, 0, 1])):
    # Ensure albedo and normal_map are float for calculations
    albedo = albedo / 255.0
    roughness = roughness / 255.0
    normal_map = (normal_map / 255.0) * 2 - 1  # Normalize normal map to range [-1, 1]

    roughness = np.mean(roughness, axis=-1)

    # Normalize the light direction
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Calculate the dot product of light direction and the normal map
    # We use np.einsum for efficient per-pixel dot product
    dot_product = np.einsum('ijk,k->ij', normal_map, light_dir)
    dot_product = np.clip(dot_product, 0, 1)  # Clamp to [0, 1]

    # Calculate diffuse shading by applying roughness to the dot product
    diffuse = dot_product * (1 - roughness) + roughness

    # Combine albedo with diffuse shading
    rendered_map = albedo * diffuse[..., np.newaxis]

    # Clip values to [0, 1] and convert back to uint8 for image representation
    rendered_map = np.clip(rendered_map * 255, 0, 255).astype(np.uint8)

    return rendered_map


def generate_texture_map_from_prompt(prompt):
    global matfuse_model

    # save dir is a temporary directory at /tmp/matfuse_texture_map
    save_dir = os.path.join("/tmp/matfuse_texture_map", str(uuid.uuid4()))
    os.makedirs(save_dir, exist_ok=True)

    input_image_emb = None
    input_image_palette = None
    sketch = None

    num_samples = 1
    image_resolution = 512
    guidance_scale = 10.0
    ddim_steps = 50
    seed = random.randint(0, 1000000)
    eta = 0.0

    result_tex = run_generation(
        matfuse_model, input_image_emb, input_image_palette, sketch, prompt,
        num_samples, image_resolution, ddim_steps, seed, eta, guidance_scale,
        save_dir=save_dir
    )[-1]

    H, W = result_tex.shape[0] // 2, result_tex.shape[1] // 2

    albedo = result_tex[:H, :W]
    roughness = result_tex[:H, W:]
    normal_map = result_tex[H:, :W]

    # Image.fromarray(albedo).save(os.path.join(save_dir, "mat_generate_albedo.png"))
    # Image.fromarray(roughness).save(os.path.join(save_dir, "mat_generate_roughness.png"))
    # Image.fromarray(normal_map).save(os.path.join(save_dir, "mat_generate_normal_map.png"))

    rendering = pseudo_render_texture_map(albedo, roughness, normal_map)

    # delete the temporary directory
    shutil.rmtree(save_dir)

    return Image.fromarray(albedo)



def generate_texture_map_from_prompt_and_sketch(prompt, sketch: np.ndarray):
    global matfuse_model

    # save dir is a temporary directory at /tmp/matfuse_texture_map
    save_dir = os.path.join("/tmp/matfuse_texture_map", str(uuid.uuid4()))
    os.makedirs(save_dir, exist_ok=True)

    input_image_emb = None
    input_image_palette = None

    num_samples = 1
    image_resolution = 512
    guidance_scale = 10.0
    ddim_steps = 50
    seed = random.randint(0, 1000000)
    eta = 0.0

    result_tex = run_generation(
        matfuse_model, input_image_emb, input_image_palette, sketch, prompt,
        num_samples, image_resolution, ddim_steps, seed, eta, guidance_scale,
        save_dir=save_dir
    )[-1]

    H, W = result_tex.shape[0] // 2, result_tex.shape[1] // 2

    albedo = result_tex[:H, :W]
    roughness = result_tex[:H, W:]
    normal_map = result_tex[H:, :W]

    # Image.fromarray(albedo).save(os.path.join(save_dir, "mat_generate_albedo.png"))
    # Image.fromarray(roughness).save(os.path.join(save_dir, "mat_generate_roughness.png"))
    # Image.fromarray(normal_map).save(os.path.join(save_dir, "mat_generate_normal_map.png"))

    rendering = pseudo_render_texture_map(albedo, roughness, normal_map)

    # delete the temporary directory
    shutil.rmtree(save_dir)

    return Image.fromarray(rendering)


def generate_texture_map_from_prompt_and_sketch_and_image(prompt, sketch: np.ndarray, image: Image.Image):
    global matfuse_model

    # save dir is a temporary directory at /tmp/matfuse_texture_map
    save_dir = os.path.join("/tmp/matfuse_texture_map", str(uuid.uuid4()))
    os.makedirs(save_dir, exist_ok=True)

    input_image_emb = image
    input_image_palette = None

    num_samples = 1
    image_resolution = 512
    guidance_scale = 10.0
    ddim_steps = 50
    seed = random.randint(0, 1000000)
    eta = 0.0

    result_tex = run_generation(
        matfuse_model, input_image_emb, input_image_palette, sketch, prompt,
        num_samples, image_resolution, ddim_steps, seed, eta, guidance_scale,
        save_dir=save_dir
    )[-1]

    H, W = result_tex.shape[0] // 2, result_tex.shape[1] // 2

    albedo = result_tex[:H, :W]
    roughness = result_tex[:H, W:]
    normal_map = result_tex[H:, :W]

    # Image.fromarray(albedo).save(os.path.join(save_dir, "mat_generate_albedo.png"))
    # Image.fromarray(roughness).save(os.path.join(save_dir, "mat_generate_roughness.png"))
    # Image.fromarray(normal_map).save(os.path.join(save_dir, "mat_generate_normal_map.png"))

    rendering = pseudo_render_texture_map(albedo, roughness, normal_map)

    # delete the temporary directory
    shutil.rmtree(save_dir)

    return Image.fromarray(rendering)


def generate_texture_map_from_prompt_and_color(prompt, color):
    global matfuse_model

    # save dir is a temporary directory at /tmp/matfuse_texture_map
    save_dir = os.path.join("/tmp/matfuse_texture_map", str(uuid.uuid4()))
    os.makedirs(save_dir, exist_ok=True)

    input_image_emb = None
    sketch = None

    num_samples = 1
    image_resolution = 512
    guidance_scale = 10.0
    ddim_steps = 50
    seed = random.randint(0, 1000000)
    eta = 0.0
    input_image_palette = Image.fromarray((np.array(color) * 255.).clip(0, 255).reshape(1, -1, 3).astype(np.uint8).repeat(512, axis=0).repeat(512, axis=1))
    input_image_palette = input_image_palette.resize((512, 512))
    
    result_tex = run_generation(
        matfuse_model, input_image_emb, input_image_palette, sketch, prompt,
        num_samples, image_resolution, ddim_steps, seed, eta, guidance_scale,
        save_dir=save_dir
    )[-1]

    H, W = result_tex.shape[0] // 2, result_tex.shape[1] // 2

    albedo = result_tex[:H, :W]
    roughness = result_tex[:H, W:]
    normal_map = result_tex[H:, :W]

    # Image.fromarray(albedo).save(os.path.join(save_dir, "mat_generate_albedo.png"))
    # Image.fromarray(roughness).save(os.path.join(save_dir, "mat_generate_roughness.png"))
    # Image.fromarray(normal_map).save(os.path.join(save_dir, "mat_generate_normal_map.png"))

    rendering = pseudo_render_texture_map(albedo, roughness, normal_map)

    # delete the temporary directory
    shutil.rmtree(save_dir)

    return Image.fromarray(rendering)



def generate_texture_map_from_prompt_and_color_palette(prompt, color_palette):
    global matfuse_model

    # save dir is a temporary directory at /tmp/matfuse_texture_map
    save_dir = os.path.join("./tmp/matfuse_texture_map/", str(uuid.uuid4()))
    os.makedirs(save_dir, exist_ok=True)

    input_image_emb = None
    sketch = None

    num_samples = 1
    image_resolution = 512
    guidance_scale = 10.0
    ddim_steps = 50
    seed = random.randint(0, 1000000)
    eta = 0.0
    color_palette = torch.from_numpy(np.array(color_palette)).float().reshape(-1, 3)

    if color_palette.shape[0] > 5:
        color_palette = color_palette[:5]
    else:
        while color_palette.shape[0] < 5:
            color_palette = torch.cat([color_palette, color_palette + torch.randn_like(color_palette) * 0.05], dim=0)
        color_palette = color_palette[:5]

    color_palette = torch.clamp(color_palette, 0, 1)

    print(color_palette*255, color_palette.shape, file=sys.stderr)
    
    result_tex = run_generation(
        matfuse_model, input_image_emb, color_palette, sketch, prompt,
        num_samples, image_resolution, ddim_steps, seed, eta, guidance_scale,
        save_dir=save_dir, direct_palette=True
    )[-1]

    H, W = result_tex.shape[0] // 2, result_tex.shape[1] // 2

    albedo = result_tex[:H, :W]
    roughness = result_tex[:H, W:]
    normal_map = result_tex[H:, :W]

    # Image.fromarray(albedo).save(os.path.join(save_dir, "mat_generate_albedo.png"))
    # Image.fromarray(roughness).save(os.path.join(save_dir, "mat_generate_roughness.png"))
    # Image.fromarray(normal_map).save(os.path.join(save_dir, "mat_generate_normal_map.png"))

    rendering = pseudo_render_texture_map(albedo, roughness, normal_map)

    # delete the temporary directory
    shutil.rmtree(save_dir)

    return Image.fromarray(rendering)



def generate_texture_map_from_prompt_and_color_and_sketch(prompt, color, sketch: np.ndarray):
    global matfuse_model

    # save dir is a temporary directory at /tmp/matfuse_texture_map
    save_dir = os.path.join("/tmp/matfuse_texture_map", str(uuid.uuid4()))
    os.makedirs(save_dir, exist_ok=True)

    input_image_emb = None

    num_samples = 1
    image_resolution = 512
    guidance_scale = 10.0
    ddim_steps = 50
    seed = random.randint(0, 1000000)
    eta = 0.0
    input_image_palette = Image.fromarray((np.array(color) * 255.).clip(0, 255).reshape(1, 1, 3).astype(np.uint8).repeat(512, axis=0).repeat(512, axis=1))

    result_tex = run_generation(
        matfuse_model, input_image_emb, input_image_palette, sketch, prompt,
        num_samples, image_resolution, ddim_steps, seed, eta, guidance_scale,
        save_dir=save_dir
    )[-1]

    H, W = result_tex.shape[0] // 2, result_tex.shape[1] // 2

    albedo = result_tex[:H, :W]
    roughness = result_tex[:H, W:]
    normal_map = result_tex[H:, :W]

    # Image.fromarray(albedo).save(os.path.join(save_dir, "mat_generate_albedo.png"))
    # Image.fromarray(roughness).save(os.path.join(save_dir, "mat_generate_roughness.png"))
    # Image.fromarray(normal_map).save(os.path.join(save_dir, "mat_generate_normal_map.png"))

    rendering = pseudo_render_texture_map(albedo, roughness, normal_map)

    # delete the temporary directory
    shutil.rmtree(save_dir)

    return Image.fromarray(rendering)