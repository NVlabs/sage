import cv2 as cv
import extcolors
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from Pylette.src.color_extraction import k_means_extraction, median_cut_extraction
from Pylette import Palette
from torch.utils.data import Sampler

def pack_maps(maps: dict) -> torch.Tensor:
    return torch.cat(
        (maps["Diffuse"], maps["Normal"], maps["Roughness"], maps["Specular"]), 0
    )

def unpack_maps(maps: torch.Tensor) -> dict:
    maps = maps.cpu()
    maps = {
        "Diffuse": maps[:, :3],
        "Normal": maps[:, 3:6],
        "Roughness": maps[:, 6:9],
        "Specular": maps[:, 9:],
    }
    return maps


def to_pil(image):
    image = image / 2 + 0.5
    image = torch.clamp(image, 0, 1)

    return TF.to_pil_image(image.squeeze())


def make_plot_maps(x):
    grid_0 = torch.cat((x["Diffuse"], x["Normal"]), 2)
    grid_1 = torch.cat((x["Roughness"], x["Specular"]), 2)

    return torch.cat((grid_0, grid_1), 3)


def map_transform_func(x, load_size=256):
    x = TF.resize(x, load_size)
    x = TF.center_crop(x, load_size)
    x = TF.to_tensor(x)
    x = TF.normalize(x, 0.5, 0.5)
    return x


def extract_palette(render):
    colors, _ = extcolors.extract_from_image(
        render.resize((64, 64)), tolerance=20, limit=3
    )
    colors = [list(c[0]) for c in colors]
    if len(colors) < 3:  # Padding color palette
        colors += [colors[-1]] * (3 - len(colors))
    palette = torch.tensor(colors).flatten()
    palette = palette.unsqueeze(0)
    return palette / 255.0


def rescale_range(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-6)


def pylette_extract_colors_mod(
    image, palette_size=5, resize=True, mode="KM", sort_mode="luminance"
):
    """
    Extracts a set of 'palette_size' colors from the given image.
    :param image: path to Image file
    :param palette_size: number of colors to extract
    :param resize: whether to resize the image before processing, yielding faster results with lower quality
    :param mode: the color quantization algorithm to use. Currently supports K-Means (KM) and Median Cut (MC)
    :param sort_mode: sort colors by luminance, or by frequency
    :return: a list of the extracted colors
    """

    # open the image
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        img = TF.to_pil_image(image)
    elif isinstance(image, Image):
        pass
    else:
        raise NotImplementedError(f"Image type {type(image)} not supported")

    if resize:
        img = img.resize((256, 256))
    width, height = img.size
    arr = np.asarray(img)

    if mode == "KM":
        colors = k_means_extraction(arr, height, width, palette_size)
    elif mode == "MC":
        colors = median_cut_extraction(arr, height, width, palette_size)
    else:
        raise NotImplementedError("Extraction mode not implemented")

    if sort_mode == "luminance":
        colors.sort(key=lambda c: c.luminance, reverse=False)
    if sort_mode == "frequency":
        colors.sort(key=lambda c: c.freq, reverse=True)
    else:
        colors.sort(reverse=True)

    return Palette(colors)
