import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from Pylette import extract_colors



class MatFuseDataset(Dataset):
    def __init__(self, data_root, size=256, output_names=["diffuse"]):
        self.data_root = Path(data_root)

        self.output_names = output_names

        self.materials = [
            {"name": x.parent.stem, "folder": x.parent}
            for x in self.data_root.glob("**/diffuse.png")
        ]

        self._length = len(self.materials)

        self.size = size

        self._prng = np.random.RandomState()

    def __len__(self):
        return self._length

    def make_placeholder_map(self, curr_map):
        image = None
        if curr_map in ["render", "basecolor", "diffuse", "specular"]:
            image = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        elif curr_map in ["normal"]:
            image = Image.new("RGB", (self.size, self.size), (127, 127, 255))
        elif curr_map in ["height", "metallic"]:
            image = Image.new("L", (self.size, self.size), (0))
        elif curr_map in ["roughness", "opacity"]:
            image = Image.new("L", (self.size, self.size), (255))
        else:
            raise NotImplementedError(
                "MatSynth.__get__item__(): No placeholder implemented for map: "
                + curr_map
            )
        return image

    def process_image(self, src, name):
        if not src.exists():
            image = self.make_placeholder_map(name)
            image = TF.to_tensor(image)
        else:
            image = Image.open(src).convert("RGB")
            image = TF.resize(image, self.size, antialias=True)
            image = TF.to_tensor(image)
            
        image = image * 2.0 - 1.0
        image = image.clamp(-1.0, 1.0)

        image = rearrange(image, "c h w -> h w c")
        return image

    def __getitem__(self, i):
        # load material
        material = self.materials[i]
        folder = material["folder"]

        example = {}
        maps = {}

        # load metadata
        metadata = json.load(open(folder / "metadata.json"))

        # load description
        example["text"] = ", ".join(metadata["tags"])

        # load render
        render_src = random.choice(list(folder.glob("renders/*.png")))
        image = self.process_image(render_src, "render")
        example["image_embed"] = image

        # load maps
        for curr_map in self.output_names:
            src = folder / (curr_map + ".png")
            image = self.process_image(src, curr_map)
            maps[curr_map] = image

        # Parse color palette
        palette = metadata.get("palette", None)
        if not palette:
            palette = extract_colors(str(render_src), palette_size=5, sort_mode="frequency")
            palette = [c.rgb for c in palette.colors]
        # if palette is not None:
        example["palette"] = self.to_rgb(palette)

        example["sketch"] = self.process_sketch(folder)

        # prepare maps
        example["maps"] = maps
        example["packed"] = torch.cat([maps[x] for x in self.output_names], dim=-1)

        return example

    def process_sketch(self, folder):
        if (folder / "sketch.png").exists():
            sketch = Image.open(folder / "sketch.png").convert("L")
            sketch = sketch.resize([self.size, self.size])
            sketch = TF.to_tensor(sketch)
            sketch = rearrange(sketch, "c h w -> h w c")
            return sketch.clamp(0, 1).float()
        else:
            return None

    def to_rgb(self, palette):
        rgb_palette = []
        for color in palette:
            if isinstance(color, str) and color[0] == "#":
                color = color[1:]
                rgb = [int(hex[i : i + 2], 16) for i in (0, 2, 4)]
            elif (isinstance(color, list) or isinstance(color, tuple)) and len(color) == 3: # RGB format
                rgb = color
            else:
                raise ValueError(f"Color format not recognized for {color}")
            rgb_palette.append(rgb)
        return torch.tensor(rgb_palette, dtype=torch.float32) / 255.0


if __name__ == "__main__":
    dset = MatFuseDataset(
        data_root="sample_materials",
        output_names=["diffuse", "normal", "specular", "roughness"],
    )

    for i in range(5):
        item = dset[i]
        TF.to_pil_image(item["image"].permute(2, 0, 1) * 0.5 + 0.5).show()
        TF.to_pil_image(item["maps"]["basecolor"].permute(2, 0, 1) * 0.5 + 0.5).show()
        print(item.keys())
