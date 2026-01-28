import torch
import torch.nn as nn

from ldm.util import instantiate_from_config


class ZeroEmbedder(nn.Module):
    """
    Dummy embedder that returns a zero tensor, to exclude conditions without breaking the code
    """

    def __init__(self, cond_type, out_ch, latent_size):
        super().__init__()
        self.cond_type = cond_type
        self.out_ch = out_ch
        self.latent_size = latent_size

    def forward(self, x):
        bs = x.shape[0]
        if self.cond_type == "local":
            shape = [bs, self.out_ch, self.latent_size, self.latent_size]
        if self.cond_type == "global":
            shape = [bs, self.out_ch]
        return torch.zeros(shape, device=x.device)


class PaletteEncoder(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, n_colors=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hid_ch),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(hid_ch * n_colors, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiConditionEncoder(nn.Module):
    def __init__(
        self,
        image_embed_config=None,
        text_embed_config=None,
        binary_encoder_config=None,
        palette_proj_config=None,
    ):
        super().__init__()
        print(f"Conditional model: Multiconditional")
        self.image_embed = instantiate_from_config(image_embed_config)
        self.text_embed = instantiate_from_config(text_embed_config)
        self.sketch_encoder = instantiate_from_config(binary_encoder_config)
        self.palette_encoder = instantiate_from_config(palette_proj_config)

        self.encoders = {
            "image_embed": self.image_embed,
            "text": self.text_embed,
            "sketch": self.sketch_encoder,
            "palette": self.palette_encoder,
        }

        self.keys = [
            k
            for k in self.encoders.keys()
            if not isinstance(self.encoders[k], ZeroEmbedder)
        ]

    def parameters(self, trainable_only=False):
        for param in self._get_params(trainable_only):
            yield param

    def _get_params(self, trainable_only=False):
        params = list(self.sketch_encoder.parameters()) + list(
            self.palette_encoder.parameters()
        )
        if not trainable_only:
            params += list(self.text_embed.parameters())
            params += list(self.image_embed.parameters())
        return params

    def forward(self, x):
        with torch.no_grad():
            image_embed = self.image_embed(x["image_embed"])
            text_embed = self.text_embed(x["text"]).unsqueeze(1)
        sketch = self.sketch_encoder(x["sketch"])
        palette = self.palette_encoder(x["palette"]).unsqueeze(1)

        c_local = sketch
        c_global = torch.cat([image_embed, text_embed, palette], dim=1)

        return {"c_crossattn": c_global, "c_concat": c_local}
