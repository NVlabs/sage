import torch
import torch.nn as nn
from torch.nn.functional import mse_loss as l2_loss

from ldm.modules.losses.rendering import GGXRenderer
from ldm.util import instantiate_from_config


class MaterialLoss(nn.Module):

    def __init__(self, imgloss):
        super().__init__()
        self.imgloss = instantiate_from_config(imgloss)

        self.renderer = GGXRenderer()

        self.discriminator = (
            self.imgloss.discriminator
        )  # needed for AutoencoderKL model

    def forward(self, codebook_loss, inputs, reconstructions, *args, **kwargs):
        # Split inputs into 3-channel images for loss (LPIPS), aggregate computed losses
        loss_out = None
        log_out = {}

        for i_c in range(0, 12, 3):
            inputs_c = inputs[:, i_c : i_c + 3, :, :]
            reconstructions_c = reconstructions[:, i_c : i_c + 3, :, :]

            loss_c, log_c = self.imgloss(
                codebook_loss, inputs_c, reconstructions_c, *args, **kwargs
            )

            if loss_out is None:
                loss_out = loss_c
                log_out = log_c
            else:
                loss_out += loss_c
                for k in log_c.keys():
                    log_out[k] += log_c[k]

        # Compute render loss (only when optimizing the reconstruction loss: optimizer_idx == 0)
        if args[0] == 0:
            inputs = inputs * 0.5 + 0.5
            reconstructions = reconstructions.clamp(-1, 1).clone()
            reconstructions = reconstructions * 0.5 + 0.5

            # # convert diffuse and specular from sRGB to linear
            inputs[:, :3] = inputs[:, :3] ** 2.2
            inputs[:, 9:12] = inputs[:, 9:12] ** 2.2

            rec_diff, rec_norm, rec_rough, rec_spec = reconstructions.chunk(4, dim=1)
            rec_diff = rec_diff**2.2
            rec_spec = rec_spec**2.2
            reconstructions = torch.cat(
                [rec_diff, rec_norm, rec_rough, rec_spec], dim=1
            )

            # convert to [-1, 1]
            inputs = inputs.permute(0, 2, 3, 1) * 2 - 1
            reconstructions = reconstructions.permute(0, 2, 3, 1) * 2 - 1

            # compute renderings
            rend_diff_in, rend_diff_rec = self.renderer.generateDiffuseRendering(
                1, 9, inputs, reconstructions
            )
            rend_spec_in, rend_spec_rec = self.renderer.generateDiffuseRendering(
                1, 9, inputs, reconstructions
            )

            # compute loss
            rec_loss_diff = l2_loss(
                rend_diff_in.contiguous(), rend_diff_rec.contiguous()
            )
            rec_loss_spec = l2_loss(
                rend_spec_in.contiguous(), rend_spec_rec.contiguous()
            )
            rec_loss = (rec_loss_diff + rec_loss_spec) / 2

            loss_out += rec_loss
            log_out[f'{kwargs["split"]}/total_loss'] += rec_loss.detach().mean()
            log_out[f'{kwargs["split"]}/rend_loss'] = rec_loss.detach().mean()

        return loss_out, log_out
