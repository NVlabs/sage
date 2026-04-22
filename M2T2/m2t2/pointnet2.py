# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Author: Wentao Yuan
'''
Network modules for PointNet++.
'''
import torch.nn as nn

from m2t2.pointnet2_modules import (
    PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG
)


class PointNet2Base(nn.Module):
    def __init__(self):
        super(PointNet2Base, self).__init__()
        self.SA_modules = nn.ModuleList()
        self.FP_modules = nn.ModuleList()

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = None
        if pc.shape[-1] > 3 and self.use_rgb:
            features = pc[..., 3:].transpose(1, 2).contiguous()
        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features, sample_ids = [xyz], [features], []
        for i in range(len(self.SA_modules)):
            li_xyz, _, li_features, sample_idx = self.SA_modules[i](
                l_xyz[i], l_features[i]
            )
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            if sample_idx[0] is not None:
                sample_ids.append(sample_idx[0])

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        l_features = {
            f'res{i}': feat for i, feat in enumerate(l_features)
            if feat is not None
        }
        l_xyz = {
            f'res{i}': xyz for i, xyz in enumerate(l_xyz) if xyz is not None
        }
        outputs = {
            'features': l_features,
            'context_pos': l_xyz,
            'sample_ids': sample_ids
        }
        return outputs


class PointNet2MSG(PointNet2Base):
    def __init__(
        self, num_points, downsample, radius,
        radius_mult, use_rgb=True, norm='BN'
    ):
        super(PointNet2MSG, self).__init__()

        self.use_rgb = use_rgb
        c_in = 3 if use_rgb else 0
        num_points = num_points // downsample
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[radius, radius * radius_mult],
                nsamples=[16, 32],
                mlps=[[c_in, 32, 32, 64], [c_in, 32, 32, 64]],
                norm=norm
            )
        )
        c_out_0 = 64 + 64
        radius = radius * radius_mult

        num_points = num_points // downsample
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[radius, radius * radius_mult],
                nsamples=[16, 32],
                mlps=[[c_out_0, 64, 64, 128], [c_out_0, 64, 64, 128]],
                norm=norm
            )
        )
        c_out_1 = 128 + 128
        radius = radius * radius_mult

        num_points = num_points // downsample
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[radius, radius * radius_mult],
                nsamples=[16, 32],
                mlps=[[c_out_1, 128, 128, 256], [c_out_1, 128, 128, 256]],
                norm=norm
            )
        )
        c_out_2 = 256 + 256
        radius = radius * radius_mult

        num_points = num_points // downsample
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[radius, radius * radius_mult],
                nsamples=[16, 32],
                mlps=[[c_out_2, 256, 256, 512], [c_out_2, 256, 256, 512]],
                norm=norm
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules.append(
            PointnetFPModule(mlp=[256 + c_in, 128, 128])
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[512 + c_out_0, 256, 256])
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[512 + c_out_1, 512, 512])
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512])
        )

        self.out_channels = {
            'res0': 128, 'res1': 256, 'res2': 512, 'res3': 512, 'res4': 1024
        }

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args['num_points'] = cfg.num_points
        args['downsample'] = cfg.downsample
        args['radius'] = cfg.radius
        args['radius_mult'] = cfg.radius_mult
        args['use_rgb'] = cfg.use_rgb
        return cls(**args)


class PointNet2MSGCls(PointNet2Base):
    def __init__(
        self, num_points, downsample, radius,
        radius_mult, use_rgb=True, norm='BN'
    ):
        super(PointNet2MSGCls, self).__init__()

        self.use_rgb = use_rgb
        c_in = 3 if use_rgb else 0
        num_points = num_points // downsample
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[radius, radius * radius_mult],
                nsamples=[16, 32],
                mlps=[[c_in, 32, 32, 64], [c_in, 32, 32, 64]],
                norm=norm
            )
        )
        c_out_0 = 64 + 64
        radius = radius * radius_mult

        num_points = num_points // downsample
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[radius, radius * radius_mult],
                nsamples=[16, 32],
                mlps=[[c_out_0, 64, 64, 128], [c_out_0, 64, 64, 128]],
                norm=norm
            )
        )
        c_out_1 = 128 + 128
        radius = radius * radius_mult

        num_points = num_points // downsample
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[radius, radius * radius_mult],
                nsamples=[16, 32],
                mlps=[[c_out_1, 128, 128, 256], [c_out_1, 128, 128, 256]],
                norm=norm
            )
        )
        c_out_2 = 256 + 256
        self.SA_modules.append(
            PointnetSAModule(mlp=[c_out_2, 256, 256, 512], norm=norm)
        )

        self.out_channels = {
            'res0': c_in, 'res1': 128, 'res2': 256, 'res3': 512, 'res4': 512
        }

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args['num_points'] = cfg.num_points
        args['downsample'] = cfg.downsample
        args['radius'] = cfg.radius
        args['radius_mult'] = cfg.radius_mult
        args['use_rgb'] = cfg.use_rgb
        return cls(**args)
