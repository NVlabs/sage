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
Modules for PointNet++.
'''
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from m2t2.pointnet2_utils import (
    QueryAndGroup,
    GroupAll,
    furthest_point_sample,
    gather_operation,
    three_nn,
    three_interpolate
)


def _get_norm(norm, dim):
    if norm == 'BN':
        return nn.BatchNorm2d(dim)
    if norm == 'GN':
        return nn.GroupNorm(16, dim)


def build_shared_mlp(mlp_spec: List[int], norm: str):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(nn.Conv2d(
            mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=norm == ''
        ))
        norm_layer = _get_norm(norm, mlp_spec[i])
        if norm_layer is not None:
            layers.append(norm_layer)
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        sample_ids : torch.Tensor
            list of (B, npoint, nsample) points indices from ball queries
        """
        if self.npoint is not None:
            new_xyz_idx = furthest_point_sample(xyz, self.npoint)
            new_xyz = (
                gather_operation(
                    xyz.transpose(1, 2).contiguous(), new_xyz_idx
                ).transpose(1, 2).contiguous()
            )
        else:
            new_xyz_idx = torch.zeros_like(xyz[:, :1, 0]).long()
            new_xyz = torch.zeros_like(xyz[:, :1])

        new_features_list, sample_ids = [], []
        for i in range(len(self.groupers)):
            new_features, sample_idx = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = new_features.max(dim=-1)[0]  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)
            sample_ids.append(sample_idx)
        features = torch.cat(new_features_list, dim=1)

        return new_xyz, new_xyz_idx, features, sample_ids


class _PointnetSAModuleVarNPts(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleVarNPts, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor],
        num_points: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (C, N) tensor of the descriptors of the the features
        num_points : list of int
            (B, )  number of points in each example

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        sample_ids : torch.Tensor
            list of (B, npoint, nsample) points indices from ball queries
        """
        all_new_features, all_sample_ids = [], []
        all_new_xyz, all_new_xyz_idx = [], []

        if features is not None:
            features = features.split(num_points, dim=-1)
        for i, xyz in enumerate(xyz.split(num_points)):
            xyz = xyz.unsqueeze(0)
            feat = None
            if features is not None:
                feat = features[i].unsqueeze(0)
            if self.npoint is not None:
                new_xyz_idx = furthest_point_sample(xyz, self.npoint)
                new_xyz = (
                    gather_operation(
                        xyz.transpose(1, 2).contiguous(), new_xyz_idx
                    ).transpose(1, 2).contiguous()
                )
                all_new_xyz.append(new_xyz)
                all_new_xyz_idx.append(new_xyz_idx)
            else:
                new_xyz_idx = torch.zeros_like(xyz[:, :1, 0]).long()
                new_xyz = torch.zeros_like(xyz[:, :1])

            new_features_list, sample_ids = [], []
            for j in range(len(self.groupers)):
                new_features, sample_idx = self.groupers[j](
                    xyz, new_xyz, feat
                )  # (1, C, npoint, nsample)
                new_features_list.append(new_features)
                sample_ids.append(sample_idx)
            all_new_features.append(new_features_list)
            all_sample_ids.append(sample_ids)

        if self.npoint is not None:
            new_xyz = torch.cat(all_new_xyz)
            new_xyz_idx = torch.cat(all_new_xyz_idx)

        new_features_list, sample_ids = [], []
        for i in range(len(self.groupers)):
            inputs = torch.cat([features[i] for features in all_new_features])
            new_features = self.mlps[i](inputs)  # (B, mlp[-1], npoint, nsample)
            new_features = new_features.max(dim=-1)[0]  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)
            sample_ids.append(torch.cat([
                sample_id[i] for sample_id in all_sample_ids
            ]))
        features = torch.cat(new_features_list, dim=1)

        return new_xyz, new_xyz_idx, features, sample_ids


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    norm : str
        Type of normalization layer (BN/GN)
    """

    def __init__(self, npoint, radii, nsamples, mlps, norm='BN', use_xyz=True):
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, norm))


class PointnetSAModuleMSGVarNPts(_PointnetSAModuleVarNPts):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    norm : str
        Type of normalization layer (BN/GN)
    """

    def __init__(self, npoint, radii, nsamples, mlps, norm='BN', use_xyz=True):
        super(PointnetSAModuleMSGVarNPts, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, norm))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    norm : str
        Type of normalization layer (BN/GN)
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, norm='BN', use_xyz=True
    ):
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            norm=norm,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    norm : str
        Type of normalization layer (BN/GN)
    """

    def __init__(self, mlp, norm='BN'):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, norm)

    def forward(self, unknown, known, unknow_feats, known_feats):
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        dist, idx = three_nn(unknown, known)
        dist_recip = 1.0 / (dist + 1e-8)
        weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)

        new_features = three_interpolate(known_feats.float(), idx, weight)
        if unknow_feats is not None:
            new_features = torch.cat(
                [new_features, unknow_feats], dim=1
            )  # (B, C2 + C1, n)

        new_features = self.mlp(new_features.unsqueeze(-1))
        return new_features.squeeze(-1)
