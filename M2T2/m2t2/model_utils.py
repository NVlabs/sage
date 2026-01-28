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
Utility functions for network.
'''
import torch
import torch.nn as nn


def repeat_new_axis(tensor, rep, dim):
    reps = [1] * len(tensor.shape)
    reps.insert(dim, rep)
    return tensor.unsqueeze(dim).repeat(*reps)


def load_control_points():
    control_points = torch.tensor([
        [ 0.05268743, -0.00005996, 0.10527314, 1.00000000],
        [ 0.05268743, -0.00005996, 0.07527314, 1.00000000],
        [ 0.00000000,  0.00000000, 0.00000000, 1.00000000],
        [-0.05268743,  0.00005996, 0.07527314, 1.00000000],
        [-0.05268743,  0.00005996, 0.10527314, 1.00000000]
    ])
    return control_points.T


def get_activation_fn(activation):
    return getattr(nn, activation)()


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers,
        activation="ReLU", dropout=0.
    ):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        layers = []
        for m, n in zip([input_dim] + h[:-1], h):
            layers.extend([nn.Linear(m, n), get_activation_fn(activation)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
