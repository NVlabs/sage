# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interface to collect and store data from the environment using format from `robomimic`."""

# needed to import for allowing type-hinting: np.ndarray | torch.Tensor
from __future__ import annotations

import h5py
import json
import numpy as np
import os
import torch
from collections.abc import Iterable
from typing import Dict, Any, Optional, Union, cast

import omni.log

class RobomimicDataReader:
    def __init__(
        self,
        data_path: str,
    ):
        """
        Initializes the data reader.

        Args:
            data_path: The path to the data hdf5 file.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self._data_path = data_path
        self._h5_file: h5py.File = h5py.File(data_path, "r")
        self._data_group: h5py.Group = self._h5_file["data"]  # type: ignore
        self._current_episode_idx = 0
        self._total_episodes = self._count_episodes()

    def __del__(self):
        """Destructor to close the HDF5 file."""
        if hasattr(self, '_h5_file') and self._h5_file is not None:
            self._h5_file.close()

    def close(self):
        """Explicitly close the HDF5 file."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def _count_episodes(self) -> int:
        """Count the number of demo episodes in the data group."""
        count = 0
        for key in self._data_group.keys():
            if key.startswith("demo_"):
                count += 1
        return count

    def total_episodes(self) -> int:
        """
        Returns the total number of episodes in the data file.
        """
        return self._total_episodes
    
    def read_next_episode(self) -> dict:
        """
        Reads the next episode from the data file.

        Returns:
            A dictionary containing the episode data. similar to env_dataset in robomimic_data_collector.py L208
        """
        if self._current_episode_idx >= self._total_episodes:
            raise IndexError(f"No more episodes to read. Current index: {self._current_episode_idx}, Total: {self._total_episodes}")
        
        demo_name = f"demo_{self._current_episode_idx}"
        demo_group = self._data_group[demo_name]
        
        # Build episode data dictionary
        episode_data = {}
        
        for key in demo_group.keys():
            if isinstance(demo_group[key], h5py.Group):
                # Handle nested groups (like "obs")
                episode_data[key] = {}
                for sub_key in demo_group[key].keys():
                    episode_data[key][sub_key] = np.array(demo_group[key][sub_key])
            else:
                # Handle direct datasets (like "actions", "rewards", "dones")
                episode_data[key] = np.array(demo_group[key])
        
        # Increment episode index for next read
        self._current_episode_idx += 1
        
        return episode_data

    def read_episode_by_index(self, episode_idx: int) -> dict:
        """
        Reads a specific episode by index.

        Args:
            episode_idx: The index of the episode to read.

        Returns:
            A dictionary containing the episode data.
        """
        if episode_idx < 0 or episode_idx >= self._total_episodes:
            raise IndexError(f"Episode index {episode_idx} out of range [0, {self._total_episodes})")
        
        demo_name = f"demo_{episode_idx}"
        demo_group = self._data_group[demo_name]
        
        # Build episode data dictionary
        episode_data = {}
        
        for key in demo_group.keys():
            if isinstance(demo_group[key], h5py.Group):
                # Handle nested groups (like "obs")
                episode_data[key] = {}
                for sub_key in demo_group[key].keys():
                    episode_data[key][sub_key] = np.array(demo_group[key][sub_key])
            else:
                # Handle direct datasets (like "actions", "rewards", "dones")
                episode_data[key] = np.array(demo_group[key])
        
        return episode_data

    def reset_episode_reader(self):
        """Reset the episode reader to start from the first episode."""
        self._current_episode_idx = 0

    def read_meta_data(self) -> dict:
        """
        Reads the meta data from the data file.

        Returns:
            A dictionary containing the meta data. similar to env_args in robomimic_data_collector.py L280
        """
        if "env_args" not in self._data_group.attrs:
            return {}
        
        env_args_json = self._data_group.attrs["env_args"]
        if isinstance(env_args_json, bytes):
            env_args_json = env_args_json.decode('utf-8')
        
        return json.loads(env_args_json)

    def get_total_samples(self) -> int:
        """
        Returns the total number of samples across all episodes.

        Returns:
            The total number of samples.
        """
        if "total" in self._data_group.attrs:
            return int(self._data_group.attrs["total"])
        return 0

    def get_episode_length(self, episode_idx: int) -> int:
        """
        Returns the number of samples in a specific episode.

        Args:
            episode_idx: The index of the episode.

        Returns:
            The number of samples in the episode.
        """
        if episode_idx < 0 or episode_idx >= self._total_episodes:
            raise IndexError(f"Episode index {episode_idx} out of range [0, {self._total_episodes})")
        
        demo_name = f"demo_{episode_idx}"
        demo_group = self._data_group[demo_name]
        
        if "num_samples" in demo_group.attrs:
            return int(demo_group.attrs["num_samples"])
        return 0
    

