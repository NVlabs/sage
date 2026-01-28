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
import shutil
import glob
from collections.abc import Iterable
from typing import Dict, Any, Optional, Union, cast

class RobomimicDataMerger:
    def __init__(
        self,
        output_dir: str,
    ):
        """
        Initializes the data merger.

        Args:
            output_dir: The path to the output directory for merged data.
        """
        self._output_dir = os.path.abspath(output_dir)
        # Ensure output directory exists
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        
        # Define output HDF5 file path
        self._output_hdf5_path = os.path.join(self._output_dir, "hdf_dataset.hdf5")
    
    def merge_directories(self, input_dirs: list[str], output_name: str = "hdf_dataset"):
        """
        Merges data from multiple directories, each containing HDF5 files and parameter files.

        Adopts parameter files from the first input directory.

        Args:
            input_dirs: List of input directory paths containing HDF5 data and parameter files.
            output_name: Name for the output HDF5 file (without extension). Defaults to "hdf_dataset".
        
        Raises:
            FileNotFoundError: If any input directories don't exist.
            ValueError: If no input directories are provided or no HDF5 files found.
        """
        if not input_dirs:
            raise ValueError("No input directories provided for merging")
        
        # Validate all input directories exist
        for dir_path in input_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Input directory not found: {dir_path}")
        
        print(f"Starting directory merge of {len(input_dirs)} directories to {self._output_dir}")
        
        # Discover HDF5 files in input directories
        hdf5_files = []
        for input_dir in input_dirs:
            hdf5_file = self._find_hdf5_file(input_dir)
            if hdf5_file:
                hdf5_files.append(hdf5_file)
                print(f"Found HDF5 file: {hdf5_file}")
            else:
                print(f"Warning: No HDF5 file found in {input_dir}")
        
        if not hdf5_files:
            raise ValueError("No HDF5 files found in any input directories")
        
        # Set output HDF5 file path
        self._output_hdf5_path = os.path.join(self._output_dir, f"{output_name}.hdf5")
        
        # Merge the HDF5 files
        self.merge_data(hdf5_files)
        
        # Copy parameter files from first directory
        if input_dirs:
            self._copy_parameter_files(input_dirs[0], self._output_dir)
        
        print(f"Directory merge completed. Output saved to: {self._output_dir}")
    
    def _find_hdf5_file(self, directory: str) -> Optional[str]:
        """
        Find the first HDF5 file in the given directory.
        
        Args:
            directory: Directory path to search in.
            
        Returns:
            Path to the first HDF5 file found, or None if none found.
        """
        hdf5_patterns = ["*.hdf5", "*.h5"]
        
        for pattern in hdf5_patterns:
            search_pattern = os.path.join(directory, pattern)
            matches = glob.glob(search_pattern)
            if matches:
                return matches[0]  # Return first match
        
        return None
    
    def _copy_parameter_files(self, source_dir: str, dest_dir: str):
        """
        Copy parameter files (non-HDF5 files) from source directory to destination directory.
        
        Args:
            source_dir: Source directory containing parameter files.
            dest_dir: Destination directory to copy files to.
        """
        print(f"Copying parameter files from {source_dir} to {dest_dir}")
        
        # Get all files in source directory
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            dest_path = os.path.join(dest_dir, item)
            
            # Skip HDF5 files and directories
            if os.path.isfile(source_path) and not item.lower().endswith(('.hdf5', '.h5')):
                try:
                    shutil.copy2(source_path, dest_path)
                    print(f"  Copied: {item}")
                except Exception as e:
                    print(f"  Warning: Failed to copy {item}: {e}")
            elif os.path.isdir(source_path):
                # Copy directories recursively (but skip if it contains HDF5 files)
                try:
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                    print(f"  Copied directory: {item}")
                except Exception as e:
                    print(f"  Warning: Failed to copy directory {item}: {e}")
    
    def get_directories_info(self, input_dirs: list[str]) -> dict:
        """
        Get information about directories to be merged without performing the merge.
        
        Args:
            input_dirs: List of input directory paths.
            
        Returns:
            Dictionary containing merge information.
        """
        info = {
            "total_directories": len(input_dirs),
            "total_demos": 0,
            "total_samples": 0,
            "directory_info": []
        }
        
        for dir_path in input_dirs:
            dir_info = {"path": dir_path, "hdf5_file": None, "demos": 0, "samples": 0, "valid": False, "param_files": []}
            
            if not os.path.exists(dir_path):
                dir_info["error"] = "Directory not found"
                info["directory_info"].append(dir_info)
                continue
            
            # Find HDF5 file
            hdf5_file = self._find_hdf5_file(dir_path)
            if hdf5_file:
                dir_info["hdf5_file"] = hdf5_file
                
                # Get HDF5 file info
                try:
                    with h5py.File(hdf5_file, "r") as f:
                        if "data" in f:
                            data_group = cast(h5py.Group, f["data"])
                            demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]  # type: ignore
                            dir_info["demos"] = len(demo_keys)
                            
                            if "total" in data_group.attrs:
                                dir_info["samples"] = int(data_group.attrs["total"])  # type: ignore
                            
                            dir_info["valid"] = True
                            info["total_demos"] += dir_info["demos"]
                            info["total_samples"] += dir_info["samples"]
                
                except Exception as e:
                    dir_info["error"] = f"Error reading HDF5: {e}"
            else:
                dir_info["error"] = "No HDF5 file found"
            
            # List parameter files
            try:
                for item in os.listdir(dir_path):
                    if os.path.isfile(os.path.join(dir_path, item)) and not item.lower().endswith(('.hdf5', '.h5')):
                        dir_info["param_files"].append(item)
            except Exception:
                pass
            
            info["directory_info"].append(dir_info)
        
        return info

    def merge_data(self, data_paths: list[str]):
        """
        Merges the data from the data files using streaming I/O.

        Adopts the meta data from the first data file.

        Args:
            data_paths: The paths to the data files.
        
        Raises:
            FileNotFoundError: If any of the input files don't exist.
            ValueError: If no data files are provided or files are invalid.
        """
        if not data_paths:
            raise ValueError("No data paths provided for merging")
        
        # Validate all input files exist
        for path in data_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input data file not found: {path}")
        
        print(f"Starting merge of {len(data_paths)} files to {self._output_hdf5_path}")
        
        # Open output file for writing
        with h5py.File(self._output_hdf5_path, "w") as output_file:
            # Create main data group
            output_data_group = output_file.create_group("data")
            
            # Initialize counters
            total_demos = 0
            total_samples = 0
            output_demo_idx = 0
            
            # Process each input file
            for file_idx, input_path in enumerate(data_paths):
                print(f"Processing file {file_idx + 1}/{len(data_paths)}: {input_path}")
                
                with h5py.File(input_path, "r") as input_file:
                    # Validate input file structure
                    if "data" not in input_file:
                        print(f"Warning: Skipping {input_path} - missing 'data' group")
                        continue
                    
                    input_data_group = cast(h5py.Group, input_file["data"])
                    
                    # Copy metadata from first file
                    if file_idx == 0:
                        self._copy_metadata(input_data_group, output_data_group)
                    
                    # Get all demo keys and sort them
                    demo_keys = [key for key in input_data_group.keys() if key.startswith("demo_")]  # type: ignore
                    demo_keys.sort(key=lambda x: int(x.split("_")[1]))
                    
                    file_demos = 0
                    file_samples = 0
                    
                    # Stream copy each demo
                    for demo_key in demo_keys:
                        input_demo_group = input_data_group[demo_key]
                        output_demo_name = f"demo_{output_demo_idx}"
                        
                        # Copy entire demo group using HDF5's efficient copy
                        input_file.copy(f"data/{demo_key}", output_data_group, output_demo_name)
                        
                        # Update sample count
                        if "num_samples" in input_demo_group.attrs:
                            demo_samples = int(input_demo_group.attrs["num_samples"])  # type: ignore
                            file_samples += demo_samples
                        
                        output_demo_idx += 1
                        file_demos += 1
                    
                    total_demos += file_demos
                    total_samples += file_samples
                    
                    print(f"  Added {file_demos} demos, {file_samples} samples from {input_path}")
            
            # Update final metadata
            output_data_group.attrs["total"] = total_samples
            
            print(f"Merge completed: {total_demos} demos, {total_samples} total samples")
            print(f"Output saved to: {self._output_hdf5_path}")
    
    def _copy_metadata(self, input_data_group: h5py.Group, output_data_group: h5py.Group):
        """
        Copy metadata from input data group to output data group.
        
        Args:
            input_data_group: Source data group.
            output_data_group: Destination data group.
        """
        # Copy all attributes except 'total' (will be recalculated)
        for attr_name in input_data_group.attrs.keys():
            if attr_name != "total":
                output_data_group.attrs[attr_name] = input_data_group.attrs[attr_name]
        
        # Initialize total to 0 (will be updated during merge)
        output_data_group.attrs["total"] = 0
    
    def get_merge_info(self, data_paths: list[str]) -> dict:
        """
        Get information about files to be merged without performing the merge.
        
        Args:
            data_paths: The paths to the data files.
            
        Returns:
            Dictionary containing merge information.
        """
        info = {
            "total_files": len(data_paths),
            "total_demos": 0,
            "total_samples": 0,
            "file_info": []
        }
        
        for path in data_paths:
            if not os.path.exists(path):
                continue
                
            file_info = {"path": path, "demos": 0, "samples": 0, "valid": False}
            
            try:
                with h5py.File(path, "r") as f:
                    if "data" not in f:
                        continue
                    
                    data_group = cast(h5py.Group, f["data"])
                    demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]  # type: ignore
                    file_info["demos"] = len(demo_keys)
                    
                    if "total" in data_group.attrs:
                        file_info["samples"] = int(data_group.attrs["total"])  # type: ignore
                    
                    file_info["valid"] = True
                    info["total_demos"] += file_info["demos"]
                    info["total_samples"] += file_info["samples"]
            
            except Exception as e:
                file_info["error"] = str(e)
            
            info["file_info"].append(file_info)
        
        return info
    
    def merge_data_with_validation(self, data_paths: list[str], validate_structure: bool = True):
        """
        Merge data files with optional structure validation.
        
        Args:
            data_paths: The paths to the data files.
            validate_structure: Whether to validate data structure consistency.
        """
        if validate_structure:
            print("Validating file structures...")
            self._validate_file_structures(data_paths)
        
        self.merge_data(data_paths)
    
    def _validate_file_structures(self, data_paths: list[str]):
        """
        Validate that all files have compatible structures.
        
        Args:
            data_paths: The paths to the data files.
            
        Raises:
            ValueError: If files have incompatible structures.
        """
        if not data_paths:
            return
        
        # Use first valid file as reference
        reference_structure = None
        reference_env_args = None
        
        for path in data_paths:
            if not os.path.exists(path):
                continue
                
            try:
                with h5py.File(path, "r") as f:
                    if "data" not in f:
                        continue
                    
                    data_group = cast(h5py.Group, f["data"])
                    
                    # Get structure from first demo
                    demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]  # type: ignore
                    if not demo_keys:
                        continue
                    
                    first_demo = cast(h5py.Group, data_group[demo_keys[0]])
                    current_structure = set(first_demo.keys())  # type: ignore
                    
                    # Check env_args consistency
                    current_env_args = None
                    if "env_args" in data_group.attrs:
                        env_args_json = data_group.attrs["env_args"]
                        if isinstance(env_args_json, bytes):
                            env_args_json = env_args_json.decode('utf-8')
                        current_env_args = json.loads(str(env_args_json))  # type: ignore
                    
                    if reference_structure is None:
                        reference_structure = current_structure
                        reference_env_args = current_env_args
                        print(f"Using {path} as structure reference")
                    else:
                        # Validate structure compatibility
                        if current_structure != reference_structure:
                            print(f"Warning: Structure mismatch in {path}")
                            print(f"  Reference: {reference_structure}")
                            print(f"  Current: {current_structure}")
                        
                        # Check env_args compatibility (warn but don't fail)
                        if current_env_args != reference_env_args:
                            print(f"Warning: Environment args differ in {path}")
            
            except Exception as e:
                print(f"Warning: Could not validate structure of {path}: {e}")
        
        print("Structure validation completed")
    
    @staticmethod
    def merge_from_parent_directory(parent_dir: str, output_name: str = "hdf_dataset"):
        """
        Convenience method to merge all subdirectories under a parent directory.
        
        Automatically discovers subdirectories containing HDF5 files, creates a merged directory,
        and merges them into the generated output directory.
        
        Args:
            parent_dir: Parent directory containing subdirectories with HDF5 files.
            output_name: Name for the output HDF5 file (without extension). Defaults to "hdf_dataset".
        
        Returns:
            Path to the created output directory containing merged data.
        
        Raises:
            FileNotFoundError: If parent directory doesn't exist.
            ValueError: If no valid subdirectories with HDF5 files are found.
        """
        if not os.path.exists(parent_dir):
            raise FileNotFoundError(f"Parent directory not found: {parent_dir}")
        
        print(f"Searching for subdirectories with HDF5 files in: {parent_dir}")
        
        # Auto-generate output directory
        output_dir = os.path.join(parent_dir, "merged")
        
        # Create merger instance
        merger = RobomimicDataMerger(output_dir)
        
        # Find all subdirectories that contain HDF5 files
        input_dirs = []
        for item in os.listdir(parent_dir):
            subdir_path = os.path.join(parent_dir, item)
            if os.path.isdir(subdir_path) and item != "merged":  # Skip the output directory itself
                hdf5_file = merger._find_hdf5_file(subdir_path)
                if hdf5_file:
                    input_dirs.append(subdir_path)
                    print(f"Found subdirectory with HDF5: {subdir_path}")
        
        if not input_dirs:
            raise ValueError(f"No subdirectories with HDF5 files found in {parent_dir}")
        
        print(f"Found {len(input_dirs)} directories to merge")
        print(f"Output will be saved to: {output_dir}")
        
        # Perform the merge
        merger.merge_directories(input_dirs, output_name)
        
        return output_dir
    
    @staticmethod
    def create_and_merge_from_parent(parent_dir: str, output_dir: str, output_name: str = "hdf_dataset"):
        """
        Static convenience method to create a merger and merge from parent directory in one call.
        
        Args:
            parent_dir: Parent directory containing subdirectories with HDF5 files.
            output_dir: Directory where merged data will be saved.
            output_name: Name for the output HDF5 file (without extension). Defaults to "hdf_dataset".
            
        Returns:
            Path to the created merged HDF5 file.
        """
        merger = RobomimicDataMerger(output_dir)
        merger.merge_from_parent_directory(parent_dir, output_name)
        return os.path.join(output_dir, f"{output_name}.hdf5")
