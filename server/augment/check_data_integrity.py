#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Check data integrity of HDF5 files for mobile manipulation dataset.
Verifies that all required keys and values are present in each file.
"""

import h5py
import os
from collections import defaultdict

# Define expected camera keys based on replay file
EXPECTED_CAMERAS = ['right', 'left', 'wrist']  # Common camera names
REQUIRED_EPISODE_KEYS = ['states', 'actions', 'obs']
OPTIONAL_EPISODE_KEYS = ['next_obs']


def check_episode_integrity(episode_group, episode_key, file_path):
    """
    Check integrity of a single episode.
    
    Returns:
        dict: Dictionary with check results
    """
    issues = []
    warnings = []
    
    # Check required keys
    for key in REQUIRED_EPISODE_KEYS:
        if key not in episode_group:
            issues.append(f"Missing required key: {key}")
    
    # Check obs group structure
    if 'obs' in episode_group:
        obs_group = episode_group['obs']
        obs_keys = list(obs_group.keys())
        
        # Find all camera pairs
        rgb_cameras = [k.replace('rgb_', '') for k in obs_keys if k.startswith('rgb_')]
        depth_cameras = [k.replace('depth_', '') for k in obs_keys if k.startswith('depth_')]
        
        # Check for matching RGB/depth pairs
        for camera in rgb_cameras:
            if camera not in depth_cameras:
                issues.append(f"RGB camera '{camera}' has no matching depth camera")
        
        for camera in depth_cameras:
            if camera not in rgb_cameras:
                issues.append(f"Depth camera '{camera}' has no matching RGB camera")
        
        # Check for expected camera names
        for expected_cam in EXPECTED_CAMERAS:
            if f'rgb_{expected_cam}' not in obs_keys:
                warnings.append(f"Expected camera 'rgb_{expected_cam}' not found")
            if f'depth_{expected_cam}' not in obs_keys:
                warnings.append(f"Expected camera 'depth_{expected_cam}' not found")
        
        # Store found cameras
        found_cameras = set(rgb_cameras) & set(depth_cameras)
    else:
        issues.append("Missing 'obs' group")
        found_cameras = set()
    
    # Check next_obs group structure
    if 'next_obs' in episode_group:
        next_obs_group = episode_group['next_obs']
        next_obs_keys = list(next_obs_group.keys())
        
        # Should have same structure as obs
        for camera in found_cameras:
            if f'rgb_{camera}' not in next_obs_keys:
                issues.append(f"next_obs missing 'rgb_{camera}'")
            if f'depth_{camera}' not in next_obs_keys:
                issues.append(f"next_obs missing 'depth_{camera}'")
    
    return {
        'issues': issues,
        'warnings': warnings,
        'cameras_found': list(found_cameras),
        'num_samples': episode_group.attrs.get('num_samples', 0)
    }


def check_file_integrity(file_path):
    """
    Check integrity of an HDF5 file.
    
    Returns:
        dict: Dictionary with check results
    """
    result = {
        'file_path': file_path,
        'exists': False,
        'has_data_group': False,
        'num_episodes': 0,
        'total_samples': 0,
        'issues': [],
        'warnings': [],
        'episodes_with_issues': [],
        'camera_summary': defaultdict(int)
    }
    
    # Check if file exists
    if not os.path.exists(file_path):
        result['issues'].append('File does not exist')
        return result
    
    result['exists'] = True
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check for data group
            if 'data' not in f:
                result['issues'].append('Missing "data" group')
                return result
            
            result['has_data_group'] = True
            data_group = f['data']
            
            # Get all episodes
            episode_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')],
                                 key=lambda x: int(x.split('_')[1]))
            
            result['num_episodes'] = len(episode_keys)
            
            if len(episode_keys) == 0:
                result['warnings'].append('No episodes found')
                return result
            
            # Check each episode
            for episode_key in episode_keys:
                episode_group = data_group[episode_key]
                episode_result = check_episode_integrity(episode_group, episode_key, file_path)
                
                result['total_samples'] += episode_result['num_samples']
                
                # Track camera usage
                for camera in episode_result['cameras_found']:
                    result['camera_summary'][camera] += 1
                
                # Store episodes with issues
                if episode_result['issues']:
                    result['episodes_with_issues'].append({
                        'episode': episode_key,
                        'issues': episode_result['issues'],
                        'warnings': episode_result['warnings']
                    })
            
            # File-level warnings if camera consistency issues
            if result['camera_summary']:
                cameras_in_all = [cam for cam, count in result['camera_summary'].items() 
                                 if count == result['num_episodes']]
                cameras_in_some = [cam for cam, count in result['camera_summary'].items() 
                                  if count < result['num_episodes']]
                
                if cameras_in_some:
                    result['warnings'].append(
                        f"Inconsistent cameras: {cameras_in_some} not in all episodes"
                    )
    
    except Exception as e:
        result['issues'].append(f"Error reading file: {str(e)}")
    
    return result


def main():
    hdf5_file_name = "hdf_dataset_3.hdf5"
    data_prefix = "/home/hongchix/main/server/robomimic_data/Isaac-Mobile-Manipulation-Obj-Scene-Franka-IK-Abs-v3-motion_plan_"
    
    file_paths = [
        f"{data_prefix}layout_fbea22a2_01/{hdf5_file_name}",
        f"{data_prefix}layout_fbea22a2_03/{hdf5_file_name}",
        f"{data_prefix}layout_fbea22a2_04/{hdf5_file_name}",
        f"{data_prefix}layout_fbea22a2_05/{hdf5_file_name}",
        f"{data_prefix}layout_fbea22a2_06/{hdf5_file_name}",
        f"{data_prefix}layout_fbea22a2_07/{hdf5_file_name}",
        f"{data_prefix}layout_4f7c8ae1_00/{hdf5_file_name}",
        f"{data_prefix}layout_4f7c8ae1_01/{hdf5_file_name}",
        f"{data_prefix}layout_4f7c8ae1_02/{hdf5_file_name}",
        f"{data_prefix}layout_4f7c8ae1_03/{hdf5_file_name}",
        f"{data_prefix}layout_4f7c8ae1_04/{hdf5_file_name}",
        f"{data_prefix}layout_4f7c8ae1_05/{hdf5_file_name}",
        f"{data_prefix}layout_4f7c8ae1_06/{hdf5_file_name}",
        f"{data_prefix}layout_4f7c8ae1_07/{hdf5_file_name}",
        f"{data_prefix}layout_7a5944dd_00/{hdf5_file_name}",
        f"{data_prefix}layout_7a5944dd_01/{hdf5_file_name}",
        f"{data_prefix}layout_7a5944dd_02/{hdf5_file_name}",
        f"{data_prefix}layout_7a5944dd_03/{hdf5_file_name}",
        f"{data_prefix}layout_7a5944dd_04/{hdf5_file_name}",
        f"{data_prefix}layout_7a5944dd_05/{hdf5_file_name}",
        f"{data_prefix}layout_7a5944dd_06/{hdf5_file_name}",
        f"{data_prefix}layout_7a5944dd_07/{hdf5_file_name}",
        f"{data_prefix}layout_94554a35_00/{hdf5_file_name}",
        f"{data_prefix}layout_94554a35_01/{hdf5_file_name}",
        f"{data_prefix}layout_94554a35_02/{hdf5_file_name}",
        f"{data_prefix}layout_94554a35_03/{hdf5_file_name}",
        f"{data_prefix}layout_94554a35_04/{hdf5_file_name}",
        f"{data_prefix}layout_94554a35_05/{hdf5_file_name}",
        f"{data_prefix}layout_94554a35_06/{hdf5_file_name}",
        f"{data_prefix}layout_94554a35_07/{hdf5_file_name}",
        f"{data_prefix}layout_81ea4f47_00/{hdf5_file_name}",
        f"{data_prefix}layout_81ea4f47_01/{hdf5_file_name}",
        f"{data_prefix}layout_81ea4f47_02/{hdf5_file_name}",
        f"{data_prefix}layout_81ea4f47_03/{hdf5_file_name}",
        f"{data_prefix}layout_81ea4f47_04/{hdf5_file_name}",
        f"{data_prefix}layout_81ea4f47_05/{hdf5_file_name}",
        f"{data_prefix}layout_81ea4f47_06/{hdf5_file_name}",
        f"{data_prefix}layout_81ea4f47_07/{hdf5_file_name}",
        f"{data_prefix}layout_f0ec1abe_00/{hdf5_file_name}",
        f"{data_prefix}layout_f0ec1abe_01/{hdf5_file_name}",
        f"{data_prefix}layout_f0ec1abe_02/{hdf5_file_name}",
        f"{data_prefix}layout_f0ec1abe_03/{hdf5_file_name}",
        f"{data_prefix}layout_f0ec1abe_04/{hdf5_file_name}",
        f"{data_prefix}layout_f0ec1abe_05/{hdf5_file_name}",
        f"{data_prefix}layout_f0ec1abe_06/{hdf5_file_name}",
        f"{data_prefix}layout_f0ec1abe_07/{hdf5_file_name}",
        f"{data_prefix}layout_5d574c48_00/{hdf5_file_name}",
        f"{data_prefix}layout_5d574c48_01/{hdf5_file_name}",
        f"{data_prefix}layout_5d574c48_02/{hdf5_file_name}",
        f"{data_prefix}layout_5d574c48_03/{hdf5_file_name}",
        f"{data_prefix}layout_5d574c48_04/{hdf5_file_name}",
        f"{data_prefix}layout_5d574c48_05/{hdf5_file_name}",
        f"{data_prefix}layout_5d574c48_06/{hdf5_file_name}",
        f"{data_prefix}layout_5d574c48_07/{hdf5_file_name}",
        f"{data_prefix}layout_fa44732e_00/{hdf5_file_name}",
        f"{data_prefix}layout_fa44732e_01/{hdf5_file_name}",
        f"{data_prefix}layout_fa44732e_02/{hdf5_file_name}",
        f"{data_prefix}layout_fa44732e_03/{hdf5_file_name}",
        f"{data_prefix}layout_fa44732e_04/{hdf5_file_name}",
        f"{data_prefix}layout_fa44732e_05/{hdf5_file_name}",
        f"{data_prefix}layout_fa44732e_06/{hdf5_file_name}",
        f"{data_prefix}layout_fa44732e_07/{hdf5_file_name}",
        f"{data_prefix}layout_05e8e319_00/{hdf5_file_name}",
        f"{data_prefix}layout_05e8e319_01/{hdf5_file_name}",
        f"{data_prefix}layout_05e8e319_02/{hdf5_file_name}",
        f"{data_prefix}layout_05e8e319_03/{hdf5_file_name}",
        f"{data_prefix}layout_05e8e319_04/{hdf5_file_name}",
        f"{data_prefix}layout_05e8e319_05/{hdf5_file_name}",
        f"{data_prefix}layout_05e8e319_06/{hdf5_file_name}",
        f"{data_prefix}layout_05e8e319_07/{hdf5_file_name}",
        f"{data_prefix}layout_96175ef0_00/{hdf5_file_name}",
        f"{data_prefix}layout_96175ef0_01/{hdf5_file_name}",
        f"{data_prefix}layout_96175ef0_02/{hdf5_file_name}",
        f"{data_prefix}layout_96175ef0_03/{hdf5_file_name}",
        f"{data_prefix}layout_96175ef0_04/{hdf5_file_name}",
        f"{data_prefix}layout_96175ef0_05/{hdf5_file_name}",
        f"{data_prefix}layout_96175ef0_06/{hdf5_file_name}",
        f"{data_prefix}layout_96175ef0_07/{hdf5_file_name}",
        f"{data_prefix}layout_bfbb9b79_02/{hdf5_file_name}",
        f"{data_prefix}layout_72c30bb3_00/{hdf5_file_name}",
        f"{data_prefix}layout_72c30bb3_01/{hdf5_file_name}",
        f"{data_prefix}layout_72c30bb3_02/{hdf5_file_name}",
        f"{data_prefix}layout_72c30bb3_03/{hdf5_file_name}",
        f"{data_prefix}layout_72c30bb3_04/{hdf5_file_name}",
        f"{data_prefix}layout_72c30bb3_05/{hdf5_file_name}",
        f"{data_prefix}layout_72c30bb3_06/{hdf5_file_name}",
        f"{data_prefix}layout_72c30bb3_07/{hdf5_file_name}",
        f"{data_prefix}layout_ae2a0794_00/{hdf5_file_name}",
        f"{data_prefix}layout_ae2a0794_01/{hdf5_file_name}",
        f"{data_prefix}layout_ae2a0794_02/{hdf5_file_name}",
        f"{data_prefix}layout_ae2a0794_03/{hdf5_file_name}",
        f"{data_prefix}layout_ae2a0794_04/{hdf5_file_name}",
        f"{data_prefix}layout_ae2a0794_05/{hdf5_file_name}",
        f"{data_prefix}layout_ae2a0794_06/{hdf5_file_name}",
        f"{data_prefix}layout_ae2a0794_07/{hdf5_file_name}",
        f"{data_prefix}layout_6606e62e_00/{hdf5_file_name}",
        f"{data_prefix}layout_6606e62e_01/{hdf5_file_name}",
        f"{data_prefix}layout_6606e62e_02/{hdf5_file_name}",
        f"{data_prefix}layout_6606e62e_03/{hdf5_file_name}",
        f"{data_prefix}layout_6606e62e_04/{hdf5_file_name}",
        f"{data_prefix}layout_6606e62e_05/{hdf5_file_name}",
        f"{data_prefix}layout_6606e62e_06/{hdf5_file_name}",
        f"{data_prefix}layout_6606e62e_07/{hdf5_file_name}",
        f"{data_prefix}layout_59524b4a_02/{hdf5_file_name}",
        f"{data_prefix}layout_59524b4a_03/{hdf5_file_name}",
        f"{data_prefix}layout_59524b4a_04/{hdf5_file_name}",
        f"{data_prefix}layout_59524b4a_05/{hdf5_file_name}",
        f"{data_prefix}layout_59524b4a_06/{hdf5_file_name}",
        f"{data_prefix}layout_59524b4a_07/{hdf5_file_name}",
        f"{data_prefix}layout_05708a38_01/{hdf5_file_name}",
        f"{data_prefix}layout_05708a38_02/{hdf5_file_name}",
        f"{data_prefix}layout_05708a38_03/{hdf5_file_name}",
        f"{data_prefix}layout_05708a38_04/{hdf5_file_name}",
        f"{data_prefix}layout_05708a38_05/{hdf5_file_name}",
        f"{data_prefix}layout_05708a38_06/{hdf5_file_name}",
        f"{data_prefix}layout_05708a38_07/{hdf5_file_name}",
        f"{data_prefix}layout_7e174907_00/{hdf5_file_name}",
        f"{data_prefix}layout_58c1c39d_00/{hdf5_file_name}",
        f"{data_prefix}layout_58c1c39d_03/{hdf5_file_name}",
        f"{data_prefix}layout_58c1c39d_06/{hdf5_file_name}",
        f"{data_prefix}layout_7f3cad85_06/{hdf5_file_name}",
        f"{data_prefix}layout_7f3cad85_07/{hdf5_file_name}",
        f"{data_prefix}layout_e3c768b5_04/{hdf5_file_name}",
        f"{data_prefix}layout_d6ff748b_00/{hdf5_file_name}",
        f"{data_prefix}layout_d6ff748b_02/{hdf5_file_name}",
        f"{data_prefix}layout_4c048b61_00/{hdf5_file_name}",
        f"{data_prefix}layout_4c048b61_01/{hdf5_file_name}",
        f"{data_prefix}layout_4c048b61_02/{hdf5_file_name}",
        f"{data_prefix}layout_4c048b61_03/{hdf5_file_name}",
        f"{data_prefix}layout_4c048b61_04/{hdf5_file_name}",
        f"{data_prefix}layout_4c048b61_05/{hdf5_file_name}",
        f"{data_prefix}layout_4c048b61_06/{hdf5_file_name}",
        f"{data_prefix}layout_4c048b61_07/{hdf5_file_name}",
        f"{data_prefix}layout_ee6baa52_00/{hdf5_file_name}",
        f"{data_prefix}layout_ee6baa52_01/{hdf5_file_name}",
        f"{data_prefix}layout_ee6baa52_02/{hdf5_file_name}",
        f"{data_prefix}layout_ee6baa52_03/{hdf5_file_name}",
        f"{data_prefix}layout_ee6baa52_04/{hdf5_file_name}",
        f"{data_prefix}layout_ee6baa52_05/{hdf5_file_name}",
        f"{data_prefix}layout_ee6baa52_06/{hdf5_file_name}",
        f"{data_prefix}layout_ee6baa52_07/{hdf5_file_name}",
        f"{data_prefix}layout_b3bba355_00/{hdf5_file_name}",
        f"{data_prefix}layout_b3bba355_01/{hdf5_file_name}",
        f"{data_prefix}layout_b3bba355_02/{hdf5_file_name}",
        f"{data_prefix}layout_b3bba355_03/{hdf5_file_name}",
        f"{data_prefix}layout_b3bba355_06/{hdf5_file_name}",
        f"{data_prefix}layout_4387750e_00/{hdf5_file_name}",
        f"{data_prefix}layout_4387750e_01/{hdf5_file_name}",
        f"{data_prefix}layout_4387750e_02/{hdf5_file_name}",
        f"{data_prefix}layout_4387750e_03/{hdf5_file_name}",
        f"{data_prefix}layout_4387750e_04/{hdf5_file_name}",
        f"{data_prefix}layout_4387750e_06/{hdf5_file_name}",
        f"{data_prefix}layout_4387750e_07/{hdf5_file_name}",
        f"{data_prefix}layout_fea26344_07/{hdf5_file_name}",
        f"{data_prefix}layout_b33077b5_05/{hdf5_file_name}",
        f"{data_prefix}layout_4861dd58_00/{hdf5_file_name}",
        f"{data_prefix}layout_4861dd58_02/{hdf5_file_name}",
        f"{data_prefix}layout_4861dd58_05/{hdf5_file_name}",
        f"{data_prefix}layout_4861dd58_06/{hdf5_file_name}",
        f"{data_prefix}layout_4861dd58_07/{hdf5_file_name}",
        f"{data_prefix}layout_fe5a0508_04/{hdf5_file_name}",
        f"{data_prefix}layout_fe5a0508_07/{hdf5_file_name}",
        f"{data_prefix}layout_9f80c7a4_06/{hdf5_file_name}",
    ]
    
    print("="*100)
    print("HDF5 Data Integrity Check")
    print("="*100)
    print(f"Total files to check: {len(file_paths)}\n")
    
    # Track overall statistics
    files_ok = 0
    files_with_issues = 0
    files_with_warnings = 0
    files_not_found = 0
    all_results = []
    
    # Check each file
    for idx, file_path in enumerate(file_paths, 1):
        print(f"\n[{idx}/{len(file_paths)}] Checking: {file_path}")
        result = check_file_integrity(file_path)
        all_results.append(result)
        
        if not result['exists']:
            print(f"  ❌ FILE NOT FOUND")
            files_not_found += 1
            continue
        
        if result['issues']:
            print(f"  ❌ ISSUES FOUND: {len(result['issues'])} issue(s)")
            for issue in result['issues']:
                print(f"     - {issue}")
            files_with_issues += 1
        elif result['warnings']:
            print(f"  ⚠️  WARNINGS: {len(result['warnings'])} warning(s)")
            for warning in result['warnings']:
                print(f"     - {warning}")
            files_with_warnings += 1
        else:
            print(f"  ✓ OK - {result['num_episodes']} episodes, {result['total_samples']} samples")
            files_ok += 1
        
        # Show camera summary
        if result['camera_summary']:
            cameras_str = ', '.join([f"{cam}:{count}" for cam, count in sorted(result['camera_summary'].items())])
            print(f"     Cameras: {cameras_str}")
        
        # Show episodes with issues
        if result['episodes_with_issues']:
            print(f"     Episodes with issues: {len(result['episodes_with_issues'])}")
            for ep_info in result['episodes_with_issues'][:3]:  # Show first 3
                print(f"       - {ep_info['episode']}: {', '.join(ep_info['issues'][:2])}")
            if len(result['episodes_with_issues']) > 3:
                print(f"       ... and {len(result['episodes_with_issues']) - 3} more")
    
    # Print summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Total files checked: {len(file_paths)}")
    print(f"✓ Files OK: {files_ok}")
    print(f"⚠️  Files with warnings: {files_with_warnings}")
    print(f"❌ Files with issues: {files_with_issues}")
    print(f"❌ Files not found: {files_not_found}")
    
    # List all files with issues
    if files_with_issues > 0:
        print(f"\n{'='*100}")
        print("FILES WITH ISSUES:")
        print("="*100)
        for result in all_results:
            if result['issues']:
                print(f"\n{result['file_path']}")
                for issue in result['issues']:
                    print(f"  - {issue}")
    
    # List all files with warnings (missing expected cameras)
    if files_with_warnings > 0:
        print(f"\n{'='*100}")
        print("FILES WITH WARNINGS (Missing Expected Cameras):")
        print("="*100)
        for result in all_results:
            if result['warnings'] and not result['issues']:
                print(f"\n{result['file_path']}")
                for warning in result['warnings']:
                    print(f"  - {warning}")


if __name__ == "__main__":
    main()