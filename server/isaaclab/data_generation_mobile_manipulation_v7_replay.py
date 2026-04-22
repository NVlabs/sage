
import torch
import os
import numpy as np
import argparse
import h5py
import json
import imageio
from PIL import Image


def depth_to_rgb(depth):
    """Convert depth image to RGB for visualization."""
    if len(depth.shape) == 2:
        depth = depth[..., None]
    depth = np.repeat(depth, 3, axis=-1)
    return depth


def find_stage_boundaries(states):
    """
    Find the boundaries for each stage based on states array.
    
    Args:
        states: numpy array of shape (N, k) where N is num samples, k is num stages
                values are -1 (not finished) or 1 (finished)
    
    Returns:
        List of (start_idx, end_idx) tuples for each stage
    """
    num_samples, num_stages = states.shape
    boundaries = []
    
    for stage_idx in range(num_stages):
        # Find where this stage transitions from -1 to 1
        stage_states = states[:, stage_idx]
        
        # Find first index where stage becomes 1
        transition_idx = None
        for i in range(num_samples):
            if stage_states[i] == 1:
                transition_idx = i
                break
        
        if transition_idx is None:
            # Stage never completed
            print(f"    Warning: Stage {stage_idx} never completed")
            continue
        
        # Start is either 0 (for first stage) or after previous stage completed
        if stage_idx == 0:
            start_idx = 0
        else:
            # Find where previous stage completed
            prev_stage_states = states[:, stage_idx - 1]
            for i in range(num_samples):
                if prev_stage_states[i] == 1:
                    start_idx = i
                    break
        
        # End is transition + 32 frames (or end of episode)
        # For the last stage, include all data until the end
        if stage_idx == num_stages - 1:
            end_idx = num_samples - 1
        else:
            end_idx = min(transition_idx + 32, num_samples - 1)
        
        boundaries.append((start_idx, end_idx))
    
    return boundaries


def create_stage_split_hdf5(source_file, output_path, stage_idx, num_stages):
    """
    Create a new HDF5 file with data split by stages.
    
    Args:
        source_file: Original HDF5 file object
        output_path: Path to save the new HDF5 file
        stage_idx: Which stage to extract (0 to k-1)
        num_stages: Total number of stages
    """
    data_group = source_file['data']
    
    # Create new HDF5 file
    with h5py.File(output_path, 'w') as new_file:
        # Create data group
        new_data_group = new_file.create_group('data')
        
        # Copy attributes
        new_data_group.attrs['total'] = 0
        if 'env_args' in data_group.attrs:
            new_data_group.attrs['env_args'] = data_group.attrs['env_args']
        
        # Get all episode keys
        episode_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')], 
                             key=lambda x: int(x.split('_')[1]))
        
        new_demo_count = 0
        
        for episode_key in episode_keys:
            episode_group = data_group[episode_key]
            
            # Check if states exist
            if 'states' not in episode_group:
                print(f"    Warning: No 'states' in {episode_key}, skipping")
                continue
            
            states = episode_group['states'][:]
            
            # Find stage boundaries
            boundaries = find_stage_boundaries(states)
            
            if stage_idx >= len(boundaries):
                print(f"    Warning: Stage {stage_idx} not found in {episode_key}")
                continue
            
            start_idx, end_idx = boundaries[stage_idx]
            num_samples_stage = end_idx - start_idx + 1
            
            # Create new episode group
            new_episode_group = new_data_group.create_group(f'demo_{new_demo_count}')
            new_episode_group.attrs['num_samples'] = num_samples_stage
            
            # Copy all datasets for this stage
            for key in episode_group.keys():
                item = episode_group[key]
                
                if isinstance(item, h5py.Group):
                    # Handle nested groups (like obs, next_obs)
                    new_group = new_episode_group.create_group(key)
                    for sub_key in item.keys():
                        data = item[sub_key][start_idx:end_idx+1]
                        new_group.create_dataset(sub_key, data=data)
                
                elif isinstance(item, h5py.Dataset):
                    data = item[start_idx:end_idx+1]
                    
                    # Special handling for actions: concatenate stage states as termination prediction
                    if key == 'actions':
                        # Get the states column for this stage and add as termination prediction
                        stage_states = states[start_idx:end_idx+1, stage_idx:stage_idx+1]
                        data = np.concatenate([data, stage_states], axis=-1)
                    
                    new_episode_group.create_dataset(key, data=data)
            
            # Update total samples
            new_data_group.attrs['total'] += num_samples_stage
            new_demo_count += 1
        
        print(f"    Created {output_path} with {new_demo_count} episodes for stage {stage_idx}")


def create_video_from_episode(episode_group, episode_key, save_path, fps=30):
    """Create a video from episode observations with RGB and depth in two rows."""
    
    # Get observation groups
    obs_group = episode_group['obs']
    
    # Find all RGB and depth camera pairs
    camera_pairs = []
    obs_keys = list(obs_group.keys())
    
    # Extract camera names from rgb keys
    for key in obs_keys:
        if key.startswith('rgb_'):
            camera_name = key.replace('rgb_', '')
            depth_key = f'depth_{camera_name}'
            if depth_key in obs_keys:
                camera_pairs.append((f'rgb_{camera_name}', depth_key, camera_name))
    
    if not camera_pairs:
        print(f"  No RGB/depth pairs found in {episode_key}")
        return False
    
    print(f"  Found {len(camera_pairs)} camera pairs: {[name for _, _, name in camera_pairs]}")
    
    # Get number of frames
    num_frames = episode_group.attrs.get('num_samples', 0)
    if num_frames == 0:
        print(f"  No samples in {episode_key}")
        return False
    
    # Load and organize frames
    frames = []
    for frame_idx in range(num_frames):
        rgb_images = []
        depth_images = []
        
        for rgb_key, depth_key, camera_name in camera_pairs:
            # Load RGB image
            rgb = obs_group[rgb_key][frame_idx]
            if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                rgb = (rgb * 255).astype(np.uint8)
            rgb_images.append(rgb)
            
            # Load depth image (already uint8)
            depth = obs_group[depth_key][frame_idx]
            # Handle depth with channel dimension
            if len(depth.shape) == 3 and depth.shape[-1] == 1:
                depth = depth[:, :, 0]
            # Convert to RGB (depth is already uint8, no normalization needed)
            depth_rgb = depth_to_rgb(depth)
            depth_images.append(depth_rgb)
        
        # Concatenate RGB images horizontally
        rgb_row = np.concatenate(rgb_images, axis=1)
        # Concatenate depth images horizontally
        depth_row = np.concatenate(depth_images, axis=1)
        # Stack RGB and depth rows vertically
        combined_frame = np.concatenate([rgb_row, depth_row], axis=0)
        
        frames.append(combined_frame)
    
    # Save video
    imageio.mimwrite(save_path, frames, fps=fps)
    print(f"  Saved video: {save_path} ({num_frames} frames)")
    return True


def create_stage_videos(source_file, replays_dir, stage_idx, max_videos=None, fps=30):
    """Create videos for a specific stage from all episodes."""
    data_group = source_file['data']
    
    # Get all episode keys
    episode_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')], 
                         key=lambda x: int(x.split('_')[1]))
    
    videos_to_create = min(max_videos, len(episode_keys)) if max_videos else len(episode_keys)
    videos_created = 0
    
    for ep_idx, episode_key in enumerate(episode_keys[:videos_to_create]):
        episode_group = data_group[episode_key]
        
        # Check if states exist
        if 'states' not in episode_group:
            print(f"    Warning: No 'states' in {episode_key}, skipping")
            continue
        
        states = episode_group['states'][:]
        boundaries = find_stage_boundaries(states)
        
        if stage_idx >= len(boundaries):
            print(f"    Warning: Stage {stage_idx} not found in {episode_key}")
            continue
        
        start_idx, end_idx = boundaries[stage_idx]
        
        # Create temporary episode group with just this stage's data
        video_filename = f"{episode_key}_stage_{stage_idx}_replay.mp4"
        video_path = os.path.join(replays_dir, video_filename)
        
        print(f"  Creating video for {episode_key} stage {stage_idx} (frames {start_idx}-{end_idx})...")
        
        # Create frames for this stage
        obs_group = episode_group['obs']
        camera_pairs = []
        obs_keys = list(obs_group.keys())
        
        for key in obs_keys:
            if key.startswith('rgb_'):
                camera_name = key.replace('rgb_', '')
                depth_key = f'depth_{camera_name}'
                if depth_key in obs_keys:
                    camera_pairs.append((f'rgb_{camera_name}', depth_key, camera_name))
        
        if not camera_pairs:
            continue
        
        frames = []
        for frame_idx in range(start_idx, end_idx + 1):
            rgb_images = []
            depth_images = []
            
            for rgb_key, depth_key, camera_name in camera_pairs:
                rgb = obs_group[rgb_key][frame_idx]
                if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                    rgb = (rgb * 255).astype(np.uint8)
                rgb_images.append(rgb)
                
                depth = obs_group[depth_key][frame_idx]
                if len(depth.shape) == 3 and depth.shape[-1] == 1:
                    depth = depth[:, :, 0]
                depth_rgb = depth_to_rgb(depth)
                depth_images.append(depth_rgb)
            
            rgb_row = np.concatenate(rgb_images, axis=1)
            depth_row = np.concatenate(depth_images, axis=1)
            combined_frame = np.concatenate([rgb_row, depth_row], axis=0)
            
            # Add red border for all frames where stage is complete (state == 1)
            if states[frame_idx, stage_idx] == 1:
                # Add thick red border (10 pixels)
                border_thickness = 10
                combined_frame = combined_frame.copy()
                combined_frame[:border_thickness, :] = [255, 0, 0]  # Top
                combined_frame[-border_thickness:, :] = [255, 0, 0]  # Bottom
                combined_frame[:, :border_thickness] = [255, 0, 0]  # Left
                combined_frame[:, -border_thickness:] = [255, 0, 0]  # Right
            
            frames.append(combined_frame)
        
        imageio.mimwrite(video_path, frames, fps=fps)
        print(f"    Saved: {video_path} ({len(frames)} frames)")
        videos_created += 1
    
    return videos_created


def print_hdf5_structure(hdf5_file, group_path='', indent=0):
    """Recursively print the structure of an HDF5 file with dataset shapes."""
    indent_str = '  ' * indent
    
    if isinstance(hdf5_file, h5py.Group):
        for key in hdf5_file.keys():
            item = hdf5_file[key]
            if isinstance(item, h5py.Group):
                print(f"{indent_str}{key}/ (Group)")
                print_hdf5_structure(item, group_path + '/' + key, indent + 1)
            elif isinstance(item, h5py.Dataset):
                print(f"{indent_str}{key}: shape={item.shape}, dtype={item.dtype}")


def main():

    # add argparse arguments
    parser = argparse.ArgumentParser(description="Collect demonstrations for Isaac Lab environments.")
    parser.add_argument("--task", type=str, default="Isaac-Mobile-Manipulation-Obj-Scene-Franka-IK-Abs-v3", help="Name of the task.")
    parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
    parser.add_argument("--post_fix", type=str, default="", help="Postfix for the log directory")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information for each episode")
    parser.add_argument("--episode_id", type=int, default=None, help="Print detailed info for specific episode (e.g., 0 for demo_0)")
    parser.add_argument("--create_videos", action="store_true", help="Create videos from RGB/depth observations")
    parser.add_argument("--video_fps", type=int, default=30, help="FPS for generated videos")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to create (default: all)")
    parser.add_argument("--split_by_stages", action="store_true", help="Split data by stages into separate HDF5 files and videos")
    parser.add_argument("--print_actions", action="store_true", help="Print actions in formatted row-by-row display")
    parser.add_argument("--max_action_rows", type=int, default=1000, help="Maximum number of action rows to print per episode (default: 50)")

    """Collect demonstrations from the environment using teleop interfaces."""
    args_cli = parser.parse_args()
    log_dir = os.path.abspath(os.path.join("./robomimic_data", args_cli.task+"-"+args_cli.post_fix))
    data_path = os.path.join(log_dir, args_cli.filename+".hdf5")

    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return

    print(f"\n{'='*80}")
    print(f"Loading HDF5 file from: {data_path}")
    print(f"{'='*80}\n")

    # Open the HDF5 file
    with h5py.File(data_path, 'r') as f:
        # Print top-level structure
        print("Top-level keys:", list(f.keys()))
        print()

        # Check if 'data' group exists
        if 'data' not in f:
            print("Error: 'data' group not found in HDF5 file")
            return

        data_group = f['data']
        
        # Print data group attributes
        print("Data group attributes:")
        for attr_name in data_group.attrs:
            attr_value = data_group.attrs[attr_name]
            print(f"  {attr_name}: {attr_value}")
        print()

        # Parse env_args if it exists
        if 'env_args' in data_group.attrs:
            try:
                env_args = json.loads(data_group.attrs['env_args'])
                print("Environment configuration:")
                print(f"  env_name: {env_args.get('env_name', 'N/A')}")
                print(f"  type: {env_args.get('type', 'N/A')}")
                print()
            except:
                pass

        # Get all episode keys
        episode_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')], 
                             key=lambda x: int(x.split('_')[1]))
        
        num_episodes = len(episode_keys)
        print(f"Total number of episodes: {num_episodes}")
        print()

        # Collect statistics
        total_samples = 0
        episode_lengths = []
        num_stages = None

        # Iterate through episodes
        for ep_idx, episode_key in enumerate(episode_keys):
            episode_group = data_group[episode_key]
            num_samples = episode_group.attrs.get('num_samples', 0)
            episode_lengths.append(num_samples)
            total_samples += num_samples
            
            # Detect number of stages from first episode
            if num_stages is None and 'states' in episode_group:
                states = episode_group['states'][:]
                num_stages = states.shape[1]
                print(f"Detected {num_stages} stages from states array\n")

            # Print summary for each episode
            if args_cli.verbose or args_cli.episode_id == ep_idx:
                print(f"\n{'-'*80}")
                print(f"{episode_key}: {num_samples} samples")
                print(f"{'-'*80}")
                print_hdf5_structure(episode_group, indent=1)
            else:
                # Just print one-line summary
                print(f"{episode_key}: {num_samples} samples")

        # Print overall statistics
        print(f"\n{'='*80}")
        print("Dataset Statistics:")
        print(f"{'='*80}")
        print(f"Total episodes: {num_episodes}")
        print(f"Total samples: {total_samples}")
        print(f"Average episode length: {total_samples / num_episodes if num_episodes > 0 else 0:.2f}")
        print(f"Min episode length: {min(episode_lengths) if episode_lengths else 0}")
        print(f"Max episode length: {max(episode_lengths) if episode_lengths else 0}")
        if num_stages:
            print(f"Number of stages: {num_stages}")
        print()

        # Print detailed structure for first episode as example
        if num_episodes > 0 and not args_cli.verbose and args_cli.episode_id is None:
            print(f"\nDetailed structure of first episode (demo_0) as example:")
            print(f"{'-'*80}")
            first_episode = data_group['demo_0']
            print_hdf5_structure(first_episode, indent=1)
            print()
            print("Tip: Use --verbose to see all episodes, or --episode_id N to see specific episode")

        # Print actions if requested
        if args_cli.print_actions:
            print(f"\n{'='*80}")
            print("Actions Display:")
            print(f"{'='*80}\n")
            
            episodes_to_print = [args_cli.episode_id] if args_cli.episode_id is not None else range(min(num_episodes, 5))
            
            for ep_idx in episodes_to_print:
                if ep_idx >= num_episodes:
                    continue
                    
                episode_key = f'demo_{ep_idx}'
                episode_group = data_group[episode_key]
                
                if 'actions' not in episode_group:
                    print(f"{episode_key}: No actions found")
                    continue
                
                actions = episode_group['actions'][:]
                num_samples = actions.shape[0]
                action_dim = actions.shape[1]
                
                print(f"\n{episode_key}: {num_samples} actions, dimension: {action_dim}")
                print(f"{'-'*80}")
                
                # Print header
                header = f"{'Step':>6} | " + " | ".join([f"a[{i}]" for i in range(action_dim)])
                print(header)
                print("-" * len(header))
                
                # Print actions
                max_rows = min(args_cli.max_action_rows, num_samples)
                for i in range(max_rows):
                    action_str = f"{i:6d} | " + " | ".join([f"{actions[i, j]:7.3f}" for j in range(action_dim)])
                    print(action_str)
                
                if num_samples > max_rows:
                    print(f"... ({num_samples - max_rows} more rows)")
            
            if args_cli.episode_id is None and num_episodes > 5:
                print(f"\nShowing first 5 episodes. Use --episode_id N to see specific episode")

        # Split by stages if requested
        if args_cli.split_by_stages:
            if num_stages is None:
                print("\nError: Cannot split by stages - no 'states' found in episodes")
                return
            
            print(f"\n{'='*80}")
            print(f"Splitting Data by {num_stages} Stages:")
            print(f"{'='*80}\n")
            
            # Create replays directory
            replays_dir = os.path.join(log_dir, "replays")
            os.makedirs(replays_dir, exist_ok=True)
            
            for stage_idx in range(num_stages):
                print(f"\nProcessing Stage {stage_idx}:")
                print(f"{'-'*80}")
                
                # Create stage-specific HDF5 file
                base_filename = args_cli.filename.replace('.hdf5', '')
                stage_filename = f"{base_filename}_{stage_idx}.hdf5"
                stage_path = os.path.join(log_dir, stage_filename)
                
                print(f"  Creating HDF5 file: {stage_filename}")
                create_stage_split_hdf5(f, stage_path, stage_idx, num_stages)
                
                # Create videos for this stage if requested
                if args_cli.create_videos:
                    print(f"  Creating videos for stage {stage_idx}...")
                    videos_created = create_stage_videos(
                        f, replays_dir, stage_idx, 
                        max_videos=args_cli.max_videos, 
                        fps=args_cli.video_fps
                    )
                    print(f"  Created {videos_created} videos for stage {stage_idx}")
            
            print(f"\n{'='*80}")
            print(f"Stage Splitting Complete!")
            print(f"{'='*80}")
            print(f"HDF5 files saved in: {log_dir}")
            print(f"Videos saved in: {replays_dir}")
        
        # Create videos if requested (full episodes, not split)
        elif args_cli.create_videos:
            print(f"\n{'='*80}")
            print("Creating Videos from Observations:")
            print(f"{'='*80}\n")
            
            # Create replays directory
            replays_dir = os.path.join(log_dir, "replays")
            os.makedirs(replays_dir, exist_ok=True)
            print(f"Saving videos to: {replays_dir}\n")
            
            # Determine how many videos to create
            max_videos = args_cli.max_videos if args_cli.max_videos is not None else num_episodes
            videos_to_create = min(max_videos, num_episodes)
            
            videos_created = 0
            for ep_idx, episode_key in enumerate(episode_keys[:videos_to_create]):
                episode_group = data_group[episode_key]
                video_filename = f"{episode_key}_replay.mp4"
                video_path = os.path.join(replays_dir, video_filename)
                
                print(f"Creating video for {episode_key}...")
                success = create_video_from_episode(
                    episode_group, 
                    episode_key, 
                    video_path, 
                    fps=args_cli.video_fps
                )
                if success:
                    videos_created += 1
            
            print(f"\nSuccessfully created {videos_created}/{videos_to_create} videos")
            print(f"Videos saved in: {replays_dir}")

if __name__ == "__main__":
    main()