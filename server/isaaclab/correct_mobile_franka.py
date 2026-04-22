import asyncio
import json
import sys
import os
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
import copy

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from models import Object, Point3D, Euler, Dimensions, FloorPlan, Room
from isaacsim.isaac_mcp.server import (
    get_isaac_connection,
    create_room_layout_scene,
    simulate_the_scene,
    create_robot,
    move_robot_to_target,
    create_physics_scene,
    get_room_layout_scene_usd,
    create_single_room_layout_scene,
    get_room_layout_scene_usd_separate_from_layout
)
from tex_utils import export_layout_to_mesh_dict_list
from glb_utils import (
    create_glb_scene,
    add_textured_mesh_to_glb_scene,
    save_glb_scene
)
from objects.object_on_top_placement import (
    get_random_placements_on_target_object, 
    filter_placements_by_physics_critic,
)
from constants import RESULTS_DIR
from utils import export_layout_to_json, dict_to_floor_plan, extract_json_from_response
from vlm import call_vlm
import argparse
import shutil


def find_closest_object_to_trajectory_line(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    candidate_objects: List[Object],
    excluded_obj_ids: set
) -> Optional[Object]:
    """
    Find the closest object to the line segment from start_pos to end_pos.
    
    Args:
        start_pos: Start position [x, y]
        end_pos: End position [x, y]
        candidate_objects: List of objects to consider
        excluded_obj_ids: Set of object IDs to exclude from consideration
        
    Returns:
        The closest object to the trajectory line, or None if no suitable object found
    """
    def point_to_line_segment_distance(point, line_start, line_end):
        """Calculate minimum distance from point to line segment"""
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq < 1e-10:
            # Line segment is actually a point
            return np.linalg.norm(point - line_start)
        
        # Vector from line_start to point
        point_vec = point - line_start
        
        # Project point onto line (parameterized by t in [0, 1])
        t = np.dot(point_vec, line_vec) / line_len_sq
        t = np.clip(t, 0.0, 1.0)
        
        # Closest point on line segment
        closest_on_line = line_start + t * line_vec
        
        # Distance from point to closest point on line
        distance = np.linalg.norm(point - closest_on_line)
        return distance
    
    min_distance = float('inf')
    closest_object = None
    
    for obj in candidate_objects:
        # Skip excluded objects
        if obj.id in excluded_obj_ids:
            continue
        
        # Get object position
        obj_pos = np.array([obj.position.x, obj.position.y])
        
        # Calculate distance from object to trajectory line
        distance = point_to_line_segment_distance(obj_pos, start_pos, end_pos)
        
        if distance < min_distance:
            min_distance = distance
            closest_object = obj
    
    return closest_object


async def correct_mobile_franka_standalone(layout: FloorPlan, room_id: str = "", temp_json_path: str = None) -> str:
    """
    Correct the placement of objects in a room for mobile franka robot tasks.
    This is a standalone version that accepts a FloorPlan object directly.
    
    Validates that the robot can complete pick-place tasks by:
    1. Sampling valid robot spawn positions
    2. Sampling valid pick object poses with reachable robot positions
    3. Planning collision-free trajectories from spawn to pick
    4. Sampling valid place locations with reachable robot positions
    5. Planning collision-free trajectories from pick to place
    6. If validation fails, removing blocking objects and retrying
    
    Args:
        layout: The FloorPlan object to correct
        room_id: The ID of the room to correct
        temp_json_path: Path to save temporary JSON during correction (optional)
    
    Returns:
        JSON string with correction results and updated object placements
    """
    current_layout = layout
    policy_analysis = current_layout.policy_analysis

    # Import mobile manipulation utilities
    from objects.object_mobile_manipulation_utils import (
        sample_robot_spawn,
        sample_pick_object_pose_with_mobile_franka_occupancy,
        sample_place_object_pose_with_mobile_franka_occupancy,
        plan_robot_traj,
        sample_robot_place_location,
        sample_robot_location,
    )
    import torch
    
    # Extract information from policy analysis
    robot_type = policy_analysis["robot_type"]
    task_decomposition = policy_analysis.get("task_decomposition", [])
    updated_task_decomposition = policy_analysis.get("updated_task_decomposition", [])
    print(f"correction: Updated task decomposition: {updated_task_decomposition}", file=sys.stderr)
    object_mapping = policy_analysis.get("object_mapping", {})
    minimum_required_objects = policy_analysis.get("minimum_required_objects", [])
    
    # Find the target room
    target_room = next((r for r in current_layout.rooms if r.id == room_id), None)
    if target_room is None:
        return json.dumps({
            "success": False,
            "error": f"Room with ID '{room_id}' not found"
        })
    
    # Use updated_task_decomposition if available, otherwise fall back to original
    decomposition_to_use = updated_task_decomposition if updated_task_decomposition else task_decomposition
    use_updated_format = bool(updated_task_decomposition)
    
    if use_updated_format:
        print(f"Using updated_task_decomposition with {len(decomposition_to_use)} steps", file=sys.stderr)
    else:
        print(f"Using original task_decomposition with {len(decomposition_to_use)} steps", file=sys.stderr)
    
    # Parse task decomposition to extract pick-place pairs
    pick_place_pairs = []
    i = 0
    while i < len(decomposition_to_use):
        task = decomposition_to_use[i]
        
        if task["action"] == "navigate":
            # Skip navigate actions
            i += 1
            continue
        elif task["action"] == "pick":
            # Look for corresponding place action
            pick_task = task
            place_task = None
            
            if i + 1 < len(decomposition_to_use):
                next_task = decomposition_to_use[i + 1]
                if next_task["action"] == "place":
                    place_task = next_task
                    i += 2
                elif i + 2 < len(decomposition_to_use) and decomposition_to_use[i + 2]["action"] == "place":
                    # Navigate might be between pick and place
                    place_task = decomposition_to_use[i + 2]
                    i += 3
                else:
                    i += 1
            else:
                i += 1
            
            if place_task:
                pick_place_pairs.append({
                    "pick": pick_task,
                    "place": place_task
                })
        else:
            i += 1
    
    print(f"Pick-place pairs: {pick_place_pairs}", file=sys.stderr)
    
    if not pick_place_pairs:
        return json.dumps({
            "success": False,
            "error": "No pick-place task pairs found in task decomposition"
        })
    
    print(f"Found {len(pick_place_pairs)} pick-place pairs to validate", file=sys.stderr)
    
    # Get layout save directory
    layout_save_dir = os.path.join(RESULTS_DIR, current_layout.id)
    debug_dir = os.path.join(layout_save_dir, "mobile_franka_correction_debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Use temp JSON path if provided, otherwise use the regular path
    if temp_json_path is None:
        temp_json_path = os.path.join(layout_save_dir, f"{current_layout.id}.json")
    
    print(f"Using temporary JSON path: {temp_json_path}", file=sys.stderr)
    
    # Save initial layout to temp JSON path before starting iterations
    # This ensures occupancy grid can load from it correctly from the start
    export_layout_to_json(current_layout, temp_json_path)
    print(f"Initial layout saved to temp path", file=sys.stderr)
    
    # Main correction loop
    max_outer_iterations = 50  # Repeat until only minimum required objects left
    max_inner_iterations = 10  # Try 10 times to find valid trajectories
    
    # Track trajectory failures across inner iterations
    trajectory_failure_log = []
    
    for outer_iter in range(max_outer_iterations):
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Outer Iteration {outer_iter + 1}/{max_outer_iterations}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # read the temp json path and get the current layout
        current_layout = dict_to_floor_plan(json.load(open(temp_json_path, "r")))
        target_room = next((r for r in current_layout.rooms if r.id == room_id), None)
        
        # Try to find valid trajectories
        valid_trajectory_found = False
        # Clear failure log for this outer iteration
        trajectory_failure_log = []
        
        for inner_iter in range(max_inner_iterations):
            print(f"\nInner Iteration {inner_iter + 1}/{max_inner_iterations}", file=sys.stderr)
            
            try:
                # Step 0: Sample robot spawn pose
                print("Step 0: Sampling robot spawn pose...", file=sys.stderr)
                spawn_pos, spawn_angles = sample_robot_spawn(
                    layout_save_dir,
                    temp_layout_name,
                    room_id,
                    num_envs=1,
                    debug_dir=debug_dir
                )
                
                # Convert spawn to position and quaternion format
                spawn_pos_np = spawn_pos.cpu().numpy()[0]  # [side, forward]
                spawn_angle_np = spawn_angles.cpu().numpy()[0, 0]  # yaw angle
                
                # Convert to support point position using inverse function
                from isaaclab.omron_franka_occupancy import support_point
                spawn_support_point = support_point(spawn_pos_np[1], spawn_pos_np[0], spawn_angle_np)
                
                # Create spawn position and quaternion
                spawn_position = torch.tensor([spawn_support_point[0], spawn_support_point[1], 0], dtype=torch.float, device="cuda")
                spawn_quat = torch.tensor([np.cos(spawn_angle_np/2), 0, 0, np.sin(spawn_angle_np/2)], dtype=torch.float, device="cuda")
                
                print(f"  Spawn: pos={spawn_position.cpu().numpy()}, yaw={np.degrees(spawn_angle_np):.1f}°", file=sys.stderr)
                
                # Save original room.objects for rollback
                original_room_objects = copy.deepcopy(target_room.objects)
                
                # Step 1: Get all pick_table and place_table ids
                print("Step 1: Collecting pick_table and place_table IDs...", file=sys.stderr)
                tables_to_resample = {}  # {table_id: [list of reachable_object_ids on this table]}
                
                for pair_idx, pair in enumerate(pick_place_pairs):
                    # Get pick and place object IDs
                    if use_updated_format:
                        pick_obj_id = pair["pick"].get("target_object_id")
                        place_obj_id = pair["place"].get("location_object_id")
                    else:
                        pick_obj_name = pair["pick"]["target_object"]
                        place_obj_name = pair["place"]["target_object"]
                        pick_mapping = object_mapping.get(pick_obj_name, {})
                        pick_obj_id = pick_mapping.get("matched_ids", [None])[0]
                        place_mapping = object_mapping.get(place_obj_name, {})
                        place_obj_id = place_mapping.get("matched_ids", [None])[0]
                    
                    if not pick_obj_id or not place_obj_id:
                        continue
                    
                    # Find pick_table_id
                    pick_obj = next((obj for obj in target_room.objects if obj.id == pick_obj_id), None)
                    if pick_obj and pick_obj.place_id not in ["floor", "wall"]:
                        pick_table_id = pick_obj.place_id
                        if pick_table_id not in tables_to_resample:
                            tables_to_resample[pick_table_id] = []
                        if pick_obj_id not in tables_to_resample[pick_table_id]:
                            tables_to_resample[pick_table_id].append(pick_obj_id)
                    
                    # Find place_table_id
                    place_obj = next((obj for obj in target_room.objects if obj.id == place_obj_id), None)
                    if place_obj:
                        if place_obj.place_id in ["floor", "wall"]:
                            # place_obj is the table itself
                            place_table_id = place_obj_id
                            tables_to_resample[place_table_id] = []
                        else:
                            # place_obj is on another object (the table)
                            place_table_id = place_obj.place_id
                            if place_table_id not in tables_to_resample:
                                tables_to_resample[place_table_id] = []
                            if place_obj_id not in tables_to_resample[place_table_id]:
                                tables_to_resample[place_table_id].append(place_obj_id)
                
                print(f"  Found {len(tables_to_resample)} tables to resample: {list(tables_to_resample.keys())}", file=sys.stderr)
                
                # Step 2: For each table, run object_augmentation_pose_object_tree_sim_correction
                from objects.object_augmentation import object_augmentation_pose_object_tree_sim_correction
                
                sampling_success = True
                # Extract temp layout name for passing to resampling (so occupancy grid uses temp file)
                temp_layout_name = os.path.splitext(os.path.basename(temp_json_path))[0]
                
                for table_id, reachable_obj_ids in tables_to_resample.items():
                    print(f"  Step 2: Resampling objects on table {table_id}, reachable objects: {reachable_obj_ids}", file=sys.stderr)
                    
                    table_obj = next((obj for obj in target_room.objects if obj.id == table_id), None)
                    if not table_obj:
                        print(f"    Warning: Table {table_id} not found", file=sys.stderr)
                        sampling_success = False
                        break
                    
                    # Run the correction function (pass temp layout name so it uses updated occupancy grid)
                    if len(reachable_obj_ids) > 0:
                        new_objects, success = object_augmentation_pose_object_tree_sim_correction(
                            current_layout,
                            target_room,
                            table_obj,
                            reachable_obj_ids,
                            layout_file_name=temp_layout_name
                        )

                        if not success or new_objects is None:
                            print(f"    ❌ Failed to resample objects on table {table_id}", file=sys.stderr)
                            sampling_success = False
                            trajectory_failure_log.append({
                                "pair_idx": -1,
                                "step": "place",
                                "start_pos": spawn_position.cpu().numpy().tolist()[:2],
                                "end_pos": [table_obj.position.x, table_obj.position.y],
                                "pick_table_id": table_id,
                                "place_table_id": table_id,
                            })
                            break

                    else:
                        # TODO
                        # sample place locations on the table_obj
                        # ensure at least one place location is reachable
                        place_robot_base_pos, place_robot_base_quat, place_locations = sample_robot_place_location(
                            layout_save_dir,
                            temp_layout_name,
                            room_id,
                            table_obj.id,
                            num_envs=1,
                            debug_dir=debug_dir
                        )

                        success = place_robot_base_pos is not None and len(place_robot_base_pos) > 0
                        
                        # if not, success = False as well

                        if not success:
                            print(f"    ❌ Failed to resample objects on table {table_id}", file=sys.stderr)
                            sampling_success = False
                            trajectory_failure_log.append({
                                "pair_idx": -1,
                                "step": "place",
                                "start_pos": spawn_position.cpu().numpy().tolist()[:2],
                                "end_pos": [table_obj.position.x, table_obj.position.y],
                                "pick_table_id": table_id,
                                "place_table_id": table_id,
                            })
                            break
                    
                    # Update room objects with new poses
                    target_room.objects = new_objects
                    print(f"    ✅ Successfully resampled objects on table {table_id}", file=sys.stderr)
                    
                    # Save to temp JSON for subsequent operations
                    export_layout_to_json(current_layout, temp_json_path)
                
                # Step 3: If sampling failed, restore original objects and break to remove objects
                if not sampling_success:
                    print(f"  ⚠️ Object resampling failed, restoring original objects and breaking to remove objects...", file=sys.stderr)
                    target_room.objects = original_room_objects
                    export_layout_to_json(current_layout, temp_json_path)
                    break
                
                print(f"  ✅ All object resampling succeeded", file=sys.stderr)
                
                # Step 4: Now run trajectory planning without resampling pick/place objects
                print("Step 4: Planning trajectories...", file=sys.stderr)
                all_trajectories_valid = True
                current_robot_pos = spawn_position
                current_robot_quat = spawn_quat
                
                trajectory_results = []
                
                for pair_idx, pair in enumerate(pick_place_pairs):
                    print(f"\n  Pick-Place Pair {pair_idx + 1}/{len(pick_place_pairs)}", file=sys.stderr)
                    
                    # Get pick and place object IDs based on format
                    if use_updated_format:
                        # Use updated format: target_object_id for pick, location_object_id for place
                        pick_obj_id = pair["pick"].get("target_object_id")
                        place_location_obj_id = pair["place"].get("location_object_id")
                        
                        if not pick_obj_id:
                            print(f"    Warning: No target_object_id in pick task", file=sys.stderr)
                            all_trajectories_valid = False
                            break
                        
                        if not place_location_obj_id:
                            print(f"    Warning: No location_object_id in place task", file=sys.stderr)
                            all_trajectories_valid = False
                            break
                        
                        place_obj_id = place_location_obj_id
                    else:
                        # Use original format: target_object with object_mapping
                        pick_obj_name = pair["pick"]["target_object"]
                        place_obj_name = pair["place"]["target_object"]
                        
                        # Find actual object IDs from mapping
                        pick_mapping = object_mapping.get(pick_obj_name, {})
                        pick_obj_ids = pick_mapping.get("matched_ids", [])
                        
                        if not pick_obj_ids:
                            print(f"    Warning: No matched objects for pick target '{pick_obj_name}'", file=sys.stderr)
                            all_trajectories_valid = False
                            break
                        
                        pick_obj_id = pick_obj_ids[0]  # Use first matched object
                        
                        place_mapping = object_mapping.get(place_obj_name, {})
                        place_obj_ids = place_mapping.get("matched_ids", [])
                        
                        if not place_obj_ids:
                            print(f"    Warning: No matched objects for place target '{place_obj_name}'", file=sys.stderr)
                            all_trajectories_valid = False
                            break
                        
                        place_obj_id = place_obj_ids[0]  # Use first matched object
                    
                    print(f"    Pick object: {pick_obj_id}", file=sys.stderr)
                    print(f"    Place object: {place_obj_id}", file=sys.stderr)
                    
                    # Find pick table
                    pick_obj = next((obj for obj in target_room.objects if obj.id == pick_obj_id), None)
                    if not pick_obj:
                        print(f"    Warning: Pick object {pick_obj_id} not found", file=sys.stderr)
                        all_trajectories_valid = False
                        break
                    
                    pick_table_id = pick_obj.place_id
                    if pick_table_id not in ["floor", "wall"]:
                        # Pick object is on another object, use that as the table
                        pick_table_obj = next((obj for obj in target_room.objects if obj.id == pick_table_id), None)
                        if pick_table_obj:
                            pick_table_id = pick_table_obj.id
                    
                    # Sample robot positions for pick (using current pick object pose)
                    print(f"    Sampling robot positions for picking {pick_obj_id}...", file=sys.stderr)
                    try:
                        pick_robot_positions, pick_robot_quats, _ = sample_robot_location(
                            layout_save_dir,
                            temp_layout_name,
                            room_id,
                            pick_obj_id,
                            pick_table_id,
                            num_envs=5,  # Sample 5 candidates
                            debug_dir=debug_dir
                        )
                    except Exception as e:
                        print(f"    ❌ Failed to sample robot positions for pick: {e}", file=sys.stderr)
                        pick_robot_positions = None
                    
                    if pick_robot_positions is None or len(pick_robot_positions) == 0:
                        print(f"    ❌ No valid pick robot poses found for {pick_obj_id}", file=sys.stderr)
                        # Get pick table position for logging
                        pick_table_pos = None
                        if pick_table_id not in ["floor", "wall"]:
                            pick_table_obj = next((obj for obj in target_room.objects if obj.id == pick_table_id), None)
                            if pick_table_obj:
                                pick_table_pos = [pick_table_obj.position.x, pick_table_obj.position.y, pick_table_obj.position.z]
                        
                        trajectory_failure_log.append({
                            "pair_idx": pair_idx,
                            "step": "pick",
                            "start_pos": current_robot_pos.cpu().numpy().tolist(),
                            "end_pos": pick_table_pos,
                            "pick_obj_id": pick_obj_id,
                            "pick_table_id": pick_table_id,
                            "place_obj_id": place_obj_id
                        })
                        all_trajectories_valid = False
                        break
                    
                    # Test trajectory from current position to each pick pose
                    print(f"    Testing trajectory from current pos to pick poses...", file=sys.stderr)
                    
                    pick_trajectory_found = False
                    selected_pick_robot_pos = None
                    selected_pick_robot_quat = None
                    
                    for i in range(len(pick_robot_positions)):
                        pick_robot_pos = pick_robot_positions[i]
                        pick_robot_quat = pick_robot_quats[i]
                        
                        # Plan trajectory
                        trajectory, plan_successful = plan_robot_traj(
                            current_robot_pos,
                            current_robot_quat,
                            pick_robot_pos,
                            pick_robot_quat,
                            layout_save_dir,
                            temp_layout_name,
                            room_id,
                            debug_dir,
                            return_plan_status=True
                        )
                        
                        if plan_successful:
                            pick_trajectory_found = True
                            selected_pick_robot_pos = pick_robot_pos
                            selected_pick_robot_quat = pick_robot_quat
                            print(f"    ✅ Valid trajectory to pick pose found", file=sys.stderr)
                            break
                    
                    if not pick_trajectory_found:
                        print(f"    ❌ No valid trajectory to any pick pose", file=sys.stderr)
                        trajectory_failure_log.append({
                            "pair_idx": pair_idx,
                            "step": "pick",
                            "start_pos": current_robot_pos.cpu().numpy().tolist(),
                            "end_pos": pick_robot_positions[0].cpu().numpy().tolist() if len(pick_robot_positions) > 0 else None,
                            "pick_obj_id": pick_obj_id,
                            "pick_table_id": pick_table_id,
                            "place_obj_id": place_obj_id
                        })
                        all_trajectories_valid = False
                        break
                    
                    # Update current robot pose to pick pose
                    current_robot_pos = selected_pick_robot_pos
                    current_robot_quat = selected_pick_robot_quat
                    
                    # Determine place_table_id
                    place_obj = next((obj for obj in target_room.objects if obj.id == place_obj_id), None)
                    if not place_obj:
                        print(f"    Warning: Place object {place_obj_id} not found", file=sys.stderr)
                        all_trajectories_valid = False
                        break
                    
                    # If placing directly on table/surface
                    if place_obj.place_id in ["floor", "wall"]:
                        place_table_id = place_obj_id
                    else:
                        # Placing on another object
                        place_table_id = place_obj.place_id
                    
                    # Sample robot positions for place (using current place object pose)
                    print(f"    Sampling robot positions for placing at {place_obj_id}...", file=sys.stderr)
                    try:
                        place_robot_positions, place_robot_quats, _ = sample_robot_location(
                            layout_save_dir,
                            temp_layout_name,
                            room_id,
                            place_obj_id,
                            place_table_id,
                            num_envs=5,  # Sample 5 candidates
                            debug_dir=debug_dir
                        )
                    except Exception as e:
                        print(f"    ❌ Failed to sample robot positions for place: {e}", file=sys.stderr)
                        place_robot_positions = None
                    
                    if place_robot_positions is None or len(place_robot_positions) == 0:
                        print(f"    ❌ No valid place robot poses found", file=sys.stderr)
                        # Get place table position for logging
                        place_table_pos = None
                        if place_table_id not in ["floor", "wall"]:
                            place_table_obj = next((obj for obj in target_room.objects if obj.id == place_table_id), None)
                            if place_table_obj:
                                place_table_pos = [place_table_obj.position.x, place_table_obj.position.y, place_table_obj.position.z]
                        
                        trajectory_failure_log.append({
                            "pair_idx": pair_idx,
                            "step": "place",
                            "start_pos": current_robot_pos.cpu().numpy().tolist(),
                            "end_pos": place_table_pos,
                            "pick_obj_id": pick_obj_id,
                            "place_obj_id": place_obj_id
                        })
                        all_trajectories_valid = False
                        break
                    
                    # Test trajectory from pick to place
                    print(f"    Testing trajectory from pick to place poses...", file=sys.stderr)
                    
                    place_trajectory_found = False
                    selected_place_robot_pos = None
                    selected_place_robot_quat = None
                    
                    for i in range(len(place_robot_positions)):
                        place_robot_pos = place_robot_positions[i]
                        place_robot_quat = place_robot_quats[i]
                        
                        # Plan trajectory
                        trajectory, plan_successful = plan_robot_traj(
                            current_robot_pos,
                            current_robot_quat,
                            place_robot_pos,
                            place_robot_quat,
                            layout_save_dir,
                            temp_layout_name,
                            room_id,
                            debug_dir,
                            return_plan_status=True
                        )
                        
                        if plan_successful:
                            place_trajectory_found = True
                            selected_place_robot_pos = place_robot_pos
                            selected_place_robot_quat = place_robot_quat
                            print(f"    ✅ Valid trajectory to place pose found", file=sys.stderr)
                            break
                    
                    if not place_trajectory_found:
                        print(f"    ❌ No valid trajectory to any place pose", file=sys.stderr)
                        trajectory_failure_log.append({
                            "pair_idx": pair_idx,
                            "step": "place",
                            "start_pos": current_robot_pos.cpu().numpy().tolist(),
                            "end_pos": place_robot_positions[0].cpu().numpy().tolist() if len(place_robot_positions) > 0 else None,
                            "pick_obj_id": pick_obj_id,
                            "place_obj_id": place_obj_id
                        })
                        all_trajectories_valid = False
                        break
                    
                    # Store results for this pair
                    trajectory_results.append({
                        "pick_robot_pos": selected_pick_robot_pos.cpu().numpy().tolist(),
                        "pick_robot_quat": selected_pick_robot_quat.cpu().numpy().tolist(),
                        "place_robot_pos": selected_place_robot_pos.cpu().numpy().tolist(),
                        "place_robot_quat": selected_place_robot_quat.cpu().numpy().tolist(),
                        "pick_obj_id": pick_obj_id,
                        "place_obj_id": place_obj_id
                    })
                    
                    # Update current position for next pick-place pair
                    current_robot_pos = selected_place_robot_pos
                    current_robot_quat = selected_place_robot_quat
                
                # Check if all trajectories are valid
                if all_trajectories_valid:
                    print(f"\n✅ All trajectories valid! Object placements were updated in step 2.", file=sys.stderr)
                    
                    # Save final layout to temp path
                    export_layout_to_json(current_layout, temp_json_path)
                    
                    valid_trajectory_found = True
                    
                    return json.dumps({
                        "success": True,
                        "message": "Mobile franka correction successful - all trajectories valid",
                        "task_decomposition_format": "updated" if use_updated_format else "original",
                        "iterations": {
                            "outer": outer_iter + 1,
                            "inner": inner_iter + 1
                        },
                        "pick_place_pairs_validated": len(pick_place_pairs),
                        "trajectory_results": trajectory_results
                    }, indent=2)
                else:
                    # Validation failed, restore original room objects
                    print(f"\n⚠️ Trajectory validation failed, restoring original room objects...", file=sys.stderr)
                    target_room.objects = original_room_objects
                    
                    # Save restored layout to temp path
                    export_layout_to_json(current_layout, temp_json_path)
                    print(f"   Layout restored to original state", file=sys.stderr)
                    
            except Exception as e:
                print(f"  Error in iteration: {str(e)}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                
                # Rollback to original room objects if they exist
                if 'original_room_objects' in locals():
                    print(f"  Rolling back to original room objects due to error...", file=sys.stderr)
                    target_room.objects = original_room_objects
                    export_layout_to_json(current_layout, temp_json_path)
                    print(f"  Layout restored to original state after error", file=sys.stderr)
                
                continue
        
        # After max_inner_iterations, remove blocking objects if needed
        if not valid_trajectory_found:
            print(f"\n⚠️ No valid trajectories found after {max_inner_iterations} attempts", file=sys.stderr)
            print(f"Attempting to remove blocking objects...", file=sys.stderr)
            
            # Get minimum required object IDs
            required_obj_ids = set()
            for req_obj in minimum_required_objects:
                matched_ids = req_obj.get("matched_object_ids", [])
                required_obj_ids.update(matched_ids)
            
            # Also exclude pick and place objects from removal
            for pair in pick_place_pairs:
                if use_updated_format:
                    pick_obj_id = pair["pick"].get("target_object_id")
                    place_obj_id = pair["place"].get("location_object_id")
                    if pick_obj_id:
                        required_obj_ids.add(pick_obj_id)
                    if place_obj_id:
                        required_obj_ids.add(place_obj_id)
                else:
                    pick_obj_name = pair["pick"]["target_object"]
                    place_obj_name = pair["place"]["target_object"]
                    
                    pick_mapping = object_mapping.get(pick_obj_name, {})
                    pick_obj_ids = pick_mapping.get("matched_ids", [])
                    required_obj_ids.update(pick_obj_ids)
                    
                    place_mapping = object_mapping.get(place_obj_name, {})
                    place_obj_ids = place_mapping.get("matched_ids", [])
                    required_obj_ids.update(place_obj_ids)
            
            # Analyze trajectory failures to find blocking objects
            print(f"  Analyzing {len(trajectory_failure_log)} trajectory failures...", file=sys.stderr)
            
            obj_to_remove = None
            
            if trajectory_failure_log:
                # Select the failure that occurs at the farthest/most advanced step
                # Priority: later pick-place pairs > earlier pairs, and place > pick within same pair
                def failure_priority(failure):
                    """
                    Calculate priority for a failure. Higher value = more advanced step.
                    Format: (pair_idx, step_priority)
                    where step_priority is 1 for place, 0 for pick
                    """
                    pair_idx = failure["pair_idx"]
                    step = failure["step"]
                    step_priority = 1 if step == "place" else 0
                    return (pair_idx, step_priority)
                
                # Sort failures by priority and select the one at the farthest step
                recent_failure = max(trajectory_failure_log, key=failure_priority)
                
                pair_idx = recent_failure["pair_idx"]
                step = recent_failure["step"]
                start_pos = recent_failure["start_pos"]
                end_pos = recent_failure["end_pos"]
                
                print(f"  Selected failure (farthest step): Pick-Place Pair {pair_idx + 1}, Step: {step}", file=sys.stderr)
                print(f"  Start pos: {start_pos}", file=sys.stderr)
                print(f"  End pos (robot): {end_pos}", file=sys.stderr)
                
                if start_pos:
                    # Convert start position to numpy array
                    start_pos_np = np.array(start_pos[:2])  # Use only x, y
                    
                    # Use the actual target object position (pick table or place table) as end_pos
                    target_obj_id = None
                    if step == "pick":
                        # For pick step, the target is the pick table (where the object sits)
                        target_obj_id = recent_failure.get("pick_table_id")
                    elif step == "place":
                        # For place step, the target is the place table/object
                        target_obj_id = recent_failure.get("place_table_id")
                    
                    # Find the target object and use its position
                    end_pos_np = None
                    if target_obj_id:
                        target_obj = next((obj for obj in target_room.objects if obj.id == target_obj_id), None)
                        if target_obj:
                            end_pos_np = np.array([target_obj.position.x, target_obj.position.y])
                            print(f"  End pos (target object {target_obj_id}): {end_pos_np}", file=sys.stderr)
                        else:
                            print(f"  Warning: Target object {target_obj_id} not found", file=sys.stderr)
                    
                    # Fallback to robot end position if target object not found
                    if end_pos_np is None and end_pos:
                        end_pos_np = np.array(end_pos[:2])
                        print(f"  Using fallback end pos (robot): {end_pos_np}", file=sys.stderr)
                    
                    if end_pos_np is not None:
                        # Get all objects on floor or wall (potential blockers)
                        floor_wall_objects = [obj for obj in target_room.objects if obj.place_id in ["floor"]]
                        
                        # Find the closest object to the failed trajectory line
                        obj_to_remove = find_closest_object_to_trajectory_line(
                            start_pos_np,
                            end_pos_np,
                            floor_wall_objects,
                            required_obj_ids
                        )
                    
                    if obj_to_remove:
                        print(f"  Found blocking object: {obj_to_remove.type} ({obj_to_remove.id})", file=sys.stderr)
                        obj_pos = np.array([obj_to_remove.position.x, obj_to_remove.position.y])
                        print(f"  Object position: {obj_pos}", file=sys.stderr)
            
            if not obj_to_remove:
                # Fallback: no valid trajectory info or no object found
                # Get all removable floor/wall objects
                floor_wall_objects = [obj for obj in target_room.objects if obj.place_id in ["floor"]]
                removable_objects = [obj for obj in floor_wall_objects if obj.id not in required_obj_ids]
                
                if not removable_objects:
                    print(f"  No removable objects left - only minimum required objects remain", file=sys.stderr)
                    return json.dumps({
                        "success": False,
                        "error": "Cannot find valid trajectories even with minimum required objects",
                        "message": "The room layout may need manual adjustment or more space",
                        "iterations": {
                            "outer": outer_iter + 1,
                            "inner": max_inner_iterations
                        }
                    }, indent=2)
                
                # Remove a random removable object as last resort
                import random
                obj_to_remove = random.choice(removable_objects)
                print(f"  Fallback: removing random object: {obj_to_remove.type} ({obj_to_remove.id})", file=sys.stderr)
            
            if obj_to_remove:
                print(f"  Removing object: {obj_to_remove.type} ({obj_to_remove.id})", file=sys.stderr)
                
                # Visualize the object removal decision
                try:
                    from objects.object_mobile_manipulation_utils import (
                        create_unified_occupancy_grid,
                        visualize_object_removal_decision
                    )
                    from datetime import datetime
                    
                    # Create occupancy grid for visualization (use temp layout name to reflect removed objects)
                    temp_layout_name = os.path.splitext(os.path.basename(temp_json_path))[0]
                    occupancy_grid, grid_x, grid_y, room_bounds, _, _, _ = create_unified_occupancy_grid(
                        layout_save_dir, temp_layout_name, room_id
                    )
                    
                    # Get the trajectory information for visualization
                    if trajectory_failure_log:
                        print(f"  Trajectory failure log: {len(trajectory_failure_log)} {trajectory_failure_log}", file=sys.stderr)
                        # Use the same logic to select the failure at the farthest step
                        def failure_priority_viz(failure):
                            pair_idx = failure["pair_idx"]
                            step = failure["step"]
                            step_priority = 1 if step == "place" else 0
                            return (pair_idx, step_priority)
                        
                        recent_failure = max(trajectory_failure_log, key=failure_priority_viz)
                        start_pos_viz = np.array(recent_failure["start_pos"][:2])
                        
                        # Use the actual target object position (pick table or place table) as end_pos
                        step = recent_failure["step"]
                        target_obj_id = None
                        
                        if step == "pick":
                            # For pick step, the target is the pick table (where the object sits)
                            target_obj_id = recent_failure.get("pick_table_id")
                        elif step == "place":
                            # For place step, the target is the place table/object
                            target_obj_id = recent_failure.get("place_table_id")
                        
                        # Find the target object and use its position
                        if target_obj_id:
                            target_obj = next((obj for obj in target_room.objects if obj.id == target_obj_id), None)
                            if target_obj:
                                end_pos_viz = np.array([target_obj.position.x, target_obj.position.y])
                            else:
                                # Fallback to robot end position if object not found
                                end_pos_viz = np.array(recent_failure["end_pos"][:2]) if recent_failure["end_pos"] else start_pos_viz
                        else:
                            # Fallback to robot end position if no target object ID
                            end_pos_viz = np.array(recent_failure["end_pos"][:2]) if recent_failure["end_pos"] else start_pos_viz
                    else:
                        print(f"  No target object ID found, using fallback trajectory", file=sys.stderr)
                        # Fallback: use a dummy trajectory (e.g., from room center to object)
                        room_center = np.array([
                            (room_bounds[0] + room_bounds[2]) / 2,
                            (room_bounds[1] + room_bounds[3]) / 2
                        ])
                        obj_pos_viz = np.array([obj_to_remove.position.x, obj_to_remove.position.y])
                        start_pos_viz = room_center
                        end_pos_viz = obj_pos_viz
                    
                    # Get all floor objects for visualization
                    floor_wall_objects = [obj for obj in target_room.objects if obj.place_id in ["floor"]]
                    
                    # Create save path with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    removal_viz_dir = os.path.join(debug_dir, "object_removal")
                    os.makedirs(removal_viz_dir, exist_ok=True)
                    save_path = os.path.join(
                        removal_viz_dir,
                        f"object_removal_{current_layout.id}_{room_id}_iter{outer_iter+1}_{timestamp}.png"
                    )
                    
                    # Call visualization function
                    visualize_object_removal_decision(
                        room_bounds=room_bounds,
                        occupancy_grid=occupancy_grid,
                        grid_x=grid_x,
                        grid_y=grid_y,
                        grid_res=0.05,  # From CollisionCheckingConfig.GRID_RES
                        start_pos=start_pos_viz,
                        end_pos=end_pos_viz,
                        all_objects=floor_wall_objects,
                        required_obj_ids=required_obj_ids,
                        closest_object=obj_to_remove,
                        layout_name=current_layout.id,
                        room_id=room_id,
                        save_path=save_path
                    )
                    
                    print(f"  Object removal visualization saved to: {save_path}", file=sys.stderr)
                    
                except Exception as e:
                    print(f"  Warning: Failed to create object removal visualization: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                
                # Remove object and its children
                objects_to_remove = [obj_to_remove.id]
                
                # Find children recursively
                for obj in target_room.objects:
                    current_obj = obj
                    while current_obj.place_id not in ["floor", "wall"]:
                        if current_obj.place_id == obj_to_remove.id:
                            objects_to_remove.append(obj.id)
                            break
                        parent = next((o for o in target_room.objects if o.id == current_obj.place_id), None)
                        if not parent:
                            break
                        current_obj = parent
                
                # Remove objects from room
                target_room.objects = [obj for obj in target_room.objects if obj.id not in objects_to_remove]
                
                # Update current_layout
                for room_idx, layout_room in enumerate(current_layout.rooms):
                    if layout_room.id == room_id:
                        layout_room.objects = target_room.objects
                        current_layout.rooms[room_idx] = layout_room
                        break
                
                # Save updated layout to temp path
                export_layout_to_json(current_layout, temp_json_path)
                
                print(f"  Removed {len(objects_to_remove)} objects, continuing...", file=sys.stderr)
                continue  # Try again in next outer iteration
            
            # If we couldn't find object to remove, just continue
            print(f"  Could not determine object to remove, continuing...", file=sys.stderr)
    
    # If we exit the loop without finding valid trajectories
    return json.dumps({
        "success": False,
        "error": "Failed to find valid trajectories after maximum iterations",
        "message": "Consider simplifying the scene or adjusting object positions manually",
        "task_decomposition_format": "updated" if use_updated_format else "original",
        "pick_place_pairs_attempted": len(pick_place_pairs),
        "iterations": {
            "outer": max_outer_iterations,
            "inner": max_inner_iterations
        }
    }, indent=2)


async def robot_task_feasibility_correction_for_room_standalone(layout: FloorPlan, room_id: str = "", do_explicit_correction: bool = False) -> str:
    """
    After generating scene with objects inside the room, check whether the robot can theoretically complete 
    the task in the room and adjust object placement if needed. This is a standalone version that accepts
    a FloorPlan object directly.

    For example:
    
    1. If the robot is franka arm for pick and place task, then all objects must be reachable by the franka arm. 
       If not, we will adjust the objects placement.
    2. If the robot is mobile franka for mobile pick and place task, then besides reachability, we also need to 
       consider whether a collision-free path exists for the mobile robot to move inside the room. If not, we need 
       to remove or adjust the location of the objects.

    Args:
        layout: The FloorPlan object to correct
        room_id: The ID of the room to analyze

    Return:
        JSON string containing the correction result
    """
    current_layout = layout
    policy_analysis = current_layout.policy_analysis

    robot_type = policy_analysis["robot_type"]
    room_type = policy_analysis["room_type"]
    minimum_required_objects = policy_analysis["minimum_required_objects"]
    task_decomposition = policy_analysis["task_decomposition"]

    # match the policy analysis with the room
    target_room = next((r for r in current_layout.rooms if r.id == room_id), None)
    if target_room is None:
        return json.dumps({
            "success": False,
            "error": f"Room with ID '{room_id}' not found"
        })
    
    # Match the objects in policy analysis with the objects in the room using LLM
    print("🔍 Matching policy analysis objects with actual room objects...", file=sys.stderr)
    
    # Prepare room objects information
    room_objects_info = []
    for obj in target_room.objects:
        room_objects_info.append({
            "id": obj.id,
            "type": obj.type,
            "description": obj.description,
            "dimensions": {
                "width": f"{obj.dimensions.width:.2f}m",
                "length": f"{obj.dimensions.length:.2f}m",
                "height": f"{obj.dimensions.height:.2f}m"
            },
            "place_id": obj.place_id
        })
    
    # Create matching prompt for LLM
    matching_prompt = f"""You are a robotics task planning expert. Match the required objects from the robot policy analysis to the actual objects present in the room.

ROBOT TASK CONTEXT:
- Robot Type: {robot_type}
- Room Type: {room_type}

POLICY ANALYSIS - REQUIRED OBJECTS:
{json.dumps(minimum_required_objects, indent=2)}

TASK DECOMPOSITION (for context):
{json.dumps(task_decomposition, indent=2)}

ACTUAL ROOM OBJECTS:
{json.dumps(room_objects_info, indent=2)}

TASK:
Create a precise mapping from each required object in the policy analysis to the most appropriate actual object(s) in the room.

MATCHING CRITERIA:
1. Match by object type similarity (e.g., "table" matches "table", "coffee_table", "dining_table", etc.)
2. Consider the placement guidance as well. The inter object relationship should be considered mapped well.
2. Respect required quantity - map to exact number of room objects
3. Consider object dimensions and descriptions for semantic appropriateness
4. Prioritize objects that make sense for the robot task context
5. If multiple objects of same type exist, select the most task-appropriate ones
6. If you can't find a suitable match, you need to find the most similar object in the room and use it as a fallback.

OUTPUT FORMAT (JSON):
```json
{{
    "mappings": [
        {{
            "required_object_type": "object_type from minimum_required_objects",
            "required_quantity": number,
            "required_placement_guidance": "placement guidance from policy",
            "matched_object_ids": ["room_object_id_1", "room_object_id_2"],
            "matched_object_types": ["actual_type_1", "actual_type_2"],
            "match_confidence": "high|medium|low",
            "match_reasoning": "Brief explanation of why these objects were matched"
        }}
    ],
    "unmatched_requirements": [
        {{
            "object_type": "required object type",
            "quantity": number,
            "reason": "Why no suitable match was found in the room"
        }}
    ],
    "updated_task_decomposition": [
        {{
            "step": the step number of the task decomposition
            "action": "pick/place/navigate",
            "target_object_id": "room_object_id",
            "location_object_id": "another_room_object_id"
        }}
    ],
    "summary": {{
        "total_required_objects": number,
        "successfully_matched": number,
        "unmatched_count": number,
        "overall_confidence": "high|medium|low"
    }}
}}
```

Be strict - only match objects that are semantically appropriate for the robot task."""

    # Call OpenAI LLM for matching
    try:
        response = call_vlm(
            vlm_type="openai",
            model="openai/gpt-oss-120b",
            max_tokens=4000,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": matching_prompt
                }
            ]
        )
        
        response_text = response.content[0].text.strip()
        
        # Parse the LLM response
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError("Could not extract JSON content from LLM response")
        
        matching_result = json.loads(response_text)
        print(f"📊 Matching result: {json.dumps(matching_result, indent=2)}", file=sys.stderr)
        
        # Create the object mapping dictionary
        object_mapping = {}
        for mapping in matching_result.get("mappings", []):
            required_type = mapping.get("required_object_type")
            matched_ids = mapping.get("matched_object_ids", [])
            object_mapping[required_type] = {
                "matched_ids": matched_ids,
                "matched_types": mapping.get("matched_object_types", []),
                "confidence": mapping.get("match_confidence", "unknown"),
                "reasoning": mapping.get("match_reasoning", "")
            }
        
        print(f"✅ Object matching complete: {len(object_mapping)} mappings created", file=sys.stderr)
        
        # Update policy analysis with actual object IDs
        updated_policy_analysis = policy_analysis.copy()
        
        # Update minimum_required_objects with matched IDs
        for req_obj in updated_policy_analysis["minimum_required_objects"]:
            obj_type = req_obj["object_type"]
            if obj_type in object_mapping:
                req_obj["matched_object_ids"] = object_mapping[obj_type]["matched_ids"]
                req_obj["matched_object_types"] = object_mapping[obj_type]["matched_types"]
                req_obj["match_confidence"] = object_mapping[obj_type]["confidence"]
            else:
                req_obj["matched_object_ids"] = []
                req_obj["matched_object_types"] = []
                req_obj["match_confidence"] = "none"
        
        # Update task_decomposition with actual object IDs where applicable
        for task in updated_policy_analysis.get("task_decomposition", []):
            target_obj_name = task.get("target_object", "")
            # Try to find matching in object_mapping
            if target_obj_name in object_mapping and object_mapping[target_obj_name]["matched_ids"]:
                task["actual_object_ids"] = object_mapping[target_obj_name]["matched_ids"]
                task["actual_object_types"] = object_mapping[target_obj_name]["matched_types"]
        
        # Process updated_task_decomposition from LLM if provided
        updated_task_decomposition = matching_result.get("updated_task_decomposition", [])
        if updated_task_decomposition:
            print(f"📋 Processing updated task decomposition with {len(updated_task_decomposition)} steps", file=sys.stderr)
            updated_policy_analysis["updated_task_decomposition"] = updated_task_decomposition
        else:
            print(f"⚠️ No updated_task_decomposition provided by LLM, will use original task decomposition", file=sys.stderr)
            updated_policy_analysis["updated_task_decomposition"] = []
        
        # Store the updated policy analysis and mapping
        updated_policy_analysis["object_mapping"] = object_mapping
        updated_policy_analysis["unmatched_requirements"] = matching_result.get("unmatched_requirements", [])
        updated_policy_analysis["matching_summary"] = matching_result.get("summary", {})
        
        # Update the current_layout's policy_analysis
        current_layout.policy_analysis = updated_policy_analysis
        policy_analysis = updated_policy_analysis

        print(f"📊 Updated policy analysis: {json.dumps(updated_policy_analysis, indent=2)}", file=sys.stderr)
        
        print(f"📊 Matching Summary: {json.dumps(matching_result.get('summary', {}), indent=2)}", file=sys.stderr)
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse LLM matching response: {str(e)}", file=sys.stderr)
        return json.dumps({
            "success": False,
            "error": f"Failed to parse LLM matching response: {str(e)}",
            "raw_response": response_text[:500] if response_text else "No response"
        })
    except Exception as e:
        print(f"❌ Error during object matching: {str(e)}", file=sys.stderr)
        return json.dumps({
            "success": False,
            "error": f"Error during object matching: {str(e)}"
        })

    # Create temp JSON path for correction process
    layout_save_dir = os.path.join(RESULTS_DIR, current_layout.id)
    temp_json_path = os.path.join(layout_save_dir, f"{current_layout.id}_temp.json")
    original_json_path = os.path.join(RESULTS_DIR, current_layout.id, f'{current_layout.id}.json')

    # save the current layout to the original_json_path
    export_layout_to_json(current_layout, original_json_path)

    if do_explicit_correction:
    
        if robot_type == "franka":
            pass
        elif robot_type == "mobile_franka":
            correction_result = await correct_mobile_franka_standalone(current_layout, room_id, temp_json_path)
        else:
            return json.dumps({
                "success": False,
                "error": f"Invalid robot type: {robot_type}"
            })
    
        # Parse correction result to check success
        correction_result_dict = json.loads(correction_result) if isinstance(correction_result, str) else correction_result

        if correction_result_dict.get("success", False):
            # cp the temp json to the original json
            shutil.copy2(temp_json_path, original_json_path)
            print(f"Copied temp JSON to original JSON: {original_json_path}")


    get_room_layout_scene_usd_separate_from_layout(
        original_json_path,
        os.path.join(RESULTS_DIR, current_layout.id, f"{current_layout.id}_usd_collection")
    )
    
    return json.dumps({
        "success": True,
        "robot_type": robot_type,
        "room_id": room_id,
        "matching_summary": updated_policy_analysis.get("matching_summary", {}),
        "task_decomposition_info": {
            "original_steps": len(task_decomposition),
            "updated_steps": len(updated_policy_analysis.get("updated_task_decomposition", [])),
            "has_updated_decomposition": bool(updated_policy_analysis.get("updated_task_decomposition", []))
        },
    }, indent=2)


async def test_load_layout(layout_id: str, room_id: str = None):
    """Test loading layout from JSON file and running mobile franka correction"""
    
    try:
        layout_save_path = os.path.join(RESULTS_DIR, layout_id, layout_id+".json")
        scene_save_dir = os.path.join(RESULTS_DIR, layout_id)

        # Load the layout directly from JSON
        print(f"Loading layout from: {layout_save_path}")
        with open(layout_save_path, 'r') as f:
            layout_dict = json.load(f)
        
        # Convert dictionary to FloorPlan object
        current_layout = dict_to_floor_plan(layout_dict)
        print(f"Layout loaded successfully: {current_layout.id}")
        
        # If room_id is not specified, use the first room
        if room_id is None:
            if not current_layout.rooms:
                print("ERROR: No rooms found in layout")
                return
            room_id = current_layout.rooms[0].id
            print(f"Using first room: {room_id}")
        
        # Check if layout has policy_analysis
        if not hasattr(current_layout, 'policy_analysis') or current_layout.policy_analysis is None:
            print("ERROR: Layout does not have policy_analysis. This script requires a layout with robot task information.")
            return
        
        print(f"\nStarting robot task feasibility correction for room: {room_id}")
        print(f"Robot type: {current_layout.policy_analysis.get('robot_type', 'N/A')}")
        print("="*80)
        
        # Run the correction (passing the layout object)
        correction_result = await robot_task_feasibility_correction_for_room_standalone(current_layout, room_id, do_explicit_correction=True)
        
        # Parse and display the result
        result_dict = json.loads(correction_result)
        print("\n" + "="*80)
        print("CORRECTION RESULT:")
        print(json.dumps(result_dict, indent=2))
        print("="*80)
        
        if result_dict.get("success", False):
            print("\n✅ Correction completed successfully!")
            
            # Get the temp JSON path from the result
            temp_json_path = result_dict.get("temp_json_path")
            
            if temp_json_path and os.path.exists(temp_json_path):
                # Copy temp file to original location
                print(f"\nOverwriting original layout JSON: {layout_save_path}")
                shutil.copy2(temp_json_path, layout_save_path)
                
                # Clean up temp file
                try:
                    os.remove(temp_json_path)
                    print(f"Cleaned up temp file: {temp_json_path}")
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_json_path}: {e}")
            else:
                print(f"Warning: Temp JSON path not found or doesn't exist")
            
            # Reload the corrected layout for GLB export
            with open(layout_save_path, 'r') as f:
                corrected_layout_dict = json.load(f)
            corrected_layout = dict_to_floor_plan(corrected_layout_dict)
            
            # Export GLB for visualization
            print("\nExporting corrected layout to GLB...")
            export_glb_path = os.path.join(scene_save_dir, "layout_corrected.glb")
            mesh_dict_list = export_layout_to_mesh_dict_list(corrected_layout)
            scene = create_glb_scene()
            for mesh_id, mesh_data in mesh_dict_list.items():
                mesh_data_dict = {
                    'vertices': mesh_data['mesh'].vertices,
                    'faces': mesh_data['mesh'].faces,
                    'vts': mesh_data['texture']['vts'],
                    'fts': mesh_data['texture']['fts'],
                    'texture_image': np.array(Image.open(mesh_data['texture']['texture_map_path']))
                }
                add_textured_mesh_to_glb_scene(mesh_data_dict, scene, material_name=f"material_{mesh_id}", mesh_name=f"mesh_{mesh_id}", preserve_coordinate_system=True)
            save_glb_scene(export_glb_path, scene)
            print(f"GLB exported to: {export_glb_path}")
        else:
            print(f"\n❌ Correction failed: {result_dict.get('error', 'Unknown error')}")
            
            # Clean up temp file if it exists
            temp_json_path = result_dict.get("temp_json_path")
            if temp_json_path and os.path.exists(temp_json_path):
                try:
                    os.remove(temp_json_path)
                    print(f"Cleaned up temp file: {temp_json_path}")
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_json_path}: {e}")
            
            print(f"Original layout JSON remains unchanged: {layout_save_path}")
        
    except Exception as e:
        print(f"ERROR: Exception occurred during test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    parser = argparse.ArgumentParser(description="Load a layout and run mobile franka correction")
    parser.add_argument("--layout_id", type=str, required=True, help="ID of the layout to load")
    parser.add_argument("--room_id", type=str, default=None, help="ID of the room to correct (defaults to first room)")
    args = parser.parse_args()
    asyncio.run(test_load_layout(args.layout_id, args.room_id))
