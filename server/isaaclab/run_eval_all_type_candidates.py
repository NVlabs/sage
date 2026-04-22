#!/usr/bin/env python3

"""
Wrapper script to evaluate all type candidates separately.
This avoids Isaac Sim simulation context conflicts by running each type candidate in a separate process.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

# Add parent directory to Python path to import constants
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import SERVER_ROOT_DIR

def get_type_candidates(scene_save_dir: str, type_aug_name: str) -> List[str]:
    """Get all available type candidate IDs."""
    metadata_path = os.path.join(scene_save_dir, type_aug_name, f"{type_aug_name}_type_candidates_metadata.json")
    
    if not os.path.exists(metadata_path):
        raise ValueError(f"Type candidates metadata file not found: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        type_metadata = json.load(f)
    
    return [candidate_info["layout_id"] for candidate_info in type_metadata["candidates"]]

def run_type_candidate_evaluation(
    type_candidate_id: str,
    type_aug_name: str,
    pose_aug_name: str,
    layout_id: str,
    room_id: str,
    target_object_name: str,
    place_object_name: str,
    table_object_name: str,
    num_envs: int = 1,
    task: str = "Isaac-Lift-Single-Obj-Scene-Franka-IK-Abs-v0",
    ckpt_path: str = None
) -> tuple[bool, dict]:
    """Run evaluation for a single type candidate."""
    
    print(f"\n{'='*60}")
    print(f"EVALUATING TYPE CANDIDATE: {type_candidate_id}")
    print(f"{'='*60}")
    
    # Construct command
    script_path = os.path.join(os.path.dirname(__file__), "data_generation_grasp_aug_pose_v2_eval_aug_type.py")
    
    cmd = [
        sys.executable, script_path,
        "--type_aug_name", type_aug_name,
        "--pose_aug_name", pose_aug_name,
        "--type_candidate_id", type_candidate_id,
        "--layout_id", layout_id,
        "--room_id", room_id,
        "--target_object_name", target_object_name,
        "--place_object_name", place_object_name,
        "--table_object_name", table_object_name,
        "--num_envs", str(num_envs),
        "--task", task,
        "--ckpt_path", ckpt_path,
        "--headless",  # Run headless to avoid GUI conflicts
        "--enable_cameras"
    ]
    
    # Expected results file path
    log_dir = os.path.abspath(os.path.join("./logs_eval", task+f"_{type_aug_name}_{pose_aug_name}_{os.path.abspath(ckpt_path).replace('/', '_')}"))
    results_file = os.path.join(log_dir, f"results_{type_candidate_id}.json")
    
    if not os.path.exists(results_file):
    
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the evaluation for this type candidate
        result = subprocess.run(cmd, timeout=3600)  # 1 hour timeout
        # result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
    
    # Load results from file
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results_data = json.load(f)

        print(f"results_data: {results_data}")

        if results_data.get("evaluation_completed", False):
            print(f"✓ Successfully completed evaluation for type candidate {type_candidate_id}")
            print(f"  Layouts: {results_data.get('total_layouts', 'Unknown')}")
            print(f"  Successes: {results_data.get('total_successes', 'Unknown')}")
            print(f"  Success Rate: {results_data.get('success_rate', 0.0):.3f}")
            return True, results_data
        else:
            return False, results_data
    else:
        print(f"✗ No results file found for type candidate {type_candidate_id}")
        print(f"  Expected file: {results_file}")
        return False, {}
            


def load_type_candidate_results(log_dir: str, type_candidate_ids: List[str]) -> Dict[str, dict]:
    """Load results from individual type candidate result files."""
    all_results = {}
    
    for type_candidate_id in type_candidate_ids:
        results_file = os.path.join(log_dir, f"results_{type_candidate_id}.json")
        
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    results_data = json.load(f)
                all_results[type_candidate_id] = results_data
                print(f"✓ Loaded results for {type_candidate_id}")
            except Exception as e:
                print(f"✗ Failed to load results for {type_candidate_id}: {e}")
                all_results[type_candidate_id] = {}
        else:
            print(f"✗ No results file found for {type_candidate_id}")
            all_results[type_candidate_id] = {}
    
    return all_results

def aggregate_results(
    log_dir: str,
    type_candidate_results: Dict[str, dict],
    type_aug_name: str,
    pose_aug_name: str
) -> Dict:
    """Aggregate results from all type candidate evaluations."""
    
    aggregated_results = {
        "type_aug_name": type_aug_name,
        "pose_aug_name": pose_aug_name,
        "type_candidates": {},
        "summary": {}
    }
    
    total_layouts = 0
    total_successes = 0
    successful_candidates = 0
    attempts_per_layout = 3
    all_success_rates = []
    
    for type_candidate_id, candidate_results in type_candidate_results.items():
        if candidate_results:  # If we have actual results
            candidate_layouts = candidate_results.get('total_layouts', 0)
            if candidate_layouts == 0:
                continue
            candidate_successes = candidate_results.get('total_successes', 0)
            candidate_success_rate = candidate_successes / (candidate_layouts * attempts_per_layout)
            
            total_layouts += candidate_layouts
            total_successes += candidate_successes
            
            if candidate_layouts > 0:
                all_success_rates.append(candidate_success_rate)
                successful_candidates += 1
            
            aggregated_results["type_candidates"][type_candidate_id] = {
                "processed": True,
                "total_layouts": candidate_layouts,
                "total_attempts": candidate_layouts * attempts_per_layout,
                "total_successes": candidate_successes,
                "success_rate": candidate_success_rate,
                "layout_results": candidate_results.get('layout_results', {})
            }
        else:
            aggregated_results["type_candidates"][type_candidate_id] = {
                "processed": False,
                "error": "Failed to complete evaluation"
            }
    
    # Calculate overall statistics
    overall_success_rate = total_successes / (total_layouts * attempts_per_layout) if total_layouts > 0 else 0.0
    avg_success_rate = sum(all_success_rates) / len(all_success_rates) if all_success_rates else 0.0
    
    aggregated_results["summary"] = {
        "total_type_candidates": len(type_candidate_results),
        "successful_candidates": successful_candidates,
        "total_layouts": total_layouts,
        "total_attempts": total_layouts * attempts_per_layout,
        "total_successes": total_successes,
        "overall_success_rate": overall_success_rate,
        "average_success_rate_per_candidate": avg_success_rate,
        "min_success_rate": min(all_success_rates) if all_success_rates else 0.0,
        "max_success_rate": max(all_success_rates) if all_success_rates else 0.0
    }
    
    return aggregated_results

def main():
    parser = argparse.ArgumentParser(description="Run evaluation for all type candidates separately.")
    parser.add_argument("--type_aug_name", type=str, required=True, help="Name of the type augmentation to evaluate.")
    parser.add_argument("--pose_aug_name", type=str, required=True, help="Name of the pose augmentation to evaluate.")
    parser.add_argument("--layout_id", type=str, default="layout_625b9812", help="Layout ID to evaluate.")
    parser.add_argument("--room_id", type=str, default="room_0736c934", help="Room ID to evaluate.")
    parser.add_argument("--target_object_name", type=str, default="room_0736c934_ceramic_mug_with_handle_dcaede53", help="Target object name.")
    parser.add_argument("--place_object_name", type=str, default="room_0736c934_ceramic_bowl_empty_opening_e3270589", help="Place object name.")
    parser.add_argument("--table_object_name", type=str, default="room_0736c934_wooden_rectangular_coffee_table_ad3b7c58", help="Table object name.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="Isaac-Lift-Single-Obj-Scene-Franka-IK-Abs-v0", help="Name of the task.")
    parser.add_argument("--specific_candidate", type=str, default=None, help="Run only a specific type candidate (for testing).")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint.")
    
    args = parser.parse_args()
    
    # Get scene directory
    scene_save_dir = os.path.join(SERVER_ROOT_DIR, f"results/{args.layout_id}")
    
    # Get all type candidate IDs
    try:
        if args.specific_candidate:
            type_candidate_ids = [args.specific_candidate]
            print(f"Running evaluation for specific type candidate: {args.specific_candidate}")
        else:
            type_candidate_ids = get_type_candidates(scene_save_dir, args.type_aug_name)
            print(f"Found {len(type_candidate_ids)} type candidates to evaluate")
    except Exception as e:
        print(f"Error loading type candidates: {e}")
        return 1
    
    # Create log directory
    log_dir = os.path.abspath(os.path.join("./logs_eval", args.task+f"_{args.type_aug_name}_{args.pose_aug_name}_{os.path.abspath(args.ckpt_path).replace('/', '_')}"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Run evaluation for each type candidate
    successful_evaluations = 0
    failed_evaluations = 0
    
    for i, type_candidate_id in enumerate(type_candidate_ids):
        print(f"\n[{i+1}/{len(type_candidate_ids)}] Processing type candidate: {type_candidate_id}")
        
        success, _ = run_type_candidate_evaluation(
            type_candidate_id=type_candidate_id,
            type_aug_name=args.type_aug_name,
            pose_aug_name=args.pose_aug_name,
            layout_id=args.layout_id,
            room_id=args.room_id,
            target_object_name=args.target_object_name,
            place_object_name=args.place_object_name,
            table_object_name=args.table_object_name,
            num_envs=args.num_envs,
            task=args.task,
            ckpt_path=args.ckpt_path
        )
        
        if success:
            successful_evaluations += 1
        else:
            failed_evaluations += 1
    
    # Load all results from files after all evaluations complete
    print(f"\n{'='*60}")
    print(f"LOADING RESULTS FROM FILES")
    print(f"{'='*60}")
    all_type_candidate_results = load_type_candidate_results(log_dir, type_candidate_ids)
    
    # Aggregate results first
    aggregated_results = {}
    aggregated_results = aggregate_results(log_dir, all_type_candidate_results, args.type_aug_name, args.pose_aug_name)
    
    # Save aggregated results
    results_path = os.path.join(log_dir, "aggregated_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(aggregated_results, f, indent=4)
    
    print(f"\nAggregated results saved to: {results_path}")
        

    
    # Print comprehensive final summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total type candidates: {len(type_candidate_ids)}")
    print(f"Successful evaluations: {successful_evaluations}")
    print(f"Failed evaluations: {failed_evaluations}")
    print(f"Evaluation success rate: {successful_evaluations/len(type_candidate_ids):.3f}")
    
    if aggregated_results and 'summary' in aggregated_results:
        summary = aggregated_results['summary']
        print(f"\n--- PERFORMANCE STATISTICS ---")
        print(f"Total layouts evaluated: {summary.get('total_layouts', 0)}")
        print(f"Total attempts evaluated: {summary.get('total_attempts', 0)}")
        print(f"Total task successes: {summary.get('total_successes', 0)}")
        print(f"Overall task success rate: {summary.get('overall_success_rate', 0.0):.3f}")
        print(f"Average success rate per candidate: {summary.get('average_success_rate_per_candidate', 0.0):.3f}")
        print(f"Best candidate success rate: {summary.get('max_success_rate', 0.0):.3f}")
        print(f"Worst candidate success rate: {summary.get('min_success_rate', 0.0):.3f}")
        
        print(f"\n--- PER-TYPE-CANDIDATE RESULTS ---")
        for type_candidate_id, candidate_data in aggregated_results['type_candidates'].items():
            if candidate_data.get('processed', False):
                layouts = candidate_data.get('total_attempts', 0)
                successes = candidate_data.get('total_successes', 0)
                success_rate = candidate_data.get('success_rate', 0.0)
                print(f"{type_candidate_id}: {success_rate:.3f} ({successes}/{layouts})")
            else:
                print(f"{type_candidate_id}: FAILED")
    else:
        print("Warning: No aggregated performance statistics available")
    
    return 0 if failed_evaluations == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 