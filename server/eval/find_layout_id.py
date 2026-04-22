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
import os
import json
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import RESULTS_DIR

def find_layout_id_by_room_id(room_id, results_dir=RESULTS_DIR):
    """Search for layout directory containing the room_id."""
    if not os.path.exists(results_dir):
        raise ValueError(f"Results directory not found: {results_dir}")
    
    # Search through layout directories
    for layout_dir in os.listdir(results_dir):
        layout_path = os.path.join(results_dir, layout_dir)
        if not os.path.isdir(layout_path):
            continue
        
        # Check for layout JSON file
        json_path = os.path.join(layout_path, f"{layout_dir}.json")
        if not os.path.exists(json_path):
            continue
        
        # Load and check if it contains the room_id
        try:
            with open(json_path, 'r') as f:
                layout_data = json.load(f)
            
            # Check if room_id exists in this layout
            for room in layout_data.get("rooms", []):
                if room.get("id") == room_id:
                    return os.path.basename(layout_path)
        except:
            continue
    
    raise ValueError(f"No layout found containing room_id: {room_id}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("layout_id", type=str)
    args = parser.parse_args()
    layout_id = args.layout_id

    layout_id = find_layout_id_by_room_id(layout_id)
    print(layout_id)