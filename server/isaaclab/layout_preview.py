import asyncio
import json
import sys
import os
import numpy as np
from PIL import Image

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from models import Object, Point3D, Euler, Dimensions
from layout import (
    get_layout_from_json,
    add_one_object_with_condition_in_room,
    get_current_layout,
    generate_room_layout,
    place_objects_in_room,
    room_physics_critic
)
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
from utils import dict_to_floor_plan
from constants import RESULTS_DIR
import argparse

class MockContext:
    """Mock context for testing MCP tools"""
    async def info(self, message: str):
        print(f"INFO: {message}")

async def test_load_layout(layout_id: str):
    """Test loading layout from JSON file"""
    
    try:

        from room_render import render_room_four_top_view
        from PIL import Image as PILImage
        
        layout_save_path = os.path.join(RESULTS_DIR, layout_id, layout_id+".json")
        scene_save_dir = os.path.join(RESULTS_DIR, layout_id)

        with open(layout_save_path, 'r') as f:
            layout_data = json.load(f)
        layout = dict_to_floor_plan(layout_data)

        preview_save_dir = os.path.join(scene_save_dir, "preview")
        os.makedirs(preview_save_dir, exist_ok=True)

        for room in layout.rooms:
            all_rgb = render_room_four_top_view(layout, room.id, resolution=1024)
            for i, rgb_array in enumerate(all_rgb):
                rgb_uint8 = (rgb_array * 255).astype(np.uint8)
                pil_image = PILImage.fromarray(rgb_uint8, 'RGB')
                preview_save_path = os.path.join(preview_save_dir, f"{room.id}_rendered_view_{i+1}.png")
                pil_image.save(preview_save_path)
        

        
    except Exception as e:
        print(f"ERROR: Exception occurred during test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_id", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(test_load_layout(args.layout_id))
