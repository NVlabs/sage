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
    get_current_layout,
)
from isaacsim.isaac_mcp.server import (
    get_room_layout_scene_usd_separate_from_layout
)
from tex_utils import export_layout_to_mesh_dict_list
from utils import dict_to_floor_plan
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
import argparse

class MockContext:
    """Mock context for testing MCP tools"""
    async def info(self, message: str):
        print(f"INFO: {message}")

async def test_load_layout(layout_id: str):
    """Test loading layout from JSON file"""
    
    try:
        # Create mock context
        ctx = MockContext()

        layout_save_path = os.path.join(RESULTS_DIR, layout_id, layout_id+".json")
        scene_save_dir = os.path.join(RESULTS_DIR, layout_id)

        usd_save_dir = os.path.join(scene_save_dir, layout_id+"_usd_collection")
        
        os.makedirs(usd_save_dir, exist_ok=True)
        
        get_room_layout_scene_usd_separate_from_layout(
            layout_save_path,
            usd_save_dir
        )



        print("saving usd to ", usd_save_dir)

        current_layout_dict = json.load(open(layout_save_path, "r"))
        
        layout_info_path = os.path.join(usd_save_dir, "layout_info.json")
        layout_info = {
            "layout_id": layout_id,
            "scene_save_dir": scene_save_dir,
            "layout": current_layout_dict
        }

        with open(layout_info_path, "w") as f:
            json.dump(layout_info, f, indent=4)

        current_layout = dict_to_floor_plan(current_layout_dict)

        export_glb_path = os.path.join(scene_save_dir, f"{layout_id}.glb")
        mesh_dict_list = export_layout_to_mesh_dict_list(current_layout)
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
        print("saving glb to ", export_glb_path)

        
    except Exception as e:
        print(f"ERROR: Exception occurred during test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    parser = argparse.ArgumentParser()
    layout_id = "layout_a669ed6f"
    asyncio.run(test_load_layout(layout_id))
