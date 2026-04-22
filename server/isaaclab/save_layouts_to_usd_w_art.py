import asyncio
import json
import sys
import os
import numpy as np
from PIL import Image

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

from isaacsim.isaac_mcp.server import (
    get_room_layout_scene_usd_separate_from_layout
)
from tex_utils import export_layout_to_mesh_dict_list
from utils import dict_to_floor_plan
from glb_utils import (
    create_glb_scene,
    add_textured_mesh_to_glb_scene,
    save_glb_scene,
    add_glb_file_to_glb_scene
)
from constants import RESULTS_DIR
import argparse
from partnet.urdf_utils import EditableURDF
import trimesh

class MockContext:
    """Mock context for testing MCP tools"""
    async def info(self, message: str):
        print(f"INFO: {message}")

async def test_load_layout(layout_id: str):
    """Test loading layout from JSON file"""
    
    try:
        # Create mock context
        ctx = MockContext()

        layout_save_path = os.path.join(RESULTS_DIR, layout_id, layout_id+"_edit_moved.json")
        scene_save_dir = os.path.join(RESULTS_DIR, layout_id)
        partnet_dir = os.path.join(scene_save_dir, "partnet")

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

        articulated_object_dict = {}
        for room in current_layout_dict["rooms"]:
            room_objects = room["objects"]
            for object_dict in room_objects:
                object_source = object_dict["source"]
                if object_source == "partnet":
                    object_id = object_dict["id"]
                    
                    source_id = object_dict["source_id"]
                    additional_info_path = os.path.join(partnet_dir, f"{source_id}_additional_info.json")
                    urdf_dir = os.path.join(usd_save_dir, f"{source_id}_urdf")
                    glb_path = os.path.join(partnet_dir, f"{source_id}.glb")
                    ply_path = os.path.join(partnet_dir, f"{source_id}.ply")
                    with open(additional_info_path, 'r') as file:
                        additional_info = json.load(file)
                    urdf_path = additional_info["urdf_path"]
                    urdf_transform_matrix_list = additional_info["transforms"]

                    articulated_object_dict[object_id] = {
                        "urdf_path": urdf_path,
                        "urdf_transform_matrix_list": urdf_transform_matrix_list,
                        "urdf_dir": urdf_dir,
                        "glb_path": glb_path,
                        "ply_path": ply_path,
                        "position": object_dict["position"],
                        "rotation": object_dict["rotation"],
                    }

        # export urdf
        for object_id, object_data in articulated_object_dict.items():
            print(f"exporting object {object_id}")
            urdf_path = object_data["urdf_path"]
            urdf_transform_matrix_list = object_data["urdf_transform_matrix_list"]
            urdf_dir = object_data["urdf_dir"]
            glb_path = object_data["glb_path"]
            editable_urdf = EditableURDF(urdf_path, urdf_transform_matrix_list)
            urdf_updated_path = editable_urdf.export_urdf(urdf_dir)
            editable_urdf = EditableURDF(urdf_updated_path, [])
            ply_path = object_data["ply_path"]
            editable_urdf.export_ply(ply_path)
            print(f"exported ply to {ply_path}")

        # # export glb

        # scene = create_glb_scene()
        # for mesh_id, mesh_data in mesh_dict_list.items():
        #     # mesh_data_dict = {
        #     #     'vertices': mesh_data['mesh'].vertices,
        #     #     'faces': mesh_data['mesh'].faces,
        #     #     'vts': mesh_data['texture']['vts'],
        #     #     'fts': mesh_data['texture']['fts'],
        #     #     'texture_image': np.array(Image.open(mesh_data['texture']['texture_map_path']))
        #     # }
        #     # add_textured_mesh_to_glb_scene(mesh_data_dict, scene, material_name=f"material_{mesh_id}", mesh_name=f"mesh_{mesh_id}", preserve_coordinate_system=True)
        #     if mesh_id in articulated_object_dict:
        #         art_object_dict = articulated_object_dict[mesh_id]
        #         art_base_transforms = art_object_dict["urdf_transform_matrix_list"]

        #         art_world_transforms = [
        #             trimesh.transformations.rotation_matrix(
        #                 np.radians(art_object_dict["rotation"]["x"]),
        #                 [1, 0, 0]
        #             ),
        #             trimesh.transformations.rotation_matrix(
        #                 np.radians(art_object_dict["rotation"]["y"]),
        #                 [0, 1, 0]
        #             ),
        #             trimesh.transformations.rotation_matrix(
        #                 np.radians(art_object_dict["rotation"]["z"]),
        #                 [0, 0, 1]
        #             ),
        #             trimesh.transformations.translation_matrix([
        #                 art_object_dict["position"]["x"],
        #                 art_object_dict["position"]["y"],
        #                 art_object_dict["position"]["z"],
        #             ]),
        #         ]

        #         editable_urdf = EditableURDF(art_object_dict["urdf_path"], art_base_transforms + art_world_transforms)
        #         mesh_joint_info = editable_urdf.print_joint_info()
        #         mesh_joint_setting = {
        #             joint_name: np.random.uniform(mesh_joint_info[joint_name]["limits"][0], mesh_joint_info[joint_name]["limits"][1]) for joint_name in mesh_joint_info.keys()
        #         }
        #         editable_urdf.set_joint_values(mesh_joint_setting)
        #         editable_urdf.export_glb(art_object_dict["glb_path"])
        #         add_glb_file_to_glb_scene(art_object_dict["glb_path"], scene)
            
        #     else:
        #         mesh_data_dict = {
        #             'vertices': mesh_data['mesh'].vertices,
        #             'faces': mesh_data['mesh'].faces,
        #             'vts': mesh_data['texture']['vts'],
        #             'fts': mesh_data['texture']['fts'],
        #             'texture_image': np.array(Image.open(mesh_data['texture']['texture_map_path']))
        #         }
        #         add_textured_mesh_to_glb_scene(mesh_data_dict, scene, material_name=f"material_{mesh_id}", mesh_name=f"mesh_{mesh_id}", preserve_coordinate_system=True)
        #     save_glb_scene(export_glb_path, scene)
        # print("saving glb to ", export_glb_path)

        
    except Exception as e:
        print(f"ERROR: Exception occurred during test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    parser = argparse.ArgumentParser()
    # layout_a669ed6f
    layout_id = "layout_a669ed6f"

    asyncio.run(test_load_layout(layout_id))
