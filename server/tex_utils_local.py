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
from models import FloorPlan, Room, Wall, Door, Object, Window, Point3D, Dimensions, Euler
import trimesh
import numpy as np
import os
import xatlas
from typing import Dict

def dict_to_room(room_data: dict) -> Room:
    """
    Convert a dictionary to a Room object.
    
    Args:
        room_data: Dictionary containing room data
        
    Returns:
        Room object reconstructed from the dictionary
    """
    try:
        # Convert position
        position = Point3D(
            x=room_data["position"]["x"],
            y=room_data["position"]["y"],
            z=room_data["position"]["z"]
        )
        
        # Convert dimensions
        dimensions = Dimensions(
            width=room_data["dimensions"]["width"],
            length=room_data["dimensions"]["length"],
            height=room_data["dimensions"]["height"]
        )
        
        # Convert walls
        walls = []
        for wall_data in room_data["walls"]:
            wall = dict_to_wall(wall_data)
            walls.append(wall)
        
        # Convert doors
        doors = []
        for door_data in room_data["doors"]:
            door = dict_to_door(door_data)
            doors.append(door)
        
        # Convert windows
        windows = []
        for window_data in room_data["windows"]:
            window = dict_to_window(window_data)
            windows.append(window)
        
        # Convert objects
        objects = []
        for object_data in room_data.get("objects", []):
            obj = dict_to_object(object_data)
            objects.append(obj)
        
        # Create Room object
        room = Room(
            id=room_data["id"],
            room_type=room_data["room_type"],
            position=position,
            dimensions=dimensions,
            walls=walls,
            doors=doors,
            objects=objects,
            windows=windows,
            floor_material=room_data.get("floor_material", "hardwood"),
            ceiling_height=room_data.get("ceiling_height", 2.7)
        )
        
        return room
        
    except KeyError as e:
        raise ValueError(f"Missing required field in room data: {e}")
    except Exception as e:
        raise ValueError(f"Error converting room data: {e}")


def dict_to_wall(wall_data: dict) -> Wall:
    """
    Convert a dictionary to a Wall object.
    
    Args:
        wall_data: Dictionary containing wall data
        
    Returns:
        Wall object reconstructed from the dictionary
    """
    try:
        start_point = Point3D(
            x=wall_data["start_point"]["x"],
            y=wall_data["start_point"]["y"],
            z=wall_data["start_point"]["z"]
        )
        
        end_point = Point3D(
            x=wall_data["end_point"]["x"],
            y=wall_data["end_point"]["y"],
            z=wall_data["end_point"]["z"]
        )
        
        wall = Wall(
            id=wall_data["id"],
            start_point=start_point,
            end_point=end_point,
            height=wall_data["height"],
            thickness=wall_data.get("thickness", 0.1),
            material=wall_data.get("material", "drywall")
        )
        
        return wall
        
    except KeyError as e:
        raise ValueError(f"Missing required field in wall data: {e}")
    except Exception as e:
        raise ValueError(f"Error converting wall data: {e}")


def dict_to_door(door_data: dict) -> Door:
    """
    Convert a dictionary to a Door object.
    
    Args:
        door_data: Dictionary containing door data
        
    Returns:
        Door object reconstructed from the dictionary
    """
    try:
        door = Door(
            id=door_data["id"],
            wall_id=door_data["wall_id"],
            position_on_wall=door_data["position_on_wall"],
            width=door_data["width"],
            height=door_data["height"],
            door_type=door_data.get("door_type", "standard"),
            opens_inward=door_data.get("opens_inward", True),
            opening=door_data.get("opening", False),
            door_material=door_data.get("door_material", "wood")
        )
        
        return door
        
    except KeyError as e:
        raise ValueError(f"Missing required field in door data: {e}")
    except Exception as e:
        raise ValueError(f"Error converting door data: {e}")


def dict_to_window(window_data: dict) -> Window:
    """
    Convert a dictionary to a Window object.
    
    Args:
        window_data: Dictionary containing window data
        
    Returns:
        Window object reconstructed from the dictionary
    """
    try:
        window = Window(
            id=window_data["id"],
            wall_id=window_data["wall_id"],
            position_on_wall=window_data["position_on_wall"],
            width=window_data["width"],
            height=window_data["height"],
            sill_height=window_data["sill_height"],
            window_type=window_data.get("window_type", "standard")
        )
        
        return window
        
    except KeyError as e:
        raise ValueError(f"Missing required field in window data: {e}")
    except Exception as e:
        raise ValueError(f"Error converting window data: {e}")


def dict_to_object(object_data: dict) -> Object:
    """
    Convert a dictionary to an Object object.
    
    Args:
        object_data: Dictionary containing object data
        
    Returns:
        Object object reconstructed from the dictionary
    """
    try:
        # Convert position
        position = Point3D(
            x=object_data["position"]["x"],
            y=object_data["position"]["y"],
            z=object_data["position"]["z"]
        )
        
        # Convert rotation
        rotation = Euler(
            x=object_data["rotation"]["x"],
            y=object_data["rotation"]["y"],
            z=object_data["rotation"]["z"]
        )
        
        # Convert dimensions
        dimensions = Dimensions(
            width=object_data["dimensions"]["width"],
            length=object_data["dimensions"]["length"],
            height=object_data["dimensions"]["height"]
        )
        
        obj = Object(
            id=object_data["id"],
            room_id=object_data["room_id"],
            type=object_data["type"],
            description=object_data["description"],
            position=position,
            rotation=rotation,
            dimensions=dimensions,
            source=object_data["source"],
            source_id=object_data["source_id"],
            place_id=object_data["place_id"],
            mass=object_data.get("mass", 1.0)
        )
        
        return obj
        
    except KeyError as e:
        raise ValueError(f"Missing required field in object data: {e}")
    except Exception as e:
        raise ValueError(f"Error converting object data: {e}")


def apply_object_transform(mesh: trimesh.Trimesh, obj: Object) -> trimesh.Trimesh:
    """
    Apply position and rotation transforms to an object mesh.
    
    Args:
        mesh: The original mesh (untransformed)
        obj: Object containing position and rotation information
        
    Returns:
        Transformed mesh positioned and rotated according to object properties
    """
    # Create a copy of the mesh to avoid modifying the original
    transformed_mesh = mesh.copy()
    
    # Convert Euler angles from degrees to radians
    rx_rad = np.radians(obj.rotation.x)
    ry_rad = np.radians(obj.rotation.y)
    rz_rad = np.radians(obj.rotation.z)
    
    # Create rotation matrices for each axis
    # Rotation order: X -> Y -> Z (Euler XYZ)
    rotation_x = trimesh.transformations.rotation_matrix(rx_rad, [1, 0, 0])
    rotation_y = trimesh.transformations.rotation_matrix(ry_rad, [0, 1, 0])
    rotation_z = trimesh.transformations.rotation_matrix(rz_rad, [0, 0, 1])
    
    # Combine rotations (order matters: Z * Y * X for XYZ Euler)
    combined_rotation = rotation_z @ rotation_y @ rotation_x
    
    # Create translation matrix
    translation = trimesh.transformations.translation_matrix([
        obj.position.x,
        obj.position.y,
        obj.position.z
    ])
    
    # Combine rotation and translation (translation after rotation)
    final_transform = translation @ combined_rotation
    
    # Apply the transform to the mesh
    transformed_mesh.apply_transform(final_transform)
    
    return transformed_mesh

def apply_object_transform_return_transform(mesh: trimesh.Trimesh, obj: Object) -> trimesh.Trimesh:
    """
    Apply position and rotation transforms to an object mesh.
    
    Args:
        mesh: The original mesh (untransformed)
        obj: Object containing position and rotation information
        
    Returns:
        Transformed mesh positioned and rotated according to object properties
    """
    # Create a copy of the mesh to avoid modifying the original
    transformed_mesh = mesh.copy()
    
    # Convert Euler angles from degrees to radians
    rx_rad = np.radians(obj.rotation.x)
    ry_rad = np.radians(obj.rotation.y)
    rz_rad = np.radians(obj.rotation.z)
    
    # Create rotation matrices for each axis
    # Rotation order: X -> Y -> Z (Euler XYZ)
    rotation_x = trimesh.transformations.rotation_matrix(rx_rad, [1, 0, 0])
    rotation_y = trimesh.transformations.rotation_matrix(ry_rad, [0, 1, 0])
    rotation_z = trimesh.transformations.rotation_matrix(rz_rad, [0, 0, 1])
    
    # Combine rotations (order matters: Z * Y * X for XYZ Euler)
    combined_rotation = rotation_z @ rotation_y @ rotation_x
    
    # Create translation matrix
    translation = trimesh.transformations.translation_matrix([
        obj.position.x,
        obj.position.y,
        obj.position.z
    ])
    
    # Combine rotation and translation (translation after rotation)
    final_transform = translation @ combined_rotation
    
    # Apply the transform to the mesh
    transformed_mesh.apply_transform(final_transform)
    
    return transformed_mesh, final_transform


def apply_object_transform_direct(mesh: trimesh.Trimesh, position: Dict[str, float], rotation: Dict[str, float]) -> trimesh.Trimesh:
    """
    Apply position and rotation transforms to an object mesh.
    
    Args:
        mesh: The original mesh (untransformed)
        obj: Object containing position and rotation information
        
    Returns:
        Transformed mesh positioned and rotated according to object properties
    """
    # Create a copy of the mesh to avoid modifying the original
    transformed_mesh = mesh.copy()
    
    # Convert Euler angles from degrees to radians
    rx_rad = np.radians(rotation["x"])
    ry_rad = np.radians(rotation["y"])
    rz_rad = np.radians(rotation["z"])
    
    # Create rotation matrices for each axis
    # Rotation order: X -> Y -> Z (Euler XYZ)
    rotation_x = trimesh.transformations.rotation_matrix(rx_rad, [1, 0, 0])
    rotation_y = trimesh.transformations.rotation_matrix(ry_rad, [0, 1, 0])
    rotation_z = trimesh.transformations.rotation_matrix(rz_rad, [0, 0, 1])
    
    # Combine rotations (order matters: Z * Y * X for XYZ Euler)
    combined_rotation = rotation_z @ rotation_y @ rotation_x
    
    # Create translation matrix
    translation = trimesh.transformations.translation_matrix([
        position["x"],
        position["y"],
        position["z"]
    ])
    
    # Combine rotation and translation (translation after rotation)
    final_transform = translation @ combined_rotation
    
    # Apply the transform to the mesh
    transformed_mesh.apply_transform(final_transform)
    
    return transformed_mesh


def dict_to_floor_plan(layout_data: dict) -> FloorPlan:
    """
    Convert a dictionary (from JSON) back to a FloorPlan object.
    
    Args:
        layout_data: Dictionary containing the floor plan data
        
    Returns:
        FloorPlan object reconstructed from the dictionary
        
    Raises:
        ValueError: If the data structure is invalid or incomplete
    """
    try:
        # Convert rooms
        rooms = []
        for room_data in layout_data["rooms"]:
            room = dict_to_room(room_data)
            rooms.append(room)
        
        # Create FloorPlan object
        floor_plan = FloorPlan(
            id=layout_data["id"],
            rooms=rooms,
            total_area=layout_data["total_area"],
            building_style=layout_data["building_style"],
            description=layout_data["description"],
            created_from_text=layout_data["created_from_text"]
        )
        
        return floor_plan
        
    except KeyError as e:
        raise ValueError(f"Missing required field in layout data: {e}")
    except Exception as e:
        raise ValueError(f"Error converting layout data: {e}")


def create_floor_mesh(room: Room) -> trimesh.Trimesh:
    """Create a floor mesh for a room."""
    pos = room.position
    dims = room.dimensions
    
    # Create floor as a thin box
    floor_thickness = 0.1
    floor_box = trimesh.creation.box(
        extents=[dims.width, dims.length, floor_thickness],
        transform=trimesh.transformations.translation_matrix([
            pos.x + dims.width/2,
            pos.y + dims.length/2,
            pos.z - floor_thickness/2
        ])
    )

    return floor_box


def create_room_meshes_with_openings(room: Room, processed_doors: set, processed_windows: set):
    """
    Create wall meshes with door and window openings cut out using boolean operations.
    
    Returns:
        Tuple of (wall_meshes, door_meshes, window_meshes)
    """
    wall_meshes = []
    door_meshes = []
    window_meshes = []

    wall_ids = []
    door_ids = []
    window_ids = []
    
    # Create each wall
    for wall in room.walls:
        wall_mesh = create_wall_mesh(wall, room)
        wall_ids.append(wall.id)
        # Find doors and windows on this wall
        wall_doors = [door for door in room.doors if door.wall_id == wall.id]
        wall_windows = [window for window in room.windows if window.wall_id == wall.id]
        
        # Create door meshes and subtract from wall
        for door in wall_doors:
            door_id = get_door_unique_id(room, door)
            if door_id not in processed_doors:
                door_mesh = create_door_mesh(wall, door, door_size_offset=0.11)
                if not door.opening:
                    door_meshes.append(door_mesh)
                processed_doors.add(door_id)
                door_ids.append(door_id)
                # Cut door opening from wall
                try:
                    wall_mesh = wall_mesh.difference(door_mesh, engine="manifold")
                except:
                    # If boolean operation fails, just subtract a simple box
                    opening_mesh = create_door_opening_mesh(wall, door)
                    try:
                        wall_mesh = wall_mesh.difference(opening_mesh, engine="manifold")
                    except:
                        print(f"Boolean operation failed for door {door.id} on wall {wall.id}")
                        pass  # Keep original wall if boolean ops fail
        
        # Create window meshes and subtract from wall
        for window in wall_windows:
            window_id = get_window_unique_id(room, window)
            if window_id not in processed_windows:
                window_mesh = create_window_mesh(wall, window)
                window_meshes.append(window_mesh)
                processed_windows.add(window_id)
                window_ids.append(window.id)
                # Cut window opening from wall
                try:
                    wall_mesh = wall_mesh.difference(window_mesh, engine="manifold")
                except:
                    # If boolean operation fails, just subtract a simple box
                    opening_mesh = create_window_opening_mesh(wall, window)
                    try:
                        wall_mesh = wall_mesh.difference(opening_mesh, engine="manifold")
                    except:
                        print(f"Boolean operation failed for window {window.id} on wall {wall.id}")
                        pass  # Keep original wall if boolean ops fail
        
        wall_meshes.append(wall_mesh)
    
    return wall_meshes, door_meshes, window_meshes, wall_ids, door_ids, window_ids



def create_wall_mesh(wall: Wall, room: Room) -> trimesh.Trimesh:
    """Create a wall mesh from wall definition."""
    import numpy as np
    
    # Calculate wall direction and length
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    
    wall_vector = end - start
    wall_length = np.linalg.norm(wall_vector)
    wall_direction = wall_vector / wall_length
    
    # Calculate room center from room position and dimensions
    room_center = np.array([
        room.position.x + room.dimensions.width / 2,
        room.position.y + room.dimensions.length / 2,
        room.position.z
    ])
    
    # Calculate wall center point at the midpoint of start-end line
    wall_center = (start + end) / 2
    
    # Calculate both possible normal directions (perpendicular to wall)
    # For a vector (x, y, z), the two perpendicular directions in XY plane are:
    normal1 = np.array([wall_direction[1], -wall_direction[0], 0])
    normal2 = np.array([-wall_direction[1], wall_direction[0], 0])
    
    # Vector from wall center to room center
    wall_to_room = room_center - wall_center
    
    # Choose the normal that points toward the room center
    # (has positive dot product with wall_to_room vector)
    if np.dot(normal1, wall_to_room) > 0:
        inward_normal = normal1
    else:
        inward_normal = normal2
    
    # Use half thickness to avoid overlapping with adjacent walls
    half_thickness = wall.thickness / 2
    
    # Set wall center Z coordinate
    wall_center[2] = wall.start_point.z + wall.height / 2
    
    # Offset the wall center by half thickness in the inward direction
    # This positions the wall mesh only on the inside of the room
    wall_center_offset = wall_center + inward_normal * (half_thickness / 2)
    
    # Create wall mesh as a box with half thickness
    wall_box = trimesh.creation.box(
        extents=[wall_length, half_thickness, wall.height]
    )
    
    # Calculate rotation to align with wall direction
    # Default box is aligned with X-axis, we need to rotate to wall direction
    if abs(wall_direction[0]) < 0.001:  # Vertical wall (Y-aligned)
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1])
    else:  # Horizontal wall (X-aligned) - no rotation needed
        rotation_matrix = np.eye(4)
    
    # Apply transformation
    transform = trimesh.transformations.translation_matrix(wall_center_offset) @ rotation_matrix
    wall_box.apply_transform(transform)
    
    return wall_box


def create_door_mesh(wall: Wall, door: Door, size_scale: float = 1.0, thickness_scale: float = 1.0, door_size_offset: float = 0.0) -> trimesh.Trimesh:
    """Create a door mesh positioned on the wall."""
    import numpy as np
    
    # Calculate door position on wall
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_vector = end - start
    
    # Position along the wall
    door_position_3d = start + wall_vector * door.position_on_wall
    door_position_3d[2] = wall.start_point.z + door.height / 2
    
    # Create door mesh
    door_box = trimesh.creation.box(
        extents=[door.width * size_scale + door_size_offset, wall.thickness * thickness_scale, door.height * size_scale + door_size_offset]  # Slightly thicker than wall
    )
    
    # Rotate if wall is vertical
    wall_direction = wall_vector / np.linalg.norm(wall_vector)
    if abs(wall_direction[0]) < 0.001:  # Vertical wall
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1])
        door_box.apply_transform(rotation_matrix)
    
    # Position door
    door_box.apply_translation(door_position_3d)
    
    return door_box


def create_door_frame_mesh(wall: Wall, door: Door, size_scale: float = 1.0, thickness_scale: float = 1.0, frame_width: float = 0.05) -> trimesh.Trimesh:
    """Create a door frame mesh with left, right, and top edges surrounding the door."""
    import numpy as np
    
    # Calculate door position on wall
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_vector = end - start
    wall_direction = wall_vector / np.linalg.norm(wall_vector)
    
    # Position along the wall
    door_position_3d = start + wall_vector * door.position_on_wall
    door_position_3d[2] = wall.start_point.z + door.height / 2
    
    # Door dimensions (scaled) - this is the actual door size that will be created
    door_width = door.width * size_scale
    door_height = door.height * size_scale
    door_thickness = wall.thickness * thickness_scale
    
    # Frame dimensions
    frame_thickness = wall.thickness * thickness_scale # Frame uses full wall thickness
    
    # Calculate the gap between scaled door and original door opening
    # The wall opening is typically the full door size, scaled door is smaller
    gap_width = (door.width - door_width) / 2
    gap_height = door.height - door_height  # Only at top since door sits on floor
    
    # The frame should be positioned completely outside the door region
    # We need to ensure no overlap with the actual door (door_width x door_height)
    
    frame_meshes = []
    
    # Create left frame piece - positioned completely outside the door region
    left_frame = trimesh.creation.box(
        extents=[frame_width, frame_thickness, door_height + gap_height + frame_width]
    )
    # Position left frame to be completely outside the door region
    # The door extends from -door_width/2 to +door_width/2
    # So the left frame should start at -door_width/2 - frame_width/2 and extend outward
    left_offset = np.array([-(door_width/2 + frame_width/2), 0, (gap_height + frame_width)/2])
    
    # Create right frame piece - positioned completely outside the door region
    right_frame = trimesh.creation.box(
        extents=[frame_width, frame_thickness, door_height + gap_height + frame_width]
    )
    # Position right frame to be completely outside the door region
    # The right frame should start at +door_width/2 + frame_width/2 and extend outward
    right_offset = np.array([door_width/2 + frame_width/2, 0, (gap_height + frame_width)/2])
    
    # Create top frame piece - positioned completely above the door region
    top_frame = trimesh.creation.box(
        extents=[door_width + 2*frame_width, frame_thickness, frame_width]
    )
    # Position top frame to be completely above the door region
    # The door extends from 0 to door_height, so top frame starts at door_height + frame_width/2
    top_offset = np.array([0, 0, door_height/2 + frame_width/2])
    
    # Apply offsets
    left_frame.apply_translation(left_offset)
    right_frame.apply_translation(right_offset)
    top_frame.apply_translation(top_offset)
    
    # Combine frame pieces
    frame_meshes = [left_frame, right_frame, top_frame]
    combined_frame = trimesh.util.concatenate(frame_meshes)
    
    # Rotate if wall is vertical
    if abs(wall_direction[0]) < 0.001:  # Vertical wall
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1])
        combined_frame.apply_transform(rotation_matrix)
    
    # Position frame at door location
    combined_frame.apply_translation(door_position_3d)
    
    return combined_frame


def create_window_mesh(wall: Wall, window: Window) -> trimesh.Trimesh:
    """Create a window mesh positioned on the wall."""
    import numpy as np
    
    # Calculate window position on wall
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_vector = end - start
    
    # Position along the wall
    window_position_3d = start + wall_vector * window.position_on_wall
    window_position_3d[2] = wall.start_point.z + window.sill_height + window.height / 2
    
    # Create window mesh
    window_box = trimesh.creation.box(
        extents=[window.width, wall.thickness * 1.0, window.height]  # Slightly thicker than wall
    )
    
    # Rotate if wall is vertical
    wall_direction = wall_vector / np.linalg.norm(wall_vector)
    if abs(wall_direction[0]) < 0.001:  # Vertical wall
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1])
        window_box.apply_transform(rotation_matrix)
    
    # Position window
    window_box.apply_translation(window_position_3d)
    
    return window_box


def create_door_opening_mesh(wall: Wall, door: Door) -> trimesh.Trimesh:
    """Create a door opening mesh for boolean subtraction."""
    return create_door_mesh(wall, door)  # Same as door mesh for cutting


def create_window_opening_mesh(wall: Wall, window: Window) -> trimesh.Trimesh:
    """Create a window opening mesh for boolean subtraction."""
    return create_window_mesh(wall, window)  # Same as window mesh for cutting


def get_door_unique_id(room: Room, door: Door) -> str:
    """Generate unique ID for a door to avoid processing bidirectional doors twice."""
    if door.door_type == "connecting":
        # For connecting doors, create ID based on position to match bidirectional pairs
        wall_id = door.wall_id
        position = door.position_on_wall
        return f"connecting_door_{wall_id}_{position:.3f}"
    else:
        return f"door_{room.id}_{door.id}"


def get_window_unique_id(room: Room, window: Window) -> str:
    """Generate unique ID for a window to avoid processing bidirectional windows twice."""
    if window.window_type == "connecting":
        # For connecting windows, create ID based on position to match bidirectional pairs
        wall_id = window.wall_id
        position = window.position_on_wall
        return f"connecting_window_{wall_id}_{position:.3f}"
    else:
        return f"window_{room.id}_{window.id}"


def create_ceiling_mesh(room: Room) -> trimesh.Trimesh:
    """Create a ceiling mesh for a room."""
    pos = room.position
    dims = room.dimensions
    
    # Create floor as a thin box
    ceiling_thickness = 0.1
    ceiling_box = trimesh.creation.box(
        extents=[dims.width, dims.length, ceiling_thickness],
        transform=trimesh.transformations.translation_matrix([
            pos.x + dims.width/2,
            pos.y + dims.length/2,
            pos.z + dims.height + ceiling_thickness/2
        ])
    )

    return ceiling_box


def export_layout_to_mesh_dict_list(layout: FloorPlan, layout_dir: str):
    """
    Export a FloorPlan object to a mesh file using trimesh.
    Creates gray boxes for walls/floors, red boxes for doors, blue boxes for windows,
    and includes actual object meshes with their transforms.
    Uses boolean operations to cut door/window openings in walls.
    
    Args:
        layout: FloorPlan object to export
    """
    import os
    import pickle

    def get_object_mesh(source, source_id):
        object_save_path = f"{layout_dir}/{source}/{source_id}.ply"
        if os.path.exists(object_save_path):
            return trimesh.load(object_save_path)
        else:
            return None
        
    def get_object_mesh_texture(source, source_id):
        tex_coords_save_path = f"{layout_dir}/{source}/{source_id}_tex_coords.pkl"
        texture_map_path = f"{layout_dir}/{source}/{source_id}_texture.png"
        if os.path.exists(tex_coords_save_path):
            with open(tex_coords_save_path, "rb") as f:
                tex_coords = pickle.load(f)
            return {
                "vts": tex_coords["vts"],
                "fts": tex_coords["fts"],
                "texture_map_path": texture_map_path
            }
        else:
            return None
    
    mesh_info_dict = {}

    # Collections for different mesh types
    floor_meshes = []
    wall_meshes = []
    door_meshes = []
    window_meshes = []
    object_meshes = []
    
    # Track processed bidirectional doors/windows to avoid duplicates
    processed_doors = set()
    processed_windows = set()
    
    # Process each room
    for room in layout.rooms:
        # Create floor mesh
        floor_mesh = create_floor_mesh(room)
        floor_mesh_texture_map_path = f"{layout_dir}/materials/{room.floor_material}.png"
        # TODO: generate tex coords for floor mesh
        floor_mesh_tex_coords = create_floor_mesh_tex_coords(floor_mesh)
        # floor_meshes.append(floor_mesh)

        mesh_info_dict[f"floor_{room.id}"] = {
            "mesh": floor_mesh,
            "static": True,
            "texture": {
                "vts": floor_mesh_tex_coords["vts"],
                "fts": floor_mesh_tex_coords["fts"],
                "texture_map_path": floor_mesh_texture_map_path
            }
        }
        
        # Create wall meshes with door/window cutouts
        room_wall_meshes, room_door_meshes, room_window_meshes, room_wall_ids, room_door_ids, room_window_ids = create_room_meshes_with_openings(
            room, processed_doors, processed_windows
        )
        
        # wall_meshes.extend(room_wall_meshes)
        # door_meshes.extend(room_door_meshes)
        # window_meshes.extend(room_window_meshes)

        wall_mesh_texture_map_path = f"{layout_dir}/materials/{room.walls[0].material}.png"

        for wall_id, wall_mesh in zip(room_wall_ids, room_wall_meshes):
            # TODO: generate tex coords for wall mesh
            wall_mesh_tex_coords = create_wall_mesh_tex_coords(wall_mesh)
            mesh_info_dict[f"{wall_id}"] = {
                "mesh": wall_mesh,
                "static": True,
                "texture": {
                    "vts": wall_mesh_tex_coords["vts"],
                    "fts": wall_mesh_tex_coords["fts"],
                    "texture_map_path": wall_mesh_texture_map_path
                }
            }

        for window_id, window_mesh in zip(room_window_ids, room_window_meshes):
            # window_mesh_tex_coords = create_window_mesh_tex_coords(window_mesh)

            window_mesh_texture_map_path = f"{layout_dir}/materials/{window_id}_texture.png"
            window_mesh_tex_coords_save_path = f"{layout_dir}/materials/{window_id}_tex_coords.pkl"
            with open(window_mesh_tex_coords_save_path, "rb") as f:
                window_mesh_tex_coords = pickle.load(f)

            mesh_info_dict[f"{window_id}"] = {
                "mesh": window_mesh,
                "static": True,
                "texture": {
                    "vts": window_mesh_tex_coords["vts"],
                    "fts": window_mesh_tex_coords["fts"],
                    "texture_map_path": window_mesh_texture_map_path
                }
            }

        ceiling_mesh = create_ceiling_mesh(room)
        ceiling_mesh_texture_map_path = f"{layout_dir}/materials/{room.walls[0].material}.png"
        # TODO: generate tex coords for floor mesh
        ceiling_mesh_tex_coords = create_wall_mesh_tex_coords(ceiling_mesh)

        mesh_info_dict[f"floor_{room.id}_ceiling"] = {
            "mesh": ceiling_mesh,
            "static": True,
            "texture": {
                "vts": ceiling_mesh_tex_coords["vts"],
                "fts": ceiling_mesh_tex_coords["fts"],
                "texture_map_path": ceiling_mesh_texture_map_path
            }
        }
            
    # Process each room
    for room in layout.rooms:
        # Create object meshes with transforms
        for obj in room.objects:
            obj_mesh = get_object_mesh(obj.source, obj.source_id)
            if obj_mesh is not None:
                # Apply transforms to the object mesh
                transformed_mesh = apply_object_transform(obj_mesh, obj)
                texture_info = get_object_mesh_texture(obj.source, obj.source_id)

                mesh_info_dict[obj.id] = {
                    "mesh": transformed_mesh,
                    "static": False,
                    "texture": texture_info
                }


    door_center_list = []

    for room in layout.rooms:
        wall_map = {wall.id: wall for wall in room.walls}

        # Add doors - position them at the inner room boundary
        for door in room.doors:

            if door.opening:
                continue

            # Get the wall this door is on
            wall = wall_map.get(door.wall_id)
            assert wall is not None, f"Wall {door.wall_id} not found"

            start_point = wall.start_point
            end_point = wall.end_point
            
            position_on_wall = door.position_on_wall
            door_center_x = start_point.x + (end_point.x - start_point.x) * position_on_wall
            door_center_y = start_point.y + (end_point.y - start_point.y) * position_on_wall

            duplicate_door = False
            for door_center_prev_x, door_center_prev_y in door_center_list:
                if abs(door_center_x - door_center_prev_x) < 0.01 and abs(door_center_y - door_center_prev_y) < 0.01:
                    duplicate_door = True
                    break
            
            if duplicate_door:
                continue

            door_center_list.append((door_center_x, door_center_y))
            
            thickness_scale = 0.99
            size_scale = 0.99

            door_width_original = door.width
            door_thickness = wall.thickness * thickness_scale

            delta_s = 0.5 * (1 - size_scale) * door_width_original
            delta_r_min = max(0, ((0.5 * door_thickness) ** 2 - delta_s ** 2) / (2 * delta_s))
            delta_r = delta_r_min * 1.1

            door_size_offset_calculated = (size_scale - 1) * door_width_original
            door_mesh = create_door_mesh(wall, door, size_scale=1.0, thickness_scale=thickness_scale, door_size_offset=door_size_offset_calculated)
            door_frame_mesh = create_door_frame_mesh(wall, door, size_scale=1.0, thickness_scale=1.05, frame_width=0.05)


            # Calculate door position on wall
            start = np.array([wall.start_point.x, wall.start_point.y, 0])
            end = np.array([wall.end_point.x, wall.end_point.y, 0])
            wall_vector = end - start
            wall_vector_norm = wall_vector / np.linalg.norm(wall_vector)
            
            # Position along the wall
            door_center_point = start + wall_vector * door.position_on_wall
            door_start_point = door_center_point - wall_vector_norm * door.width / 2

            door_rotate_axis_point_lower = door_start_point + wall_vector_norm * (delta_s + delta_r)
            door_rotate_axis_point_lower[2] = 0.
            door_rotate_axis_point_upper = door_start_point + wall_vector_norm * (delta_s + delta_r)
            door_rotate_axis_point_upper[2] = door.height

            door_mesh_texture_map_path = f"{layout_dir}/materials/{door.door_material}_texture.png"

            door_mesh_tex_coords_save_path = f"{layout_dir}/materials/{door.door_material}_tex_coords.pkl"
            with open(door_mesh_tex_coords_save_path, "rb") as f:
                door_mesh_tex_coords = pickle.load(f)

            texture_info = {
                "vts": door_mesh_tex_coords["vts"],
                "fts": door_mesh_tex_coords["fts"],
                "texture_map_path": door_mesh_texture_map_path
            }


            mesh_info_dict[f"{door.id}"] = {
                "mesh": door_mesh,
                "static": False,
                "articulation": (door_rotate_axis_point_lower, door_rotate_axis_point_upper),
                "texture": texture_info
            }

            # Add door frame mesh to the dictionary
            # Use door-specific frame texture based on door material
            door_frame_texture_map_path = f"{layout_dir}/materials/{door.door_material}_frame_texture.png"
            door_frame_tex_coords_save_path = f"{layout_dir}/materials/{door.door_material}_frame_tex_coords.pkl"
            
            # Check if door frame texture files exist, if not use door material as fallback
            if not os.path.exists(door_frame_tex_coords_save_path):
                door_frame_texture_map_path = door_mesh_texture_map_path
                door_frame_tex_coords_save_path = door_mesh_tex_coords_save_path
            
            with open(door_frame_tex_coords_save_path, "rb") as f:
                door_frame_tex_coords = pickle.load(f)

            door_frame_texture_info = {
                "vts": door_frame_tex_coords["vts"],
                "fts": door_frame_tex_coords["fts"],
                "texture_map_path": door_frame_texture_map_path
            }

            mesh_info_dict[f"{door.id}_frame"] = {
                "mesh": door_frame_mesh,
                "static": True,  # Door frame is static, doesn't move with door
                "texture": door_frame_texture_info
            }


    return mesh_info_dict




def export_single_room_layout_to_mesh_dict_list(layout: FloorPlan, room_id: str):
    """
    Export a FloorPlan object to a mesh file using trimesh.
    Creates gray boxes for walls/floors, red boxes for doors, blue boxes for windows,
    and includes actual object meshes with their transforms.
    Uses boolean operations to cut door/window openings in walls.
    
    Args:
        layout: FloorPlan object to export
    """
    import os
    import pickle

    layout_save_dir = RESULTS_DIR
    
    def get_object_mesh(source, source_id, layout_id):
        object_save_path = f"{layout_save_dir}/{layout_id}/{source}/{source_id}.ply"
        if os.path.exists(object_save_path):
            return trimesh.load(object_save_path)
        else:
            return None
        
    def get_object_mesh_texture(source, source_id, layout_id):
        tex_coords_save_path = f"{layout_save_dir}/{layout_id}/{source}/{source_id}_tex_coords.pkl"
        texture_map_path = f"{layout_save_dir}/{layout_id}/{source}/{source_id}_texture.png"
        if os.path.exists(tex_coords_save_path):
            with open(tex_coords_save_path, "rb") as f:
                tex_coords = pickle.load(f)
            return {
                "vts": tex_coords["vts"],
                "fts": tex_coords["fts"],
                "texture_map_path": texture_map_path
            }
        else:
            return None
    
    mesh_info_dict = {}

    
    # Track processed bidirectional doors/windows to avoid duplicates
    processed_doors = set()
    processed_windows = set()
    
    # Process each room
    all_rooms = layout.rooms
    room = next(room for room in all_rooms if room.id == room_id)
    
    # Create floor mesh
    floor_mesh = create_floor_mesh(room)
    floor_mesh_texture_map_path = f"{layout_save_dir}/{layout.id}/materials/{room.floor_material}.png"
    # TODO: generate tex coords for floor mesh
    floor_mesh_tex_coords = create_floor_mesh_tex_coords(floor_mesh)
    # floor_meshes.append(floor_mesh)

    mesh_info_dict[f"floor_{room.id}"] = {
        "mesh": floor_mesh,
        "static": True,
        "texture": {
            "vts": floor_mesh_tex_coords["vts"],
            "fts": floor_mesh_tex_coords["fts"],
            "texture_map_path": floor_mesh_texture_map_path
        }
    }
    
    # Create wall meshes with door/window cutouts
    room_wall_meshes, room_door_meshes, room_window_meshes, room_wall_ids, room_door_ids, room_window_ids = create_room_meshes_with_openings(
        room, processed_doors, processed_windows
    )
    
    # wall_meshes.extend(room_wall_meshes)
    # door_meshes.extend(room_door_meshes)
    # window_meshes.extend(room_window_meshes)

    wall_mesh_texture_map_path = f"{layout_save_dir}/{layout.id}/materials/{room.walls[0].material}.png"

    for wall_id, wall_mesh in zip(room_wall_ids, room_wall_meshes):
        # TODO: generate tex coords for wall mesh
        wall_mesh_tex_coords = create_wall_mesh_tex_coords(wall_mesh)
        mesh_info_dict[f"{wall_id}"] = {
            "mesh": wall_mesh,
            "static": True,
            "texture": {
                "vts": wall_mesh_tex_coords["vts"],
                "fts": wall_mesh_tex_coords["fts"],
                "texture_map_path": wall_mesh_texture_map_path
            }
        }

    for window_id, window_mesh in zip(room_window_ids, room_window_meshes):
        # window_mesh_tex_coords = create_window_mesh_tex_coords(window_mesh)

        window_mesh_texture_map_path = f"{layout_save_dir}/{layout.id}/materials/{window_id}_texture.png"
        window_mesh_tex_coords_save_path = f"{layout_save_dir}/{layout.id}/materials/{window_id}_tex_coords.pkl"
        with open(window_mesh_tex_coords_save_path, "rb") as f:
            window_mesh_tex_coords = pickle.load(f)

        mesh_info_dict[f"{window_id}"] = {
            "mesh": window_mesh,
            "static": True,
            "texture": {
                "vts": window_mesh_tex_coords["vts"],
                "fts": window_mesh_tex_coords["fts"],
                "texture_map_path": window_mesh_texture_map_path
            }
        }

    # Process each room
    # Create object meshes with transforms
    for obj in room.objects:
        obj_mesh = get_object_mesh(obj.source, obj.source_id, layout.id)
        if obj_mesh is not None:
            # Apply transforms to the object mesh
            transformed_mesh = apply_object_transform(obj_mesh, obj)
            texture_info = get_object_mesh_texture(obj.source, obj.source_id, layout.id)

            mesh_info_dict[obj.id] = {
                "mesh": transformed_mesh,
                "static": False,
                "texture": texture_info
            }


    door_center_list = []


    wall_map = {wall.id: wall for wall in room.walls}

    # Add doors - position them at the inner room boundary
    for door in room.doors:

            if door.opening:
                continue

            # Get the wall this door is on
            wall = wall_map.get(door.wall_id)
            assert wall is not None, f"Wall {door.wall_id} not found"

            start_point = wall.start_point
            end_point = wall.end_point
            
            position_on_wall = door.position_on_wall
            door_center_x = start_point.x + (end_point.x - start_point.x) * position_on_wall
            door_center_y = start_point.y + (end_point.y - start_point.y) * position_on_wall

            duplicate_door = False
            for door_center_prev_x, door_center_prev_y in door_center_list:
                if abs(door_center_x - door_center_prev_x) < 0.01 and abs(door_center_y - door_center_prev_y) < 0.01:
                    duplicate_door = True
                    break
            
            if duplicate_door:
                continue

            door_center_list.append((door_center_x, door_center_y))
            
            thickness_scale = 0.95
            size_scale = 0.95

            door_width_original = door.width
            door_thickness = wall.thickness * thickness_scale

            delta_s = 0.5 * (1 - size_scale) * door_width_original
            delta_r_min = max(0, ((0.5 * door_thickness) ** 2 - delta_s ** 2) / (2 * delta_s))
            delta_r = delta_r_min * 1.1

            door_size_offset_calculated = (size_scale - 1) * door_width_original
            door_mesh = create_door_mesh(wall, door, size_scale=1.0, thickness_scale=thickness_scale, door_size_offset=door_size_offset_calculated)
            door_frame_mesh = create_door_frame_mesh(wall, door, size_scale=1.0, thickness_scale=1.05, frame_width=0.05)

            # Calculate door position on wall
            start = np.array([wall.start_point.x, wall.start_point.y, 0])
            end = np.array([wall.end_point.x, wall.end_point.y, 0])
            wall_vector = end - start
            wall_vector_norm = wall_vector / np.linalg.norm(wall_vector)
            
            # Position along the wall
            door_center_point = start + wall_vector * door.position_on_wall
            door_start_point = door_center_point - wall_vector_norm * door.width / 2

            door_rotate_axis_point_lower = door_start_point + wall_vector_norm * (delta_s + delta_r)
            door_rotate_axis_point_lower[2] = 0.
            door_rotate_axis_point_upper = door_start_point + wall_vector_norm * (delta_s + delta_r)
            door_rotate_axis_point_upper[2] = door.height

            # print("door_rotate_axis_point_lower: ", door_rotate_axis_point_lower)
            # print("door_rotate_axis_point_upper: ", door_rotate_axis_point_upper)

            # print("wall start: ", wall.start_point)
            # print("wall end: ", wall.end_point)
            # print("wall vector: ", wall_vector)
            # print("wall vector norm: ", wall_vector_norm)
            # print("door center point: ", door_center_point)
            # print("door start point: ", door_start_point)
            # print("door width: ", door.width)


            door_mesh_texture_map_path = f"{layout_save_dir}/{layout.id}/materials/{door.door_material}_texture.png"

            door_mesh_tex_coords_save_path = f"{layout_save_dir}/{layout.id}/materials/{door.door_material}_tex_coords.pkl"
            with open(door_mesh_tex_coords_save_path, "rb") as f:
                door_mesh_tex_coords = pickle.load(f)

            texture_info = {
                "vts": door_mesh_tex_coords["vts"],
                "fts": door_mesh_tex_coords["fts"],
                "texture_map_path": door_mesh_texture_map_path
            }


            mesh_info_dict[f"{door.id}"] = {
                "mesh": door_mesh,
                "static": False,
                "articulation": (door_rotate_axis_point_lower, door_rotate_axis_point_upper),
                "texture": texture_info
            }

            # Add door frame mesh to the dictionary
            # Use door-specific frame texture based on door material
            door_frame_texture_map_path = f"{layout_save_dir}/{layout.id}/materials/{door.door_material}_frame_texture.png"
            door_frame_tex_coords_save_path = f"{layout_save_dir}/{layout.id}/materials/{door.door_material}_frame_tex_coords.pkl"
            
            # Check if door frame texture files exist, if not use door material as fallback
            if not os.path.exists(door_frame_tex_coords_save_path):
                door_frame_texture_map_path = door_mesh_texture_map_path
                door_frame_tex_coords_save_path = door_mesh_tex_coords_save_path
            
            with open(door_frame_tex_coords_save_path, "rb") as f:
                door_frame_tex_coords = pickle.load(f)

            door_frame_texture_info = {
                "vts": door_frame_tex_coords["vts"],
                "fts": door_frame_tex_coords["fts"],
                "texture_map_path": door_frame_texture_map_path
            }

            mesh_info_dict[f"{door.id}_frame"] = {
                "mesh": door_frame_mesh,
                "static": True,  # Door frame is static, doesn't move with door
                "texture": door_frame_texture_info
            }


    return mesh_info_dict



def create_floor_mesh_tex_coords(floor_mesh: trimesh.Trimesh) -> dict:
    """
    Generate texture coordinates for a floor mesh using xatlas.
    
    Args:
        floor_mesh: Trimesh object representing the floor
        
    Returns:
        Dictionary with 'vts' (texture coordinates) and 'fts' (face texture indices)
    """
    try:
        # Create xatlas mesh from trimesh
        atlas = xatlas.Atlas()
        
        # Convert trimesh to xatlas format
        vertices = floor_mesh.vertices.astype(np.float32)
        faces = floor_mesh.faces.astype(np.uint32)
        
        # Add mesh to atlas
        atlas.add_mesh(vertices, faces)
        
        # Generate UV coordinates
        atlas.generate()
        
        # Get the UV coordinates and face indices
        vmapping, indices, uvs = atlas.get_mesh(0)
        
        # Convert to the expected format
        # vts: texture coordinates (u, v) for each vertex
        vts = uvs
        
        # fts: face texture indices - map each face to texture coordinates
        fts = indices
        
        return {
            "vts": vts,
            "fts": fts
        }
        
    except Exception as e:
        print(f"Error generating texture coordinates for floor mesh: {e}")
        # Fallback: simple planar UV mapping
        return _simple_planar_uv_mapping(floor_mesh)


def create_wall_mesh_tex_coords(wall_mesh: trimesh.Trimesh) -> dict:
    """
    Generate texture coordinates for a wall mesh using xatlas.
    
    Args:
        wall_mesh: Trimesh object representing the wall
        
    Returns:
        Dictionary with 'vts' (texture coordinates) and 'fts' (face texture indices)
    """
    try:
        # Create xatlas mesh from trimesh
        atlas = xatlas.Atlas()
        
        # Convert trimesh to xatlas format
        vertices = wall_mesh.vertices.astype(np.float32)
        faces = wall_mesh.faces.astype(np.uint32)
        
        # Add mesh to atlas
        atlas.add_mesh(vertices, faces)
        
        # Generate UV coordinates
        atlas.generate()
        
        # Get the UV coordinates and face indices
        vmapping, indices, uvs = atlas.get_mesh(0)
        
        # Convert to the expected format
        # vts: texture coordinates (u, v) for each vertex
        vts = uvs
        
        # fts: face texture indices - map each face to texture coordinates
        fts = indices
        
        return {
            "vts": vts,
            "fts": fts
        }
        
    except Exception as e:
        print(f"Error generating texture coordinates for wall mesh: {e}")
        # Fallback: simple planar UV mapping
        return _simple_planar_uv_mapping(wall_mesh)


def _simple_planar_uv_mapping(mesh: trimesh.Trimesh) -> dict:
    """
    Fallback function for simple planar UV mapping when xatlas fails.
    
    Args:
        mesh: Trimesh object
        
    Returns:
        Dictionary with 'vts' (texture coordinates) and 'fts' (face texture indices)
    """
    # Get mesh bounds
    bounds = mesh.bounds
    min_coords = bounds[0]
    max_coords = bounds[1]
    
    # Calculate UV coordinates by projecting vertices onto XY plane
    vertices = mesh.vertices
    u = (vertices[:, 0] - min_coords[0]) / (max_coords[0] - min_coords[0])
    v = (vertices[:, 1] - min_coords[1]) / (max_coords[1] - min_coords[1])
    
    # Clamp to [0, 1] range
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    # Create texture coordinates
    vts = np.column_stack([u, v])
    
    # Face texture indices are the same as vertex indices
    fts = mesh.faces
    
    return {
        "vts": vts,
        "fts": fts
    }

def get_textured_object_mesh(layout, room, room_id, object_id):

    import os
    import pickle

    layout_save_dir = RESULTS_DIR
    
    def get_object_mesh(source, source_id, layout_id):
        object_save_path = f"{layout_save_dir}/{layout_id}/{source}/{source_id}.ply"
        if os.path.exists(object_save_path):
            return trimesh.load(object_save_path)
        else:
            return None
        
    def get_object_mesh_texture(source, source_id, layout_id):
        tex_coords_save_path = f"{layout_save_dir}/{layout_id}/{source}/{source_id}_tex_coords.pkl"
        texture_map_path = f"{layout_save_dir}/{layout_id}/{source}/{source_id}_texture.png"
        if os.path.exists(tex_coords_save_path):
            with open(tex_coords_save_path, "rb") as f:
                tex_coords = pickle.load(f)
            return {
                "vts": tex_coords["vts"],
                "fts": tex_coords["fts"],
                "texture_map_path": texture_map_path
            }
        else:
            return None
    

    target_room = room
    if target_room is None:
        raise ValueError(f"Room {room_id} not found in layout")
    target_object = next(obj for obj in target_room.objects if obj.id == object_id)
    if target_object is None:
        raise ValueError(f"Object {object_id} not found in room {room_id}")

    target_object_mesh = get_object_mesh(target_object.source, target_object.source_id, layout.id)
    target_object_mesh = apply_object_transform(target_object_mesh, target_object)
    target_object_texture = get_object_mesh_texture(target_object.source, target_object.source_id, layout.id)

    mesh_info_dict = {
        "mesh": target_object_mesh,
        "static": False,
        "texture": {
            "vts": target_object_texture["vts"],
            "fts": target_object_texture["fts"],
            "texture_map_path": target_object_texture["texture_map_path"]
        }
    }

    return mesh_info_dict

def get_textured_object_mesh_from_layout_id(layout_id, room, room_id, object_id):

    import os
    import pickle

    layout_save_dir = RESULTS_DIR
    
    def get_object_mesh(source, source_id, layout_id):
        object_save_path = f"{layout_save_dir}/{layout_id}/{source}/{source_id}.ply"
        if os.path.exists(object_save_path):
            return trimesh.load(object_save_path)
        else:
            return None
        
    def get_object_mesh_texture(source, source_id, layout_id):
        tex_coords_save_path = f"{layout_save_dir}/{layout_id}/{source}/{source_id}_tex_coords.pkl"
        texture_map_path = f"{layout_save_dir}/{layout_id}/{source}/{source_id}_texture.png"
        if os.path.exists(tex_coords_save_path):
            with open(tex_coords_save_path, "rb") as f:
                tex_coords = pickle.load(f)
            return {
                "vts": tex_coords["vts"],
                "fts": tex_coords["fts"],
                "texture_map_path": texture_map_path
            }
        else:
            return None
    

    target_room = room
    if target_room is None:
        raise ValueError(f"Room {room_id} not found in layout")
    target_object = next(obj for obj in target_room.objects if obj.id == object_id)
    if target_object is None:
        raise ValueError(f"Object {object_id} not found in room {room_id}")

    target_object_mesh = get_object_mesh(target_object.source, target_object.source_id, layout_id)
    target_object_mesh = apply_object_transform(target_object_mesh, target_object)
    target_object_texture = get_object_mesh_texture(target_object.source, target_object.source_id, layout_id)

    mesh_info_dict = {
        "mesh": target_object_mesh,
        "static": False,
        "texture": {
            "vts": target_object_texture["vts"],
            "fts": target_object_texture["fts"],
            "texture_map_path": target_object_texture["texture_map_path"]
        }
    }

    return mesh_info_dict

def get_textured_object_mesh_from_object(layout: FloorPlan, object: Object):
    import os
    import pickle

    layout_save_dir = RESULTS_DIR
    
    def get_object_mesh(source, source_id, layout_id):
        object_save_path = f"{layout_save_dir}/{layout_id}/{source}/{source_id}.ply"
        if os.path.exists(object_save_path):
            return trimesh.load(object_save_path)
        else:
            return None
        
    def get_object_mesh_texture(source, source_id, layout_id):
        tex_coords_save_path = f"{layout_save_dir}/{layout_id}/{source}/{source_id}_tex_coords.pkl"
        texture_map_path = f"{layout_save_dir}/{layout_id}/{source}/{source_id}_texture.png"
        if os.path.exists(tex_coords_save_path):
            with open(tex_coords_save_path, "rb") as f:
                tex_coords = pickle.load(f)
            return {
                "vts": tex_coords["vts"],
                "fts": tex_coords["fts"],
                "texture_map_path": texture_map_path
            }
        else:
            return None

    source = object.source
    source_id = object.source_id
    layout_id = layout.id

    target_object_mesh = get_object_mesh(source, source_id, layout_id)

    # if target_object_mesh is not None:
    #     # Apply transforms to the object mesh
    #     target_object_mesh = apply_object_transform(target_object_mesh, object)
    # else:
    #     raise ValueError(f"Object {object.id} not found in layout {layout.id}")

    # get texture
    target_object_texture = get_object_mesh_texture(source, source_id, layout_id)
    if target_object_texture is None:
        raise ValueError(f"Texture for object {object.id} not found in layout {layout.id}")

    mesh_info_dict = {
        "mesh": target_object_mesh,
        "static": False,
        "texture": {
            "vts": target_object_texture["vts"],
            "fts": target_object_texture["fts"],
            "texture_map_path": target_object_texture["texture_map_path"]
        }
    }

    return mesh_info_dict