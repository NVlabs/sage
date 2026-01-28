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
This file contains the constraints for the optimization.

the constraints are defined inside

1. Objects
2. Walls
3. Doors
4. Windows
5. Floor

the objective is to optize the positions and the rotations of the objects to best fit the constraints

the constraints are defined as follows:

category 1: objects and walls

positions:

1.1 against(object, [wall]): 
object should be against any one of the walls in the list

1.2 center_aligned(object, [wall]):
object should be center aligned with the any one of the walls in the list

1.3 corner(object, wall_1, wall_2):
object should be at the corner of the two walls in the list

rotations:

1.4 relative_rotation(object, [wall], ("face_to", "face_away", "face_left", "face_right"))
object should be rotated to face the wall in the list or away from the wall in the list, or face the left or right of the wall in the list

category 2: objects and doors

positions:

2.1 avoid_collision(object, [door]):
object should not block the opening of any one of the doors in the list to avoid collision

2.2 close_to(object, [door]):
object should be close to any one of the doors and against the wall in the list

2.3 center_aligned(object, [door]):
object should be center aligned with the any one of the doors in the list

rotations:

2.4 relative_rotation(object, [door], ("face_to", "face_away", "face_left", "face_right"))
object should be rotated to face the door in the list or away from the door in the list, or face the left or right of the door in the list

category 3: objects and windows

positions:

3.1 center_aligned(object, [window]):
object should be center aligned with the any one of the windows in the list

rotations:

3.2 relative_rotation(object, [window], ("face_to", "face_away", "face_left", "face_right"))
object should be rotated to face the window in the list or away from the window in the list, or face the left or right of the window in the list

category 4: objects and room

positions:

4.1 center_aligned(object, [room]):
object should be center aligned with the room

positions + rotations:

4.2 inside(object, [room]):
object should be inside the room

cetegory 5: objects and objects:

positions:

5.1 relative_planer_position(object_1, object_2, ("left", "right", "front", "back"))
object_1 should be at the left, right, front, or back of object_2

5.2 center_aligned(object_1, object_2):
object_1 should be center aligned with object_2, center(object_1) - center(object_2) is perpendicular to the any one of the walls in the room

5.3 near(object_1, object_2, criterion_distance):
object_1 should be near object_2, distance(object_1, object_2) < criterion_distance

5.4 far(object_1, object_2, criterion_distance):
object_1 should be far from object_2, distance(object_1, object_2) > criterion_distance


rotations:

5.5 point_to(object_1, object_2):
object_1 should be pointing to object_2, object_1.normal is parallel to center(object_2) - center(object_1)

5.6 relative_rotation(object_1, object_2, direction):
object_1 should be rotated to face the object_2 or away from the object_2, or face the left or right of the object_2

some key notes:
1. when consider the distance between two objects, you need to consider the oriented bounding boxes and rotations of objects, not just the positions of objects.
2. when consider the contacts, also need to consider the oriented bounding boxes
3. walls, doors, and windows have thickness - constraints should use inner faces (room side) for positioning and alignment
4. door collision avoidance should account for door thickness and opening swing area
5. all distance calculations use oriented bounding boxes for accuracy with rotated objects
"""

import torch
import numpy as np
from primitives import ObjectGeometry, WallGeometry, DoorGeometry, WindowGeometry, RoomGeometry, OrientedBoundingBox, distance_obbs, distance_obbs_simple

# Helper functions
def to_tensor(data, device='cpu'):
    """Convert numpy array or list to torch tensor."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return torch.tensor(data, device=device, dtype=torch.float32)

# Distance function selection for optimization
USE_SIMPLE_DISTANCE = False  # Set to True for faster but less accurate distance calculations

def differentiable_distance_obbs(obb1, obb2):
    """Choose between accurate and simple distance functions based on USE_SIMPLE_DISTANCE flag."""
    if USE_SIMPLE_DISTANCE:
        return distance_obbs_simple(obb1, obb2)
    else:
        return distance_obbs(obb1, obb2)

# Loss function definitions
def loss_against_wall(obj: ObjectGeometry, walls: list[WallGeometry]) -> torch.Tensor:
    """
    Loss for object being against any wall in the list.
    Returns 0 if object is touching at least one wall's inner face, positive value otherwise.
    """
    min_distance = torch.tensor(float('inf'), device=obj.device)
    for wall in walls:
        # Use wall's inner face position (room side) accounting for thickness
        wall_normal = to_tensor(wall.normal, obj.device)
        wall_inner_face = to_tensor(wall.inner_face_position, obj.device)
        
        # Distance from object center to wall inner face plane
        obj_center = obj.center
        distance_to_wall_plane = torch.dot(obj_center - wall_inner_face, wall_normal)
        
        # Account for object's oriented bounding box projection onto wall normal
        # Get rotation matrix from object
        rotation_matrix = obj._euler_to_rotation_matrix(obj.rotation)
        
        # Project object's half extents onto wall normal considering rotation
        obj_half_extents = obj.extents / 2
        
        # Calculate the projection of the rotated bounding box onto the wall normal
        projected_extent = torch.tensor(0.0, device=obj.device)
        for i in range(3):
            axis = rotation_matrix[:, i]  # i-th axis of object's local coordinate system
            projected_extent += torch.abs(torch.dot(axis * obj_half_extents[i], wall_normal))
        
        # Distance from object surface to wall inner face
        surface_distance = distance_to_wall_plane - projected_extent
        min_distance = torch.min(min_distance, surface_distance)
    
    return torch.clamp(min_distance, min=0) ** 2

def loss_center_aligned_wall(obj: ObjectGeometry, walls: list[WallGeometry]) -> float:
    """
    Loss for object being center aligned with any wall in the list.
    Alignment is relative to the wall's inner face (room side).
    """
    min_alignment_error = float('inf')
    for wall in walls:
        # Calculate alignment error relative to wall's inner face
        obj_center = obj.center
        wall_direction = (wall.end_point - wall.start_point) / wall.length
        
        # Use wall's inner face as reference for alignment
        wall_inner_start = wall.start_point + wall.normal * (wall.thickness / 2)
        wall_inner_end = wall.end_point + wall.normal * (wall.thickness / 2)
        
        # Project object center onto wall inner face line
        to_obj = obj_center - wall_inner_start
        projection_length = np.dot(to_obj, wall_direction)
        projected_point = wall_inner_start + projection_length * wall_direction
        
        # Distance from object center to wall inner face line (perpendicular distance)
        wall_normal = wall.normal
        to_projected = obj_center - projected_point
        perpendicular_distance = abs(np.dot(to_projected, wall_normal))
        
        # Also consider alignment along the wall length
        along_wall_error = np.linalg.norm(to_projected - np.dot(to_projected, wall_normal) * wall_normal)
        
        # Combined alignment error (perpendicular distance is more important)
        alignment_error = perpendicular_distance + 0.1 * along_wall_error
        min_alignment_error = min(min_alignment_error, alignment_error)
    
    return min_alignment_error ** 2

def loss_corner(obj: ObjectGeometry, wall1: WallGeometry, wall2: WallGeometry) -> float:
    """
    Loss for object being at the corner of two walls.
    Uses wall inner faces to account for thickness.
    """
    # Calculate inner face positions for both walls
    wall1_inner_start = wall1.start_point + wall1.normal * (wall1.thickness / 2)
    wall1_inner_end = wall1.end_point + wall1.normal * (wall1.thickness / 2)
    wall2_inner_start = wall2.start_point + wall2.normal * (wall2.thickness / 2)
    wall2_inner_end = wall2.end_point + wall2.normal * (wall2.thickness / 2)
    
    # Find intersection point of wall inner faces
    wall1_dir = (wall1_inner_end - wall1_inner_start) / wall1.length
    wall2_dir = (wall2_inner_end - wall2_inner_start) / wall2.length
    
    # Find corner candidates (endpoints of inner faces)
    corner_candidates = [wall1_inner_start, wall1_inner_end, wall2_inner_start, wall2_inner_end]
    
    min_distance = float('inf')
    for corner in corner_candidates:
        # Check if this point is close to both wall inner faces
        # Distance to wall1 inner face line
        to_corner1 = corner - wall1_inner_start
        proj1 = np.dot(to_corner1, wall1_dir)
        closest1 = wall1_inner_start + np.clip(proj1, 0, wall1.length) * wall1_dir
        dist1 = np.linalg.norm(corner - closest1)
        
        # Distance to wall2 inner face line
        to_corner2 = corner - wall2_inner_start
        proj2 = np.dot(to_corner2, wall2_dir)
        closest2 = wall2_inner_start + np.clip(proj2, 0, wall2.length) * wall2_dir
        dist2 = np.linalg.norm(corner - closest2)
        
        if dist1 < 0.15 and dist2 < 0.15:  # Both walls are close to this point
            # Calculate distance from object surface to corner
            obj_obb = obj.oriented_bounding_box
            corner_obb = OrientedBoundingBox(corner, np.array([0, 0, 0]), np.array([0.1, 0.1, 0.1]))
            obj_distance = distance_obbs(obj_obb, corner_obb)
            min_distance = min(min_distance, max(0, obj_distance))
    
    return min_distance ** 2

def loss_relative_rotation_wall(obj: ObjectGeometry, walls: list[WallGeometry], direction: str) -> float:
    """
    Loss for object rotation relative to wall.
    """
    min_rotation_error = float('inf')
    
    for wall in walls:
        wall_direction = (wall.end_point - wall.start_point) / wall.length
        wall_normal = wall.normal
        
        obj_direction = obj.direction
        
        if direction == "face_to":
            # Object should face the wall (direction aligned with wall normal)
            target_direction = wall_normal
        elif direction == "face_away":
            # Object should face away from wall
            target_direction = -wall_normal
        elif direction == "face_left":
            # Object should face left relative to wall direction
            target_direction = np.cross(wall_direction, np.array([0, 0, 1]))[:2]
            target_direction = np.append(target_direction, [0])
        elif direction == "face_right":
            # Object should face right relative to wall direction
            target_direction = -np.cross(wall_direction, np.array([0, 0, 1]))[:2]
            target_direction = np.append(target_direction, [0])
        else:
            continue
        
        # Calculate angle difference
        dot_product = np.dot(obj_direction, target_direction)
        angle_error = 1 - dot_product  # 0 when aligned, 2 when opposite
        min_rotation_error = min(min_rotation_error, angle_error)
    
    return min_rotation_error ** 2

def loss_avoid_collision_door(obj: ObjectGeometry, doors: list[DoorGeometry]) -> float:
    """
    Loss for object avoiding collision with door opening.
    Accounts for door thickness and opening swing area.
    """
    total_collision = 0
    
    for door in doors:
        # Door opening area includes the door thickness and swing space
        door_center = door.world_position
        door_normal = door.normal
        door_width = door.width
        door_thickness = door.thickness
        
        # Create opening zone in front of door (accounting for door thickness)
        # The door itself occupies space, so opening starts from door surface
        opening_depth = 1.2  # 1.2 meter opening depth from door surface
        door_surface_center = door_center + door_normal * (door_thickness / 2)
        opening_center = door_surface_center + door_normal * (opening_depth / 2)
        
        # Check if object's oriented bounding box intersects with opening zone
        obj_obb = obj.oriented_bounding_box
        
        # Create opening zone as an oriented bounding box
        # Opening zone dimensions: width x opening_depth x door_height
        opening_extents = np.array([door_width + 0.2, opening_depth, door.height])  # Add small buffer
        opening_position = opening_center
        opening_rotation = np.array([0, 0, 0])  # Aligned with world coordinates
        
        # Calculate door orientation based on door normal
        if abs(door_normal[0]) > abs(door_normal[1]):  # Door faces x direction
            if door_normal[0] > 0:
                opening_rotation[2] = 0  # Facing +x
            else:
                opening_rotation[2] = 180  # Facing -x
        else:  # Door faces y direction
            if door_normal[1] > 0:
                opening_rotation[2] = 90  # Facing +y
            else:
                opening_rotation[2] = -90  # Facing -y
        
        from primitives import OrientedBoundingBox, distance_obbs
        opening_obb = OrientedBoundingBox(opening_position, opening_rotation, opening_extents)
        
        # Calculate distance between object and opening zone
        distance = distance_obbs(obj_obb, opening_obb)
        
        if distance < 0:  # Objects are intersecting
            # Collision penalty based on penetration depth
            penetration_penalty = abs(distance)
            total_collision += penetration_penalty
    
    return total_collision ** 2

def loss_close_to_door(obj: ObjectGeometry, doors: list[DoorGeometry]) -> float:
    """
    Loss for object being close to any door.
    Distance is measured to the door's inner surface (room side).
    """
    min_distance = float('inf')
    
    for door in doors:
        # Calculate distance to door's inner surface (room side)
        door_inner_surface = door.world_position - door.normal * (door.thickness / 2)
        
        # Use oriented bounding box distance for accuracy
        obj_obb = obj.oriented_bounding_box
        
        # Create a thin bounding box representing the door surface
        door_surface_extents = np.array([door.width, door.thickness * 0.1, door.height])
        door_surface_obb = OrientedBoundingBox(door_inner_surface, np.array([0, 0, 0]), door_surface_extents)
        
        distance = distance_obbs(obj_obb, door_surface_obb)
        min_distance = min(min_distance, max(0, distance))  # Ensure non-negative
    
    return min_distance ** 2

def loss_center_aligned_door(obj: ObjectGeometry, doors: list[DoorGeometry]) -> float:
    """
    Loss for object being center aligned with any door.
    Alignment considers door thickness and positions relative to door's inner surface.
    """
    min_alignment_error = float('inf')
    
    for door in doors:
        # Use door's inner surface position for alignment (room side)
        door_inner_surface = door.world_position - door.normal * (door.thickness / 2)
        
        # Calculate alignment error in the plane perpendicular to door normal
        obj_center = obj.center
        door_normal = door.normal
        
        # Project both positions onto a plane perpendicular to door normal
        # Remove the component along door normal for alignment calculation
        obj_projected = obj_center - np.dot(obj_center - door_inner_surface, door_normal) * door_normal
        door_projected = door_inner_surface
        
        # Calculate alignment error in x-y plane
        alignment_error = np.linalg.norm(obj_projected[:2] - door_projected[:2])
        min_alignment_error = min(min_alignment_error, alignment_error)
    
    return min_alignment_error ** 2

def loss_relative_rotation_door(obj: ObjectGeometry, doors: list[DoorGeometry], direction: str) -> float:
    """
    Loss for object rotation relative to door.
    """
    min_rotation_error = float('inf')
    
    for door in doors:
        door_normal = door.normal
        obj_direction = obj.direction
        
        if direction == "face_to":
            target_direction = door_normal
        elif direction == "face_away":
            target_direction = -door_normal
        elif direction == "face_left":
            target_direction = np.cross(door_normal, np.array([0, 0, 1]))[:2]
            target_direction = np.append(target_direction, [0])
        elif direction == "face_right":
            target_direction = -np.cross(door_normal, np.array([0, 0, 1]))[:2]
            target_direction = np.append(target_direction, [0])
        else:
            continue
        
        dot_product = np.dot(obj_direction, target_direction)
        angle_error = 1 - dot_product
        min_rotation_error = min(min_rotation_error, angle_error)
    
    return min_rotation_error ** 2

def loss_center_aligned_window(obj: ObjectGeometry, windows: list[WindowGeometry]) -> float:
    """
    Loss for object being center aligned with any window.
    Alignment considers window thickness and positions relative to window's inner surface.
    """
    min_alignment_error = float('inf')
    
    for window in windows:
        # Use window's inner surface position for alignment (room side)
        window_inner_surface = window.world_position - window.normal * (window.thickness / 2)
        
        # Calculate alignment error in the plane perpendicular to window normal
        obj_center = obj.center
        window_normal = window.normal
        
        # Project both positions onto a plane perpendicular to window normal
        obj_projected = obj_center - np.dot(obj_center - window_inner_surface, window_normal) * window_normal
        window_projected = window_inner_surface
        
        # Calculate alignment error in x-y plane
        alignment_error = np.linalg.norm(obj_projected[:2] - window_projected[:2])
        min_alignment_error = min(min_alignment_error, alignment_error)
    
    return min_alignment_error ** 2

def loss_relative_rotation_window(obj: ObjectGeometry, windows: list[WindowGeometry], direction: str) -> float:
    """
    Loss for object rotation relative to window.
    """
    min_rotation_error = float('inf')
    
    for window in windows:
        window_normal = window.normal
        obj_direction = obj.direction
        
        if direction == "face_to":
            target_direction = window_normal
        elif direction == "face_away":
            target_direction = -window_normal
        elif direction == "face_left":
            target_direction = np.cross(window_normal, np.array([0, 0, 1]))[:2]
            target_direction = np.append(target_direction, [0])
        elif direction == "face_right":
            target_direction = -np.cross(window_normal, np.array([0, 0, 1]))[:2]
            target_direction = np.append(target_direction, [0])
        else:
            continue
        
        dot_product = np.dot(obj_direction, target_direction)
        angle_error = 1 - dot_product
        min_rotation_error = min(min_rotation_error, angle_error)
    
    return min_rotation_error ** 2

def loss_center_aligned_room(obj: ObjectGeometry, rooms: list[RoomGeometry]) -> float:
    """
    Loss for object being center aligned with room.
    """
    min_alignment_error = float('inf')
    
    for room in rooms:
        alignment_error = np.linalg.norm(obj.center[:2] - room.center[:2])
        min_alignment_error = min(min_alignment_error, alignment_error)
    
    return min_alignment_error ** 2

def loss_inside_room(obj: ObjectGeometry, rooms: list[RoomGeometry]) -> float:
    """
    Loss for object being inside room.
    """
    min_outside_penalty = float('inf')
    
    for room in rooms:
        # Check if object is within room bounds
        room_center = room.center
        room_half_width = room.width / 2
        room_half_length = room.length / 2
        room_height = room.height
        
        obj_center = obj.center
        obj_half_extents = obj.extents / 2
        
        # Calculate how much object extends outside room bounds
        x_outside = max(0, abs(obj_center[0] - room_center[0]) + obj_half_extents[0] - room_half_width)
        y_outside = max(0, abs(obj_center[1] - room_center[1]) + obj_half_extents[1] - room_half_length)
        z_outside = max(0, obj_center[2] + obj_half_extents[2] - room_height)
        
        outside_penalty = x_outside + y_outside + z_outside
        min_outside_penalty = min(min_outside_penalty, outside_penalty)
    
    return min_outside_penalty ** 2

def loss_relative_planar_position(obj1: ObjectGeometry, obj2: ObjectGeometry, direction: str) -> float:
    """
    Loss for relative planar position between objects.
    """
    obj1_center = obj1.center
    obj2_center = obj2.center
    relative_pos = obj1_center - obj2_center
    
    obj2_direction = obj2.direction
    obj2_right = np.cross(obj2_direction, np.array([0, 0, 1]))[:2]
    obj2_right = np.append(obj2_right, [0])
    
    if direction == "left":
        # obj1 should be to the left of obj2
        target_direction = -obj2_right
    elif direction == "right":
        # obj1 should be to the right of obj2
        target_direction = obj2_right
    elif direction == "front":
        # obj1 should be in front of obj2
        target_direction = obj2_direction
    elif direction == "back":
        # obj1 should be behind obj2
        target_direction = -obj2_direction
    else:
        return 0
    
    # Check if relative position aligns with target direction
    dot_product = np.dot(relative_pos, target_direction)
    if dot_product > 0:
        return 0  # Correct relative position
    else:
        return (-dot_product) ** 2  # Penalty for wrong direction

def loss_center_aligned_objects(obj1: ObjectGeometry, obj2: ObjectGeometry) -> float:
    """
    Loss for objects being center aligned.
    """
    alignment_error = np.linalg.norm(obj1.center[:2] - obj2.center[:2])
    return alignment_error ** 2

def loss_near_objects(obj1: ObjectGeometry, obj2: ObjectGeometry, criterion_distance: float) -> torch.Tensor:
    """
    Loss for objects being near each other.
    """
    distance = distance_obbs(obj1.oriented_bounding_box, obj2.oriented_bounding_box)
    criterion_tensor = torch.tensor(criterion_distance, device=obj1.device, dtype=torch.float32)
    return torch.clamp(distance - criterion_tensor, min=0) ** 2

def loss_far_objects(obj1: ObjectGeometry, obj2: ObjectGeometry, criterion_distance: float) -> float:
    """
    Loss for objects being far from each other.
    """
    distance = distance_obbs(obj1.oriented_bounding_box, obj2.oriented_bounding_box)
    if distance > criterion_distance:
        return 0
    else:
        return (criterion_distance - distance) ** 2

def loss_point_to(obj1: ObjectGeometry, obj2: ObjectGeometry) -> float:
    """
    Loss for obj1 pointing to obj2.
    """
    obj1_direction = obj1.direction
    to_obj2 = obj2.center - obj1.center
    to_obj2_normalized = to_obj2 / np.linalg.norm(to_obj2)
    
    # Calculate angle between obj1's direction and direction to obj2
    dot_product = np.dot(obj1_direction, to_obj2_normalized)
    angle_error = 1 - dot_product
    
    return angle_error ** 2

def loss_relative_rotation_objects(obj1: ObjectGeometry, obj2: ObjectGeometry, direction: str) -> float:
    """
    Loss for relative rotation between objects.
    """
    obj1_direction = obj1.direction
    obj2_direction = obj2.direction
    
    if direction == "face_to":
        to_obj2 = obj2.center - obj1.center
        target_direction = to_obj2 / np.linalg.norm(to_obj2)
    elif direction == "face_away":
        to_obj2 = obj2.center - obj1.center
        target_direction = -to_obj2 / np.linalg.norm(to_obj2)
    elif direction == "face_left":
        obj2_right = np.cross(obj2_direction, np.array([0, 0, 1]))[:2]
        target_direction = -np.append(obj2_right, [0])
    elif direction == "face_right":
        obj2_right = np.cross(obj2_direction, np.array([0, 0, 1]))[:2]
        target_direction = np.append(obj2_right, [0])
    else:
        return 0
    
    dot_product = np.dot(obj1_direction, target_direction)
    angle_error = 1 - dot_product
    
    return angle_error ** 2

# Constraints to losses mapping
constraints_to_losses = {
    # Category 1: Objects and Walls
    "against": loss_against_wall,
    "center_aligned_wall": loss_center_aligned_wall,
    "corner": loss_corner,
    "relative_rotation_wall": loss_relative_rotation_wall,
    
    # Category 2: Objects and Doors
    "avoid_collision": loss_avoid_collision_door,
    "close_to_door": loss_close_to_door,
    "center_aligned_door": loss_center_aligned_door,
    "relative_rotation_door": loss_relative_rotation_door,
    
    # Category 3: Objects and Windows
    "center_aligned_window": loss_center_aligned_window,
    "relative_rotation_window": loss_relative_rotation_window,
    
    # Category 4: Objects and Room
    "center_aligned_room": loss_center_aligned_room,
    "inside": loss_inside_room,
    
    # Category 5: Objects and Objects
    "relative_planar_position": loss_relative_planar_position,
    "center_aligned_objects": loss_center_aligned_objects,
    "near": loss_near_objects,
    "far": loss_far_objects,
    "point_to": loss_point_to,
    "relative_rotation_objects": loss_relative_rotation_objects,
}

# Utility classes and functions for constraint evaluation
class Constraint:
    """Represents a single constraint with its type, objects, and parameters."""
    
    def __init__(self, constraint_type: str, objects: list, target_objects: list = None, 
                 parameters: dict = None, weight: float = 1.0):
        """
        Initialize a constraint.
        
        Args:
            constraint_type: Type of constraint (key in constraints_to_losses)
            objects: List of primary objects involved in the constraint
            target_objects: List of target objects (walls, doors, windows, rooms, or other objects)
            parameters: Additional parameters for the constraint (e.g., distance, direction)
            weight: Weight of this constraint in the total loss
        """
        self.constraint_type = constraint_type
        self.objects = objects if isinstance(objects, list) else [objects]
        self.target_objects = target_objects if target_objects is None or isinstance(target_objects, list) else [target_objects]
        self.parameters = parameters or {}
        self.weight = weight
    
    def evaluate(self) -> float:
        """Evaluate the constraint and return the weighted loss."""
        if self.constraint_type not in constraints_to_losses:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")
        
        loss_function = constraints_to_losses[self.constraint_type]
        
        # Handle different constraint signatures
        if self.constraint_type in ["against", "center_aligned_wall", "relative_rotation_wall"]:
            # Object-wall constraints
            total_loss = 0
            for obj in self.objects:
                if self.constraint_type == "relative_rotation_wall":
                    loss = loss_function(obj, self.target_objects, self.parameters.get("direction", "face_to"))
                else:
                    loss = loss_function(obj, self.target_objects)
                total_loss += loss
            return self.weight * total_loss
            
        elif self.constraint_type == "corner":
            # Corner constraint needs two walls
            if len(self.target_objects) != 2:
                raise ValueError("Corner constraint requires exactly 2 walls")
            total_loss = 0
            for obj in self.objects:
                loss = loss_function(obj, self.target_objects[0], self.target_objects[1])
                total_loss += loss
            return self.weight * total_loss
            
        elif self.constraint_type in ["avoid_collision", "close_to_door", "center_aligned_door", "relative_rotation_door"]:
            # Object-door constraints
            total_loss = 0
            for obj in self.objects:
                if self.constraint_type == "relative_rotation_door":
                    loss = loss_function(obj, self.target_objects, self.parameters.get("direction", "face_to"))
                else:
                    loss = loss_function(obj, self.target_objects)
                total_loss += loss
            return self.weight * total_loss
            
        elif self.constraint_type in ["center_aligned_window", "relative_rotation_window"]:
            # Object-window constraints
            total_loss = 0
            for obj in self.objects:
                if self.constraint_type == "relative_rotation_window":
                    loss = loss_function(obj, self.target_objects, self.parameters.get("direction", "face_to"))
                else:
                    loss = loss_function(obj, self.target_objects)
                total_loss += loss
            return self.weight * total_loss
            
        elif self.constraint_type in ["center_aligned_room", "inside"]:
            # Object-room constraints
            total_loss = 0
            for obj in self.objects:
                loss = loss_function(obj, self.target_objects)
                total_loss += loss
            return self.weight * total_loss
            
        elif self.constraint_type in ["relative_planar_position", "center_aligned_objects", "point_to", "relative_rotation_objects"]:
            # Object-object constraints
            if len(self.objects) != 1 or len(self.target_objects) != 1:
                raise ValueError(f"Constraint {self.constraint_type} requires exactly 1 object and 1 target object")
            
            obj1 = self.objects[0]
            obj2 = self.target_objects[0]
            
            if self.constraint_type in ["relative_planar_position", "relative_rotation_objects"]:
                loss = loss_function(obj1, obj2, self.parameters.get("direction", "front"))
            else:
                loss = loss_function(obj1, obj2)
                
            return self.weight * loss
            
        elif self.constraint_type in ["near", "far"]:
            # Distance constraints
            if len(self.objects) != 1 or len(self.target_objects) != 1:
                raise ValueError(f"Constraint {self.constraint_type} requires exactly 1 object and 1 target object")
                
            obj1 = self.objects[0]
            obj2 = self.target_objects[0]
            criterion_distance = self.parameters.get("distance", 1.0)
            
            loss = loss_function(obj1, obj2, criterion_distance)
            return self.weight * loss
            
        else:
            raise ValueError(f"Unhandled constraint type: {self.constraint_type}")

class ConstraintSystem:
    """Manages a collection of constraints and evaluates the total loss."""
    
    def __init__(self):
        self.constraints = []
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the system."""
        self.constraints.append(constraint)
    
    def add_against_wall(self, obj: ObjectGeometry, walls: list[WallGeometry], weight: float = 1.0):
        """Convenience method to add an 'against wall' constraint."""
        self.add_constraint(Constraint("against", obj, walls, weight=weight))
    
    def add_center_aligned_wall(self, obj: ObjectGeometry, walls: list[WallGeometry], weight: float = 1.0):
        """Convenience method to add a 'center aligned with wall' constraint."""
        self.add_constraint(Constraint("center_aligned_wall", obj, walls, weight=weight))
    
    def add_corner(self, obj: ObjectGeometry, wall1: WallGeometry, wall2: WallGeometry, weight: float = 1.0):
        """Convenience method to add a corner constraint."""
        self.add_constraint(Constraint("corner", obj, [wall1, wall2], weight=weight))
    
    def add_inside_room(self, obj: ObjectGeometry, room: RoomGeometry, weight: float = 1.0):
        """Convenience method to add an 'inside room' constraint."""
        self.add_constraint(Constraint("inside", obj, room, weight=weight))
    
    def add_near_objects(self, obj1: ObjectGeometry, obj2: ObjectGeometry, distance: float, weight: float = 1.0):
        """Convenience method to add a 'near objects' constraint."""
        self.add_constraint(Constraint("near", obj1, obj2, {"distance": distance}, weight=weight))
    
    def add_far_objects(self, obj1: ObjectGeometry, obj2: ObjectGeometry, distance: float, weight: float = 1.0):
        """Convenience method to add a 'far objects' constraint."""
        self.add_constraint(Constraint("far", obj1, obj2, {"distance": distance}, weight=weight))
    
    def add_point_to(self, obj1: ObjectGeometry, obj2: ObjectGeometry, weight: float = 1.0):
        """Convenience method to add a 'point to' constraint."""
        self.add_constraint(Constraint("point_to", obj1, obj2, weight=weight))
    
    def add_relative_position(self, obj1: ObjectGeometry, obj2: ObjectGeometry, direction: str, weight: float = 1.0):
        """Convenience method to add a relative position constraint."""
        self.add_constraint(Constraint("relative_planar_position", obj1, obj2, {"direction": direction}, weight=weight))
    
    def evaluate_total_loss(self) -> float:
        """Evaluate the total weighted loss for all constraints."""
        total_loss = 0
        for constraint in self.constraints:
            try:
                loss = constraint.evaluate()
                total_loss += loss
            except Exception as e:
                print(f"Error evaluating constraint {constraint.constraint_type}: {e}")
                continue
        return total_loss
    
    def evaluate_individual_losses(self) -> dict:
        """Evaluate individual losses for debugging purposes."""
        losses = {}
        for i, constraint in enumerate(self.constraints):
            try:
                loss = constraint.evaluate()
                constraint_id = f"{constraint.constraint_type}_{i}"
                losses[constraint_id] = loss
            except Exception as e:
                print(f"Error evaluating constraint {constraint.constraint_type}: {e}")
                losses[f"{constraint.constraint_type}_{i}"] = float('inf')
        return losses
    
    def clear_constraints(self):
        """Clear all constraints."""
        self.constraints.clear()
    
    def remove_constraint(self, index: int):
        """Remove a constraint by index."""
        if 0 <= index < len(self.constraints):
            self.constraints.pop(index)
    
    def get_constraint_count(self) -> int:
        """Get the number of constraints."""
        return len(self.constraints)