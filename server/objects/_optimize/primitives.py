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
import trimesh
import torch
import numpy as np
from models import Room, Wall, Door, Window
from scipy.spatial.transform import Rotation as R

class OrientedBoundingBox:
    def __init__(self, position, rotation, extents, device='cpu'):
        """
        Args:
            position: torch.tensor([x, y, z]), the position of the oriented bounding box
            rotation: torch.tensor([x, y, z]), the rotation of the oriented bounding box, euler angles in degrees
            extents: torch.tensor([width, length, height]), the extents of the oriented bounding box
            device: torch device ('cpu' or 'cuda')
        """
        self.device = device
        self.position = self._to_tensor(position)
        self.rotation = self._to_tensor(rotation)
        self.extents = self._to_tensor(extents)
    
    def _to_tensor(self, data):
        """Convert input to torch tensor if not already."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return torch.tensor(data, dtype=torch.float32, device=self.device)

    def bounding_box(self):
        """
        Calculate the axis aligned bounding box of the oriented bounding box.
        
        Returns:
            torch.tensor: Shape (2, 3) where first row is min corner [x_min, y_min, z_min]
                         and second row is max corner [x_max, y_max, z_max]
        """
        # Half extents for easier corner calculation
        half_extents = self.extents / 2.0
        
        # Generate all 8 corners of the OBB in local coordinates
        corners = torch.tensor([
            [-half_extents[0], -half_extents[1], -half_extents[2]],  # ---
            [+half_extents[0], -half_extents[1], -half_extents[2]],  # +--
            [-half_extents[0], +half_extents[1], -half_extents[2]],  # -+-
            [+half_extents[0], +half_extents[1], -half_extents[2]],  # ++-
            [-half_extents[0], -half_extents[1], +half_extents[2]],  # --+
            [+half_extents[0], -half_extents[1], +half_extents[2]],  # +-+
            [-half_extents[0], +half_extents[1], +half_extents[2]],  # -++
            [+half_extents[0], +half_extents[1], +half_extents[2]]   # +++
        ], device=self.device, dtype=torch.float32)
        
        # Create rotation matrix from euler angles using torch
        rotation_matrix = self._euler_to_rotation_matrix(self.rotation)
        
        # Transform corners: rotate then translate
        transformed_corners = torch.matmul(corners, rotation_matrix.T) + self.position
        
        # Find min and max bounds across all transformed corners
        min_corner = torch.min(transformed_corners, dim=0)[0]
        max_corner = torch.max(transformed_corners, dim=0)[0]
        
        return torch.stack([min_corner, max_corner])
    
    def _euler_to_rotation_matrix(self, euler_angles):
        """Convert euler angles (in degrees) to rotation matrix using torch."""
        # Convert degrees to radians
        angles = euler_angles * torch.pi / 180.0
        x, y, z = angles[0], angles[1], angles[2]
        
        # Create rotation matrices for each axis
        cos_x, sin_x = torch.cos(x), torch.sin(x)
        cos_y, sin_y = torch.cos(y), torch.sin(y)
        cos_z, sin_z = torch.cos(z), torch.sin(z)
        
        # Rotation matrix around X axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ], device=self.device, dtype=torch.float32)
        
        # Rotation matrix around Y axis
        Ry = torch.tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], device=self.device, dtype=torch.float32)
        
        # Rotation matrix around Z axis
        Rz = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        # Combined rotation matrix (ZYX order)
        return torch.matmul(torch.matmul(Rz, Ry), Rx)

def distance_obbs(obb1: OrientedBoundingBox, obb2: OrientedBoundingBox):
    """
    Calculate the closest distance between the surfaces of two oriented bounding boxes.
    
    Returns:
        torch.tensor: Distance between OBBs
                     - Negative if intersecting (penetration depth)
                     - Zero if touching
                     - Positive if separated
    """
    # Create rotation matrices using torch
    rot1 = obb1._euler_to_rotation_matrix(obb1.rotation)
    rot2 = obb2._euler_to_rotation_matrix(obb2.rotation)
    
    # Position difference
    pos_diff = obb2.position - obb1.position
    
    # Half extents
    half_ext1 = obb1.extents / 2.0
    half_ext2 = obb2.extents / 2.0
    
    # Rotation matrix from obb1 to obb2 coordinate system
    R12 = torch.matmul(rot1.T, rot2)
    
    # Absolute values of rotation matrix elements for separating axis test
    abs_R12 = torch.abs(R12)
    
    # Position of obb2 center in obb1's coordinate system
    pos_in_obb1 = torch.matmul(rot1.T, pos_diff)
    
    # Collect all separation distances (no early returns for differentiability)
    separations = []
    
    # Test axes aligned with obb1's local axes (3 axes)
    for i in range(3):
        # Project both OBBs onto obb1's i-th axis
        proj1 = half_ext1[i]
        proj2 = torch.sum(abs_R12[i, :] * half_ext2)
        separation = torch.abs(pos_in_obb1[i]) - (proj1 + proj2)
        separations.append(separation)
    
    # Test axes aligned with obb2's local axes (3 axes)
    for j in range(3):
        # Project both OBBs onto obb2's j-th axis (expressed in obb1's coordinates)
        proj1 = torch.sum(abs_R12[:, j] * half_ext1)
        proj2 = half_ext2[j]
        separation = torch.abs(torch.sum(R12[:, j] * pos_in_obb1)) - (proj1 + proj2)
        separations.append(separation)
    
    # Test cross product axes (9 axes) - use smooth weighting instead of continue
    for i in range(3):
        for j in range(3):
            # Smooth weighting for near-parallel axes instead of hard skip
            parallel_weight = torch.sigmoid(10.0 * (0.99999 - torch.abs(R12[i, j])))
            
            # Cross product of obb1's i-th axis with obb2's j-th axis
            # Project both OBBs onto this cross product axis
            
            # Components for projection calculation
            proj1_components = torch.zeros(3, device=obb1.device)
            proj1_components[(i+1)%3] = abs_R12[(i+2)%3, j] * half_ext1[(i+2)%3]
            proj1_components[(i+2)%3] = abs_R12[(i+1)%3, j] * half_ext1[(i+1)%3]
            proj1 = proj1_components[(i+1)%3] + proj1_components[(i+2)%3]
            
            proj2_components = torch.zeros(3, device=obb1.device)
            proj2_components[(j+1)%3] = abs_R12[i, (j+2)%3] * half_ext2[(j+2)%3]
            proj2_components[(j+2)%3] = abs_R12[i, (j+1)%3] * half_ext2[(j+1)%3]
            proj2 = proj2_components[(j+1)%3] + proj2_components[(j+2)%3]
            
            # Position projection onto cross product axis
            pos_proj = torch.abs(pos_in_obb1[(i+1)%3] * R12[(i+2)%3, j] - 
                               pos_in_obb1[(i+2)%3] * R12[(i+1)%3, j])
            
            separation = pos_proj - (proj1 + proj2)
            
            # Weight the separation by how non-parallel the axes are
            weighted_separation = separation * parallel_weight
            separations.append(weighted_separation)
    
    # Use smooth maximum to find the most separating axis (differentiable)
    separations_tensor = torch.stack(separations)
    
    # Smooth maximum using LogSumExp trick for differentiability
    alpha = 10.0  # Sharpness parameter
    max_separation = torch.logsumexp(alpha * separations_tensor, dim=0) / alpha
    
    return max_separation



class ObjectGeometry:
    def __init__(self, mesh: trimesh.Trimesh, device='cpu', requires_grad=True) -> None:

        self.mesh = mesh
        self.device = device
        
        # Initialize position and rotation as trainable parameters
        self.position = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=requires_grad, dtype=torch.float32)
        self.rotation = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=requires_grad, dtype=torch.float32)

        # check whether the mesh is over x-y plane
        if not np.all(mesh.vertices[:, 2] >= 0):
            raise ValueError("Mesh is not over x-y plane")
        
        # some background:
        # object is a 3d mesh, which is a collection of vertices and faces
        # each vertex has a 3d position
        # every vertex is over z=0 plane
        # the object center is (0, 0, 0) when the position is (0, 0, 0)
        # the object has no scale property
        # the object is in the unit of meter

        # the object's bounding box is a 3d box that contains the object
        # which centers at (0, 0, 0.5 * z_max)
        # which means that the object's minimum z is 0, and maximum z is z_max

        # object rotation is a 3d rotation, which is a combination of x, y, z rotation euler angles
        # object is facing +x axis when the rotation is 0, 0, 0
        # object is facing -x axis when the rotation is 0, 0, 180
        # object is facing +y axis when the rotation is 0, 0, 90
        # object is facing -y axis when the rotation is 0, 0, -90

        # Store mesh dimensions as tensors for consistent computation
        self.height = torch.tensor(mesh.vertices[:, 2].max(), device=device, dtype=torch.float32)
        self.width = torch.tensor(mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min(), device=device, dtype=torch.float32)
        self.length = torch.tensor(mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min(), device=device, dtype=torch.float32)

    @property
    def center(self):
        return self.position + torch.tensor([0, 0, 0.5], device=self.device) * self.height
    
    @property
    def extents(self):
        return torch.stack([self.width, self.length, self.height])
    
    @property
    def oriented_bounding_box(self):
        return OrientedBoundingBox(self.position, self.rotation, self.extents, device=self.device)
    
    @property
    def direction(self):
        # originally the direction is facing +x:
        direction = torch.tensor([1.0, 0.0, 0.0], device=self.device)

        # the rotation is a 3d rotation of euler angles
        # multiply the direction by the rotation matrix
        rotation_matrix = self._euler_to_rotation_matrix(self.rotation)
        direction = torch.matmul(rotation_matrix, direction)

        return direction
    
    def _euler_to_rotation_matrix(self, euler_angles):
        """Convert euler angles (in degrees) to rotation matrix using torch."""
        # Convert degrees to radians
        angles = euler_angles * torch.pi / 180.0
        x, y, z = angles[0], angles[1], angles[2]
        
        # Create rotation matrices for each axis
        cos_x, sin_x = torch.cos(x), torch.sin(x)
        cos_y, sin_y = torch.cos(y), torch.sin(y)
        cos_z, sin_z = torch.cos(z), torch.sin(z)
        
        # Rotation matrix around X axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ], device=self.device, dtype=torch.float32)
        
        # Rotation matrix around Y axis
        Ry = torch.tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], device=self.device, dtype=torch.float32)
        
        # Rotation matrix around Z axis
        Rz = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        # Combined rotation matrix (ZYX order)
        return torch.matmul(torch.matmul(Rz, Ry), Rx)
    
class ObjectGeometryGroup:
    def __init__(self, objects: list[ObjectGeometry], device='cpu'):
        self.objects = objects
        self.device = device
        self.position = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        self.rotation = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        
    @property
    def oriented_bounding_box(self):
        """Calculate the oriented bounding box that encompasses all objects in the group."""
        if not self.objects:
            return OrientedBoundingBox(self.position, self.rotation, np.array([0, 0, 0]))
        
        # Get all object bounding boxes
        all_corners = []
        for obj in self.objects:
            obb = obj.oriented_bounding_box
            bbox = obb.bounding_box()
            # Add all 8 corners of the bounding box
            corners = [
                [bbox[0][0], bbox[0][1], bbox[0][2]],  # min corner
                [bbox[1][0], bbox[0][1], bbox[0][2]],  # max x, min y,z
                [bbox[0][0], bbox[1][1], bbox[0][2]],  # min x, max y, min z
                [bbox[0][0], bbox[0][1], bbox[1][2]],  # min x,y, max z
                [bbox[1][0], bbox[1][1], bbox[0][2]],  # max x,y, min z
                [bbox[1][0], bbox[0][1], bbox[1][2]],  # max x, min y, max z
                [bbox[0][0], bbox[1][1], bbox[1][2]],  # min x, max y,z
                [bbox[1][0], bbox[1][1], bbox[1][2]]   # max corner
            ]
            all_corners.extend(corners)
        
        all_corners = torch.stack([torch.stack(corner) for corner in all_corners])
        min_corner = torch.min(all_corners, dim=0)[0]
        max_corner = torch.max(all_corners, dim=0)[0]
        
        # Calculate group bounding box
        group_center = (min_corner + max_corner) / 2
        group_extents = max_corner - min_corner
        
        return OrientedBoundingBox(group_center, self.rotation, group_extents, device=self.device)
    
    @property
    def bounding_box(self):
        """Get axis-aligned bounding box."""
        return self.oriented_bounding_box.bounding_box()
    
    @property
    def extents(self):
        return self.oriented_bounding_box.extents
    
    @property
    def center(self):
        bbox = self.bounding_box
        return (bbox[0] + bbox[1]) / 2

    @property
    def width(self):
        return self.extents[0]
    
    @property
    def length(self):
        return self.extents[1]
    
    @property
    def height(self):
        return self.extents[2]
    
    @property
    def direction(self):
        direction = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        rotation_matrix = self._euler_to_rotation_matrix(self.rotation)
        direction = torch.matmul(rotation_matrix, direction)
        return direction
    
    def _euler_to_rotation_matrix(self, euler_angles):
        """Convert euler angles (in degrees) to rotation matrix using torch."""
        # Convert degrees to radians
        angles = euler_angles * torch.pi / 180.0
        x, y, z = angles[0], angles[1], angles[2]
        
        # Create rotation matrices for each axis
        cos_x, sin_x = torch.cos(x), torch.sin(x)
        cos_y, sin_y = torch.cos(y), torch.sin(y)
        cos_z, sin_z = torch.cos(z), torch.sin(z)
        
        # Rotation matrix around X axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ], device=self.device, dtype=torch.float32)
        
        # Rotation matrix around Y axis
        Ry = torch.tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], device=self.device, dtype=torch.float32)
        
        # Rotation matrix around Z axis
        Rz = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        # Combined rotation matrix (ZYX order)
        return torch.matmul(torch.matmul(Rz, Ry), Rx)
    

class DoorGeometry:
    def __init__(self, door: Door, orientation: str, wall_geometry=None):
        """
        Initialize door geometry with Door object and orientation.
        
        Args:
            door: Door object from models
            orientation: Direction the door faces inwards to the room ("north", "south", "east", "west")
            wall_geometry: WallGeometry object that contains this door (optional)
        """
        self.door = door
        self.orientation = orientation
        self.width = door.width
        self.height = door.height
        self.door_type = door.door_type
        self.thickness = 0.05  # Default door thickness (5cm)
        
        # Calculate door normal vector based on orientation
        self.normal = self._calculate_inward_normal()
        
        # Calculate world coordinates if wall geometry is provided
        self.world_position = np.array([0, 0, 0])
        self.left_position = np.array([0, 0, 0])
        self.right_position = np.array([0, 0, 0])
        
        if wall_geometry:
            self._calculate_world_coordinates(wall_geometry)
    
    def _calculate_inward_normal(self) -> np.array:
        """Calculate the normal vector pointing inward to the room."""
        orientation_to_normal = {
            "north": np.array([0, -1, 0]),  # door faces north, normal points south (inward)
            "south": np.array([0, 1, 0]),   # door faces south, normal points north (inward)
            "east": np.array([-1, 0, 0]),   # door faces east, normal points west (inward)
            "west": np.array([1, 0, 0])     # door faces west, normal points east (inward)
        }
        return orientation_to_normal.get(self.orientation.lower(), np.array([0, 0, 0]))
    
    def _calculate_world_coordinates(self, wall_geometry):
        """Calculate world coordinates based on wall geometry and position on wall."""
        # Calculate position along the wall based on position_on_wall (0-1 ratio)
        wall_vector = wall_geometry.end_point - wall_geometry.start_point
        door_position_along_wall = wall_geometry.start_point + (wall_vector * self.door.position_on_wall)
        
        # Door center position (at floor level)
        self.world_position = np.array([
            door_position_along_wall[0],
            door_position_along_wall[1],
            self.height / 2  # Center height of door
        ])
        
        # Calculate left and right positions relative to wall direction
        wall_direction = wall_vector / np.linalg.norm(wall_vector)
        door_half_width = self.width / 2
        
        # Left and right positions along the wall
        self.left_position = door_position_along_wall - (wall_direction * door_half_width)
        self.right_position = door_position_along_wall + (wall_direction * door_half_width)
        
        # Add z-coordinate (floor level)
        self.left_position = np.append(self.left_position[:2], [0])
        self.right_position = np.append(self.right_position[:2], [0])

class WindowGeometry:
    def __init__(self, window: Window, orientation: str, wall_geometry=None):
        """
        Initialize window geometry with Window object and orientation.
        
        Args:
            window: Window object from models
            orientation: Direction the window faces inwards to the room ("north", "south", "east", "west")
            wall_geometry: WallGeometry object that contains this window (optional)
        """
        self.window = window
        self.orientation = orientation
        self.width = window.width
        self.height = window.height
        self.sill_height = window.sill_height
        self.window_type = window.window_type
        self.thickness = 0.1  # Default window thickness (10cm)
        
        # Calculate window normal vector based on orientation
        self.normal = self._calculate_inward_normal()
        
        # Calculate world coordinates if wall geometry is provided
        self.world_position = np.array([0, 0, 0])
        self.left_position = np.array([0, 0, 0])
        self.right_position = np.array([0, 0, 0])
        self.sill_position = np.array([0, 0, 0])
        self.top_position = np.array([0, 0, 0])
        
        if wall_geometry:
            self._calculate_world_coordinates(wall_geometry)
    
    def _calculate_inward_normal(self) -> np.array:
        """Calculate the normal vector pointing inward to the room."""
        orientation_to_normal = {
            "north": np.array([0, -1, 0]),  # window faces north, normal points south (inward)
            "south": np.array([0, 1, 0]),   # window faces south, normal points north (inward)
            "east": np.array([-1, 0, 0]),   # window faces east, normal points west (inward)
            "west": np.array([1, 0, 0])     # window faces west, normal points east (inward)
        }
        return orientation_to_normal.get(self.orientation.lower(), np.array([0, 0, 0]))
    
    def _calculate_world_coordinates(self, wall_geometry):
        """Calculate world coordinates based on wall geometry and position on wall."""
        # Calculate position along the wall based on position_on_wall (0-1 ratio)
        wall_vector = wall_geometry.end_point - wall_geometry.start_point
        window_position_along_wall = wall_geometry.start_point + (wall_vector * self.window.position_on_wall)
        
        # Window center position
        self.world_position = np.array([
            window_position_along_wall[0],
            window_position_along_wall[1],
            self.sill_height + (self.height / 2)  # Center height of window
        ])
        
        # Calculate left and right positions relative to wall direction
        wall_direction = wall_vector / np.linalg.norm(wall_vector)
        window_half_width = self.width / 2
        
        # Left and right positions along the wall
        self.left_position = window_position_along_wall - (wall_direction * window_half_width)
        self.right_position = window_position_along_wall + (wall_direction * window_half_width)
        
        # Sill and top positions
        self.sill_position = window_position_along_wall.copy()
        self.top_position = window_position_along_wall.copy()
        
        # Add z-coordinates
        self.left_position = np.append(self.left_position[:2], [self.sill_height])
        self.right_position = np.append(self.right_position[:2], [self.sill_height])
        self.sill_position = np.append(self.sill_position[:2], [self.sill_height])
        self.top_position = np.append(self.top_position[:2], [self.sill_height + self.height])

class WallGeometry:
    def __init__(self, wall: Wall, orientation: str):
        """
        Initialize wall geometry with Wall object and orientation.
        
        Args:
            wall: Wall object from models
            orientation: Direction the wall faces inwards to the room ("north", "south", "east", "west")
        """
        self.wall = wall
        self.orientation = orientation
        self.thickness = wall.thickness
        self.height = wall.height
        self.material = wall.material
        
        # Calculate wall dimensions and positions
        self.start_point = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
        self.end_point = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
        self.length = np.linalg.norm(self.end_point - self.start_point)
        
        # Calculate wall center and positions
        self.center = (self.start_point + self.end_point) / 2
        self.world_position = self.center.copy()
        self.world_position[2] = self.height / 2  # Center height of wall
        
        # Calculate left and right positions (relative to wall direction)
        wall_direction = (self.end_point - self.start_point) / self.length
        self.left_position = self.start_point.copy()
        self.right_position = self.end_point.copy()
        
        # Calculate inward and outward face positions
        self.normal = self._calculate_inward_normal()
        self.thickness_offset = self.normal * (self.thickness / 2)
        
        # Inner face (room side) and outer face positions
        self.inner_face_position = self.center + self.thickness_offset
        self.outer_face_position = self.center - self.thickness_offset
        
        # Set z-coordinates for face positions
        self.inner_face_position[2] = self.height / 2
        self.outer_face_position[2] = self.height / 2
        
        # Calculate wall corners (bottom and top)
        self.bottom_left = self.start_point.copy()
        self.bottom_right = self.end_point.copy()
        self.top_left = self.start_point.copy()
        self.top_right = self.end_point.copy()
        
        self.bottom_left[2] = 0
        self.bottom_right[2] = 0
        self.top_left[2] = self.height
        self.top_right[2] = self.height
        
    def _calculate_inward_normal(self) -> np.array:
        """Calculate the normal vector pointing inward to the room."""
        # Wall direction vector
        wall_vector = self.end_point - self.start_point
        wall_vector = wall_vector / np.linalg.norm(wall_vector)  # normalize
        
        # Calculate perpendicular vector (normal) - assuming walls are vertical
        # For a wall from start to end, the inward normal depends on room layout
        # We'll use the orientation to determine the correct inward direction
        orientation_to_normal = {
            "north": np.array([0, -1, 0]),  # wall faces north, normal points south (inward)
            "south": np.array([0, 1, 0]),   # wall faces south, normal points north (inward)
            "east": np.array([-1, 0, 0]),   # wall faces east, normal points west (inward)
            "west": np.array([1, 0, 0])     # wall faces west, normal points east (inward)
        }
        return orientation_to_normal.get(self.orientation.lower(), np.array([0, 0, 0]))

class RoomGeometry:
    def __init__(self, room: Room) -> None:
        """
        Initialize room geometry with Room object.
        
        Args:
            room: Room object from models
        """
        self.room = room
        self.center = np.array([room.position.x, room.position.y, room.position.z])
        self.width = room.dimensions.width
        self.length = room.dimensions.length
        self.height = room.dimensions.height
        self.ceiling_height = room.ceiling_height
        
        # Calculate wall thickness from walls if available
        if room.walls:
            self.wall_thickness = room.walls[0].thickness
        else:
            self.wall_thickness = 0.1  # default thickness

    def get_walls(self):
        """
        Get a list of WallGeometry objects with orientations pointing inward to the room.
        
        Returns:
            List of WallGeometry objects
        """
        wall_geometries = []
        
        for wall in self.room.walls:
            # Determine wall orientation based on wall position relative to room center
            orientation = self._determine_wall_orientation(wall)
            wall_geometry = WallGeometry(wall, orientation)
            wall_geometries.append(wall_geometry)
            
        return wall_geometries
    
    def get_doors(self):
        """
        Get a list of DoorGeometry objects with orientations pointing inward to the room.
        
        Returns:
            List of DoorGeometry objects
        """
        door_geometries = []
        wall_geometries = {wall.wall.id: wall for wall in self.get_walls()}
        
        for door in self.room.doors:
            # Find the wall this door belongs to
            wall_geometry = wall_geometries.get(door.wall_id)
            if wall_geometry:
                # Use the same orientation as the wall
                orientation = wall_geometry.orientation
                door_geometry = DoorGeometry(door, orientation, wall_geometry)
                door_geometries.append(door_geometry)
                
        return door_geometries
    
    def get_windows(self):
        """
        Get a list of WindowGeometry objects with orientations pointing inward to the room.
        
        Returns:
            List of WindowGeometry objects
        """
        window_geometries = []
        wall_geometries = {wall.wall.id: wall for wall in self.get_walls()}
        
        for window in self.room.windows:
            # Find the wall this window belongs to
            wall_geometry = wall_geometries.get(window.wall_id)
            if wall_geometry:
                # Use the same orientation as the wall
                orientation = wall_geometry.orientation
                window_geometry = WindowGeometry(window, orientation, wall_geometry)
                window_geometries.append(window_geometry)
                
        return window_geometries
    
    def _find_wall_by_id(self, wall_id: str) -> Wall:
        """Find a wall by its ID."""
        for wall in self.room.walls:
            if wall.id == wall_id:
                return wall
        return None
    
    def _determine_wall_orientation(self, wall: Wall) -> str:
        """
        Determine wall orientation relative to room center.
        This is a simplified implementation - in practice, you might need more 
        sophisticated logic based on your room layout conventions.
        """
        wall_start = np.array([wall.start_point.x, wall.start_point.y])
        wall_end = np.array([wall.end_point.x, wall.end_point.y])
        wall_center = (wall_start + wall_end) / 2
        room_center_2d = np.array([self.center[0], self.center[1]])
        
        # Vector from room center to wall center
        to_wall = wall_center - room_center_2d
        
        # Determine primary direction
        if abs(to_wall[1]) > abs(to_wall[0]):  # More vertical displacement
            if to_wall[1] > 0:
                return "north"  # Wall is north of room center
            else:
                return "south"  # Wall is south of room center
        else:  # More horizontal displacement
            if to_wall[0] > 0:
                return "east"   # Wall is east of room center
            else:
                return "west"   # Wall is west of room center



    