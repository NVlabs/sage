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
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Point3D:
    """Represents a 3D coordinate point."""
    x: float
    y: float
    z: float

@dataclass
class Euler:
    """Represents a 3D rotation in Euler angles (x, y, z) in degrees."""
    x: float
    y: float
    z: float

@dataclass
class Dimensions:
    """Represents 3D dimensions."""
    width: float
    length: float
    height: float

@dataclass
class Wall:
    """Represents a wall in the room."""
    id: str
    start_point: Point3D
    end_point: Point3D
    height: float
    thickness: float = 0.1
    material: str = "drywall"

@dataclass
class Window:
    """Represents a window on a wall."""
    id: str
    wall_id: str
    position_on_wall: float  # 0-1, position along the wall
    width: float
    height: float
    sill_height: float  # height from floor to window sill
    window_type: str = "standard"

@dataclass
class Door:
    """Represents a door on a wall."""
    id: str
    wall_id: str
    position_on_wall: float  # 0-1, position along the wall
    width: float
    height: float
    door_type: str = "standard"
    opens_inward: bool = True
    opening: bool = False # if opening is true, then it is a permanent opening without any actual door at the space
    door_material: str = "wood"

@dataclass
class Object:
    """Represents an object/furniture item in a room."""
    id: str # unique id for the object
    room_id: str # id of the room the object is in
    type: str # type of the object
    description: str # description of the object
    position: Point3D # position of the object in the room
    rotation: Euler # rotation of the object in the room
    dimensions: Dimensions # dimensions of the object
    source: str # "objaverse", "generation", etc.
    source_id: str # id of the object in the source
    place_id: str # id of the place the object is in; could be a wall (wall_id), a floor (room_id), or another object (object_id)
    place_guidance: str = "Standard placement for the object" # guidance on where to place the object in the room
    mass: float = 1.0 # mass of the object in kg

@dataclass
class Room:
    """Represents a room in the layout."""
    id: str
    room_type: str
    position: Point3D
    dimensions: Dimensions
    walls: List[Wall]
    doors: List[Door]
    objects: List[Object]
    windows: List[Window]
    floor_material: str = "hardwood"
    ceiling_height: float = 2.7  # meters

@dataclass
class FloorPlan:
    """Represents the complete floor plan layout."""
    id: str
    rooms: List[Room]
    total_area: float
    building_style: str
    description: str
    created_from_text: str 