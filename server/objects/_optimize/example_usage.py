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
Example usage of the constraint system for furniture optimization.

This example demonstrates how to use the constraint system to optimize
furniture placement in a room with walls, doors, and windows.
"""

import numpy as np
import trimesh
from constraints import ConstraintSystem, Constraint
from primitives import ObjectGeometry, WallGeometry, DoorGeometry, WindowGeometry, RoomGeometry
from models import Room, Wall, Door, Window, Point3D, Dimensions3D

def create_example_room():
    """Create an example room with walls, doors, and windows."""
    
    # Create room dimensions
    room_dims = Dimensions3D(width=5.0, length=4.0, height=3.0)
    room_pos = Point3D(x=0.0, y=0.0, z=0.0)
    
    # Create walls
    wall1 = Wall(
        id="wall1",
        start_point=Point3D(x=-2.5, y=-2.0, z=0.0),
        end_point=Point3D(x=2.5, y=-2.0, z=0.0),
        thickness=0.2,
        height=3.0,
        material="drywall"
    )
    
    wall2 = Wall(
        id="wall2", 
        start_point=Point3D(x=2.5, y=-2.0, z=0.0),
        end_point=Point3D(x=2.5, y=2.0, z=0.0),
        thickness=0.2,
        height=3.0,
        material="drywall"
    )
    
    wall3 = Wall(
        id="wall3",
        start_point=Point3D(x=2.5, y=2.0, z=0.0),
        end_point=Point3D(x=-2.5, y=2.0, z=0.0),
        thickness=0.2,
        height=3.0,
        material="drywall"
    )
    
    wall4 = Wall(
        id="wall4",
        start_point=Point3D(x=-2.5, y=2.0, z=0.0),
        end_point=Point3D(x=-2.5, y=-2.0, z=0.0),
        thickness=0.2,
        height=3.0,
        material="drywall"
    )
    
    # Create door
    door1 = Door(
        id="door1",
        wall_id="wall1",
        position_on_wall=0.3,  # 30% along the wall
        width=0.8,
        height=2.0,
        door_type="hinged"
    )
    
    # Create window
    window1 = Window(
        id="window1",
        wall_id="wall3",
        position_on_wall=0.5,  # 50% along the wall
        width=1.2,
        height=1.0,
        sill_height=1.0,
        window_type="casement"
    )
    
    # Create room
    room = Room(
        id="room1",
        position=room_pos,
        dimensions=room_dims,
        ceiling_height=3.0,
        walls=[wall1, wall2, wall3, wall4],
        doors=[door1],
        windows=[window1]
    )
    
    return room


def create_example_furniture():
    """Create example furniture objects."""
    
    # Create a simple box mesh for a table
    table_mesh = trimesh.creation.box(extents=[1.2, 0.6, 0.8])
    table_mesh.vertices[:, 2] = np.maximum(table_mesh.vertices[:, 2], 0)  # Ensure it's above ground
    table = ObjectGeometry(table_mesh)
    
    # Create a simple box mesh for a chair
    chair_mesh = trimesh.creation.box(extents=[0.5, 0.5, 1.0])
    chair_mesh.vertices[:, 2] = np.maximum(chair_mesh.vertices[:, 2], 0)  # Ensure it's above ground
    chair = ObjectGeometry(chair_mesh)
    
    # Create a simple box mesh for a bookshelf
    bookshelf_mesh = trimesh.creation.box(extents=[0.8, 0.3, 2.0])
    bookshelf_mesh.vertices[:, 2] = np.maximum(bookshelf_mesh.vertices[:, 2], 0)  # Ensure it's above ground
    bookshelf = ObjectGeometry(bookshelf_mesh)
    
    return table, chair, bookshelf


def optimize_furniture_placement():
    """Example of optimizing furniture placement using constraints."""
    
    # Create room and furniture
    room = create_example_room()
    table, chair, bookshelf = create_example_furniture()
    
    # Create geometry objects
    room_geometry = RoomGeometry(room)
    walls = room_geometry.get_walls()
    doors = room_geometry.get_doors()
    windows = room_geometry.get_windows()
    
    # Create constraint system
    constraint_system = ConstraintSystem()
    
    # Add constraints
    
    # 1. All furniture should be inside the room
    constraint_system.add_inside_room(table, room_geometry, weight=10.0)
    constraint_system.add_inside_room(chair, room_geometry, weight=10.0)
    constraint_system.add_inside_room(bookshelf, room_geometry, weight=10.0)
    
    # 2. Bookshelf should be against a wall
    constraint_system.add_against_wall(bookshelf, walls, weight=5.0)
    
    # 3. Table should be center-aligned with the room
    constraint_system.add_center_aligned_room(table, [room_geometry], weight=3.0)
    
    # 4. Chair should be near the table
    constraint_system.add_near_objects(chair, table, distance=1.5, weight=4.0)
    
    # 5. Chair should point towards the table
    constraint_system.add_point_to(chair, table, weight=2.0)
    
    # 6. Furniture should not block the door
    if doors:
        constraint_system.add_constraint(
            Constraint("avoid_collision", [table, chair, bookshelf], doors, weight=8.0)
        )
    
    # 7. Bookshelf should be positioned to the left of the table
    constraint_system.add_relative_position(bookshelf, table, "left", weight=1.0)
    
    # 8. Objects should not be too close to each other (avoid overlap)
    constraint_system.add_far_objects(table, bookshelf, distance=0.5, weight=6.0)
    constraint_system.add_far_objects(chair, bookshelf, distance=0.3, weight=6.0)
    
    print(f"Created constraint system with {constraint_system.get_constraint_count()} constraints")
    
    # Initial positions (random)
    table.position = np.array([0.5, 0.5, 0.0])
    chair.position = np.array([-0.5, -0.5, 0.0])
    bookshelf.position = np.array([1.0, 1.0, 0.0])
    
    # Initial rotations
    table.rotation = np.array([0.0, 0.0, 0.0])
    chair.rotation = np.array([0.0, 0.0, 45.0])
    bookshelf.rotation = np.array([0.0, 0.0, 0.0])
    
    # Evaluate initial loss
    initial_loss = constraint_system.evaluate_total_loss()
    print(f"Initial total loss: {initial_loss:.4f}")
    
    # Show individual constraint losses for debugging
    individual_losses = constraint_system.evaluate_individual_losses()
    print("\nIndividual constraint losses:")
    for constraint_id, loss in individual_losses.items():
        print(f"  {constraint_id}: {loss:.4f}")
    
    # Simple optimization loop (gradient-free)
    print("\nPerforming simple optimization...")
    
    learning_rate = 0.01
    max_iterations = 100
    
    for iteration in range(max_iterations):
        current_loss = constraint_system.evaluate_total_loss()
        
        # Simple gradient approximation
        objects = [table, chair, bookshelf]
        
        for obj in objects:
            # Store original position and rotation
            orig_pos = obj.position.copy()
            orig_rot = obj.rotation.copy()
            
            # Try small movements in each direction
            for i in range(3):  # x, y, z
                # Positive direction
                obj.position[i] += learning_rate
                loss_pos = constraint_system.evaluate_total_loss()
                
                # Negative direction
                obj.position[i] -= 2 * learning_rate
                loss_neg = constraint_system.evaluate_total_loss()
                
                # Calculate gradient
                gradient = (loss_pos - loss_neg) / (2 * learning_rate)
                
                # Update position
                obj.position[i] = orig_pos[i] - learning_rate * gradient
                
            # Try small rotations
            for i in range(3):  # rx, ry, rz
                # Positive direction
                obj.rotation[i] += learning_rate * 10  # Larger step for rotation
                loss_pos = constraint_system.evaluate_total_loss()
                
                # Negative direction
                obj.rotation[i] -= 2 * learning_rate * 10
                loss_neg = constraint_system.evaluate_total_loss()
                
                # Calculate gradient
                gradient = (loss_pos - loss_neg) / (2 * learning_rate * 10)
                
                # Update rotation
                obj.rotation[i] = orig_rot[i] - learning_rate * 10 * gradient
        
        # Print progress
        if iteration % 20 == 0:
            new_loss = constraint_system.evaluate_total_loss()
            print(f"Iteration {iteration}: Loss = {new_loss:.4f}")
    
    # Final evaluation
    final_loss = constraint_system.evaluate_total_loss()
    print(f"\nFinal total loss: {final_loss:.4f}")
    print(f"Improvement: {initial_loss - final_loss:.4f}")
    
    # Print final positions
    print("\nFinal object positions:")
    print(f"Table: position={table.position}, rotation={table.rotation}")
    print(f"Chair: position={chair.position}, rotation={chair.rotation}")
    print(f"Bookshelf: position={bookshelf.position}, rotation={bookshelf.rotation}")
    
    # Show final individual constraint losses
    final_individual_losses = constraint_system.evaluate_individual_losses()
    print("\nFinal individual constraint losses:")
    for constraint_id, loss in final_individual_losses.items():
        print(f"  {constraint_id}: {loss:.4f}")
    
    return constraint_system, [table, chair, bookshelf]


def demonstrate_constraint_types():
    """Demonstrate different types of constraints."""
    
    print("=== Constraint Types Demonstration ===")
    
    # Create simple test objects
    mesh1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    mesh2 = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    
    obj1 = ObjectGeometry(mesh1)
    obj2 = ObjectGeometry(mesh2)
    
    # Position objects
    obj1.position = np.array([0.0, 0.0, 0.0])
    obj2.position = np.array([2.0, 0.0, 0.0])
    
    # Test individual constraint types
    constraint_system = ConstraintSystem()
    
    # Distance constraints
    constraint_system.add_near_objects(obj1, obj2, distance=1.0, weight=1.0)
    near_loss = constraint_system.evaluate_total_loss()
    print(f"Near constraint (distance=1.0, actual≈2.0): {near_loss:.4f}")
    
    constraint_system.clear_constraints()
    constraint_system.add_far_objects(obj1, obj2, distance=3.0, weight=1.0)
    far_loss = constraint_system.evaluate_total_loss()
    print(f"Far constraint (distance=3.0, actual≈2.0): {far_loss:.4f}")
    
    # Alignment constraint
    constraint_system.clear_constraints()
    constraint_system.add_constraint(
        Constraint("center_aligned_objects", obj1, obj2, weight=1.0)
    )
    alignment_loss = constraint_system.evaluate_total_loss()
    print(f"Center alignment constraint: {alignment_loss:.4f}")
    
    # Point to constraint
    constraint_system.clear_constraints()
    obj1.rotation = np.array([0.0, 0.0, 90.0])  # Point in +y direction
    constraint_system.add_point_to(obj1, obj2, weight=1.0)
    point_to_loss = constraint_system.evaluate_total_loss()
    print(f"Point to constraint (obj1 pointing +y, obj2 at +x): {point_to_loss:.4f}")
    
    # Relative position constraint
    constraint_system.clear_constraints()
    constraint_system.add_relative_position(obj2, obj1, "right", weight=1.0)
    relative_pos_loss = constraint_system.evaluate_total_loss()
    print(f"Relative position constraint (obj2 right of obj1): {relative_pos_loss:.4f}")


if __name__ == "__main__":
    print("=== Furniture Optimization Example ===")
    optimize_furniture_placement()
    
    print("\n" + "="*50 + "\n")
    
    demonstrate_constraint_types() 