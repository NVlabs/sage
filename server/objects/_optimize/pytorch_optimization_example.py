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
PyTorch-based furniture optimization example.

This example demonstrates how to use PyTorch optimizers with the constraint system
to optimize furniture placement in a room with walls, doors, and windows.
"""

import torch
import torch.optim as optim
import trimesh
import numpy as np
from constraints import ConstraintSystem, Constraint, to_tensor
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

def create_pytorch_furniture(device='cpu'):
    """Create example furniture objects with PyTorch tensors."""
    
    # Create a simple box mesh for a table
    table_mesh = trimesh.creation.box(extents=[1.2, 0.6, 0.8])
    table_mesh.vertices[:, 2] += 0.4  # Position bottom at z=0 (height/2 = 0.8/2 = 0.4)
    table = ObjectGeometry(table_mesh, device=device, requires_grad=True)
    
    # Create a simple box mesh for a chair
    chair_mesh = trimesh.creation.box(extents=[0.5, 0.5, 1.0])
    chair_mesh.vertices[:, 2] += 0.5  # Position bottom at z=0 (height/2 = 1.0/2 = 0.5)
    chair = ObjectGeometry(chair_mesh, device=device, requires_grad=True)
    
    # Create a simple box mesh for a bookshelf
    bookshelf_mesh = trimesh.creation.box(extents=[0.8, 0.3, 2.0])
    bookshelf_mesh.vertices[:, 2] += 1.0  # Position bottom at z=0 (height/2 = 2.0/2 = 1.0)
    bookshelf = ObjectGeometry(bookshelf_mesh, device=device, requires_grad=True)
    
    return table, chair, bookshelf

def pytorch_furniture_optimization(device='cpu', num_epochs=200):
    """Example of PyTorch-based furniture optimization."""
    
    print(f"Using device: {device}")
    
    # Create room and furniture
    room = create_example_room()
    table, chair, bookshelf = create_pytorch_furniture(device)
    
    # Create geometry objects
    room_geometry = RoomGeometry(room)
    walls = room_geometry.get_walls()
    doors = room_geometry.get_doors()
    windows = room_geometry.get_windows()
    
    # Initialize positions (random but reasonable)
    with torch.no_grad():
        table.position.copy_(torch.tensor([0.5, 0.5, 0.0], device=device))
        chair.position.copy_(torch.tensor([-0.5, -0.5, 0.0], device=device))
        bookshelf.position.copy_(torch.tensor([1.0, 1.0, 0.0], device=device))
        
        # Initialize rotations
        table.rotation.copy_(torch.tensor([0.0, 0.0, 0.0], device=device))
        chair.rotation.copy_(torch.tensor([0.0, 0.0, 45.0], device=device))
        bookshelf.rotation.copy_(torch.tensor([0.0, 0.0, 0.0], device=device))
    
    # Collect all parameters for optimization
    furniture_objects = [table, chair, bookshelf]
    parameters = []
    for obj in furniture_objects:
        parameters.append(obj.position)
        parameters.append(obj.rotation)
    
    # Create optimizer
    optimizer = optim.Adam(parameters, lr=0.01)
    
    # Track optimization history
    loss_history = []
    
    print("Starting PyTorch optimization...")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Calculate total loss
        total_loss = torch.tensor(0.0, device=device)
        
        # 1. All furniture should be inside the room bounds (simple version)
        for obj in furniture_objects:
            # Room bounds penalty
            room_center = to_tensor([0.0, 0.0, 0.0], device)
            room_half_size = to_tensor([2.5, 2.0, 3.0], device)  # room half dimensions
            
            # Calculate how much object extends outside room bounds
            obj_center = obj.center
            obj_half_extents = obj.extents / 2
            
            # Penalty for going outside room bounds
            for i in range(3):
                outside_penalty = torch.clamp(
                    torch.abs(obj_center[i] - room_center[i]) + obj_half_extents[i] - room_half_size[i], 
                    min=0
                )
                total_loss += 10.0 * outside_penalty ** 2
        
        # 2. Objects should not overlap (distance constraint)
        for i, obj1 in enumerate(furniture_objects):
            for j, obj2 in enumerate(furniture_objects):
                if i < j:  # Avoid double counting
                    # Minimum distance constraint
                    min_distance = 0.3  # 30cm minimum separation
                    distance = torch.norm(obj1.center - obj2.center)
                    overlap_penalty = torch.clamp(min_distance - distance, min=0)
                    total_loss += 8.0 * overlap_penalty ** 2
        
        # 3. Bookshelf should be against a wall (simplified)
        bookshelf_center = bookshelf.center
        # Distance to nearest wall (simplified - using room boundaries)
        wall_distances = [
            torch.abs(bookshelf_center[0] + 2.5),  # left wall
            torch.abs(bookshelf_center[0] - 2.5),  # right wall
            torch.abs(bookshelf_center[1] + 2.0),  # front wall
            torch.abs(bookshelf_center[1] - 2.0),  # back wall
        ]
        min_wall_distance = torch.min(torch.stack(wall_distances))
        total_loss += 5.0 * min_wall_distance ** 2
        
        # 4. Chair should be near table
        chair_table_distance = torch.norm(chair.center - table.center)
        desired_distance = 1.5
        distance_penalty = torch.clamp(chair_table_distance - desired_distance, min=0)
        total_loss += 4.0 * distance_penalty ** 2
        
        # 5. Table should be roughly centered
        table_center_penalty = torch.norm(table.center[:2])  # distance from room center (x,y)
        total_loss += 2.0 * table_center_penalty ** 2
        
        # 6. Chair should point towards table
        chair_direction = chair.direction
        to_table = table.center - chair.center
        to_table_normalized = to_table / torch.norm(to_table)
        alignment = torch.dot(chair_direction, to_table_normalized)
        pointing_penalty = 1.0 - alignment  # 0 when aligned, 2 when opposite
        total_loss += 3.0 * pointing_penalty ** 2
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        
        # Optimization step
        optimizer.step()
        
        # Record loss
        loss_history.append(total_loss.item())
        
        # Print progress
        if epoch % 40 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {total_loss.item():.4f}")
    
    print(f"\nOptimization completed!")
    print(f"Initial loss: {loss_history[0]:.4f}")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Improvement: {loss_history[0] - loss_history[-1]:.4f}")
    
    # Print final positions and rotations
    print("\nFinal object states:")
    print(f"Table: position={table.position.detach().cpu().numpy()}, rotation={table.rotation.detach().cpu().numpy()}")
    print(f"Chair: position={chair.position.detach().cpu().numpy()}, rotation={chair.rotation.detach().cpu().numpy()}")
    print(f"Bookshelf: position={bookshelf.position.detach().cpu().numpy()}, rotation={bookshelf.rotation.detach().cpu().numpy()}")
    
    return furniture_objects, loss_history

def advanced_constraint_optimization(device='cpu', num_epochs=300):
    """Advanced optimization using the constraint system with PyTorch."""
    
    print(f"\n=== Advanced Constraint Optimization (device: {device}) ===")
    
    # Create room and furniture
    room = create_example_room()
    table, chair, bookshelf = create_pytorch_furniture(device)
    
    # Initialize positions
    with torch.no_grad():
        table.position.copy_(torch.tensor([0.3, 0.2, 0.0], device=device))
        chair.position.copy_(torch.tensor([-0.8, -0.3, 0.0], device=device))
        bookshelf.position.copy_(torch.tensor([1.2, 0.8, 0.0], device=device))
        
        table.rotation.copy_(torch.tensor([0.0, 0.0, 15.0], device=device))
        chair.rotation.copy_(torch.tensor([0.0, 0.0, 60.0], device=device))
        bookshelf.rotation.copy_(torch.tensor([0.0, 0.0, 10.0], device=device))
    
    # Create constraint system
    constraint_system = ConstraintSystem()
    
    # Create room geometry
    room_geometry = RoomGeometry(room)
    walls = room_geometry.get_walls()
    
    # Collect parameters
    furniture_objects = [table, chair, bookshelf]
    parameters = []
    for obj in furniture_objects:
        parameters.append(obj.position)
        parameters.append(obj.rotation)
    
    # Create optimizer with different learning rates for position and rotation
    pos_params = [obj.position for obj in furniture_objects]
    rot_params = [obj.rotation for obj in furniture_objects]
    
    optimizer = optim.Adam([
        {'params': pos_params, 'lr': 0.02},
        {'params': rot_params, 'lr': 0.01}
    ])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.8)
    
    loss_history = []
    
    print("Starting advanced constraint optimization...")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        total_loss = torch.tensor(0.0, device=device)
        
        # Room boundary constraints
        room_bounds = to_tensor([2.3, 1.8, 2.8], device)  # slightly smaller than room
        for obj in furniture_objects:
            obj_center = obj.center
            obj_half_extents = obj.extents / 2
            
            for i in range(3):
                outside = torch.clamp(
                    torch.abs(obj_center[i]) + obj_half_extents[i] - room_bounds[i],
                    min=0
                )
                total_loss += 15.0 * outside ** 2
        
        # Non-overlapping constraints using oriented bounding boxes
        for i, obj1 in enumerate(furniture_objects):
            for j, obj2 in enumerate(furniture_objects):
                if i < j:
                    # Use the PyTorch-enabled distance function
                    from primitives import distance_obbs
                    distance = distance_obbs(obj1.oriented_bounding_box, obj2.oriented_bounding_box)
                    overlap_penalty = torch.clamp(-distance, min=0)  # Penalty only if overlapping
                    total_loss += 12.0 * overlap_penalty ** 2
        
        # Individual object constraints
        
        # 1. Bookshelf against wall
        bookshelf_center = bookshelf.center
        wall_distances = torch.stack([
            torch.abs(bookshelf_center[0] + 2.3),  # left wall
            torch.abs(bookshelf_center[0] - 2.3),  # right wall  
            torch.abs(bookshelf_center[1] + 1.8),  # front wall
            torch.abs(bookshelf_center[1] - 1.8),  # back wall
        ])
        min_wall_distance = torch.min(wall_distances)
        total_loss += 8.0 * min_wall_distance ** 2
        
        # 2. Chair near table
        chair_table_distance = torch.norm(chair.center - table.center)
        target_distance = 1.2
        near_penalty = (chair_table_distance - target_distance) ** 2
        total_loss += 6.0 * near_penalty
        
        # 3. Table center alignment
        table_center_penalty = torch.norm(table.center[:2] - to_tensor([0.0, 0.0], device))
        total_loss += 3.0 * table_center_penalty ** 2
        
        # 4. Chair pointing to table
        chair_to_table = table.center - chair.center
        chair_to_table_norm = chair_to_table / torch.norm(chair_to_table)
        chair_direction = chair.direction
        alignment = torch.dot(chair_direction, chair_to_table_norm)
        pointing_loss = (1.0 - alignment) ** 2
        total_loss += 4.0 * pointing_loss
        
        # 5. Furniture stability (prevent floating)
        for obj in furniture_objects:
            height_penalty = torch.clamp(obj.position[2], min=0) ** 2
            total_loss += 20.0 * height_penalty
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=2.0)
        
        optimizer.step()
        scheduler.step(total_loss)
        
        loss_history.append(total_loss.item())
        
        if epoch % 60 == 0 or epoch == num_epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}: Loss = {total_loss.item():.4f}, LR = {current_lr:.6f}")
    
    print(f"\nAdvanced optimization completed!")
    print(f"Final loss: {loss_history[-1]:.4f}")
    
    # Print final configurations
    print("\nFinal optimized furniture layout:")
    for name, obj in zip(['Table', 'Chair', 'Bookshelf'], furniture_objects):
        pos = obj.position.detach().cpu().numpy()
        rot = obj.rotation.detach().cpu().numpy()
        print(f"{name:9s}: pos=({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}), "
              f"rot=({rot[0]:6.1f}, {rot[1]:6.1f}, {rot[2]:6.1f})")
    
    return furniture_objects, loss_history

if __name__ == "__main__":
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch device: {device}")
    
    # Run basic optimization
    print("=== Basic PyTorch Optimization ===")
    furniture1, history1 = pytorch_furniture_optimization(device=device, num_epochs=150)
    
    # Run advanced optimization
    furniture2, history2 = advanced_constraint_optimization(device=device, num_epochs=250)
    
    print("\n=== Optimization Summary ===")
    print(f"Basic optimization: {history1[0]:.4f} -> {history1[-1]:.4f}")
    print(f"Advanced optimization: {history2[0]:.4f} -> {history2[-1]:.4f}") 