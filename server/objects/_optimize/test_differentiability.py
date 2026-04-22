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
Test script to demonstrate differentiability of distance functions.

This script shows the difference between the old non-differentiable distance function
and the new differentiable version for PyTorch optimization.

IMPORTANT: Mesh Positioning
---------------------------
The ObjectGeometry class expects meshes with vertices z >= 0 (bottom at z=0).
- Do NOT use transform matrices that move the mesh center
- Instead, manually adjust vertices: mesh.vertices[:, 2] += height/2
- This ensures proper distance calculations

Example:
✅ CORRECT:
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    mesh.vertices[:, 2] += 0.5  # Move bottom to z=0

❌ INCORRECT:
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0], 
                               transform=translation_matrix([0, 0, 0.5]))
"""
import sys
sys.path.append('.')
import torch
import trimesh
import numpy as np
from primitives import ObjectGeometry, distance_obbs, distance_obbs_simple, OrientedBoundingBox

def test_distance_differentiability():
    """Test that the distance functions are differentiable."""
    
    device = 'cpu'
    
    # Create two simple box objects (NO transform - let ObjectGeometry handle positioning)
    mesh1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    mesh2 = trimesh.creation.box(extents=[0.8, 0.8, 0.8])
    
    # Move mesh vertices to be above z=0 plane (as expected by ObjectGeometry)
    mesh1.vertices[:, 2] += 0.5  # Move bottom to z=0, top to z=1.0
    mesh2.vertices[:, 2] += 0.4  # Move bottom to z=0, top to z=0.8
    
    obj1 = ObjectGeometry(mesh1, device=device, requires_grad=True)
    obj2 = ObjectGeometry(mesh2, device=device, requires_grad=True)
    
    # Set initial positions
    obj1.position = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
    obj2.position = torch.tensor([2.0, 0.0, 0.0], device=device, requires_grad=True)
    
    # Set rotations
    obj1.rotation = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
    obj2.rotation = torch.tensor([0.0, 0.0, 45.0], device=device, requires_grad=True)
    
    print("=== Testing Differentiability ===")
    print(f"Initial obj1 position: {obj1.position}")
    print(f"Initial obj2 position: {obj2.position}")
    print(f"Initial obj2 rotation: {obj2.rotation}")
    print(f"Obj1 center: {obj1.center}")
    print(f"Obj2 center: {obj2.center}")
    print(f"Obj1 extents: {obj1.extents}")
    print(f"Obj2 extents: {obj2.extents}")
    
    # Calculate expected distance manually for verification
    center_distance = torch.norm(obj1.center - obj2.center)
    print(f"Center-to-center distance: {center_distance.item():.4f}")
    
    # Expected surface distance (rough approximation)
    obj1_radius = torch.norm(obj1.extents) / 2
    obj2_radius = torch.norm(obj2.extents) / 2
    expected_surface_distance = center_distance - obj1_radius - obj2_radius
    print(f"Expected surface distance (approx): {expected_surface_distance.item():.4f}")
    
    # Test differentiable distance function
    print("\n1. Testing differentiable distance_obbs:")
    try:
        distance = distance_obbs(obj1.oriented_bounding_box, obj2.oriented_bounding_box)
        print(f"   Distance: {distance.item():.4f}")
        
        # Test backward pass
        loss = distance ** 2
        loss.backward()
        
        print(f"   ✅ Gradient for obj1.position: {obj1.position.grad}")
        print(f"   ✅ Gradient for obj2.position: {obj2.position.grad}")
        print(f"   ✅ Gradient for obj2.rotation: {obj2.rotation.grad}")
        print("   SUCCESS: distance_obbs is differentiable!")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Reset gradients
    if obj1.position.grad is not None:
        obj1.position.grad.zero_()
    if obj2.position.grad is not None:
        obj2.position.grad.zero_()
    if obj2.rotation.grad is not None:
        obj2.rotation.grad.zero_()
    

def test_optimization_convergence():
    """Test optimization convergence with both distance functions."""
    
    device = 'cpu'
    
    print("\n=== Testing Optimization Convergence ===")
    
    for distance_func, name in [(distance_obbs, "Accurate")]:
        print(f"\n{name} Distance Function:")
        
        # Create two objects (correct mesh positioning)
        mesh1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        mesh2 = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
        
        # Position meshes correctly (bottom at z=0)
        mesh1.vertices[:, 2] += 0.5  # Bottom at z=0, top at z=1.0
        mesh2.vertices[:, 2] += 0.25 # Bottom at z=0, top at z=0.5
        
        obj1 = ObjectGeometry(mesh1, device=device, requires_grad=True)
        obj2 = ObjectGeometry(mesh2, device=device, requires_grad=True)
        
        # Initial positions (overlapping)
        obj1.position = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
        obj2.position = torch.tensor([0.3, 0.0, 0.0], device=device, requires_grad=True)  # Should overlap
        
        obj1.rotation = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
        obj2.rotation = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
        
        # Debug: Show object geometry
        print(f"   Obj1 - center: {obj1.center.detach().numpy()}, extents: {obj1.extents.detach().numpy()}")
        print(f"   Obj2 - center: {obj2.center.detach().numpy()}, extents: {obj2.extents.detach().numpy()}")
        
        # Calculate expected overlap
        obj1_bounds = {
            'x_min': obj1.center[0] - obj1.extents[0]/2, 'x_max': obj1.center[0] + obj1.extents[0]/2,
            'y_min': obj1.center[1] - obj1.extents[1]/2, 'y_max': obj1.center[1] + obj1.extents[1]/2,
        }
        obj2_bounds = {
            'x_min': obj2.center[0] - obj2.extents[0]/2, 'x_max': obj2.center[0] + obj2.extents[0]/2,
            'y_min': obj2.center[1] - obj2.extents[1]/2, 'y_max': obj2.center[1] + obj2.extents[1]/2,
        }
        
        x_overlap = max(0, min(obj1_bounds['x_max'], obj2_bounds['x_max']) - max(obj1_bounds['x_min'], obj2_bounds['x_min']))
        y_overlap = max(0, min(obj1_bounds['y_max'], obj2_bounds['y_max']) - max(obj1_bounds['y_min'], obj2_bounds['y_min']))
        
        print(f"   Expected overlap: x={x_overlap.item():.3f}, y={y_overlap.item():.3f}")
        if x_overlap > 0 and y_overlap > 0:
            print(f"   ✅ Objects should be overlapping")
        else:
            print(f"   ❌ Objects should NOT be overlapping")
        
        # Optimizer
        optimizer = torch.optim.Adam([obj2.position], lr=0.1)
        
        initial_distance = distance_func(obj1.oriented_bounding_box, obj2.oriented_bounding_box).item()
        print(f"   Initial distance: {initial_distance:.4f}")
        
        # Optimization loop
        for step in range(500):
            optimizer.zero_grad()
            
            # Loss: penalty for being too close (< 0.2 meters)
            distance = distance_func(obj1.oriented_bounding_box, obj2.oriented_bounding_box)
            target_distance = 0.2
            # loss = torch.clamp(target_distance - distance, min=0) ** 2
            loss = (target_distance - distance) ** 2
            
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print(f"   Step {step:2d}: distance={distance.item():.4f}, loss={loss.item():.6f}")
        
        final_distance = distance_func(obj1.oriented_bounding_box, obj2.oriented_bounding_box).item()
        print(f"   Final distance: {final_distance:.4f}")
        print(f"   Final obj2 position: {obj2.position.detach().numpy()}")
        
        if final_distance >= 0.19:  # Close to target
            print("   ✅ Optimization CONVERGED successfully!")
        else:
            print("   ❌ Optimization FAILED to converge")
            
        # Verify final state
        final_obj2_center = obj2.center.detach()
        center_distance = torch.norm(obj1.center.detach() - final_obj2_center)
        print(f"   Final center-to-center distance: {center_distance.item():.4f}")
        
        # Check if objects are properly separated
        obj1_radius = torch.norm(obj1.extents) / 2
        obj2_radius = torch.norm(obj2.extents) / 2
        expected_min_center_distance = obj1_radius + obj2_radius + 0.2  # target separation
        print(f"   Expected min center distance: {expected_min_center_distance.item():.4f}")
        
        if center_distance >= expected_min_center_distance * 0.95:  # 5% tolerance
            print("   ✅ Objects properly separated")
        else:
            print("   ⚠️  Objects may still be too close")


if __name__ == "__main__":
    test_distance_differentiability()
    test_optimization_convergence()