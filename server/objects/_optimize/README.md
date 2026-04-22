# PyTorch-Enabled Constraint-Based Furniture Optimization System

This module implements a comprehensive constraint-based optimization system for furniture placement in 3D rooms using **PyTorch tensors** for gradient-based optimization. The system allows you to define spatial relationships between objects, walls, doors, windows, and rooms, then optimize object positions and rotations using modern deep learning optimizers.

## Key Components

### 1. Primitives (`primitives.py`)
- **ObjectGeometry**: Represents 3D furniture objects with position, rotation, and oriented bounding boxes
- **WallGeometry**: Represents room walls with inward-facing normals
- **DoorGeometry**: Represents doors with opening directions
- **WindowGeometry**: Represents windows with orientations
- **RoomGeometry**: Represents rooms containing walls, doors, and windows
- **OrientedBoundingBox**: Handles 3D oriented bounding box calculations
- **distance_obbs()**: Calculates accurate distances between oriented bounding boxes

### 2. Constraints (`constraints.py`)
- **Loss Functions**: 15+ different constraint types with smooth, differentiable loss functions
- **Constraint Class**: Wrapper for individual constraints with parameters and weights
- **ConstraintSystem**: Manages collections of constraints and evaluates total loss

### 3. Example Usage (`example_usage.py`, `pytorch_optimization_example.py`)
- **`example_usage.py`**: Original NumPy-based examples and constraint demonstrations
- **`pytorch_optimization_example.py`**: Advanced PyTorch-based optimization with Adam optimizer, learning rate scheduling, and GPU support
- Demonstrates all constraint types with gradient-based optimization

## Constraint Types

### Category 1: Object-Wall Constraints
- `against(object, walls)`: Object should touch any wall in the list
- `center_aligned_wall(object, walls)`: Object center aligned with any wall
- `corner(object, wall1, wall2)`: Object positioned at corner of two walls
- `relative_rotation_wall(object, walls, direction)`: Object rotation relative to wall

### Category 2: Object-Door Constraints
- `avoid_collision(object, doors)`: Object shouldn't block door opening
- `close_to_door(object, doors)`: Object positioned near any door
- `center_aligned_door(object, doors)`: Object center aligned with any door
- `relative_rotation_door(object, doors, direction)`: Object rotation relative to door

### Category 3: Object-Window Constraints
- `center_aligned_window(object, windows)`: Object center aligned with any window
- `relative_rotation_window(object, windows, direction)`: Object rotation relative to window

### Category 4: Object-Room Constraints
- `center_aligned_room(object, room)`: Object center aligned with room
- `inside(object, room)`: Object must be inside room bounds

### Category 5: Object-Object Constraints
- `relative_planar_position(obj1, obj2, direction)`: Relative positioning (left/right/front/back)
- `center_aligned_objects(obj1, obj2)`: Objects center aligned
- `near(obj1, obj2, distance)`: Objects should be close (distance < threshold)
- `far(obj1, obj2, distance)`: Objects should be far (distance > threshold)
- `point_to(obj1, obj2)`: Object 1 should point toward object 2
- `relative_rotation_objects(obj1, obj2, direction)`: Relative rotation between objects

## Usage Examples

### PyTorch-based Optimization (Recommended)

```python
import torch
import torch.optim as optim
from constraints import ConstraintSystem
from primitives import ObjectGeometry, RoomGeometry
import trimesh

# Create furniture with PyTorch tensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'
table_mesh = trimesh.creation.box(extents=[1.2, 0.6, 0.8])
table_mesh.vertices[:, 2] += 0.4  # Position bottom at z=0 (height/2 = 0.8/2 = 0.4)
table = ObjectGeometry(table_mesh, device=device, requires_grad=True)

# Initialize position and rotation
table.position = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
table.rotation = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)

# Create optimizer
parameters = [table.position, table.rotation]
optimizer = optim.Adam(parameters, lr=0.01)

# Optimization loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Calculate constraint losses (automatically differentiable)
    total_loss = calculate_constraint_losses(table, room, walls, doors)
    
    total_loss.backward()
    optimizer.step()
```

### Traditional Constraint System

```python
from constraints import ConstraintSystem
from primitives import ObjectGeometry, RoomGeometry
import trimesh

# Create furniture
table_mesh = trimesh.creation.box(extents=[1.2, 0.6, 0.8])
table = ObjectGeometry(table_mesh)

# Create constraint system
constraints = ConstraintSystem()

# Add constraints
constraints.add_inside_room(table, room_geometry, weight=10.0)
constraints.add_against_wall(table, walls, weight=5.0)
constraints.add_near_objects(chair, table, distance=1.5, weight=4.0)

# Evaluate loss
total_loss = constraints.evaluate_total_loss()
```

## Key Features

1. **🚀 PyTorch Integration**: Full PyTorch tensor support with automatic differentiation
2. **⚡ GPU Acceleration**: CUDA support for faster optimization on compatible hardware
3. **🎯 Advanced Optimizers**: Support for Adam, SGD, and other PyTorch optimizers with learning rate scheduling
4. **📐 Oriented Bounding Boxes**: Accurate collision and distance calculations considering object rotations
5. **🔄 Differentiable Loss Functions**: Smooth, gradient-friendly functions for efficient optimization
6. **⚖️ Weighted Constraints**: Each constraint can have different importance weights
7. **🔧 Flexible Architecture**: Easy to add new constraint types
8. **🐛 Debug Support**: Individual constraint evaluation for troubleshooting
9. **🏠 Room Awareness**: Considers walls, doors, windows, and room boundaries
10. **📊 Optimization Monitoring**: Loss tracking and convergence analysis

## Technical Notes

- **Thickness Awareness**: All wall, door, and window constraints account for element thickness
  - Wall constraints use inner face positions (room side) for accurate positioning
  - Door collision avoidance considers door thickness and opening swing area
  - Window alignment uses inner surface positions
- **Fully Differentiable**: All distance and constraint functions are PyTorch-differentiable
  - **`distance_obbs()`**: Accurate OBB distance using smooth maximum (LogSumExp)
  - **`distance_obbs_simple()`**: Fast approximation using center-to-center distance
  - No early returns or hard conditionals that break gradient flow
  - Smooth weighting for edge cases (e.g., near-parallel axes)
- **Oriented Bounding Boxes**: All distance calculations use oriented bounding boxes for accuracy
- **Smooth Optimization**: All operations designed for gradient-based optimization
  - Rotation constraints use dot products for smooth, differentiable angle calculations
  - Loss functions use `torch.clamp()` instead of hard conditions
  - Smooth approximations replace discontinuous operations
- **Flexible Design**: The system supports both individual object constraints and object-pair constraints
- **Automatic Orientation**: Wall, door, and window orientations are automatically calculated based on room geometry
- **Performance Options**: Choose between accuracy and speed based on your needs
- **Mesh Requirements**: Objects must have vertices with z ≥ 0 (use `mesh.vertices[:, 2] += height/2` to position correctly)

## Distance Function Selection

The system provides two differentiable distance functions for oriented bounding boxes:

### `distance_obbs()` - Accurate (Recommended)
- **Accuracy**: High - implements full separating axis theorem
- **Performance**: Moderate - more computationally intensive 
- **Differentiability**: ✅ Fully differentiable using LogSumExp smooth maximum
- **Use cases**: Final optimization, high-precision requirements

### `distance_obbs_simple()` - Fast
- **Accuracy**: Approximate - uses center-to-center distance minus extents
- **Performance**: Fast - simple vector operations
- **Differentiability**: ✅ Fully differentiable 
- **Use cases**: Initial optimization, real-time applications, large scenes

### Usage Example
```python
# In constraints.py, set the global flag:
USE_SIMPLE_DISTANCE = True  # For faster optimization
USE_SIMPLE_DISTANCE = False # For accurate optimization (default)

# Or use directly:
distance1 = distance_obbs(obb1, obb2)        # Accurate
distance2 = distance_obbs_simple(obb1, obb2) # Fast
```

### Differentiability Verification
Run the test script to verify differentiability:
```bash
python test_differentiability.py
```

## Extending the System

To add new constraint types:

1. Implement a loss function in `constraints.py`
2. Add the function to the `constraints_to_losses` mapping
3. Update the `Constraint.evaluate()` method to handle the new constraint
4. Optionally add a convenience method to `ConstraintSystem`

The loss functions should:
- Return 0 when the constraint is satisfied
- Return positive values when violated
- Be smooth and differentiable
- Handle edge cases gracefully 