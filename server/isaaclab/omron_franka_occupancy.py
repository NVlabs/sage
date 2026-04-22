import numpy as np


def occupancy_map(forward, side, yaw, offset=0.0):
    """
    It will return a function that can calculate the x-y occupancy for the robot.

    x-y occupancy means the area that the robot takes up in the x-y plane.

    The robot is defined in /home/hongchix/codes/curobo/src/curobo/content/assets/robot/omron_franka/composed_robot.urdf

    We only need to consider the robot's base.

    We assume that the robot is spawn at the origin without any rotation.

    You need to get detailed information from the urdf file.

    For example, the base's rectangle location, size;
    
    the forward / side translation direction.

    the rotation center when yaw is not 0.

    Finally the return is a function, which will take 2d x-y point array (N,2) as input

    and return a boolean array (N,) indicating whether the point is in the robot's occupancy.
    """
    
    # Based on URDF analysis:
    # - base_wheeled_base has collision box: size="0.7 0.5 0.38" at xyz="0 0 0"
    # - base_joint_mobile_forward: axis="0 1 0" (Y-axis), origin xyz="-0.21 0 0"
    # - base_joint_mobile_side: axis="1 0 0" (X-axis) 
    # - base_joint_mobile_yaw: axis="0 0 1" (Z-axis)
    # - base_jointfix_3_8: origin xyz="0.21 0 0" (offset compensation)
    # - base_jointfix_2_0: origin xyz="-0.20 0 0.192" (wheeled_base offset)
    
    # Robot base dimensions (from collision box)
    base_width = 0.7 + 2 * offset   # X dimension
    base_length = 0.5 + 2 * offset  # Y dimension
    
    # Transform chain analysis:
    # 1. Forward movement: moves in Y direction with -0.21m initial offset
    # 2. Side movement: moves in X direction 
    # 3. Yaw rotation: rotates around Z axis
    # 4. Fixed offset: +0.21m in X to base_base
    # 5. Wheeled base offset: -0.20m in X from base_base
    
    # Net offset from origin to wheeled_base center:
    # X: -0.21 (forward joint) + 0.21 (fixed joint) - 0.20 (wheeled_base) = -0.20m
    # Y: 0 (no net Y offset in fixed joints)
    base_center_offset_x = -0.20
    base_center_offset_y = 0.0
    
    def check_occupancy(points):
        """
        Check if 2D points (N,2) are inside the robot's occupancy footprint.
        
        Args:
            points: numpy array of shape (N, 2) with x,y coordinates
            
        Returns:
            boolean array of shape (N,) indicating occupancy
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        # Calculate the robot base center position after transformations
        # Forward movement is in Y direction
        robot_center_x = side + base_center_offset_x
        robot_center_y = forward + base_center_offset_y
        
        # Transform points to robot's local coordinate system
        # First translate to robot center
        local_points = points - np.array([robot_center_x, robot_center_y])
        
        # Then rotate by negative yaw to get points in robot's frame
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw],
                                   [sin_yaw, cos_yaw]])
        
        rotated_points = local_points @ rotation_matrix.T
        
        # Check if points are inside the rectangular base footprint
        half_width = base_width / 2.0
        half_length = base_length / 2.0
        
        # Add small tolerance for floating-point precision
        tolerance = 1e-10
        inside_x = np.abs(rotated_points[:, 0]) <= (half_width + tolerance)
        inside_y = np.abs(rotated_points[:, 1]) <= (half_length + tolerance)
        
        return inside_x & inside_y
    
    return check_occupancy


def test_occupancy_function():
    """
    Simple test function to verify the occupancy implementation.
    """
    print("Testing occupancy function...")
    
    # Test 1: Robot at origin, no rotation
    occupancy_fn = occupancy_map(forward=0.0, side=0.0, yaw=0.0)
    
    # Points that should be inside (robot center is at [-0.2, 0.0])
    inside_points = np.array([
        [-0.2, 0.0],      # Center of robot
        [-0.2 - 0.3, 0.0], # Left edge
        [-0.2 + 0.3, 0.0], # Right edge  
        [-0.2, -0.2],     # Front edge
        [-0.2, 0.2]       # Back edge
    ])
    
    # Points that should be outside
    outside_points = np.array([
        [-0.2 - 0.4, 0.0], # Too far left
        [-0.2 + 0.4, 0.0], # Too far right
        [-0.2, -0.3],      # Too far forward
        [-0.2, 0.3],       # Too far back
        [1.0, 1.0]         # Far away
    ])
    
    inside_results = occupancy_fn(inside_points)
    outside_results = occupancy_fn(outside_points)
    
    print(f"Inside points results: {inside_results}")
    print(f"Outside points results: {outside_results}")
    print(f"All inside points detected: {np.all(inside_results)}")
    print(f"All outside points rejected: {np.all(~outside_results)}")
    
    # Test 2: Robot with translation
    occupancy_fn2 = occupancy_map(forward=1.0, side=0.5, yaw=0.0)
    
    # Robot center should now be at [0.5 - 0.2, 1.0] = [0.3, 1.0]
    test_point = np.array([[0.3, 1.0]])
    result = occupancy_fn2(test_point)
    print(f"Translated robot center occupancy: {result[0]}")
    
    # Test 3: Robot with rotation
    occupancy_fn3 = occupancy_map(forward=0.0, side=0.0, yaw=np.pi/2)
    
    # After 90-degree rotation, the robot's local x-axis points in world y direction
    rotated_test_points = np.array([
        [-0.2, 0.0],      # Still at robot center
        [-0.2, 0.3],      # Should be inside (was robot's +x direction)
        [-0.2, -0.3]      # Should be inside (was robot's -x direction)
    ])
    
    rotated_results = occupancy_fn3(rotated_test_points)
    print(f"Rotated robot results: {rotated_results}")
    
    print("Testing complete!")

def support_point(forward, side, yaw):
    """
    Given the joint positions of forward, side, yaw, it will return the x-y coordinates of the link of base_support

    return: [x, y]
    """
    
    # Transformation chain analysis to base_support:
    # 1. world -> base_link0_5: origin="0 0 0"
    # 2. base_link0_5 -> base_link0_6 (forward joint): origin="-0.21 0 0", axis="0 1 0"
    # 3. base_link0_6 -> base_link0_7 (side joint): origin="0 0 0", axis="1 0 0" 
    # 4. base_link0_7 -> base_link0_8 (yaw joint): origin="0 0 0", axis="0 0 1"
    # 5. base_link0_8 -> base_base: origin="0.21 0 0"
    # 6. base_base -> base_fixed_support: origin="-0.05 0 0.5"
    # 7. base_fixed_support -> base_link0_3 (height joint): origin="0.05 0 0.2"
    # 8. base_link0_3 -> base_support: origin="0 0 0"
    
    # Calculate position step by step:
    # Start at world origin
    x, y = 0.0, 0.0
    
    # Step 1: world -> base_link0_5 (no offset)
    # x, y remain 0, 0
    
    # Step 2: base_joint_mobile_forward (prismatic in Y direction)
    # Initial offset: xyz="-0.21 0 0"
    # Joint movement: forward * axis[1] = forward * 1 in Y direction
    x += -0.21  # Initial X offset
    y += forward  # Forward movement in Y direction
    
    # Step 3: base_joint_mobile_side (prismatic in X direction)  
    # No initial offset: xyz="0 0 0"
    # Joint movement: side * axis[0] = side * 1 in X direction
    x += side  # Side movement in X direction
    
    # Step 4: base_joint_mobile_yaw (rotation around Z axis)
    # No translation offset, only rotation affects subsequent transformations
    # Current position before rotation
    pos_before_rotation = np.array([x, y])
    
    # Step 5: base_link0_8 -> base_base fixed joint
    # Add offset: xyz="0.21 0 0"
    x += 0.21
    y += 0.0
    
    # Step 6: base_base -> base_fixed_support fixed joint
    # Add offset: xyz="-0.05 0 0.5" (we only care about x,y)
    x += -0.05
    y += 0.0
    
    # Step 7: base_fixed_support -> base_link0_3 (height joint)
    # Add offset: xyz="0.05 0 0.2" (we only care about x,y)
    # Note: height joint moves in Z direction, doesn't affect x,y
    x += 0.05
    y += 0.0
    
    # Step 8: base_link0_3 -> base_support fixed joint
    # No offset: xyz="0 0 0"
    # x, y unchanged
    
    # Now apply the yaw rotation around the rotation center
    # The rotation happens at step 4, so we need to rotate the position
    # relative to where the rotation center was at that point
    
    # Position of rotation center (after steps 1-3, before step 4)
    rotation_center_x = -0.21 + side
    rotation_center_y = forward
    
    # Final position before applying rotation
    final_pos_before_rotation = np.array([x, y])
    
    # Apply yaw rotation around the rotation center
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # Translate to rotation center
    relative_pos = final_pos_before_rotation - np.array([rotation_center_x, rotation_center_y])
    
    # Apply rotation
    rotated_relative_pos = np.array([
        cos_yaw * relative_pos[0] - sin_yaw * relative_pos[1],
        sin_yaw * relative_pos[0] + cos_yaw * relative_pos[1]
    ])
    
    # Translate back
    final_pos = rotated_relative_pos + np.array([rotation_center_x, rotation_center_y])
    
    return [final_pos[0], final_pos[1]]


def get_forward_side_from_support_point_and_yaw(support_point, yaw):
    """
    Given the support point (x, y) and yaw, it will return the forward and side movements.
    
    This is the inverse of the support_point function.
    
    Args:
        support_point: [x, y] coordinates of the desired support point location
        yaw: Rotation around Z axis (radians)
    
    Returns:
        [forward, side]: The forward and side movements needed to achieve the support point
    """
    
    # Convert support_point to numpy array for easier manipulation
    support_pos = np.array(support_point)
    
    # From the support_point function, we know the transformation chain:
    # 1. Start with forward/side movements
    # 2. Apply yaw rotation around rotation center
    # 3. Add fixed offsets to get final support position
    
    # The fixed offsets from the transformation chain:
    # Net X offset: -0.21 (forward joint) + 0.21 (fixed joint) - 0.05 (base to fixed_support) + 0.05 (fixed_support to support) = 0.0
    # Net Y offset: 0.0
    base_center_offset_x = 0.0  # Net X offset to support point
    base_center_offset_y = 0.0  # Net Y offset to support point
    
    # The rotation center is at the position after forward/side movements but before yaw rotation
    # From support_point function: rotation_center = [-0.21 + side, forward]
    
    # We need to solve for forward and side given:
    # support_pos = rotation_center + R(yaw) * offset_from_rotation_center
    # where offset_from_rotation_center = [0.21, 0.0] (the fixed offsets after rotation)
    
    # The offset vector from rotation center to support point in the robot's frame
    # (before applying yaw rotation)
    offset_from_rotation_center = np.array([0.21, 0.0])
    
    # Apply yaw rotation to get the offset in world frame
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([[cos_yaw, -sin_yaw],
                               [sin_yaw, cos_yaw]])
    
    rotated_offset = rotation_matrix @ offset_from_rotation_center
    
    # Now we can solve for the rotation center:
    # support_pos = rotation_center + rotated_offset
    # rotation_center = support_pos - rotated_offset
    rotation_center = support_pos - rotated_offset
    
    # From the rotation center, we can extract forward and side:
    # rotation_center = [-0.21 + side, forward]
    side = rotation_center[0] + 0.21
    forward = rotation_center[1]
    
    return [forward, side]

def visualize_robot(forward, side, yaw, plot_range=2.0, resolution=0.05, save_path=None):
    """
    Visualize the robot's occupancy footprint and support point in 2D, saving to file.
    
    Args:
        forward: Forward movement (Y direction)
        side: Side movement (X direction) 
        yaw: Rotation around Z axis (radians)
        plot_range: Half-width of the plot area (meters)
        resolution: Grid resolution for occupancy visualization (meters)
        save_path: Path to save the plot (if None, generates automatic filename)
    
    Returns:
        str: Path to the saved file
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os
    from datetime import datetime
    
    # Set matplotlib backend to non-interactive
    plt.switch_backend('Agg')
    
    # Get the occupancy function and support point
    occupancy_fn = occupancy_map(forward, side, yaw)
    support_pos = support_point(forward, side, yaw)
    
    # Create a grid of points to test occupancy
    x_range = np.linspace(-plot_range, plot_range, int(2 * plot_range / resolution))
    y_range = np.linspace(-plot_range, plot_range, int(2 * plot_range / resolution))
    X, Y = np.meshgrid(x_range, y_range)
    
    # Flatten the grid to test occupancy
    points = np.column_stack([X.ravel(), Y.ravel()])
    occupancy = occupancy_fn(points)
    
    # Reshape back to grid
    occupancy_grid = occupancy.reshape(X.shape)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot occupancy as a filled contour
    ax.contourf(X, Y, occupancy_grid.astype(int), levels=[0, 0.5, 1], 
                colors=['white', 'lightblue'], alpha=0.7)
    
    # Plot occupancy boundary
    ax.contour(X, Y, occupancy_grid.astype(int), levels=[0.5], 
               colors=['blue'], linewidths=2)
    
    # Calculate robot center for reference
    robot_center_x = side + (-0.20)  # Same as in occupancy_map
    robot_center_y = forward + 0.0
    
    # Plot robot center
    ax.plot(robot_center_x, robot_center_y, 'ro', markersize=8, 
            label='Robot Center (base_wheeled_base)')
    
    # Plot support point
    ax.plot(support_pos[0], support_pos[1], 'gs', markersize=10, 
            label='Support Point (base_support)')
    
    # Plot coordinate axes for reference
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add robot orientation arrow from center
    arrow_length = 0.3
    arrow_dx = arrow_length * np.cos(yaw)
    arrow_dy = arrow_length * np.sin(yaw)
    ax.arrow(robot_center_x, robot_center_y, arrow_dx, arrow_dy,
             head_width=0.05, head_length=0.05, fc='red', ec='red',
             label='Robot Orientation')
    
    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Robot Occupancy Visualization\n'
                f'Forward: {forward:.2f}m, Side: {side:.2f}m, Yaw: {yaw:.2f}rad ({np.degrees(yaw):.1f}°)',
                fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Set axis limits
    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    
    # Add text with robot dimensions
    textstr = f'Robot Base: 0.7m × 0.5m\nSupport at: ({support_pos[0]:.2f}, {support_pos[1]:.2f})'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Generate save path if not provided
    if save_path is None:
        vis_dir = "/home/hongchix/codes/curobo/src/curobo/content/assets/robot/omron_franka/.vis"
        os.makedirs(vis_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robot_viz_f{forward:.2f}_s{side:.2f}_y{np.degrees(yaw):.0f}deg_{timestamp}.png"
        save_path = os.path.join(vis_dir, filename)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # Close the figure to free memory
    
    return save_path


def test_support_point_function():
    """
    Test function to verify the support_point implementation.
    """
    print("Testing support_point function...")
    
    # Test 1: Robot at origin, no rotation
    support_pos = support_point(forward=0.0, side=0.0, yaw=0.0)
    print(f"Support point at origin: {support_pos}")
    
    # Expected: x = -0.21 + 0 + 0.21 - 0.05 + 0.05 = 0.0
    #           y = 0 + 0 = 0.0
    expected = [0.0, 0.0]
    print(f"Expected: {expected}")
    print(f"Match: {np.allclose(support_pos, expected)}")
    
    # Test 2: Robot with forward movement
    support_pos2 = support_point(forward=1.0, side=0.0, yaw=0.0)
    print(f"Support point with forward=1.0: {support_pos2}")
    
    # Expected: x = 0.0 (same as before)
    #           y = 1.0 (forward movement)
    expected2 = [0.0, 1.0]
    print(f"Expected: {expected2}")
    print(f"Match: {np.allclose(support_pos2, expected2)}")
    
    # Test 3: Robot with side movement
    support_pos3 = support_point(forward=0.0, side=0.5, yaw=0.0)
    print(f"Support point with side=0.5: {support_pos3}")
    
    # Expected: x = 0.0 + 0.5 = 0.5
    #           y = 0.0
    expected3 = [0.5, 0.0]
    print(f"Expected: {expected3}")
    print(f"Match: {np.allclose(support_pos3, expected3)}")
    
    # Test 4: Robot with yaw rotation (90 degrees)
    support_pos4 = support_point(forward=0.0, side=0.0, yaw=np.pi/2)
    print(f"Support point with yaw=π/2: {support_pos4}")
    
    # The rotation center is at [-0.21, 0], and support point before rotation is at [0, 0]
    # Relative to rotation center: [0.21, 0]
    # After 90° rotation: [0, 0.21]
    # Back to world: [-0.21, 0] + [0, 0.21] = [-0.21, 0.21]
    expected4 = [-0.21, 0.21]
    print(f"Expected: {expected4}")
    print(f"Match: {np.allclose(support_pos4, expected4, atol=1e-10)}")
    
    # Test 5: Combined movement and rotation
    support_pos5 = support_point(forward=1.0, side=0.5, yaw=np.pi/4)
    print(f"Support point with forward=1.0, side=0.5, yaw=π/4: {support_pos5}")
    
    print("Support point testing complete!")


def test_inverse_function():
    """
    Test the inverse function get_forward_side_from_support_point_and_yaw.
    """
    print("Testing inverse function (get_forward_side_from_support_point_and_yaw)...")
    
    # Test configurations to verify round-trip accuracy
    test_configs = [
        (0.0, 0.0, 0.0, "Origin"),
        (1.0, 0.0, 0.0, "Forward movement"),
        (0.0, 0.5, 0.0, "Side movement"),
        (0.0, 0.0, np.pi/2, "90° rotation"),
        (1.0, 0.5, np.pi/4, "Combined movement and 45° rotation"),
        (0.5, -0.3, -np.pi/6, "Complex configuration"),
        (-0.5, 1.2, np.pi, "Negative movements with 180° rotation")
    ]
    
    all_passed = True
    
    for forward, side, yaw, description in test_configs:
        print(f"\nTest: {description}")
        print(f"  Original: forward={forward:.3f}, side={side:.3f}, yaw={yaw:.3f}")
        
        # Forward: get support point from forward/side/yaw
        support_pos = support_point(forward, side, yaw)
        print(f"  Support point: [{support_pos[0]:.6f}, {support_pos[1]:.6f}]")
        
        # Inverse: get forward/side from support point and yaw
        recovered_forward, recovered_side = get_forward_side_from_support_point_and_yaw(support_pos, yaw)
        print(f"  Recovered: forward={recovered_forward:.6f}, side={recovered_side:.6f}")
        
        # Check if we recovered the original values
        forward_match = np.allclose([recovered_forward], [forward], atol=1e-10)
        side_match = np.allclose([recovered_side], [side], atol=1e-10)
        
        if forward_match and side_match:
            print(f"  ✓ PASS: Round-trip successful")
        else:
            print(f"  ✗ FAIL: Round-trip failed")
            print(f"    Forward error: {abs(recovered_forward - forward):.2e}")
            print(f"    Side error: {abs(recovered_side - side):.2e}")
            all_passed = False
    
    print(f"\nInverse function test result: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    
    # Additional test: verify with known support point
    print(f"\nAdditional verification test:")
    target_support = [1.0, 2.0]
    target_yaw = np.pi/3
    
    # Get forward/side for this support point
    forward, side = get_forward_side_from_support_point_and_yaw(target_support, target_yaw)
    print(f"  Target support point: {target_support}")
    print(f"  Calculated forward={forward:.6f}, side={side:.6f}, yaw={target_yaw:.6f}")
    
    # Verify by computing support point
    actual_support = support_point(forward, side, target_yaw)
    print(f"  Actual support point: [{actual_support[0]:.6f}, {actual_support[1]:.6f}]")
    
    verification_match = np.allclose(actual_support, target_support, atol=1e-10)
    print(f"  Verification: {'✓ PASS' if verification_match else '✗ FAIL'}")
    
    print("Inverse function testing complete!")


def test_visualization_function():
    """
    Test the visualization function by creating a sample plot.
    """
    print("Testing visualization function...")
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
        
        # Test creating a simple visualization
        print("✓ Creating test visualization...")
        test_path = quick_viz(0.0, 0.0, 0.0)
        print(f"✓ Test visualization saved to: {test_path}")
        print("✓ Visualization function is ready to use!")
        print()
        print("Available functions:")
        print("  • visualize_robot(forward, side, yaw) - Single detailed plot")
        print("  • quick_viz(forward, side, yaw) - Quick shorthand")
        print("  • viz_examples() - Grid of 6 example configurations")
        print("  • demo_visualizations() - Create 6 demo plots")
        print()
        
    except ImportError:
        print("✗ matplotlib not available - visualization function requires matplotlib")
        print("Install with: pip install matplotlib")
    except Exception as e:
        print(f"✗ Error testing visualization: {e}")


def demo_visualizations():
    """
    Create demonstration visualizations for different robot configurations, saves all to files.
    
    Returns:
        list: List of paths to the saved files
    """
    print("Robot Visualization Demo")
    print("=======================")
    print("Creating visualizations for several robot configurations...\n")
    
    # Demo configurations
    configs = [
        (0.0, 0.0, 0.0, "Robot at origin"),
        (1.0, 0.0, 0.0, "Forward movement"),
        (0.0, 0.5, 0.0, "Side movement"),
        (0.0, 0.0, np.pi/4, "45° rotation"),
        (1.0, 0.5, np.pi/2, "Combined movement and 90° rotation"),
        (0.5, -0.3, -np.pi/6, "Complex configuration")
    ]
    
    saved_files = []
    
    for i, (forward, side, yaw, description) in enumerate(configs, 1):
        print(f"Demo {i}/{len(configs)}: {description}")
        print(f"  Configuration: forward={forward}m, side={side}m, yaw={yaw:.3f}rad ({np.degrees(yaw):.1f}°)")
        
        # Generate custom filename for this demo
        import os
        from datetime import datetime
        vis_dir = "/home/hongchix/codes/curobo/src/curobo/content/assets/robot/omron_franka/.vis"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_{i:02d}_{description.lower().replace(' ', '_').replace('°', 'deg')}_{timestamp}.png"
        save_path = os.path.join(vis_dir, filename)
        
        result_path = visualize_robot(forward, side, yaw, save_path=save_path)
        saved_files.append(result_path)
        print(f"  ✓ Saved to: {result_path}\n")
    
    print("Demo complete!")
    print(f"All {len(saved_files)} visualizations saved to .vis directory")
    return saved_files


def quick_viz(forward=0.0, side=0.0, yaw=0.0, save_path=None):
    """
    Quick visualization function with default parameters, saves to file.
    
    Args:
        forward: Forward movement (default: 0.0)
        side: Side movement (default: 0.0) 
        yaw: Yaw rotation in radians (default: 0.0)
        save_path: Path to save the plot (if None, generates automatic filename)
    
    Returns:
        str: Path to the saved file
    """
    return visualize_robot(forward, side, yaw, save_path=save_path)


def viz_examples(save_path=None):
    """
    Create a grid showing multiple robot configuration examples, saves to file.
    
    Args:
        save_path: Path to save the plot (if None, generates automatic filename)
    
    Returns:
        str: Path to the saved file
    """
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    
    # Set matplotlib backend to non-interactive
    plt.switch_backend('Agg')
    
    print("Creating example visualizations...")
    
    # Create a 2x3 subplot showing different configurations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Robot Configuration Examples', fontsize=16)
    
    configs = [
        (0.0, 0.0, 0.0, "Origin"),
        (1.0, 0.0, 0.0, "Forward +1m"),
        (0.0, 1.0, 0.0, "Side +1m"),
        (0.0, 0.0, np.pi/2, "90° Rotation"),
        (1.0, 1.0, 0.0, "Diagonal Move"),
        (0.5, 0.5, np.pi/4, "Complex Config")
    ]
    
    for idx, (forward, side, yaw, title) in enumerate(configs):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        # Create occupancy grid
        plot_range = 2.0
        resolution = 0.1
        x_range = np.linspace(-plot_range, plot_range, int(2 * plot_range / resolution))
        y_range = np.linspace(-plot_range, plot_range, int(2 * plot_range / resolution))
        X, Y = np.meshgrid(x_range, y_range)
        
        # Get occupancy and support point
        occupancy_fn = occupancy_map(forward, side, yaw)
        support_pos = support_point(forward, side, yaw)
        
        points = np.column_stack([X.ravel(), Y.ravel()])
        occupancy = occupancy_fn(points)
        occupancy_grid = occupancy.reshape(X.shape)
        
        # Plot
        ax.contourf(X, Y, occupancy_grid.astype(int), levels=[0, 0.5, 1], 
                   colors=['white', 'lightblue'], alpha=0.7)
        ax.contour(X, Y, occupancy_grid.astype(int), levels=[0.5], 
                  colors=['blue'], linewidths=1)
        
        # Robot center and support point
        robot_center_x = side + (-0.20)
        robot_center_y = forward + 0.0
        ax.plot(robot_center_x, robot_center_y, 'ro', markersize=6)
        ax.plot(support_pos[0], support_pos[1], 'gs', markersize=8)
        
        # Orientation arrow
        arrow_length = 0.2
        arrow_dx = arrow_length * np.cos(yaw)
        arrow_dy = arrow_length * np.sin(yaw)
        ax.arrow(robot_center_x, robot_center_y, arrow_dx, arrow_dy,
                head_width=0.05, head_length=0.05, fc='red', ec='red')
        
        # Formatting
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{title}\n({forward:.1f}, {side:.1f}, {np.degrees(yaw):.0f}°)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    plt.tight_layout()
    
    # Generate save path if not provided
    if save_path is None:
        vis_dir = "/home/hongchix/codes/curobo/src/curobo/content/assets/robot/omron_franka/.vis"
        os.makedirs(vis_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robot_examples_grid_{timestamp}.png"
        save_path = os.path.join(vis_dir, filename)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # Close the figure to free memory
    
    print(f"Example visualizations saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    print("🤖 Omron Franka Robot Occupancy & Visualization System")
    print("=" * 55)
    
    # Run basic tests
    test_occupancy_function()
    print("\n" + "="*50 + "\n")
    test_support_point_function()
    print("\n" + "="*50 + "\n")
    test_inverse_function()
    print("\n" + "="*50 + "\n")
    test_visualization_function()
    
    print("=" * 55)
    print("🧮 Core Functions Available:")
    print("  • occupancy_map(forward, side, yaw) - Returns occupancy checker function")
    print("  • support_point(forward, side, yaw) - Returns support point [x, y]")
    print("  • get_forward_side_from_support_point_and_yaw(point, yaw) - Inverse function")
    print()
    print("📊 Visualization Functions (Save to .vis directory):")
    print("  • visualize_robot(forward, side, yaw) - Single detailed plot")
    print("  • quick_viz(forward, side, yaw) - Quick shorthand function")  
    print("  • demo_visualizations() - Create 6 demo plots")
    print("  • viz_examples() - Grid of 6 example configurations")
    print()
    print("💡 Example usage:")
    print("  from occupancy import *")
    print("  import numpy as np")
    print("  # Forward: get support point from movements")
    print("  support_pos = support_point(1.0, 0.5, np.pi/4)")
    print("  # Inverse: get movements from desired support point")
    print("  forward, side = get_forward_side_from_support_point_and_yaw([1.5, 1.0], np.pi/6)")
    print("  # Visualization")
    print("  path = visualize_robot(forward, side, np.pi/6)")
    print(f"  📁 All plots saved to: .vis directory")
    print("=" * 55)