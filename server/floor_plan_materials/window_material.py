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
import numpy as np
import os
import torch
import trimesh
import cv2
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from floor_plan_materials.material_generator import (
    material_generate_from_prompt, 
    generate_texture_map_from_prompt_and_sketch, 
    generate_texture_map_from_prompt_and_sketch_and_image,
    generate_texture_map_from_prompt_and_color,
    generate_texture_map_from_prompt_and_color_and_sketch,
    # generate_texture_map_from_prompt_and_sketch_controlnet
)
from constants import SERVER_ROOT_DIR

class WindowMaterialGenerator:
    def __init__(self):
        # Define standard window type mappings
        # Format: (nx, ny) where nx=horizontal grid, ny=vertical grid
        self.window_type_grids = {
            "fixed": (1, 1),
            "slider": (2, 1),  # 2 horizontal panes, 1 vertical
            "hung": (1, 2),    # 1 horizontal pane, 2 vertical (stacked)
            "bay_small": (2, 2),
            "bay_medium": (3, 2),
            "bay_large": (3, 3),
            "picture": (1, 1),
            "casement": (1, 1),
            "double_hung": (1, 2),  # 1 horizontal, 2 vertical (stacked)
            "single_hung": (1, 2),  # 1 horizontal, 2 vertical (stacked)
        }
        
        # Define default colors
        self.default_glass_color = [0.8, 0.9, 1.0]  # Light blue tint
        self.default_frame_color = [0.3, 0.3, 0.3]  # Dark gray

    def generate_window_by_type(self, window_type, color_glass=None, color_frame=None):
        """
        Generate window material by window type name.
        
        Args:
            window_type: String window type ("fixed", "slider", "hung", "bay_small", etc.)
            color_glass: RGB color for glass regions (optional, uses default if None)
            color_frame: RGB color for frame regions (optional, uses default if None)
            
        Returns:
            dict: Same as generate_window_material
        """
        if window_type not in self.window_type_grids:
            raise ValueError(f"Unknown window type: {window_type}. Available types: {list(self.window_type_grids.keys())}")
        
        grid_xy = self.window_type_grids[window_type]
        
        # Use default colors if not provided
        if color_glass is None:
            color_glass = self.default_glass_color
        if color_frame is None:
            color_frame = self.default_frame_color
        
        return self.generate_window_material(grid_xy, color_glass, color_frame)

    def generate_window_material(self, window_grid_xy, color_glass, color_frame):
        """
        Generate window material with glass and frame regions.
        
        Args:
            window_grid_xy: Tuple (nx, ny) defining the grid layout
                           - Bay: (n, m) grid  
                           - Hung: (1, 2) grid
                           - Slider: (2, 1) grid
                           - Fixed: (1, 1) grid
            color_glass: RGB color for glass regions (3-element array/list)
            color_frame: RGB color for frame regions (3-element array/list)
            
        Returns:
            dict: {
                "texture_image": numpy array of texture image,
                "vts": texture coordinates (N, 2),
                "fts": face texture indices (M, 3)
            }
        """
        
        # Parse grid layout
        nx, ny = window_grid_xy
        
        # Validate inputs
        if nx <= 0 or ny <= 0:
            raise ValueError(f"Grid dimensions must be positive: got {window_grid_xy}")
        
        # Convert colors to numpy arrays
        color_glass = np.array(color_glass, dtype=np.float32)
        color_frame = np.array(color_frame, dtype=np.float32)
        
        # Validate colors
        if color_glass.shape != (3,) or color_frame.shape != (3,):
            raise ValueError("Colors must be 3-element arrays (RGB)")
        
        # Clamp colors to valid range
        color_glass = np.clip(color_glass, 0.0, 1.0)
        color_frame = np.clip(color_frame, 0.0, 1.0)
        
        # Create comprehensive texture layout for all 6 faces of the window box
        # Window box faces: front/back (width x height), left/right (thickness x height), top/bottom (width x thickness)
        
        # Define texture dimensions for the main window area
        window_area_width = 512
        window_area_height = 512
        
        # Define edge strip dimensions (for side faces)
        # Since thickness << width, height, we use smaller strips
        edge_strip_width = 64  # For left/right faces (thickness x height)
        edge_strip_height = 64  # For top/bottom faces (width x thickness)
        
        # Create comprehensive texture layout:
        # [window_area] [edge_strip] 
        # [edge_strip_bottom]
        total_texture_width = window_area_width + edge_strip_width
        total_texture_height = window_area_height + edge_strip_height
        
        # Create texture image (start with frame color)
        window_texture_image = np.ones((total_texture_height, total_texture_width, 3), dtype=np.float32)
        window_texture_image[:, :, :] = color_frame
        
        # 1. Create main window area (for front/back faces)
        # Calculate frame thickness as a ratio of grid cell size
        frame_thickness_ratio = 0.05  # 5% of grid cell size
        
        # Calculate grid cell dimensions for main window area
        cell_width = window_area_width // nx
        cell_height = window_area_height // ny
        
        # Calculate frame thickness in pixels
        frame_thickness_x = max(1, int(cell_width * frame_thickness_ratio))
        frame_thickness_y = max(1, int(cell_height * frame_thickness_ratio))
        
        # Fill main window area with frame color first
        window_texture_image[:window_area_height, :window_area_width, :] = color_frame
        
        # Create glass regions in main window area
        for i in range(nx):
            for j in range(ny):
                # Calculate glass region bounds for this grid cell
                x_start = i * cell_width + frame_thickness_x
                x_end = (i + 1) * cell_width - frame_thickness_x
                y_start = j * cell_height + frame_thickness_y  
                y_end = (j + 1) * cell_height - frame_thickness_y
                
                # Ensure bounds are valid
                x_start = max(0, min(x_start, window_area_width - 1))
                x_end = max(x_start + 1, min(x_end, window_area_width))
                y_start = max(0, min(y_start, window_area_height - 1))
                y_end = max(y_start + 1, min(y_end, window_area_height))
                
                # Fill glass region
                window_texture_image[y_start:y_end, x_start:x_end, :] = color_glass
        
        # Add horizontal frame lines in main window area
        for j in range(1, ny):
            y_center = j * cell_height
            y_start = max(0, y_center - frame_thickness_y // 2)
            y_end = min(window_area_height, y_center + frame_thickness_y // 2)
            window_texture_image[y_start:y_end, :window_area_width, :] = color_frame
        
        # Add vertical frame lines in main window area
        for i in range(1, nx):
            x_center = i * cell_width
            x_start = max(0, x_center - frame_thickness_x // 2)
            x_end = min(window_area_width, x_center + frame_thickness_x // 2)
            window_texture_image[:window_area_height, x_start:x_end, :] = color_frame
        
        # 2. Fill edge areas with solid frame color (for side faces)
        # Right edge strip (for left/right faces - thickness x height)
        window_texture_image[:window_area_height, window_area_width:, :] = color_frame
        
        # Bottom edge strip (for top/bottom faces - width x thickness)
        window_texture_image[window_area_height:, :window_area_width, :] = color_frame
        
        # Bottom-right corner (for consistency)
        window_texture_image[window_area_height:, window_area_width:, :] = color_frame
        
        # Generate texture coordinates for all 6 faces of the window box
        # Window box: extents=[window.width, wall.thickness, window.height]  
        # X=width, Y=thickness, Z=height
        
        # Create a temporary box to understand the face structure
        temp_box = trimesh.creation.box([1, 1, 1])  # Unit box for reference
        
        # Get face normals
        face_normals = temp_box.face_normals
        
        # Calculate UV coordinate ranges for different areas
        # Main window area (for front/back faces)
        window_u_max = window_area_width / total_texture_width
        window_v_max = window_area_height / total_texture_height
        
        # Right edge strip (for left/right faces - thickness x height)
        edge_right_u_min = window_area_width / total_texture_width
        edge_right_u_max = 1.0
        
        # Bottom edge strip (for top/bottom faces - width x thickness)
        edge_bottom_v_min = window_area_height / total_texture_height
        edge_bottom_v_max = 1.0
        
        # Create unique texture coordinates for each face
        unique_vts = []
        unique_fts = []
        
        for face_idx, (face, normal) in enumerate(zip(temp_box.faces, face_normals)):
            # Get vertices for this face
            face_vertices = temp_box.vertices[face]
            
            # Determine texture coordinates based on face orientation
            if abs(normal[1]) > 0.9:  # Front/back faces (Y-axis - thickness direction)
                # Map to main window area
                # For window faces, we use X (width) and Z (height) coordinates
                # Get the face bounds in local coordinates
                min_x, max_x = face_vertices[:, 0].min(), face_vertices[:, 0].max()
                min_z, max_z = face_vertices[:, 2].min(), face_vertices[:, 2].max()
                
                # Map face vertices to window texture UV coordinates
                # X direction (width) maps to U, Z direction (height) maps to V
                # IMPORTANT: Flip V coordinate since texture V increases downward but Z increases upward
                tex_coords = np.array([
                    [(face_vertices[i, 0] - min_x) / (max_x - min_x) * window_u_max,
                     (face_vertices[i, 2] - min_z) / (max_z - min_z) * window_v_max]
                    for i in range(3)
                ])
                
            elif abs(normal[0]) > 0.9:  # Left/right faces (X-axis normals)
                # Map to right edge strip (thickness x height)
                # Use Y (thickness) and Z (height) coordinates
                min_y, max_y = face_vertices[:, 1].min(), face_vertices[:, 1].max()
                min_z, max_z = face_vertices[:, 2].min(), face_vertices[:, 2].max()
                
                tex_coords = np.array([
                    [edge_right_u_min + (face_vertices[i, 1] - min_y) / (max_y - min_y) * (edge_right_u_max - edge_right_u_min),
                     (face_vertices[i, 2] - min_z) / (max_z - min_z) * window_v_max]
                    for i in range(3)
                ])
                
            elif abs(normal[2]) > 0.9:  # Top/bottom faces (Z-axis normals)
                # Map to bottom edge strip (width x thickness)
                # Use X (width) and Y (thickness) coordinates
                min_x, max_x = face_vertices[:, 0].min(), face_vertices[:, 0].max()
                min_y, max_y = face_vertices[:, 1].min(), face_vertices[:, 1].max()
                
                tex_coords = np.array([
                    [(face_vertices[i, 0] - min_x) / (max_x - min_x) * window_u_max,
                     edge_bottom_v_min + (face_vertices[i, 1] - min_y) / (max_y - min_y) * (edge_bottom_v_max - edge_bottom_v_min)]
                    for i in range(3)
                ])
            else:
                # Fallback (should not happen with box mesh)
                tex_coords = np.array([
                    [0.0, 0.0],
                    [0.0, 0.0], 
                    [0.0, 0.0]
                ])
            
            # Add to unique arrays
            start_idx = len(unique_vts)
            unique_vts.extend(tex_coords)
            unique_fts.append([start_idx, start_idx + 1, start_idx + 2])
        
        # Convert to numpy arrays
        vts = np.array(unique_vts, dtype=np.float32)
        fts = np.array(unique_fts, dtype=np.int32)

        # print(vts.shape, fts.shape)
        # print(vts)
        # print(fts)

        vts[:, 1] = 1 - vts[:, 1]
        
        return {
            # "texture_image": window_texture_image[:int(window_u_max * window_texture_image.shape[0]), :int(window_v_max * window_texture_image.shape[1])],
            "texture_image": window_texture_image,
            "vts": vts,
            "fts": fts,
            "grid_layout": window_grid_xy,
            "dimensions": (total_texture_width, total_texture_height)
        }

    def generate_window_material_color_diffusion(self, window_grid_xy, color_glass, color_frame):
        """
        Generate window material with glass and frame regions using color-based texture generation.
        
        Args:
            window_grid_xy: Tuple (nx, ny) defining the grid layout
                           - Bay: (n, m) grid  
                           - Hung: (1, 2) grid
                           - Slider: (2, 1) grid
                           - Fixed: (1, 1) grid
            color_glass: RGB color for glass regions (3-element array/list)
            color_frame: RGB color for frame regions (3-element array/list)
            
        Returns:
            dict: {
                "texture_image": numpy array of texture image,
                "vts": texture coordinates (N, 2),
                "fts": face texture indices (M, 3)
            }
        """
        
        # Parse grid layout
        nx, ny = window_grid_xy
        
        # Validate inputs
        if nx <= 0 or ny <= 0:
            raise ValueError(f"Grid dimensions must be positive: got {window_grid_xy}")
        
        # Convert colors to numpy arrays and normalize to 0-1 range
        color_glass = np.array(color_glass, dtype=np.float32)
        color_frame = np.array(color_frame, dtype=np.float32)
        
        # Validate colors
        if color_glass.shape != (3,) or color_frame.shape != (3,):
            raise ValueError("Colors must be 3-element arrays (RGB)")
        
        # Clamp colors to valid range
        color_glass = np.clip(color_glass, 0.0, 1.0)
        color_frame = np.clip(color_frame, 0.0, 1.0)
        
        # Create comprehensive texture layout for all 6 faces of the window box
        # Window box faces: front/back (width x height), left/right (thickness x height), top/bottom (width x thickness)
        
        # Define texture dimensions for the main window area
        window_area_width = 512
        window_area_height = 512
        
        # Define edge strip dimensions (for side faces)
        # Since thickness << width, height, we use smaller strips
        edge_strip_width = 64  # For left/right faces (thickness x height)
        edge_strip_height = 64  # For top/bottom faces (width x thickness)
        
        # Create comprehensive texture layout:
        # [window_area] [edge_strip] 
        # [edge_strip_bottom]
        total_texture_width = window_area_width + edge_strip_width
        total_texture_height = window_area_height + edge_strip_height
        
        # Step 1: Create glass and frame region masks
        # Initialize masks (True = glass region, False = frame region)
        glass_mask = np.zeros((total_texture_height, total_texture_width), dtype=bool)
        frame_mask = np.ones((total_texture_height, total_texture_width), dtype=bool)  # Start with all frame
        
        # 1. Create main window area masks (for front/back faces)
        # Calculate frame thickness as a ratio of grid cell size
        frame_thickness_ratio = 0.05  # 5% of grid cell size
        
        # Calculate grid cell dimensions for main window area
        cell_width = window_area_width // nx
        cell_height = window_area_height // ny
        
        # Calculate frame thickness in pixels
        frame_thickness_x = max(1, int(cell_width * frame_thickness_ratio))
        frame_thickness_y = max(1, int(cell_height * frame_thickness_ratio))
        
        # Create glass regions in main window area
        for i in range(nx):
            for j in range(ny):
                # Calculate glass region bounds for this grid cell
                x_start = i * cell_width + frame_thickness_x
                x_end = (i + 1) * cell_width - frame_thickness_x
                y_start = j * cell_height + frame_thickness_y  
                y_end = (j + 1) * cell_height - frame_thickness_y
                
                # Ensure bounds are valid
                x_start = max(0, min(x_start, window_area_width - 1))
                x_end = max(x_start + 1, min(x_end, window_area_width))
                y_start = max(0, min(y_start, window_area_height - 1))
                y_end = max(y_start + 1, min(y_end, window_area_height))
                
                # Mark glass region in mask
                glass_mask[y_start:y_end, x_start:x_end] = True
                frame_mask[y_start:y_end, x_start:x_end] = False
        
        # Add horizontal frame lines in main window area (override glass regions)
        for j in range(1, ny):
            y_center = j * cell_height
            y_start = max(0, y_center - frame_thickness_y // 2)
            y_end = min(window_area_height, y_center + frame_thickness_y // 2)
            glass_mask[y_start:y_end, :window_area_width] = False
            frame_mask[y_start:y_end, :window_area_width] = True
        
        # Add vertical frame lines in main window area (override glass regions)
        for i in range(1, nx):
            x_center = i * cell_width
            x_start = max(0, x_center - frame_thickness_x // 2)
            x_end = min(window_area_width, x_center + frame_thickness_x // 2)
            glass_mask[:window_area_height, x_start:x_end] = False
            frame_mask[:window_area_height, x_start:x_end] = True
        
        # 2. Edge areas are all frame (already set in frame_mask initialization)
        # Right edge strip and bottom edge strip remain as frame regions

        # print(f"glass_mask: {glass_mask}", file=sys.stderr)
        
        # Step 2: Generate glass texture using generate_texture_map_from_prompt_and_color
        glass_texture_pil = generate_texture_map_from_prompt_and_color("glass", color_glass.tolist())

        # print(f"glass_texture_pil: {glass_texture_pil}", file=sys.stderr)
        
        # Step 3: Generate frame texture using generate_texture_map_from_prompt_and_color  
        frame_texture_pil = generate_texture_map_from_prompt_and_color("frame", color_frame.tolist())

        # print(f"frame_texture_pil: {frame_texture_pil}", file=sys.stderr)
        
        # Step 4: Resize textures to match our texture dimensions
        glass_texture_pil = glass_texture_pil.resize((total_texture_width, total_texture_height), Image.LANCZOS)
        frame_texture_pil = frame_texture_pil.resize((total_texture_width, total_texture_height), Image.LANCZOS)
        
        # Convert PIL images to numpy arrays
        glass_texture = np.array(glass_texture_pil).astype(np.float32) / 255.0
        frame_texture = np.array(frame_texture_pil).astype(np.float32) / 255.0
        
        # Step 5: Combine textures using region masks
        # Start with frame texture as base
        window_texture_image = frame_texture.copy()
        
        # Apply glass texture to glass regions
        # Expand mask to 3 channels for RGB
        glass_mask_3d = np.stack([glass_mask, glass_mask, glass_mask], axis=2)
        window_texture_image[glass_mask_3d] = glass_texture[glass_mask_3d]
        
        # Generate texture coordinates for all 6 faces of the window box
        # Window box: extents=[window.width, wall.thickness, window.height]  
        # X=width, Y=thickness, Z=height
        
        # Create a temporary box to understand the face structure
        temp_box = trimesh.creation.box([1, 1, 1])  # Unit box for reference
        
        # Get face normals
        face_normals = temp_box.face_normals
        
        # Calculate UV coordinate ranges for different areas
        # Main window area (for front/back faces)
        window_u_max = window_area_width / total_texture_width
        window_v_max = window_area_height / total_texture_height
        
        # Right edge strip (for left/right faces - thickness x height)
        edge_right_u_min = window_area_width / total_texture_width
        edge_right_u_max = 1.0
        
        # Bottom edge strip (for top/bottom faces - width x thickness)
        edge_bottom_v_min = window_area_height / total_texture_height
        edge_bottom_v_max = 1.0
        
        # Create unique texture coordinates for each face
        unique_vts = []
        unique_fts = []
        
        for face_idx, (face, normal) in enumerate(zip(temp_box.faces, face_normals)):
            # Get vertices for this face
            face_vertices = temp_box.vertices[face]
            
            # Determine texture coordinates based on face orientation
            if abs(normal[1]) > 0.9:  # Front/back faces (Y-axis - thickness direction)
                # Map to main window area
                # For window faces, we use X (width) and Z (height) coordinates
                # Get the face bounds in local coordinates
                min_x, max_x = face_vertices[:, 0].min(), face_vertices[:, 0].max()
                min_z, max_z = face_vertices[:, 2].min(), face_vertices[:, 2].max()
                
                # Map face vertices to window texture UV coordinates
                # X direction (width) maps to U, Z direction (height) maps to V
                # IMPORTANT: Flip V coordinate since texture V increases downward but Z increases upward
                tex_coords = np.array([
                    [(face_vertices[i, 0] - min_x) / (max_x - min_x) * window_u_max,
                     (face_vertices[i, 2] - min_z) / (max_z - min_z) * window_v_max]
                    for i in range(3)
                ])
                
            elif abs(normal[0]) > 0.9:  # Left/right faces (X-axis normals)
                # Map to right edge strip (thickness x height)
                # Use Y (thickness) and Z (height) coordinates
                min_y, max_y = face_vertices[:, 1].min(), face_vertices[:, 1].max()
                min_z, max_z = face_vertices[:, 2].min(), face_vertices[:, 2].max()
                
                tex_coords = np.array([
                    [edge_right_u_min + (face_vertices[i, 1] - min_y) / (max_y - min_y) * (edge_right_u_max - edge_right_u_min),
                     (face_vertices[i, 2] - min_z) / (max_z - min_z) * window_v_max]
                    for i in range(3)
                ])
                
            elif abs(normal[2]) > 0.9:  # Top/bottom faces (Z-axis normals)
                # Map to bottom edge strip (width x thickness)
                # Use X (width) and Y (thickness) coordinates
                min_x, max_x = face_vertices[:, 0].min(), face_vertices[:, 0].max()
                min_y, max_y = face_vertices[:, 1].min(), face_vertices[:, 1].max()
                
                tex_coords = np.array([
                    [(face_vertices[i, 0] - min_x) / (max_x - min_x) * window_u_max,
                     edge_bottom_v_min + (face_vertices[i, 1] - min_y) / (max_y - min_y) * (edge_bottom_v_max - edge_bottom_v_min)]
                    for i in range(3)
                ])
            else:
                # Fallback (should not happen with box mesh)
                tex_coords = np.array([
                    [0.0, 0.0],
                    [0.0, 0.0], 
                    [0.0, 0.0]
                ])
            
            # Add to unique arrays
            start_idx = len(unique_vts)
            unique_vts.extend(tex_coords)
            unique_fts.append([start_idx, start_idx + 1, start_idx + 2])
        
        # Convert to numpy arrays
        vts = np.array(unique_vts, dtype=np.float32)
        fts = np.array(unique_fts, dtype=np.int32)

        # print(vts.shape, fts.shape)
        # print(vts)
        # print(fts)

        vts[:, 1] = 1 - vts[:, 1]
        
        return {
            # "texture_image": window_texture_image[:int(window_u_max * window_texture_image.shape[0]), :int(window_v_max * window_texture_image.shape[1])],
            "texture_image": window_texture_image,
            "vts": vts,
            "fts": fts,
            "grid_layout": window_grid_xy,
            "dimensions": (total_texture_width, total_texture_height)
        }

    
    def generate_window_material_from_prompt(self, window_grid_xy, prompt):
        """
        Generate window material with glass and frame regions using texture generation from prompt and sketch.
        
        Args:
            window_grid_xy: Tuple (nx, ny) defining the grid layout
                           - Bay: (n, m) grid  
                           - Hung: (1, 2) grid
                           - Slider: (2, 1) grid
                           - Fixed: (1, 1) grid
            prompt: Text prompt for texture generation
            
        Returns:
            dict: {
                "texture_image": numpy array of texture image,
                "vts": texture coordinates (N, 2),
                "fts": face texture indices (M, 3)
            }
        """
        
        # Parse grid layout
        nx, ny = window_grid_xy
        
        # Validate inputs
        if nx <= 0 or ny <= 0:
            raise ValueError(f"Grid dimensions must be positive: got {window_grid_xy}")
        
        # Use full texture dimensions - no separate edge strips
        # The entire texture will be used for the window
        total_texture_width = 512
        total_texture_height = 512
        window_area_width = total_texture_width
        window_area_height = total_texture_height
        
        # 1. Create edge map for the main window area based on window grid layout
        # Calculate frame thickness as a ratio of grid cell size
        frame_thickness_ratio = 0.01  # 1% of grid cell size for slim frame
        
        # Calculate grid cell dimensions for main window area
        cell_width = window_area_width // nx
        cell_height = window_area_height // ny
        
        # Calculate frame thickness in pixels
        frame_thickness_x = max(1, int(cell_width * frame_thickness_ratio))
        frame_thickness_y = max(1, int(cell_height * frame_thickness_ratio))
        
        # Create binary mask for edge detection (white=glass, black=frame)
        edge_mask = np.zeros((window_area_height, window_area_width), dtype=np.uint8)
        
        # Fill glass regions in the mask
        for i in range(nx):
            for j in range(ny):
                # Calculate glass region bounds for this grid cell
                x_start = i * cell_width + frame_thickness_x
                x_end = (i + 1) * cell_width - frame_thickness_x
                y_start = j * cell_height + frame_thickness_y  
                y_end = (j + 1) * cell_height - frame_thickness_y
                
                # Ensure bounds are valid
                x_start = max(0, min(x_start, window_area_width - 1))
                x_end = max(x_start + 1, min(x_end, window_area_width))
                y_start = max(0, min(y_start, window_area_height - 1))
                y_end = max(y_start + 1, min(y_end, window_area_height))
                
                # Fill glass region with white (255)
                edge_mask[y_start:y_end, x_start:x_end] = 255
        
        # Apply Canny edge detection to get boundaries between glass and frame regions
        edges = cv2.Canny(edge_mask, 50, 150)
        
        # Add outermost boundary manually since Canny might miss the outer frame
        # Add top and bottom boundaries
        edges[1:5, :] = 255  # Top boundary
        edges[-5:-1, :] = 255  # Bottom boundary
        
        # Add left and right boundaries
        edges[:, 1:5] = 255  # Left boundary
        edges[:, -5:-1] = 255  # Right boundary
        
        # Use the edges directly as the full texture layout (no centering needed)
        full_edge_map = edges
        
        # Generate texture using the prompt and edge sketch
        # guide_image = Image.open(os.path.join(SERVER_ROOT_DIR, "floor_plan_materials/window.png"))
        # texture_pil = generate_texture_map_from_prompt_and_sketch_and_image(prompt, full_edge_map, guide_image)
        # debug save the edge map as png
        # edge_map_path = f"./vis/edge_map_{prompt.replace(' ', '_')}.png"
        # cv2.imwrite(edge_map_path, full_edge_map)
        # print(f"Saved edge map to {edge_map_path}")

        texture_pil = generate_texture_map_from_prompt_and_sketch(prompt, full_edge_map)
        
        # Convert PIL image to numpy array
        window_texture_image = np.array(texture_pil).astype(np.float32) / 255.0
        
        # Ensure the texture has the expected dimensions
        if window_texture_image.shape[:2] != (total_texture_height, total_texture_width):
            # Resize if necessary
            texture_pil_resized = texture_pil.resize((total_texture_width, total_texture_height), Image.LANCZOS)
            window_texture_image = np.array(texture_pil_resized).astype(np.float32) / 255.0
        
        # Generate texture coordinates for all 6 faces of the window box
        # Window box: extents=[window.width, wall.thickness, window.height]  
        # X=width, Y=thickness, Z=height
        
        # Create a temporary box to understand the face structure
        temp_box = trimesh.creation.box([1, 1, 1])  # Unit box for reference
        
        # Get face normals
        face_normals = temp_box.face_normals
        
        # Calculate UV coordinate ranges - use full texture space
        # All faces use the full texture (0.0 to 1.0)
        window_u_min = 0.0
        window_u_max = 1.0
        window_v_min = 0.0
        window_v_max = 1.0
        
        # Create unique texture coordinates for each face
        unique_vts = []
        unique_fts = []
        
        for face_idx, (face, normal) in enumerate(zip(temp_box.faces, face_normals)):
            # Get vertices for this face
            face_vertices = temp_box.vertices[face]
            
            # Determine texture coordinates based on face orientation
            # All faces now use the full texture space (0.0 to 1.0)
            if abs(normal[1]) > 0.9:  # Front/back faces (Y-axis - thickness direction)
                # Map to full texture using X (width) and Z (height) coordinates
                min_x, max_x = face_vertices[:, 0].min(), face_vertices[:, 0].max()
                min_z, max_z = face_vertices[:, 2].min(), face_vertices[:, 2].max()
                
                tex_coords = np.array([
                    [(face_vertices[i, 0] - min_x) / (max_x - min_x),
                     (face_vertices[i, 2] - min_z) / (max_z - min_z)]
                    for i in range(3)
                ])
                
            elif abs(normal[0]) > 0.9:  # Left/right faces (X-axis normals)
                # Map to full texture using Y (thickness) and Z (height) coordinates
                min_y, max_y = face_vertices[:, 1].min(), face_vertices[:, 1].max()
                min_z, max_z = face_vertices[:, 2].min(), face_vertices[:, 2].max()
                
                tex_coords = np.array([
                    [(face_vertices[i, 1] - min_y) / (max_y - min_y),
                     (face_vertices[i, 2] - min_z) / (max_z - min_z)]
                    for i in range(3)
                ])
                
            elif abs(normal[2]) > 0.9:  # Top/bottom faces (Z-axis normals)
                # Map to full texture using X (width) and Y (thickness) coordinates
                min_x, max_x = face_vertices[:, 0].min(), face_vertices[:, 0].max()
                min_y, max_y = face_vertices[:, 1].min(), face_vertices[:, 1].max()
                
                tex_coords = np.array([
                    [(face_vertices[i, 0] - min_x) / (max_x - min_x),
                     (face_vertices[i, 1] - min_y) / (max_y - min_y)]
                    for i in range(3)
                ])
            else:
                # Fallback (should not happen with box mesh)
                tex_coords = np.array([
                    [0.0, 0.0],
                    [0.0, 0.0], 
                    [0.0, 0.0]
                ])
            
            # Add to unique arrays
            start_idx = len(unique_vts)
            unique_vts.extend(tex_coords)
            unique_fts.append([start_idx, start_idx + 1, start_idx + 2])
        
        # Convert to numpy arrays
        vts = np.array(unique_vts, dtype=np.float32)
        fts = np.array(unique_fts, dtype=np.int32)

        # Flip V coordinates for proper texture mapping
        vts[:, 1] = 1 - vts[:, 1]
        
        return {
            "texture_image": window_texture_image,
            "vts": vts,
            "fts": fts,
            "grid_layout": window_grid_xy,
            "dimensions": (total_texture_width, total_texture_height)
        }


    def save_window_material(self, window_material, save_path_prefix):
        """
        Save window material to disk.
        
        Args:
            window_material: Result from generate_window_material
            save_path_prefix: Path prefix for saving files
            
        Returns:
            dict: Paths to saved files
        """
        import compress_pickle
        
        # Save texture image
        texture_path = f"{save_path_prefix}_texture.png"
        texture_image = window_material["texture_image"]
        texture_image_pil = Image.fromarray((texture_image * 255).astype(np.uint8))
        texture_image_pil.save(texture_path)
        
        # Save texture coordinates
        coords_path = f"{save_path_prefix}_tex_coords.pkl"
        tex_coords_dict = {
            "vts": window_material["vts"],
            "fts": window_material["fts"],
            "grid_layout": window_material["grid_layout"],
            "dimensions": window_material["dimensions"]
        }
        compress_pickle.dump(tex_coords_dict, coords_path)
        
        return {
            "texture_path": texture_path,
            "coords_path": coords_path
        }

    def get_available_window_types(self):
        """Get list of available window types."""
        return list(self.window_type_grids.keys())
    
if __name__ == "__main__":
    def test_window_material_from_prompt():
        """Test the new generate_window_material_from_prompt method"""
        
        print("Testing WindowMaterialGenerator.generate_window_material_from_prompt...")
        
        # Create generator instance
        generator = WindowMaterialGenerator()
        
        # Test different window grid layouts
        test_cases = [
            ((1, 1), "classic white painted wooden window frame with frosted glass"),
            ((2, 1), "modern black aluminum window frame with translucent glass panels"),
            ((2, 2), "classic white window frame with frosted glass panels"),
            ((3, 2), "classic natural wood grain window frame with frosted glass panes")
        ]
        
        for i, (grid_xy, prompt) in enumerate(test_cases):
            print(f"\nTest case {i+1}: Grid {grid_xy}, Prompt: '{prompt}'")
            
            try:
                # Generate window material from prompt
                result = generator.generate_window_material_from_prompt(grid_xy, prompt)
                
                # Validate result structure
                required_keys = ["texture_image", "vts", "fts", "grid_layout", "dimensions"]
                for key in required_keys:
                    if key not in result:
                        print(f"ERROR: Missing key '{key}' in result")
                        continue
                
                # Validate data types and shapes
                texture_image = result["texture_image"]
                vts = result["vts"]
                fts = result["fts"]
                
                print(f"  Texture image shape: {texture_image.shape}")
                print(f"  Texture coordinates shape: {vts.shape}")
                print(f"  Face indices shape: {fts.shape}")
                print(f"  Grid layout: {result['grid_layout']}")
                print(f"  Dimensions: {result['dimensions']}")
                
                # Basic validation
                assert len(texture_image.shape) == 3, "Texture image should be 3D (H, W, C)"
                assert texture_image.shape[2] == 3, "Texture image should have 3 channels"
                assert len(vts.shape) == 2, "Texture coordinates should be 2D"
                assert vts.shape[1] == 2, "Texture coordinates should have 2 components (u, v)"
                assert len(fts.shape) == 2, "Face indices should be 2D"
                assert fts.shape[1] == 3, "Face indices should have 3 vertices per face"
                assert result['grid_layout'] == grid_xy, "Grid layout should match input"
                
                print(f"  ✓ Test case {i+1} passed!")
                
                texture_image_pil = Image.fromarray((texture_image * 255).astype(np.uint8))
                texture_image_pil.save(f"./vis/test_window_{i+1}.png")
                print(f"  Saved texture image to ./vis/test_window_{i+1}.png")
                
            except Exception as e:
                print(f"  ✗ Test case {i+1} failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*50)
        print("Testing complete!")
    
    # # Run tests
    # window_material_generator = WindowMaterialGenerator()
    
    # # Test original method
    # print("Testing original generate_window_material method...")
    # window_material = window_material_generator.generate_window_material((1, 1), [0.8, 0.9, 1.0], [0.3, 0.3, 0.3])
    # window_material_generator.save_window_material(window_material, "test_window")
    # print("✓ Original method test completed!")
    
    # print("\n" + "="*50)
    
    # Test new prompt-based method
    test_window_material_from_prompt()