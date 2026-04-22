from __future__ import annotations
import os
import random

import sys
sys.path.append("./scripts")
sys.path.append("./scripts/matfuse_sd/src")

from matfuse_sd.src.utils.inference_helpers import run_generation, get_model

from pathlib import Path
from collections import namedtuple

from nerfstudio.utils.eval_utils import eval_setup
import torch
import numpy as np
from PIL import Image
import json
import pickle
import transformations
import trimesh
from torchvision import transforms

from pytorch3d.renderer import TexturesUV, TexturesVertex, RasterizationSettings, MeshRenderer, MeshRendererWithFragments, MeshRasterizer, HardPhongShader, PointLights, FoVPerspectiveCameras, AmbientLights, SoftSilhouetteShader
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.io import save_obj

import torch.nn.functional as F
import subprocess
import cv2

import xatlas
import pywavefront
import time
from tqdm import tqdm
import nvdiffrast.torch as dr
import pymeshlab as pml

import argparse

def read_textured_obj(obj_file_path):

    scene = pywavefront.Wavefront(obj_file_path, collect_faces=True)

    verts = np.array(scene.vertices).reshape(-1, 3).astype(np.float32)
    verts_texture = np.array(scene.parser.tex_coords).reshape(-1, 2).astype(np.float32)

    faces = []
    for mesh in scene.mesh_list:
        for face in mesh.faces:
            faces.append([face[0], face[1], face[2]])  # Vertex indices
    faces = np.array(faces).astype(np.int32).reshape(-1, 3)

    mat_vertices = scene.parser.material.vertices
    format = scene.parser.material.vertex_format
    assert format == "TIDX_T2F_NIDX_N3F_V3F", format
    faces_texture = np.array(mat_vertices).reshape(-1, 3, 10)[..., :1].astype(np.int32).reshape(-1, 3)

    return {
        "verts": verts,
        "faces": faces,
        "verts_texture": verts_texture,
        "faces_texture": faces_texture,
    }

def filter_mesh_from_vertices(keep, mesh_points, faces):
    filter_mapping = np.arange(keep.shape[0])[keep]
    filter_unmapping = -np.ones((keep.shape[0]))
    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    mesh_points = mesh_points[keep]
    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = np.logical_and(keep_0, keep_1)
    keep_faces = np.logical_and(keep_faces, keep_2)
    faces = faces[keep_faces]
    face_mapping = np.arange(keep_faces.shape[0])[keep_faces]
    faces[:, 0] = filter_unmapping[faces[:, 0]]
    faces[:, 1] = filter_unmapping[faces[:, 1]]
    faces[:, 2] = filter_unmapping[faces[:, 2]]
    return mesh_points, faces, face_mapping

def generate_uvs(vertices, faces):
    # Create a mesh in xatlas format
    _, indices, uvs = xatlas.parametrize(vertices, faces)
    verts_texture = uvs.reshape(-1, 2).astype(np.float32)
    faces_texture = indices.reshape(-1, 3).astype(np.int32)

    return {
        "verts": vertices,
        "faces": faces,
        "verts_texture": verts_texture,
        "faces_texture": faces_texture,
    }

def area(triangles):
    # Extract the vertices of the triangles
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]

    # Compute the lengths of the sides of the triangles
    a = np.linalg.norm(B - C, axis=1, ord=2)
    b = np.linalg.norm(C - A, axis=1, ord=2)
    c = np.linalg.norm(A - B, axis=1, ord=2)

    # Compute the semi-perimeter of each triangle
    s = (a + b + c) / 2

    # Compute the area of each triangle using Heron's formula
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))

    return area

def subdivide_textured_mesh(mesh_dict):
    verts = mesh_dict["verts"]
    faces = mesh_dict["faces"]
    vt = mesh_dict["verts_texture"]
    ft = mesh_dict["faces_texture"]

    area_to_subdivide = 1e-3

    while True:
        triangles = verts[faces.reshape(-1)].reshape(-1, 3, 3)
        triangles_texture = vt[ft.reshape(-1)].reshape(-1, 3, 2)

        areas = area(triangles)
        if np.all(areas <= area_to_subdivide):
            break
        face_to_subdivide = (areas > area_to_subdivide)

        mesh_faces_subdivided = faces[face_to_subdivide]
        triangles_subdivided = verts[mesh_faces_subdivided.reshape(-1)].reshape(-1, 3, 3)

        mesh_ft_subdivided = ft[face_to_subdivide]
        triangles_tex_subdivided = vt[mesh_ft_subdivided.reshape(-1)].reshape(-1, 3, 2)

        mesh_verts_added = np.concatenate([
            (triangles_subdivided[:, 0] + triangles_subdivided[:, 1]) / 2,
            (triangles_subdivided[:, 0] + triangles_subdivided[:, 2]) / 2,
            (triangles_subdivided[:, 1] + triangles_subdivided[:, 2]) / 2,
        ], axis=0)

        mesh_vts_added = np.concatenate([
            (triangles_tex_subdivided[:, 0] + triangles_tex_subdivided[:, 1]) / 2,
            (triangles_tex_subdivided[:, 0] + triangles_tex_subdivided[:, 2]) / 2,
            (triangles_tex_subdivided[:, 1] + triangles_tex_subdivided[:, 2]) / 2,
        ], axis=0)

        # for faces

        num_verts_before = verts.shape[0]
        num_subdivided_faces = triangles_subdivided.shape[0]

        verts_a_idxs = num_verts_before + np.arange(num_subdivided_faces)
        verts_b_idxs = num_verts_before + num_subdivided_faces + np.arange(num_subdivided_faces)
        verts_c_idxs = num_verts_before + num_subdivided_faces * 2 + np.arange(num_subdivided_faces)

        verts_0_idxs = mesh_faces_subdivided[:, 0]
        verts_1_idxs = mesh_faces_subdivided[:, 1]
        verts_2_idxs = mesh_faces_subdivided[:, 2]

        faces_0ab = np.stack([verts_0_idxs, verts_a_idxs, verts_b_idxs], axis=-1)
        faces_1ca = np.stack([verts_1_idxs, verts_c_idxs, verts_a_idxs], axis=-1)
        faces_2bc = np.stack([verts_2_idxs, verts_b_idxs, verts_c_idxs], axis=-1)
        faces_acb = np.stack([verts_a_idxs, verts_c_idxs, verts_b_idxs], axis=-1)

        faces[face_to_subdivide] = faces_acb
        faces = np.concatenate([
            faces,
            faces_0ab, faces_1ca, faces_2bc
        ], axis=0)

        # for ft

        num_vts_before = vt.shape[0]
        num_subdivided_fts = triangles_tex_subdivided.shape[0]

        vts_a_idxs = num_vts_before + np.arange(num_subdivided_fts)
        vts_b_idxs = num_vts_before + num_subdivided_fts + np.arange(num_subdivided_fts)
        vts_c_idxs = num_vts_before + num_subdivided_fts * 2 + np.arange(num_subdivided_fts)

        vts_0_idxs = mesh_ft_subdivided[:, 0]
        vts_1_idxs = mesh_ft_subdivided[:, 1]
        vts_2_idxs = mesh_ft_subdivided[:, 2]

        fts_0ab = np.stack([vts_0_idxs, vts_a_idxs, vts_b_idxs], axis=-1)
        fts_1ca = np.stack([vts_1_idxs, vts_c_idxs, vts_a_idxs], axis=-1)
        fts_2bc = np.stack([vts_2_idxs, vts_b_idxs, vts_c_idxs], axis=-1)
        fts_acb = np.stack([vts_a_idxs, vts_c_idxs, vts_b_idxs], axis=-1)

        ft[face_to_subdivide] = fts_acb
        ft = np.concatenate([
            ft,
            fts_0ab, fts_1ca, fts_2bc
        ], axis=0)

        verts = np.concatenate([
            verts,
            mesh_verts_added
        ], axis=0)

        vt = np.concatenate([
            vt,
            mesh_vts_added
        ], axis=0)

    return {
        "verts": verts,
        "faces": faces,
        "verts_texture": vt,
        "faces_texture": ft,
    }



@torch.no_grad()
def get_normal_map(uv_mesh, uv_size, glctx):
    verts = uv_mesh["verts"]
    faces = uv_mesh["faces"]
    vt = uv_mesh["verts_texture"]
    ft = uv_mesh["faces_texture"]

    mesh = Meshes(verts=torch.from_numpy(verts).unsqueeze(0), faces=torch.from_numpy(faces).unsqueeze(0))
    normals = mesh.faces_normals_packed().reshape(-1, 3)

    device = "cuda"
    vt = torch.from_numpy(vt.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft.astype(np.int64)).int().to(device)

    # render uv maps
    uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)  # [N, 4]

    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (uv_size, uv_size))  # [1, h, w, 4]
    valid_pixel = (rast[..., -1] > 0).reshape(uv_size, uv_size)
    face_id = (rast[..., -1] - 1).reshape(uv_size, uv_size).long()
    valid_face_id = face_id[valid_pixel].reshape(-1)
    # bary_coords = rast[..., :2].reshape(uv_size, uv_size, 2)
    # bary_coords = torch.cat([bary_coords, 1 - bary_coords[..., 0:1] - bary_coords[..., 1:2]], dim=-1)
    # valid_bary_coords = bary_coords[valid_pixel].reshape(-1, 3)
    valid_normals = normals.to(device)[valid_face_id]
    normals = torch.ones((uv_size, uv_size, 3)).reshape(-1, 3).cpu()
    normals[valid_pixel.reshape(-1).cpu()] = valid_normals.cpu()
    normals = normals.reshape(uv_size, uv_size, 3)
    normals = torch.flip(normals, [0])

    return normals

def edge_detect(input_img):
    # Convert to graycsale
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    # Blur the image for better edge detection
    # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # # Sobel Edge Detection
    # sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    # sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    # Canny Edge Detection
    edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)  # Canny Edge Detection
    return edges


def filter_faces_from_verts(keep, faces):

    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = torch.logical_or(keep_0, keep_1)
    keep_faces = torch.logical_or(keep_faces, keep_2)

    return keep_faces

def point_transform(point, transform):
    point_shape = point.shape
    point_pad = np.pad(point.reshape(-1, 3), ((0, 0), (0, 1)), constant_values=(0, 1))
    point_transformed_pad = point_pad @ transform.T
    point_transformed = point_transformed_pad[:, :3] / point_transformed_pad[:, 3:]
    return point_transformed.reshape(point_shape)

def depth_blend(renderer, mesh_list, resolution):
    H, W = resolution
    pix_to_face_list = []
    bary_coords_list = []
    for mesh_i, mesh in enumerate(mesh_list):
        image_i, frags = renderer(mesh)
        image_i = image_i[..., :3].reshape(H, W, 3)
        depth_i = frags.zbuf.reshape(H, W)
        depth_i[depth_i < 0] = 1e5 + mesh_i
        pix_to_face_i = frags.pix_to_face.reshape(H, W).clone()
        bary_coords_i = frags.bary_coords.reshape(H, W, 3).clone()
        pix_to_face_list.append(pix_to_face_i)
        bary_coords_list.append(bary_coords_i)
        if mesh_i == 0:
            depth_buf = depth_i
            image = image_i
            idx_buf = torch.zeros((H, W), dtype=torch.int64).to(depth_i.device)
            idx_buf[depth_i >= 0] = 1

        else:
            image = torch.where(depth_i.unsqueeze(-1).expand(-1, -1, 3) < depth_buf.unsqueeze(-1).expand(-1, -1, 3), image_i, image)
            idx_buf = torch.where(depth_i < depth_buf, torch.ones((H, W), dtype=torch.int64).to(depth_i.device) * (mesh_i + 1), idx_buf)

            depth_buf = torch.where(depth_i < depth_buf, depth_i, depth_buf)
    return {
        "rgb": image,
        "depth": depth_buf,
        "idx": idx_buf,
        "pix_to_face": pix_to_face_list,
        "bary_coords": bary_coords_list,
    }

def pseudo_render_texture_map(result_tex, light_dir=np.array([0, 0, 1])):

    H, W = result_tex.shape[0] // 2, result_tex.shape[1] // 2

    albedo = result_tex[:H, :W]
    roughness = result_tex[:H, W:]
    normal_map = result_tex[H:, :W]

    # Ensure albedo and normal_map are float for calculations
    albedo = albedo / 255.0
    roughness = roughness / 255.0
    normal_map = (normal_map / 255.0) * 2 - 1  # Normalize normal map to range [-1, 1]

    roughness = np.mean(roughness, axis=-1)

    # Normalize the light direction
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Calculate the dot product of light direction and the normal map
    # We use np.einsum for efficient per-pixel dot product
    dot_product = np.einsum('ijk,k->ij', normal_map, light_dir)
    dot_product = np.clip(dot_product, 0, 1)  # Clamp to [0, 1]

    # Calculate diffuse shading by applying roughness to the dot product
    diffuse = dot_product * (1 - roughness) + roughness

    # Combine albedo with diffuse shading
    rendered_map = albedo * diffuse[..., np.newaxis]

    # Clip values to [0, 1] and convert back to uint8 for image representation
    rendered_map = np.clip(rendered_map * 255, 0, 255).astype(np.uint8)

    return rendered_map

from skimage import measure


def extract_mesh_from_occupancy_and_boxes(x_min, x_max, y_min, y_max, z_min, z_max, resolution, boxes_list):
    """
    Extract a watertight mesh from an occupancy grid defined by a list of boxes.
    
    Parameters:
    -----------
    x_min, x_max, y_min, y_max, z_min, z_max : float
        The bounds of the occupancy grid.
    resolution : int
        The resolution of the grid (grid will be resolution^3).
    boxes_list : list
        List of (scale, translate) tuples where:
        - scale = (scale_x, scale_y, scale_z)
        - translate = (translate_x, translate_y, translate_z)
        Each box is a scaled and translated unit box [-0.5, 0.5]^3.
    
    Returns:
    --------
    mesh : trimesh.Trimesh
        The extracted watertight mesh.
    """
    # Create a 3D grid with the specified resolution
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    
    # Create grid of coordinate points
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Initialize occupancy grid
    occupancy = np.zeros((resolution, resolution, resolution), dtype=bool)
    
    # Fill occupancy grid based on boxes
    for box in boxes_list:
        scale, translate = box
        
        # Transform points to local box coordinates
        # First subtract translation to center, then divide by scale
        local_points = points.copy()
        local_points[:, 0] = (local_points[:, 0] - translate[0]) / scale[0]
        local_points[:, 1] = (local_points[:, 1] - translate[1]) / scale[1]
        local_points[:, 2] = (local_points[:, 2] - translate[2]) / scale[2]
        
        # Check which points are inside the unit box [-0.5, 0.5]^3
        inside = np.all(np.abs(local_points) <= 0.5, axis=1)
        
        # Update occupancy grid
        occupancy_flat = occupancy.flatten()
        occupancy_flat = occupancy_flat | inside
        occupancy = occupancy_flat.reshape((resolution, resolution, resolution))
    
    # Use marching cubes to extract the surface mesh
    verts, faces, normals, values = measure.marching_cubes(occupancy, level=0.5)
    
    # Scale vertices back to original coordinate system
    verts[:, 0] = verts[:, 0] * (x_max - x_min) / (resolution - 1) + x_min
    verts[:, 1] = verts[:, 1] * (y_max - y_min) / (resolution - 1) + y_min
    verts[:, 2] = verts[:, 2] * (z_max - z_min) / (resolution - 1) + z_min
    
    faces = faces[..., ::-1]
    
    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)
    
    return mesh

def create_drawer_internal_box_mesh_occupancy(D, Sy, Sz, thickness, m, n, internal_inverse, top=True, height_offset=0.0, depth_offset=0.0):
    original_D = D + 1e-10
    D = D - depth_offset
    half_D = D / 2
    half_Sy = Sy / 2
    half_Sz = Sz / 2

    # Create the main parts of the box
    outer_parts = []

    y_0 = half_Sy - thickness / 2 - height_offset / 2
    z_0 = half_Sz - thickness / 2 - height_offset / 2

    y_extent = [D, thickness, Sz - 2 * thickness - height_offset]
    z_extent = [D, Sy - height_offset, thickness]
    
    outer_parts.append((z_extent, [0, 0, -z_0]))

    if top:
        outer_parts.append((z_extent, [0, 0, z_0]))
    
    outer_parts.append((y_extent, [0, -y_0, 0]))
    
    outer_parts.append((y_extent, [0, y_0, 0]))
    
    outer_parts.append(([thickness, Sy - height_offset - 2 * thickness, Sz - 2 * thickness - height_offset], [-half_D + thickness / 2, 0, 0]))
    
    column_width = 2 * y_0 / m
    row_height = 2 * z_0 / n
    
    for i in range(1, m):
        y = -y_0 + i * column_width
        outer_parts.append(([D - thickness, thickness, Sz - 2 * thickness - height_offset], [thickness / 2, y, 0]))
    
    for j in range(1, n):
        z = -z_0 + j * row_height
        outer_parts.append(([D - thickness, Sy - 2 * thickness - height_offset, thickness], [thickness / 2, 0, z]))

    final_mesh = extract_mesh_from_occupancy_and_boxes(
        -original_D*2.0, original_D*2.0, 
        -half_Sy*2.0, half_Sy*2.0,
        -half_Sz*2.0, half_Sz*2.0,
        256, outer_parts
    )
    
    verts = np.array(final_mesh.vertices).reshape(-1, 3)
    verts = verts + np.array([-original_D / 2, 0, 0]).reshape(1, 3)

    faces = np.array(final_mesh.faces).reshape(-1, 3)

    if internal_inverse:
        verts[:, 0] *= -1
        faces = faces[:, ::-1]

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!
    ms.meshing_isotropic_explicit_remeshing(iterations=6)
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    final_mesh.vertices = verts
    final_mesh.faces = faces

    return final_mesh
    
    

def create_drawer_internal_box_mesh(D, Sy, Sz, thickness, m, n, internal_inverse, top=True, height_offset=0.0, depth_offset=0.0):
    original_D = D + 1e-10
    D = D - depth_offset
    half_D = D / 2
    half_Sy = Sy / 2
    half_Sz = Sz / 2

    # Create the main parts of the box
    outer_parts = []

    y_0 = half_Sy - thickness / 2 - height_offset / 2
    z_0 = half_Sz - thickness / 2 - height_offset / 2

    y_extent = [D, thickness, Sz - 2 * thickness - height_offset]
    z_extent = [D, Sy - height_offset, thickness]

    # Bottom face
    bottom = trimesh.creation.box(extents=z_extent,
                                  transform=trimesh.transformations.translation_matrix(
                                      [0, 0, -z_0]))
    outer_parts.append(bottom)

    # Top face
    if top:
        top = trimesh.creation.box(extents=z_extent,
                                   transform=trimesh.transformations.translation_matrix(
                                       [0, 0, z_0]))
        outer_parts.append(top)

    # Left face (towards -y axis)
    left = trimesh.creation.box(extents=y_extent,
                                transform=trimesh.transformations.translation_matrix(
                                    [0, -y_0, 0]))
    outer_parts.append(left)

    # Right face (towards +y axis)
    right = trimesh.creation.box(extents=y_extent,
                                 transform=trimesh.transformations.translation_matrix(
                                     [0, y_0, 0]))
    outer_parts.append(right)

    # Back face (towards -x axis)
    back = trimesh.creation.box(
        extents=[thickness, Sy - height_offset - 2 * thickness, Sz - 2 * thickness - height_offset],
        transform=trimesh.transformations.translation_matrix([-half_D + thickness / 2, 0, 0]))
    outer_parts.append(back)

    # Combine outer parts to form the outer shell
    outer_shell = trimesh.util.concatenate(outer_parts)

    # Add grid inside the box
    grid_meshes = []
    column_width = 2 * y_0 / m
    row_height = 2 * z_0 / n

    # Create vertical dividers (parallel to Y axis)
    for i in range(1, m):
        y = -y_0 + i * column_width
        grid_meshes.append(trimesh.creation.box(extents=[D - thickness, thickness, Sz - 2 * thickness - height_offset],
                                                transform=trimesh.transformations.translation_matrix([thickness / 2, y, 0])))

    # Create horizontal dividers (parallel to Z axis)
    for j in range(1, n):
        z = -z_0 + j * row_height
        grid_meshes.append(trimesh.creation.box(extents=[D - thickness, Sy - 2 * thickness - height_offset, thickness],
                                                transform=trimesh.transformations.translation_matrix([thickness / 2, 0, z])))

    # Combine all grid meshes
    if len(grid_meshes) > 0:
        grid_mesh = trimesh.util.concatenate(grid_meshes)
        final_mesh = trimesh.util.concatenate([outer_shell, grid_mesh])
    else:
        final_mesh = outer_shell



    verts = np.array(final_mesh.vertices).reshape(-1, 3)
    verts = verts + np.array([-original_D / 2, 0, 0]).reshape(1, 3)

    faces = np.array(final_mesh.faces).reshape(-1, 3)

    if internal_inverse:
        verts[:, 0] *= -1
        faces = faces[:, ::-1]

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!
    ms.meshing_isotropic_explicit_remeshing(iterations=6)
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    final_mesh.vertices = verts
    final_mesh.faces = faces

    return final_mesh

def create_drawer_door(D, Sy, Sz, internal_inverse, height_offset=0., depth_offset=0.):
    half_D = D / 2
    door_mesh = trimesh.creation.box(extents=[D, Sy - height_offset, Sz - height_offset],
                                     transform=trimesh.transformations.translation_matrix(
                                         [-half_D - depth_offset, 0, 0]))

    verts = np.array(door_mesh.vertices).reshape(-1, 3)
    faces = np.array(door_mesh.faces).reshape(-1, 3)

    door_front = np.abs(verts[:, 0]+depth_offset) < 1e-6
    faces_not_front = np.logical_not(np.all(door_front[faces.reshape(-1)].reshape(-1, 3), axis=-1)).reshape(-1)
    faces = faces[faces_not_front]

    if internal_inverse:
        verts[:, 0] *= -1

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!
    ms.meshing_isotropic_explicit_remeshing(iterations=6)
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    door_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    return door_mesh

def max_histogram_area(histogram):
    stack = []
    max_area = 0
    left_index = -1
    right_index = -1
    for i in range(len(histogram)):
        while stack and histogram[stack[-1]] >= histogram[i]:
            height = histogram[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            area = height * width
            if area > max_area:
                max_area = area
                right_index = i - 1
                left_index = stack[-1] + 1 if stack else 0
        stack.append(i)

    while stack:
        height = histogram[stack.pop()]
        width = len(histogram) if not stack else len(histogram) - stack[-1] - 1
        area = height * width
        if area > max_area:
            max_area = area
            right_index = len(histogram) - 1
            left_index = stack[-1] + 1 if stack else 0

    return max_area, left_index, right_index

def max_rectangle_in_bool_matrix(matrix):
    if matrix.size == 0:
        return 0, None, None

    nrows, ncols = matrix.shape
    histogram = np.zeros(ncols, dtype=int)
    max_area = 0
    max_rectangle = (0, 0, 0, 0)  # (top-left row, top-left col, bottom-right row, bottom-right col)

    for i in range(nrows):
        for j in range(ncols):
            histogram[j] = histogram[j] + 1 if matrix[i][j] else 0

        area, left, right = max_histogram_area(histogram)
        if area > max_area:
            max_area = area
            max_rectangle = (i - histogram[left] + 1, left, i, right)

    return max_area, max_rectangle

def add_random_curve(image):
    H, W = image.shape
    radius = 0.7
    y = []
    x = []
    for theta in np.linspace(0, 2 * np.pi, 12):
        h = (np.cos(theta) * radius * 0.5 + 0.5) * H
        w = (np.sin(theta) * radius * 0.5 + 0.5) * W
        y.append(h)
        x.append(w)

    y = np.array(y)
    x = np.array(x)

    lspace = np.linspace(0, H - 1, W)
    z = np.polyfit(x, y, 2)
    line_fitx = z[0] * lspace ** 2 + z[1] * lspace + z[2]

    verts = np.array(list(zip(line_fitx.astype(int), lspace.astype(int))))

    image = image[..., None]
    image = np.repeat(image, 3, -1)
    cv2.polylines(image, [verts], False, (255, 255, 255), thickness=1)
    image = image[..., 0]

    return image

parser = argparse.ArgumentParser()
parser.add_argument("--splat_dir", type=str, required=True)
parser.add_argument("--sdf_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--ckpt", type=str, help="Path to the MatFuse model")
parser.add_argument("--config", type=str, help="Path to the MatFuse config")

args = parser.parse_args()

matfuse_model = get_model(args.config, args.ckpt)

load_config = os.path.join(args.splat_dir, "config.yml")
bakedsdf_dir = args.sdf_dir
output_dir = args.output_dir

seed = 42
np.random.seed(seed)

door_depth_segment = 0.2
revolute_door_offset_scaling = 2.0


textured_mesh_path = os.path.join(bakedsdf_dir, 'texture_mesh/mesh-simplify.obj')
texture_image_path = os.path.join(bakedsdf_dir, 'texture_mesh/mesh-simplify.png')


config, pipeline, checkpoint_path, _ = eval_setup(Path(load_config))
model = pipeline.model.eval()

with open(os.path.join(bakedsdf_dir, "drawers", "trace.json"), 'r') as f:
    trace_data = json.load(f)
drawer_files = [name for name in os.listdir(os.path.join(bakedsdf_dir, "drawers", "results")) if name.endswith(".ply")]
total_drawers_num = len(drawer_files)

textured_mesh_dict = read_textured_obj(textured_mesh_path)

textured_mesh_verts = torch.from_numpy(textured_mesh_dict["verts"].copy()).cuda()
textured_mesh_faces = torch.from_numpy(textured_mesh_dict["faces"].copy()).cuda()
textured_mesh_verts_texture = torch.from_numpy(textured_mesh_dict["verts_texture"]).cuda()
textured_mesh_faces_texture = torch.from_numpy(textured_mesh_dict["faces_texture"]).cuda()

texture = torch.from_numpy(np.array(Image.open(texture_image_path)).astype(np.float32) / 255.0).unsqueeze(0).cuda()
textured_mesh_uvtex = TexturesUV(maps=texture, faces_uvs=textured_mesh_faces_texture.unsqueeze(0), verts_uvs=textured_mesh_verts_texture.unsqueeze(0))

# get prim info
full_mesh_verts = textured_mesh_dict["verts"].copy()
drawer_sub_verts_all = np.zeros_like(full_mesh_verts[..., 0]).astype(np.bool_)
full_mesh_verts_pad = np.pad(full_mesh_verts, ((0, 0), (0, 1)), constant_values=(0, 1))

mesh_door_indices = []
for prim_i in range(total_drawers_num):
    with open(os.path.join(bakedsdf_dir, "drawers", "results", f"drawer_{prim_i}.pkl"), 'rb') as f:
        prim_transform = pickle.load(f)
        interact_type = prim_transform["interact"]
        prim_transform = prim_transform["transform"]
    means_pad_transformed = full_mesh_verts_pad @ np.linalg.inv(prim_transform).T
    means_transformed = means_pad_transformed[:, :3] / means_pad_transformed[:, 3:]
    scale_limit = np.array([1e4 * door_depth_segment, 1, 1]).reshape(1, 3)
    means_transformed = means_transformed / scale_limit

    inside_y = (np.abs(means_transformed[:, 1:2]) < 0.5)
    inside_z = np.logical_and(
        means_transformed[:, 2:] < 0.48,
        means_transformed[:, 2:] > -0.5
    )
    if interact_type in ["1.1"]:
        inside_x = np.logical_and(
            means_transformed[:, 0:1] > -0.05,
            means_transformed[:, 0:1] < 0.3)
        # inside_y = np.logical_and(
        #     means_transformed[:, 1:2] < 0.5,
        #     means_transformed[:, 1:2] > -0.4)
    elif interact_type in ["1.2"]:
        inside_x = np.logical_and(
            means_transformed[:, 0:1] < 0.05,
            means_transformed[:, 0:1] > -0.3)
        # inside_y = np.logical_and(
        #     means_transformed[:, 1:2] < 0.4,
        #     means_transformed[:, 1:2] > -0.5)
    else:
        inside_x = np.logical_and(
            means_transformed[:, 0:1] > -0.05,
            means_transformed[:, 0:1] < 0.3)

        # inside_y = np.logical_and(
        #     means_transformed[:, 1:2] < 0.5,
        #     means_transformed[:, 1:2] > -0.4)

    inside = np.concatenate([inside_x, inside_y, inside_z], axis=-1)

    prim_i_means_indices = np.all(inside, axis=1).reshape(-1)
    prim_i_means_indices = torch.from_numpy(prim_i_means_indices).cuda()
    for prim_j in range(prim_i):
        prim_j_means_indices = mesh_door_indices[prim_j]
        prim_i_means_indices = torch.logical_and(prim_i_means_indices, torch.logical_not(prim_j_means_indices))
    print(f"prim_i_means_indices {prim_i}: {torch.count_nonzero(prim_i_means_indices)}")
    mesh_door_indices.append(prim_i_means_indices)

drawer_dir = os.path.join(bakedsdf_dir, "drawers", "results")
drawer_internal_dir = os.path.join(bakedsdf_dir, "drawers", "internal")

glcx = dr.RasterizeGLContext(output_db=False)

subprocess.run(f"rm -rf {output_dir}/*", shell=True)
os.makedirs(output_dir, exist_ok=True)
with torch.no_grad():
    for drawer_i in range(total_drawers_num):

        prim_i = drawer_i

        prim_mark = f"{drawer_i:0>2d}"
        trace_data_prim = trace_data[prim_mark]

        drawer_path = os.path.join(drawer_dir, f"drawer_{drawer_i}.pkl")
        with open(drawer_path, 'rb') as f:
            prim_info = pickle.load(f)
            prim_transform = prim_info["transform"]
            interact_type = prim_info["interact"]

        # remove the original mesh door from the scene mesh
        full_mesh_verts_pad_transformed = full_mesh_verts_pad @ np.linalg.inv(prim_transform).T
        full_mesh_verts_transformed = full_mesh_verts_pad_transformed[:, :3] / full_mesh_verts_pad_transformed[:, 3:]
        scale_limit = np.array([1e4 * 0.05, 1, 1]).reshape(1, 3)
        full_mesh_verts_transformed = full_mesh_verts_transformed / scale_limit
        drawer_sub_verts_local = np.all((np.abs(full_mesh_verts_transformed) < 0.5), axis=1).reshape(-1)
        door_verts, door_faces, _ = filter_mesh_from_vertices(drawer_sub_verts_local, textured_mesh_dict["verts"],
                                                              textured_mesh_dict["faces"])
        drawer_sub_verts_local = np.logical_not(drawer_sub_verts_local)
        no_drawer_mesh_verts, no_drawer_mesh_faces, _ = filter_mesh_from_vertices(drawer_sub_verts_local,
                                                                                  textured_mesh_dict["verts"],
                                                                                  textured_mesh_dict["faces"])

        # get the door transform
        scale, _, angles, trans, _ = transformations.decompose_matrix(prim_transform)
        prim_rotation = transformations.euler_matrix(axes='sxyz', *angles).reshape(4, 4)
        prim_translation = np.eye(4)
        prim_translation[:3, 3] = trans
        prim_rot_trans_original = prim_translation @ prim_rotation

        box_grid_mn = [1, 1]
        if interact_type in ["1.2", "3.2", "1.1", "3.1"]:
            if scale[1] > scale[2]:
                # box_grid_mn[0] += 1
                rto = scale[1] / scale[2]
                if abs(rto - int(rto)) < abs(rto - int(rto) - 1):
                    box_grid_mn[0] = int(rto)
                else:
                    box_grid_mn[0] = int(rto) + 1
            else:
                # box_grid_mn[1] += 1
                rto = scale[2] / scale[1]
                if abs(rto - int(rto)) < abs(rto - int(rto) - 1):
                    box_grid_mn[1] = int(rto)
                else:
                    box_grid_mn[1] = int(rto) + 1
        print("prim_i: ", prim_i, "box_grid_mn: ", box_grid_mn)

        door_depth = 0.005
        depth_offset_base = 1e-6
        
        if interact_type in ["1.2", "3.2"]:
            internal_inverse = True
            top = True
            depth = scale[1] * 1.5
            thickness = scale[1] * 1.5 / 50
            height_offset = 0.002 * scale[2]
            depth_offset = depth_offset_base * revolute_door_offset_scaling * depth

        elif interact_type in ["1.1", "3.1"]:
            internal_inverse = False
            top = True
            depth = scale[1] * 1.5
            thickness = scale[1] * 1.5 / 50
            height_offset = 0.002 * scale[2]
            depth_offset = depth_offset_base * revolute_door_offset_scaling * depth
        else:
            internal_inverse = False
            top = False
            depth = scale[1] * 1.2
            thickness = scale[1] * 1.2 / 40
            height_offset = 0.04 * scale[2]
            depth_offset = depth_offset_base * 2.5 * depth

        drawer_prim_internal = create_drawer_internal_box_mesh_occupancy(D=depth, Sy=scale[1], Sz=scale[2], thickness=thickness,
                                                               m=box_grid_mn[0], n=box_grid_mn[1],
                                                               internal_inverse=internal_inverse,
                                                               top=top, height_offset=height_offset, depth_offset=door_depth*2+depth_offset*2+0.001)

        uv_size = 1024
        internal_mesh = generate_uvs(drawer_prim_internal.vertices, drawer_prim_internal.faces)

        internal_mesh = subdivide_textured_mesh(internal_mesh)
        drawer_prim_internal.vertices = internal_mesh["verts"]
        drawer_prim_internal.faces = internal_mesh["faces"]

        normal_map = get_normal_map(internal_mesh, uv_size, glcx)
        normal_map_image = np.clip(normal_map.numpy() * 255.0, 0, 255).astype(np.uint8)

        normal_map_edge = edge_detect(normal_map_image)
        sketch = np.array(Image.fromarray(normal_map_edge).resize((1024, 1024)), dtype=np.uint8)

        sketch = add_random_curve(sketch)

        # door mesh
        drawer_door_mesh = create_drawer_door(D=door_depth, Sy=scale[1], Sz=scale[2], internal_inverse=internal_inverse, height_offset=height_offset, depth_offset=0.+depth_offset)
        uv_size = 1024
        drawer_door = generate_uvs(drawer_door_mesh.vertices, drawer_door_mesh.faces)

        drawer_door = subdivide_textured_mesh(drawer_door)
        drawer_door_mesh.vertices = drawer_door["verts"]
        drawer_door_mesh.faces = drawer_door["faces"]

        normal_map_door = get_normal_map(drawer_door, uv_size, glcx)
        normal_map_image_door = np.clip(normal_map_door.numpy() * 255.0, 0, 255).astype(np.uint8)

        normal_map_edge_door = edge_detect(normal_map_image_door)
        sketch_door = np.array(Image.fromarray(normal_map_edge_door).resize((1024, 1024)), dtype=np.uint8)

        sketch_door = add_random_curve(sketch_door)

        # internal

        drawer_prim_internal_verts = np.array(drawer_prim_internal.vertices)

        drawer_prim_internal_boundary = np.stack(
            [drawer_prim_internal_verts.min(axis=0), drawer_prim_internal_verts.max(axis=0)], axis=0).reshape(2, 3)

        prim_rot_trans_original_inverse = np.linalg.inv(prim_rot_trans_original)
        full_mesh_verts_pad_prim_internal_transformed = full_mesh_verts_pad @ prim_rot_trans_original_inverse.T
        full_mesh_verts_prim_internal_transformed = full_mesh_verts_pad_prim_internal_transformed[:,
                                                    :3] / full_mesh_verts_pad_prim_internal_transformed[:, 3:]

        full_mesh_verts_prim_internal_inside = np.logical_and(
            np.all(full_mesh_verts_prim_internal_transformed > drawer_prim_internal_boundary[0:1], axis=-1),
            np.all(full_mesh_verts_prim_internal_transformed < drawer_prim_internal_boundary[1:2], axis=-1)
        )

        drawer_prim_internal_verts_pad = np.pad(drawer_prim_internal_verts, ((0, 0), (0, 1)),
                                                constant_values=(0, 1))
        drawer_prim_internal_verts_transformed_pad = drawer_prim_internal_verts_pad @ prim_rot_trans_original.T
        drawer_prim_internal_verts_transformed = drawer_prim_internal_verts_transformed_pad[:,
                                                 :3] / drawer_prim_internal_verts_transformed_pad[:, 3:]
        drawer_prim_internal.vertices = drawer_prim_internal_verts_transformed

        internal_mesh["verts"] = drawer_prim_internal.vertices

        # door
        drawer_door_verts = drawer_door["verts"]
        drawer_door_verts_pad = np.pad(drawer_door_verts, ((0, 0), (0, 1)),
                                                constant_values=(0, 1))
        drawer_door_verts_transformed_pad = drawer_door_verts_pad @ prim_rot_trans_original.T
        drawer_door_verts_transformed = drawer_door_verts_transformed_pad[:,
                                                 :3] / drawer_door_verts_transformed_pad[:, 3:]
        drawer_door_mesh.vertices = drawer_door_verts_transformed
        drawer_door["verts"] = drawer_door_mesh.vertices

        internal_verts = np.array(drawer_prim_internal.vertices)
        internal_faces = np.array(drawer_prim_internal.faces).astype(np.int32)

        uv_size = 1024
        internal_verts_torch = torch.from_numpy(internal_verts).float().cuda()
        internal_faces_torch = torch.from_numpy(internal_faces).float().cuda()

        internal_mesh_faces_texture = torch.from_numpy(internal_mesh["faces_texture"]).cuda()
        internal_mesh_verts_texture = torch.from_numpy(internal_mesh["verts_texture"]).cuda()
        internal_mesh_p3d = Meshes(
            verts=[internal_verts_torch],
            faces=[internal_faces_torch],
            # textures=internal_mesh_uvtex
        ).to("cuda")

        # textured mesh composition
        textured_mesh_p3d = Meshes(
            verts=[textured_mesh_verts],
            faces=[textured_mesh_faces],
            textures=textured_mesh_uvtex,
        ).to("cuda")
        F_cnt = textured_mesh_faces.shape[0]
        door_keep_faces = filter_faces_from_verts(mesh_door_indices[drawer_i].cpu(), textured_mesh_faces.cpu())
        base_keep_faces = torch.logical_not(door_keep_faces)

        full_mesh_verts_prim_internal_inside_torch = torch.from_numpy(full_mesh_verts_prim_internal_inside.copy())
        base_keep_faces = torch.logical_and(base_keep_faces, torch.logical_not(
            filter_faces_from_verts(full_mesh_verts_prim_internal_inside_torch.cpu(), textured_mesh_faces.cpu())))

        door_keep_faces_idx = torch.arange(F_cnt)[door_keep_faces]
        base_keep_faces_idx = torch.arange(F_cnt)[base_keep_faces]

        textured_mesh_base_p3d = textured_mesh_p3d.submeshes([[base_keep_faces_idx]])
        textured_mesh_door_p3d = textured_mesh_p3d.submeshes([[door_keep_faces_idx]])

        door_verts_original = textured_mesh_door_p3d.verts_packed().reshape(-1, 3).clone()
        internal_verts_original = internal_mesh_p3d.verts_packed().reshape(-1, 3).clone()

        validation_threshold = 0.2

        train_cameras = pipeline.datamanager.train_dataset.cameras
        num_cameras = train_cameras.camera_to_worlds.shape[0]

        max_area_glb = 0.
        camera_params = None
        for camera_i in tqdm(range(num_cameras)):

            target_camera = train_cameras[camera_i:camera_i+1]
            fx = float(target_camera.fx)
            fy = float(target_camera.fy)
            cx = float(target_camera.cx)
            cy = float(target_camera.cy)
            H = int(target_camera.height)
            W = int(target_camera.width)
            c2w = target_camera.camera_to_worlds.reshape(3, 4).clone()
            resolution = [H, W]

            # znear = 0.01
            # zfar = 1e10
            # fov_x = 2 * np.arctan(W / (2 * fx)) * (180.0 / np.pi)  # in degrees
            # fov_y = 2 * np.arctan(H / (2 * fy)) * (180.0 / np.pi)  # in degrees
            # aspect_ratio = (W / H) * (fov_y / fov_x)
            # fov = fov_y  # average the fovs
            #
            # c2w[:3, 1:3] *= -1 # gl -> cv
            # c2w[:3, 0:2] *= -1 # cv -> p3d
            # w2c = torch.inverse(torch.cat([c2w, torch.tensor([0, 0, 0, 1.0]).reshape(1, 4)], dim=0))
            # R = w2c[:3, :3].T.reshape(1, 3, 3)
            # tvec = w2c[:3, 3].reshape(1, 3)
            #
            # camera_p3d = FoVPerspectiveCameras(
            #     znear=znear,
            #     zfar=zfar,
            #     aspect_ratio=aspect_ratio,
            #     fov=fov,
            #     degrees=True,
            #     R=R,
            #     T=tvec,
            #     device='cuda'
            # )

            intrinsic = torch.from_numpy(
                np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
            ).reshape(3, 3).float()
            c2w[:3, 1:3] *= -1  # gl -> cv
            w2c = torch.inverse(torch.cat([c2w, torch.tensor([0, 0, 0, 1.0]).reshape(1, 4)], dim=0))

            camera_p3d = cameras_from_opencv_projection(
                R=w2c[..., :3, :3].reshape(1, 3, 3),
                tvec=w2c[..., :3, 3].reshape(1, 3),
                camera_matrix=intrinsic.reshape(1, 3, 3),
                image_size=torch.from_numpy(np.array(resolution).astype(np.int32)).reshape(1, 2)
            ).to('cuda')

            composed_mesh = [textured_mesh_base_p3d, textured_mesh_door_p3d]

            lights = AmbientLights(device='cuda')

            raster_settings = RasterizationSettings(
                image_size=resolution,
                blur_radius=1e-10,
                faces_per_pixel=1,
                perspective_correct=False,
                cull_backfaces=False
            )

            renderer = MeshRendererWithFragments(
                rasterizer=MeshRasterizer(
                    cameras=camera_p3d,
                    raster_settings=raster_settings
                ),
                # shader=SoftSilhouetteShader()
                shader=HardPhongShader(device="cuda", cameras=camera_p3d, lights=lights)
            )

            output = depth_blend(renderer, composed_mesh, resolution)

            internal_face_cnt = \
                torch.unique(output["pix_to_face"][len(composed_mesh) - 1][output["idx"] == len(composed_mesh)]).shape[0]

            if internal_face_cnt > validation_threshold * textured_mesh_door_p3d.faces_packed().shape[0]:
                mask = (output["idx"] == len(composed_mesh)).detach().cpu().numpy()
                # rgb = np.clip(output["rgb"].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
                # Image.fromarray(rgb).save(os.path.join(output_dir, f"{prim_i:0>2d}_meshrgb.png"))

                max_area, max_rectangle = max_rectangle_in_bool_matrix(mask)
                h_min, w_min, h_max, w_max = max_rectangle

                if max_area > max_area_glb:
                    max_area_glb = max_area
                    camera_params = {
                        'mask': mask,
                        "max_rectangle": max_rectangle,
                        "camera": target_camera,
                        "camera_i": camera_i,
                    }

                    break

        os.makedirs(os.path.join(output_dir, f"{prim_i:0>2d}"), exist_ok=True)
        assert camera_params is not None
        target_camera = camera_params['camera']
        h_min, w_min, h_max, w_max = camera_params["max_rectangle"]
        camera_i = camera_params["camera_i"]

        # output = model.get_outputs(target_camera.to("cuda"))
        rendering = np.array(Image.open(pipeline.datamanager.train_dataset.image_filenames[camera_i])).astype(np.uint8)

        input_image_palette = Image.fromarray(rendering[h_min:h_max, w_min:w_max]).resize((1024, 1024))
        # color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2)
        # input_image_palette = color_jitter(input_image_palette)
        input_image_palette.save(os.path.join(output_dir, f"{prim_i:0>2d}", "maskrgb.png"))
        input_image_emb = None
        prompt = "A material of wood"
        # prompt = "A material of steel"

        Image.fromarray(sketch).save(os.path.join(output_dir, f"{prim_i:0>2d}", "sketch.png"))
        Image.fromarray(sketch_door).save(os.path.join(output_dir, f"{prim_i:0>2d}", "sketch_door.png"))

        num_samples = 1
        image_resolution = 512
        guidance_scale = 5.0
        ddim_steps = 100
        seed = 42
        eta = 0.0

        # seed_everything(seed)

        # internal
        result_tex = run_generation(
            matfuse_model, input_image_emb, input_image_palette, sketch, prompt,
            num_samples, image_resolution, ddim_steps, seed, eta, guidance_scale,
            save_dir=os.path.join(output_dir, f"{prim_i:0>2d}")
        )[-1]

        result_tex = pseudo_render_texture_map(result_tex)

        save_obj(os.path.join(output_dir, f"{prim_i:0>2d}", f"mesh_{prim_i:0>2d}.obj"),
                 verts=torch.from_numpy(internal_mesh["verts"]),
                 faces=torch.from_numpy(internal_mesh["faces"]),
                 verts_uvs=torch.from_numpy(internal_mesh["verts_texture"]),
                 faces_uvs=torch.from_numpy(internal_mesh["faces_texture"]),
                 texture_map=torch.from_numpy(np.array(result_tex) / 255.0)
                 )

        # door
        result_tex_door = run_generation(
            matfuse_model, input_image_emb, input_image_palette, sketch_door, prompt,
            num_samples, image_resolution, ddim_steps, seed, eta, guidance_scale,
            save_dir=os.path.join(output_dir, f"{prim_i:0>2d}")
        )[-1]

        result_tex_door = pseudo_render_texture_map(result_tex_door)

        save_obj(os.path.join(output_dir, f"{prim_i:0>2d}", f"mesh_door_{prim_i:0>2d}.obj"),
                 verts=torch.from_numpy(drawer_door["verts"]),
                 faces=torch.from_numpy(drawer_door["faces"]),
                 verts_uvs=torch.from_numpy(drawer_door["verts_texture"]),
                 faces_uvs=torch.from_numpy(drawer_door["faces_texture"]),
                 texture_map=torch.from_numpy(np.array(result_tex_door) / 255.0)
                 )

        # breakpoint()

