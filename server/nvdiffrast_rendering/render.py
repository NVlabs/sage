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
import nvdiffrast.torch as dr
import torch
import trimesh
import torch.nn.functional as F

def render_textured_mesh(
        barycentric_coords,  # (H, W, 3)
        triangle_ids,  # (H, W)
        mask,  # (H, W)
        vt,  # (V', 2) texture vertices
        ft,  # (F', 3) texture faces
        texture_map,  # (H', W', 3) texture image
):
    # Get image dimensions
    H, W = mask.shape

    # Initialize output RGB image with zeros
    rgb = torch.zeros(H, W, 3, device=barycentric_coords.device)

    # Get the valid pixels (where mask is True)
    valid_pixels = torch.where(mask)

    if valid_pixels[0].numel() == 0:
        return rgb  # No valid pixels to render

    # Get triangle IDs and barycentric coordinates for valid pixels
    valid_triangle_ids = triangle_ids[valid_pixels]
    valid_bary_coords = barycentric_coords[valid_pixels]

    # Get texture face indices for each valid pixel
    texture_faces = ft[valid_triangle_ids]  # (N, 3)

    # Get texture vertex indices
    texture_vertex_indices = texture_faces

    # Get texture coordinates (u,v) for each vertex of each face
    texture_uvs = vt[texture_vertex_indices]  # (N, 3, 2)

    # Use barycentric coordinates to interpolate texture coordinates
    bary_coords_expanded = valid_bary_coords.unsqueeze(-1)  # (N, 3, 1)

    # Interpolate texture coordinates using barycentric weights
    interpolated_uvs = (texture_uvs * bary_coords_expanded).sum(dim=1)  # (N, 2)

    # Scale to texture image coordinates
    H_tex, W_tex = texture_map.shape[0], texture_map.shape[1]

    # Method 2: Implement bilinear interpolation manually
    u = interpolated_uvs[:, 0] * (W_tex - 1)
    v = interpolated_uvs[:, 1] * (H_tex - 1)

    # Get the four neighboring pixels
    u0 = torch.floor(u).long()
    v0 = torch.floor(v).long()
    u1 = torch.min(u0 + 1, torch.tensor(W_tex - 1, device=u.device))
    v1 = torch.min(v0 + 1, torch.tensor(H_tex - 1, device=v.device))

    # Calculate interpolation weights
    w_u1 = u - u0.float()
    w_v1 = v - v0.float()
    w_u0 = 1 - w_u1
    w_v0 = 1 - w_v1

    # Sample colors at the four corners
    c00 = texture_map[v0, u0]  # (N, 3)
    c01 = texture_map[v0, u1]  # (N, 3)
    c10 = texture_map[v1, u0]  # (N, 3)
    c11 = texture_map[v1, u1]  # (N, 3)

    # Perform bilinear interpolation
    c0 = w_u0.unsqueeze(-1) * c00 + w_u1.unsqueeze(-1) * c01  # Interpolate top row
    c1 = w_u0.unsqueeze(-1) * c10 + w_u1.unsqueeze(-1) * c11  # Interpolate bottom row
    c = w_v0.unsqueeze(-1) * c0 + w_v1.unsqueeze(-1) * c1  # Interpolate between rows

    # Assign interpolated colors to valid pixels in output image
    rgb[valid_pixels] = c

    return rgb

def rasterize_mesh_with_uv(mesh_dict, projection, glctx, resolution, c2w=None):

    vertices = mesh_dict['vertices']
    pos_idx = mesh_dict['pos_idx']
    vt = mesh_dict['vt']
    ft = mesh_dict['ft']
    texture_map = mesh_dict['texture_map']

    projection = projection.to(vertices.device)

    vertices_clip = torch.matmul(vertices, torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, pos_idx, resolution=resolution)

    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)
    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)

    bary_coords = rast_out[..., :2].reshape(H, W, 2)
    bary_coords = torch.cat([bary_coords, 1 - bary_coords.sum(dim=-1, keepdim=True)], dim=-1)

    rgb = render_textured_mesh(
        bary_coords, triangle_id, valid, vt, ft, texture_map)

    if c2w is None:
        depth = None
    else:
        w2c = torch.inverse(torch.cat([c2w[:3, :4], torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(c2w.device)], dim=0))
        vert_cam = (w2c.to(vertices.device) @ vertices.permute(1, 0)).permute(1, 0)
        vert_cam = vert_cam[..., :3] / vert_cam[..., 3:4]
        depth = vert_cam[..., -1:]
        depth_inverse = 1 / (depth + 1e-20)
        depth_inverse, _ = dr.interpolate(depth_inverse.unsqueeze(0).contiguous(), rast_out, pos_idx)
        depth = 1 / (depth_inverse + 1e-20)
        depth = depth.reshape(H, W)


    return valid, triangle_id, depth, rgb


def rasterize_mesh(mesh_dict, projection, glctx, resolution, c2w=None):

    vertices = mesh_dict['vertices']
    pos_idx = mesh_dict['pos_idx']

    vertices_clip = torch.matmul(vertices, torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, pos_idx, resolution=resolution)

    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)
    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)


    if c2w is None:
        depth = None
    else:
        w2c = torch.inverse(torch.cat([c2w[:3, :4], torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(c2w.device)], dim=0))
        vert_cam = (w2c.to(vertices.device) @ vertices.permute(1, 0)).permute(1, 0)
        vert_cam = vert_cam[..., :3] / vert_cam[..., 3:4]
        depth = vert_cam[..., -1:]
        depth_inverse = 1 / (depth + 1e-20)
        depth_inverse, _ = dr.interpolate(depth_inverse.unsqueeze(0).contiguous(), rast_out, pos_idx)
        depth = 1 / (depth_inverse + 1e-20)
        depth = depth.reshape(H, W)


    return valid, triangle_id, depth


def rasterize_mesh_dict_list(mesh_dict_list, projection, glctx, resolution, c2w=None):
    faces_cnts = [0]
    all_objs_meshes = []

    for mesh_dict in mesh_dict_list:
        mesh = trimesh.Trimesh(
            mesh_dict['vertices'][:, :3].cpu().numpy(),
            mesh_dict['pos_idx'].cpu().numpy(), process=False)

        all_objs_meshes.append(mesh)
        faces_cnts.append(faces_cnts[-1] + mesh.faces.shape[0])

    all_objs_mesh = trimesh.util.concatenate(all_objs_meshes)
    mesh_dict = {
        'vertices': F.pad(
            torch.from_numpy(all_objs_mesh.vertices).float().to("cuda").contiguous(),
            pad=(0, 1), value=1.0, mode='constant'),
        'pos_idx': torch.from_numpy(all_objs_mesh.faces).int().to("cuda").contiguous()
    }

    valid, triangle_id, depth = rasterize_mesh(mesh_dict, projection, glctx, resolution, c2w)
    instance_id = torch.zeros_like(triangle_id)
    for mesh_i in range(len(faces_cnts) - 1):
        instance_id[torch.logical_and(triangle_id >= faces_cnts[mesh_i], triangle_id < faces_cnts[mesh_i + 1])] = mesh_i

    # instance_id[torch.logical_not(valid)] = -1

    return valid, triangle_id, depth, instance_id

def rasterize_mesh_dict_list_with_uv(mesh_dict_list, projection, glctx, resolution, c2w=None):
    valid, _, _, instance_id = rasterize_mesh_dict_list(mesh_dict_list, projection, glctx, resolution, c2w)

    H, W = resolution
    rgb = torch.zeros(H, W, 3).cuda()
    for mesh_i, mesh_dict in enumerate(mesh_dict_list):
        _, _, _, rgb_obj_i = rasterize_mesh_with_uv(mesh_dict, projection, glctx, resolution, c2w)
        rgb[instance_id == mesh_i] = rgb_obj_i[instance_id == mesh_i]

    return valid, instance_id, rgb

def rasterize_mesh_dict_list_with_uv_efficient(mesh_dict_list, projection, glctx, resolution, c2w=None):
    if not mesh_dict_list:
        H, W = resolution
        if c2w is None:
            return torch.zeros(H, W, dtype=torch.bool).cuda(), torch.zeros(H, W, dtype=torch.long).cuda(), torch.zeros(H, W, 3).cuda()
        else:
            return torch.zeros(H, W, dtype=torch.bool).cuda(), torch.zeros(H, W, dtype=torch.long).cuda(), torch.zeros(H, W, 3).cuda(), torch.zeros(H, W).cuda()
    if c2w is not None:
        c2w = c2w.to(torch.device("cuda"))

    # Concatenate all meshes and track their texture information
    faces_cnts = [0]
    all_vertices = []
    all_faces = []
    all_vts = []
    all_fts = []
    all_texture_maps = []
    
    vertex_offset = 0
    vt_offset = 0
    
    for mesh_dict in mesh_dict_list:
        vertices = mesh_dict['vertices']
        faces = mesh_dict['pos_idx']
        vt = mesh_dict['vt']
        ft = mesh_dict['ft']
        texture_map = mesh_dict['texture_map']
        
        # Append vertices and texture vertices
        all_vertices.append(vertices)
        all_vts.append(vt)
        
        # Adjust face indices to account for vertex offset
        adjusted_faces = faces + vertex_offset
        adjusted_fts = ft + vt_offset
        
        all_faces.append(adjusted_faces)
        all_fts.append(adjusted_fts)
        all_texture_maps.append(texture_map)
        
        faces_cnts.append(faces_cnts[-1] + faces.shape[0])
        
        # Update offsets for next mesh
        vertex_offset += vertices.shape[0]
        vt_offset += vt.shape[0]
    
    # Concatenate all geometry and texture data
    all_vertices = torch.cat(all_vertices, dim=0)
    all_faces = torch.cat(all_faces, dim=0)
    all_vts = torch.cat(all_vts, dim=0)
    all_fts = torch.cat(all_fts, dim=0)
    
    # Single rasterization pass for all meshes
    projection = projection.to(all_vertices.device)
    vertices_clip = torch.matmul(all_vertices, torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, all_faces, resolution=resolution)
    
    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)
    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)
    
    # Calculate barycentric coordinates
    bary_coords = rast_out[..., :2].reshape(H, W, 2)
    bary_coords = torch.cat([bary_coords, 1 - bary_coords.sum(dim=-1, keepdim=True)], dim=-1)
    
    # Create instance ID map to track which mesh each pixel belongs to
    instance_id = torch.zeros_like(triangle_id)
    for mesh_i in range(len(faces_cnts) - 1):
        mask = torch.logical_and(triangle_id >= faces_cnts[mesh_i], triangle_id < faces_cnts[mesh_i + 1])
        instance_id[mask] = mesh_i
    
    # Compute depth if c2w is provided
    if c2w is None:
        depth = None
    else:
        w2c = torch.inverse(torch.cat([c2w[:3, :4], torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(c2w.device)], dim=0))
        vert_cam = (w2c.to(all_vertices.device) @ all_vertices.permute(1, 0)).permute(1, 0)
        vert_cam = vert_cam[..., :3] / vert_cam[..., 3:4]
        depth = vert_cam[..., -1:]
        depth, _ = dr.interpolate(depth.unsqueeze(0).contiguous(), rast_out, all_faces)
        depth = depth.reshape(H, W)
    
    # Initialize RGB output
    rgb = torch.zeros(H, W, 3, device=all_vertices.device)
    
    # Get valid pixels
    valid_pixels = torch.where(valid)
    if valid_pixels[0].numel() == 0:
        if c2w is None:
            return valid, instance_id, rgb
        else:
            return valid, instance_id, rgb, depth
    
    # Get data for valid pixels
    valid_triangle_ids = triangle_id[valid_pixels]
    valid_bary_coords = bary_coords[valid_pixels]
    valid_instance_ids = instance_id[valid_pixels]
    
    # Process each mesh's texture separately but efficiently
    for mesh_i, mesh_dict in enumerate(mesh_dict_list):
        # Find pixels belonging to this mesh
        mesh_mask = valid_instance_ids == mesh_i
        if not mesh_mask.any():
            continue
            
        # Get triangle IDs and barycentric coordinates for this mesh's pixels
        mesh_triangle_ids = valid_triangle_ids[mesh_mask]
        mesh_bary_coords = valid_bary_coords[mesh_mask]
        
        # Adjust triangle IDs to be relative to this mesh (subtract face offset)
        mesh_triangle_ids = mesh_triangle_ids - faces_cnts[mesh_i]
        
        # Get texture information for this mesh
        vt = mesh_dict['vt']
        ft = mesh_dict['ft']
        texture_map = mesh_dict['texture_map']
        
        # Get texture face indices for valid pixels of this mesh
        texture_faces = ft[mesh_triangle_ids]  # (N, 3)
        
        # Get texture coordinates (u,v) for each vertex of each face
        texture_uvs = vt[texture_faces]  # (N, 3, 2)
        
        # Use barycentric coordinates to interpolate texture coordinates
        bary_coords_expanded = mesh_bary_coords.unsqueeze(-1)  # (N, 3, 1)
        interpolated_uvs = (texture_uvs * bary_coords_expanded).sum(dim=1)  # (N, 2)
        
        # Bilinear interpolation in texture space
        H_tex, W_tex = texture_map.shape[0], texture_map.shape[1]
        u = interpolated_uvs[:, 0] * (W_tex - 1)
        v = interpolated_uvs[:, 1] * (H_tex - 1)
        
        # Get the four neighboring pixels
        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()
        u1 = torch.min(u0 + 1, torch.tensor(W_tex - 1, device=u.device))
        v1 = torch.min(v0 + 1, torch.tensor(H_tex - 1, device=v.device))
        
        # Calculate interpolation weights
        w_u1 = u - u0.float()
        w_v1 = v - v0.float()
        w_u0 = 1 - w_u1
        w_v0 = 1 - w_v1
        
        # Sample colors at the four corners
        c00 = texture_map[v0, u0]  # (N, 3)
        c01 = texture_map[v0, u1]  # (N, 3)
        c10 = texture_map[v1, u0]  # (N, 3)
        c11 = texture_map[v1, u1]  # (N, 3)
        
        # Perform bilinear interpolation
        c0 = w_u0.unsqueeze(-1) * c00 + w_u1.unsqueeze(-1) * c01  # Interpolate top row
        c1 = w_u0.unsqueeze(-1) * c10 + w_u1.unsqueeze(-1) * c11  # Interpolate bottom row
        c = w_v0.unsqueeze(-1) * c0 + w_v1.unsqueeze(-1) * c1  # Interpolate between rows
        
        # Assign interpolated colors to valid pixels in output image
        mesh_valid_pixels = (valid_pixels[0][mesh_mask], valid_pixels[1][mesh_mask])
        rgb[mesh_valid_pixels] = c
    
    if c2w is None:
        return valid, instance_id, rgb
    else:
        return valid, instance_id, rgb, depth



def rasterize_mesh_dict_list_with_uv_efficient_uv_diff(mesh_dict_list, projection, glctx, resolution, c2w=None):
    if not mesh_dict_list:
        H, W = resolution
        if c2w is None:
            return torch.zeros(H, W, dtype=torch.bool).cuda(), torch.zeros(H, W, dtype=torch.long).cuda(), torch.zeros(H, W, 3).cuda()
        else:
            return torch.zeros(H, W, dtype=torch.bool).cuda(), torch.zeros(H, W, dtype=torch.long).cuda(), torch.zeros(H, W, 3).cuda(), torch.zeros(H, W).cuda()
    if c2w is not None:
        c2w = c2w.to(torch.device("cuda"))

    # Concatenate all meshes and track their texture information
    faces_cnts = [0]
    all_vertices = []
    all_faces = []
    all_vts = []
    all_fts = []
    all_texture_maps = []
    
    vertex_offset = 0
    vt_offset = 0
    
    for mesh_dict in mesh_dict_list:
        vertices = mesh_dict['vertices']
        faces = mesh_dict['pos_idx']
        vt = mesh_dict['vt']
        ft = mesh_dict['ft']
        texture_map = mesh_dict['texture_map']
        
        # Append vertices and texture vertices
        all_vertices.append(vertices)
        all_vts.append(vt)
        
        # Adjust face indices to account for vertex offset
        adjusted_faces = faces + vertex_offset
        adjusted_fts = ft + vt_offset
        
        all_faces.append(adjusted_faces)
        all_fts.append(adjusted_fts)
        all_texture_maps.append(texture_map)
        
        faces_cnts.append(faces_cnts[-1] + faces.shape[0])
        
        # Update offsets for next mesh
        vertex_offset += vertices.shape[0]
        vt_offset += vt.shape[0]
    
    # Concatenate all geometry and texture data
    all_vertices = torch.cat(all_vertices, dim=0)
    all_faces = torch.cat(all_faces, dim=0)
    all_vts = torch.cat(all_vts, dim=0)
    all_fts = torch.cat(all_fts, dim=0)
    
    # Single rasterization pass for all meshes
    projection = projection.to(all_vertices.device)
    vertices_clip = torch.matmul(all_vertices, torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, all_faces, resolution=resolution)
    
    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)
    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)
    
    # Calculate barycentric coordinates
    bary_coords = rast_out[..., :2].reshape(H, W, 2)
    bary_coords = torch.cat([bary_coords, 1 - bary_coords.sum(dim=-1, keepdim=True)], dim=-1)
    
    # Create instance ID map to track which mesh each pixel belongs to
    instance_id = torch.zeros_like(triangle_id)
    for mesh_i in range(len(faces_cnts) - 1):
        mask = torch.logical_and(triangle_id >= faces_cnts[mesh_i], triangle_id < faces_cnts[mesh_i + 1])
        instance_id[mask] = mesh_i
    
    # Compute depth if c2w is provided
    if c2w is None:
        depth = None
    else:
        w2c = torch.inverse(torch.cat([c2w[:3, :4], torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(c2w.device)], dim=0))
        vert_cam = (w2c.to(all_vertices.device) @ all_vertices.permute(1, 0)).permute(1, 0)
        vert_cam = vert_cam[..., :3] / vert_cam[..., 3:4]
        depth = vert_cam[..., -1:]
        depth, _ = dr.interpolate(depth.unsqueeze(0).contiguous(), rast_out, all_faces)
        depth = depth.reshape(H, W)
    
    # Initialize RGB output
    rgb = torch.zeros(H, W, 3, device=all_vertices.device)
    
    # Get valid pixels
    valid_pixels = torch.where(valid)
    if valid_pixels[0].numel() == 0:
        if c2w is None:
            return valid, instance_id, rgb
        else:
            return valid, instance_id, rgb, depth
    
    # Get data for valid pixels
    valid_triangle_ids = triangle_id[valid_pixels]
    valid_bary_coords = bary_coords[valid_pixels]
    valid_instance_ids = instance_id[valid_pixels]
    
    # Process each mesh's texture separately but efficiently
    for mesh_i, mesh_dict in enumerate(mesh_dict_list):
        # Find pixels belonging to this mesh
        mesh_mask = valid_instance_ids == mesh_i
        if not mesh_mask.any():
            continue
            
        # Get triangle IDs and barycentric coordinates for this mesh's pixels
        mesh_triangle_ids = valid_triangle_ids[mesh_mask]
        mesh_bary_coords = valid_bary_coords[mesh_mask]
        
        # Adjust triangle IDs to be relative to this mesh (subtract face offset)
        mesh_triangle_ids = mesh_triangle_ids - faces_cnts[mesh_i]
        
        # Get texture information for this mesh
        vt = mesh_dict['vt']
        ft = mesh_dict['ft']
        texture_map = mesh_dict['texture_map']
        
        # Get texture face indices for valid pixels of this mesh
        texture_faces = ft[mesh_triangle_ids]  # (N, 3)
        
        # Get texture coordinates (u,v) for each vertex of each face
        texture_uvs = vt[texture_faces]  # (N, 3, 2)
        
        # Use barycentric coordinates to interpolate texture coordinates
        bary_coords_expanded = mesh_bary_coords.unsqueeze(-1)  # (N, 3, 1)
        interpolated_uvs = (texture_uvs * bary_coords_expanded).sum(dim=1)  # (N, 2)
        
        # Bilinear interpolation in texture space
        H_tex, W_tex = texture_map.shape[0], texture_map.shape[1]
        u = interpolated_uvs[:, 0] * (W_tex - 1)
        v = interpolated_uvs[:, 1] * (H_tex - 1)
        
        # Get the four neighboring pixels
        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()
        u1 = torch.min(u0 + 1, torch.tensor(W_tex - 1, device=u.device))
        v1 = torch.min(v0 + 1, torch.tensor(H_tex - 1, device=v.device))
        
        # Calculate interpolation weights
        w_u1 = u - u0.float()
        w_v1 = v - v0.float()
        w_u0 = 1 - w_u1
        w_v0 = 1 - w_v1
        
        # Sample colors at the four corners
        c00 = texture_map[v0, u0]  # (N, 3)
        c01 = texture_map[v0, u1]  # (N, 3)
        c10 = texture_map[v1, u0]  # (N, 3)
        c11 = texture_map[v1, u1]  # (N, 3)
        
        # Perform bilinear interpolation
        c0 = w_u0.unsqueeze(-1) * c00 + w_u1.unsqueeze(-1) * c01  # Interpolate top row
        c1 = w_u0.unsqueeze(-1) * c10 + w_u1.unsqueeze(-1) * c11  # Interpolate bottom row
        c = w_v0.unsqueeze(-1) * c0 + w_v1.unsqueeze(-1) * c1  # Interpolate between rows
        
        # Assign interpolated colors to valid pixels in output image
        mesh_valid_pixels = (valid_pixels[0][mesh_mask], valid_pixels[1][mesh_mask])
        rgb[mesh_valid_pixels] = c
    
    if c2w is None:
        return valid, instance_id, rgb
    else:
        return valid, instance_id, rgb, depth

