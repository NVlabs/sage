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
import os

from plyfile import PlyData

import omni
# import argparse
# from isaaclab.app import AppLauncher
# parser = argparse.ArgumentParser()
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()
# # args_cli.headless = True
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app
from isaacsim import SimulationApp


simulation_app = SimulationApp({
    "headless": False,
    "multi_gpu": False,
})
import trimesh
import numpy as np
# simulation_app = None

# def start_simulation_app(headless=True):

#     global simulation_app
#     if simulation_app is None:
#         simulation_app = SimulationApp({"headless": headless, "multi_gpu": False})

def detect_collision(base_meshes, test_mesh):
    """
    Detect collisions between a test mesh and a series of base meshes.
    Uses edge-based ray casting to detect intersections.

    Parameters:
    -----------
    base_meshes : List of tuples, each containing (vertices, faces)
        List of base meshes to check against.
        vertices: (n, 3) float32 - Vertex coordinates
        faces: (m, 3) int32 - Face indices
        face_normals: (m, 3) float32 - Face normals (optional, can be None)

    test_mesh : Tuple of (vertices, faces)
        The mesh to test for collisions.
        vertices: (n, 3) float32 - Vertex coordinates
        faces: (m, 3) int32 - Face indices

    Returns:
    --------
    contact_points : np.ndarray
        (k, 3) array of contact point coordinates
    contact_mesh_id : np.ndarray
        (k,) array of indices indicating which base mesh had the contact
    contact_face_id : np.ndarray
        (k,) array of face indices in the base mesh
    """
    # Convert test mesh to trimesh object
    test_vertices, test_faces = test_mesh
    test_trimesh = trimesh.Trimesh(vertices=test_vertices, faces=test_faces)

    # Extract edges from test mesh
    edges = test_trimesh.edges_unique

    # Get edge vertices
    edge_points = test_vertices[edges]

    # Create ray origins and directions from edges
    ray_origins = edge_points[:, 0]
    ray_directions = edge_points[:, 1] - edge_points[:, 0]

    # Normalize ray directions
    ray_lengths = np.linalg.norm(ray_directions, axis=1)
    ray_directions = ray_directions / ray_lengths[:, np.newaxis]

    # Lists to store contact information
    all_contact_points = []
    all_contact_mesh_ids = []
    all_contact_face_ids = []
    all_contact_face_normals = []

    # Check collision with each base mesh
    for mesh_id, (base_vertices, base_faces, base_face_normals) in enumerate(base_meshes):
        # Create trimesh object for the base mesh
        base_trimesh = trimesh.Trimesh(vertices=base_vertices, faces=base_faces)

        # Use ray_mesh intersection to find contacts
        # locations, index_ray, index_tri = base_trimesh.ray.intersects_location(
        #     ray_origins=ray_origins,
        #     ray_directions=ray_directions
        # )

        locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(base_trimesh).intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        if len(locations) > 0:
            # Calculate distances from ray origins to intersection points
            distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)

            # Only consider intersections that fall within the edge length
            valid_indices = distances <= ray_lengths[index_ray]

            if np.any(valid_indices):
                contact_points = locations[valid_indices]
                contact_face_ids = index_tri[valid_indices]
                contact_mesh_ids = np.full(len(contact_points), mesh_id)
                contact_normals = base_face_normals[contact_face_ids]

                all_contact_points.append(contact_points)
                all_contact_mesh_ids.append(contact_mesh_ids)
                all_contact_face_ids.append(contact_face_ids)
                all_contact_face_normals.append(contact_normals)

    # Combine results from all base meshes
    if all_contact_points:
        contact_points = np.vstack(all_contact_points)
        contact_mesh_id = np.concatenate(all_contact_mesh_ids)
        contact_face_id = np.concatenate(all_contact_face_ids)
        contact_face_normals = np.vstack(all_contact_face_normals)
    else:
        # Return empty arrays if no contacts found
        contact_points = np.empty((0, 3), dtype=np.float32)
        contact_mesh_id = np.empty(0, dtype=np.int32)
        contact_face_id = np.empty(0, dtype=np.int32)
        contact_face_normals = np.empty((0, 3), dtype=np.float32)

    return contact_points, contact_mesh_id, contact_face_id, contact_face_normals


def AddTranslate(top, offset):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    top.AddTranslateOp().Set(value=offset)

def convert_mesh_to_usd(stage, usd_internal_path, verts, faces, collision_approximation, static, physics_iter=(255, 255)):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]

    points = verts

    # bbox_max = np.max(points, axis=0)
    # bbox_min = np.min(points, axis=0)
    # center = (bbox_max + bbox_min) / 2
    # points = points - center
    # center = (center[0], center[1], center[2])

    vertex_counts = np.ones(n_faces).astype(np.int32) * 3

    mesh = UsdGeom.Mesh.Define(stage, usd_internal_path)

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    # mesh.CreateDisplayColorPrimvar("vertex")
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    mesh.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

    # tilt = mesh.AddRotateXOp(opSuffix='tilt')
    # tilt.Set(value=-90)
    # AddTranslate(mesh, center)

    prim = stage.GetPrimAtPath(usd_internal_path)
    if not static:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        ps_rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        ps_rigid_api.CreateSolverPositionIterationCountAttr(physics_iter[0])
        ps_rigid_api.CreateSolverVelocityIterationCountAttr(physics_iter[1])
        ps_rigid_api.CreateEnableCCDAttr(True)
        ps_rigid_api.CreateEnableSpeculativeCCDAttr(True)

    UsdPhysics.CollisionAPI.Apply(prim)
    ps_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    ps_collision_api.CreateContactOffsetAttr(0.4)
    ps_collision_api.CreateRestOffsetAttr(0.)
    # collider.CreateApproximationAttr("convexDecomposition")
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    # physx_rigid_body.CreateLinearDampingAttr(50.0)
    # physx_rigid_body.CreateAngularDampingAttr(200.0)
    physx_rigid_body.CreateLinearDampingAttr(2.0)
    physx_rigid_body.CreateAngularDampingAttr(2.0)

    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(prim)
    physxSceneAPI.CreateGpuTempBufferCapacityAttr(16 * 1024 * 1024 * 2)
    physxSceneAPI.CreateGpuHeapCapacityAttr(64 * 1024 * 1024 * 2)

    if collision_approximation == "sdf":
        physx_sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        physx_sdf.CreateSdfResolutionAttr(256)
        collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
        collider.CreateApproximationAttr("sdf")
    elif collision_approximation == "convexDecomposition":
        convexdecomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
        collider.CreateApproximationAttr("convexDecomposition")

    mat = UsdPhysics.MaterialAPI.Apply(prim)
    # mat.CreateDynamicFrictionAttr(1e20)
    # mat.CreateStaticFrictionAttr(1e20)
    mat.CreateDynamicFrictionAttr(2.0)  # Increased from 0.4 for better grasping
    mat.CreateStaticFrictionAttr(2.0)   # Increased from 0.4 for better grasping

    return stage

# Step 2: Get object according to given USD prim path
def get_prim(prim_path):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Prim at path {prim_path} is not valid.")
        return None
    return prim


# Helper function to extract position and orientation from the transformation matrix
def extract_position_orientation(transform):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    position = Gf.Vec3d(transform.ExtractTranslation())
    rotation = transform.ExtractRotationQuat()
    orientation = Gf.Quatd(rotation.GetReal(), *rotation.GetImaginary())
    return position, orientation


def quaternion_angle(q1, q2):
    """
    Calculate the angle between two quaternions.

    Parameters:
    q1, q2: Lists or arrays of shape [w, x, y, z] representing quaternions

    Returns:
    angle: The angle in radians between the two quaternions
    """
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    # Convert lists to numpy arrays if they aren't already
    q1 = np.array(q1)
    q2 = np.array(q2)

    # Normalize the quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Calculate the relative quaternion: q_rel = q2 * q1^(-1)
    q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])  # Inverse of a normalized quaternion

    # Quaternion multiplication for q_rel = q2 * q1_inv
    q_rel = np.array([
        q2[0] * q1_inv[0] - q2[1] * q1_inv[1] - q2[2] * q1_inv[2] - q2[3] * q1_inv[3],
        q2[0] * q1_inv[1] + q2[1] * q1_inv[0] + q2[2] * q1_inv[3] - q2[3] * q1_inv[2],
        q2[0] * q1_inv[2] - q2[1] * q1_inv[3] + q2[2] * q1_inv[0] + q2[3] * q1_inv[1],
        q2[0] * q1_inv[3] + q2[1] * q1_inv[2] - q2[2] * q1_inv[1] + q2[3] * q1_inv[0]
    ])

    # The angle can be calculated from the scalar part (real part) of the relative quaternion
    angle = 2 * np.arccos(min(abs(q_rel[0]), 1.0))

    return angle * 180 / np.pi  # Convert to degrees



# Step 3: Start simulation and trace position, orientation, and speed of the object
def start_simulation_and_trace(prims, prim_paths, duration=1.0, dt=1.0 / 60.0):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    global simulation_app
    # Define a list to store the traced data
    traced_data = {}
    init_data = {}

    # Get the timeline and start the simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Initialize variables for tracking the previous position for speed calculation
    prev_position = None
    elapsed_time = 0.0
    init = True

    while elapsed_time < duration:

        # Get the current time code
        current_time_code = Usd.TimeCode.Default()

        # Get current position and orientation
        traced_data_frame_prims = []
        for prim in prims:
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(current_time_code)
            traced_data_frame_prim = extract_position_orientation(transform)
            traced_data_frame_prims.append(traced_data_frame_prim)

        for prim_i, (position, orientation) in enumerate(traced_data_frame_prims):
            # Calculate speed if previous position is available

            prim_path = prim_paths[prim_i]

            # Store the data for current frame
            traced_data_prim = traced_data.get(prim_path, [])

            if init:
                init_data[prim_path] = {}
                init_data[prim_path]["position"] = [position[0], position[1], position[2]]
                init_data[prim_path]["orientation"] = [orientation.GetReal(),
                                                         orientation.GetImaginary()[0],
                                                         orientation.GetImaginary()[1],
                                                         orientation.GetImaginary()[2]
                                                         ]
                relative_position = np.array([0, 0, 0])
                relative_orientation = 0.

            else:
                position_cur = np.array([position[0], position[1], position[2]])
                position_init = np.array([init_data[prim_path]["position"][0],
                                          init_data[prim_path]["position"][1],
                                          init_data[prim_path]["position"][2]])

                orientation_cur = np.array([orientation.GetReal(),
                                            orientation.GetImaginary()[0],
                                            orientation.GetImaginary()[1],
                                            orientation.GetImaginary()[2]
                                            ])
                orientation_init = np.array([init_data[prim_path]["orientation"][0],
                                             init_data[prim_path]["orientation"][1],
                                             init_data[prim_path]["orientation"][2],
                                             init_data[prim_path]["orientation"][3]
                                             ])

                relative_position = position_cur - position_init
                relative_orientation = quaternion_angle(orientation_cur, orientation_init)

            relative_position = relative_position.tolist()
            relative_orientation = float(relative_orientation)

            # traced_data_prim.append({
            #     "time": elapsed_time,
            #     "position": relative_position,
            #     "orientation": relative_orientation,
            # })

            # traced_data[prim_path] = traced_data_prim

        if init:
            init = False
        print(f"\relapsed_time: {elapsed_time:.5f}", end="")
        # Step the simulation
        simulation_app.update()
        # Increment the elapsed time
        elapsed_time += dt

    # Stop the simulation
    timeline.stop()

    return traced_data



# Step 3: Start simulation and trace position, orientation, and speed of the object
def start_simulation_and_track(prims, prim_paths, duration=1.0, dt=1.0 / 60.0):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    global simulation_app
    # Define a list to store the traced data
    traced_data = {}
    init_data = {}

    # Get the timeline and start the simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Initialize variables for tracking the previous position for speed calculation
    prev_position = None
    elapsed_time = 0.0
    init = True

    while elapsed_time < duration:

        # Get the current time code
        current_time_code = Usd.TimeCode.Default()

        # Get current position and orientation
        traced_data_frame_prims = []
        for prim in prims:
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(current_time_code)
            traced_data_frame_prim = extract_position_orientation(transform)
            traced_data_frame_prims.append(traced_data_frame_prim)

        for prim_i, (position, orientation) in enumerate(traced_data_frame_prims):
            # Calculate speed if previous position is available

            prim_path = prim_paths[prim_i]


            if init:
                init_data[prim_path] = {}
                init_data[prim_path]["position"] = [position[0], position[1], position[2]]
                init_data[prim_path]["orientation"] = [orientation.GetReal(),
                                                         orientation.GetImaginary()[0],
                                                         orientation.GetImaginary()[1],
                                                         orientation.GetImaginary()[2]
                                                         ]
                relative_position = np.array([0, 0, 0])
                relative_orientation = 0.
                
                position_cur = np.array([init_data[prim_path]["position"][0],
                                          init_data[prim_path]["position"][1],
                                          init_data[prim_path]["position"][2]])
                
                orientation_cur = np.array([init_data[prim_path]["orientation"][0],
                                             init_data[prim_path]["orientation"][1],
                                             init_data[prim_path]["orientation"][2],
                                             init_data[prim_path]["orientation"][3]
                                             ])

            else:
                position_cur = np.array([position[0], position[1], position[2]])
                position_init = np.array([init_data[prim_path]["position"][0],
                                          init_data[prim_path]["position"][1],
                                          init_data[prim_path]["position"][2]])

                orientation_cur = np.array([orientation.GetReal(),
                                            orientation.GetImaginary()[0],
                                            orientation.GetImaginary()[1],
                                            orientation.GetImaginary()[2]
                                            ])
                orientation_init = np.array([init_data[prim_path]["orientation"][0],
                                             init_data[prim_path]["orientation"][1],
                                             init_data[prim_path]["orientation"][2],
                                             init_data[prim_path]["orientation"][3]
                                             ])
                
                position_last = traced_data[prim_path]["position_last"]
                orientation_last = traced_data[prim_path]["orientation_last"]
                
                relative_position_last = position_cur - position_last
                relative_orientation_last = quaternion_angle(orientation_cur, orientation_last)
                
                relative_position_last = float(np.linalg.norm(relative_position_last))
                relative_orientation_last = float(relative_orientation_last)
                
                relative_position = position_cur - position_init
                relative_orientation = quaternion_angle(orientation_cur, orientation_init)

                relative_position = float(np.linalg.norm(relative_position))
                relative_orientation = float(relative_orientation)

            # traced_data_prim.append({
            #     "time": elapsed_time,
            #     "position": relative_position,
            #     "orientation": relative_orientation,
            # })

            # traced_data[prim_path] = traced_data_prim
            
            traced_data[prim_path] = {
                "time": elapsed_time,
                "position": position_cur,
                "orientation": orientation_cur,
                "d_position": relative_position,
                "d_orientation": relative_orientation,
                "position_last": position_cur,
                "orientation_last": orientation_cur,
            }
            
            if not init:
                traced_data[prim_path]["relative_position_last"] = relative_position_last
                traced_data[prim_path]["relative_orientation_last"] = relative_orientation_last
                if relative_position_last < 1e-6 and relative_orientation_last < 1e-6:
                    traced_data[prim_path]["stable"] = True
                else:
                    traced_data[prim_path]["stable"] = False
                    
                if prim_i == len(prims) - 1:
                    print(f"prim_path: {prim_path}, relative_position_last: {relative_position_last:.5f}, "
                          f"relative_orientation_last: {relative_orientation_last:.5f}, "
                          f"stable: {traced_data[prim_path]['stable']}")
                    
            

        if init:
            init = False
        print(f"\relapsed_time: {elapsed_time:.5f}", end="")
        # Step the simulation
        simulation_app.update()
        # Increment the elapsed time
        elapsed_time += dt

    # Stop the simulation
    timeline.stop()

    return traced_data

# Step 3: Start simulation and trace position, orientation, and speed of the object
def start_simulation_and_track_past_n(prims, prim_paths, duration=1.0, dt=1.0 / 60.0, past_n=10):

    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    global simulation_app
    # Define a list to store the traced data
    traced_data_all = {}
    init_data = {}

    stable_position_limit = 0.2
    stable_rotation_limit = 8.0

    # Get the timeline and start the simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Initialize variables for tracking the previous position for speed calculation
    prev_position = None
    elapsed_time = 0.0
    init = True

    while elapsed_time < duration:

        # Get the current time code
        current_time_code = Usd.TimeCode.Default()

        # Get current position and orientation
        traced_data_frame_prims = []
        for prim in prims:
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(current_time_code)
            traced_data_frame_prim = extract_position_orientation(transform)
            traced_data_frame_prims.append(traced_data_frame_prim)

        for prim_i, (position, orientation) in enumerate(traced_data_frame_prims):
            # Calculate speed if previous position is available

            prim_path = prim_paths[prim_i]
            
            traced_data = traced_data_all.get(prim_path, [])


            if init:
                init_data[prim_path] = {}
                init_data[prim_path]["position"] = [position[0], position[1], position[2]]
                init_data[prim_path]["orientation"] = [orientation.GetReal(),
                                                         orientation.GetImaginary()[0],
                                                         orientation.GetImaginary()[1],
                                                         orientation.GetImaginary()[2]
                                                         ]
                relative_position = 0.
                relative_orientation = 0.
                
                position_cur = np.array([init_data[prim_path]["position"][0],
                                          init_data[prim_path]["position"][1],
                                          init_data[prim_path]["position"][2]])
                
                orientation_cur = np.array([init_data[prim_path]["orientation"][0],
                                             init_data[prim_path]["orientation"][1],
                                             init_data[prim_path]["orientation"][2],
                                             init_data[prim_path]["orientation"][3]
                                             ])

            else:
                position_cur = np.array([position[0], position[1], position[2]])
                position_init = np.array([init_data[prim_path]["position"][0],
                                          init_data[prim_path]["position"][1],
                                          init_data[prim_path]["position"][2]])

                orientation_cur = np.array([orientation.GetReal(),
                                            orientation.GetImaginary()[0],
                                            orientation.GetImaginary()[1],
                                            orientation.GetImaginary()[2]
                                            ])
                orientation_init = np.array([init_data[prim_path]["orientation"][0],
                                             init_data[prim_path]["orientation"][1],
                                             init_data[prim_path]["orientation"][2],
                                             init_data[prim_path]["orientation"][3]
                                             ])
                
                position_last = traced_data[0]["position_last"]
                orientation_last = traced_data[0]["orientation_last"]
                
                relative_position_last = position_cur - position_last
                relative_orientation_last = quaternion_angle(orientation_cur, orientation_last)
                
                relative_position_last = float(np.linalg.norm(relative_position_last))
                relative_orientation_last = float(relative_orientation_last)
                
                relative_position = position_cur - position_init
                relative_orientation = quaternion_angle(orientation_cur, orientation_init)

                relative_position = float(np.linalg.norm(relative_position))
                relative_orientation = float(relative_orientation)

            
            traced_data.append({
                "time": elapsed_time,
                "position": position_cur.copy(),
                "orientation": orientation_cur.copy(),
                "d_position": relative_position,
                "d_orientation": relative_orientation,
                "position_last": position_cur.copy(),
                "orientation_last": orientation_cur.copy(),
            })

            if traced_data[-1]["d_position"] > stable_position_limit or \
               traced_data[-1]["d_orientation"] > stable_rotation_limit:
                traced_data[-1]["d_stable"] = False
            else:
                traced_data[-1]["d_stable"] = True
            
            if not init:
                traced_data[-1]["relative_position_last"] = relative_position_last
                traced_data[-1]["relative_orientation_last"] = relative_orientation_last
                if relative_position_last < 1e-3 and relative_orientation_last < 1e-3:
                    traced_data[-1]["stable"] = True
                else:
                    traced_data[-1]["stable"] = False
                    
                                  
            if len(traced_data) > past_n:
                traced_data.pop(0)
                
                longterm_stable = True
                for trace_item in traced_data:
                    longterm_stable = longterm_stable and trace_item["stable"]
                    
                traced_data[-1]["longterm_stable"] = longterm_stable
            else:
                traced_data[-1]["longterm_stable"] = False
            traced_data_all[prim_path] = traced_data
        
        all_longterm_stable = True
                    
        for prim_i, traced_data in traced_data_all.items():
            all_longterm_stable = all_longterm_stable and traced_data[-1]["longterm_stable"]

        if all_longterm_stable:
            # print(f"\nAll prims are longterm stable at elapsed_time: {elapsed_time:.5f}")
            timeline.stop()

            return traced_data_all
        
        
        existing_stable = True
        
        for prim_i, traced_data in traced_data_all.items():
            if not traced_data[-1]["d_stable"]:
                existing_stable = False
                # print(f"prim_path: {prim_i}, d_position: {traced_data[-1]['d_position']:.5f}, "
                #       f"d_orientation: {traced_data[-1]['d_orientation']:.5f}, "
                #       f"stable: {traced_data[-1]['stable']}")
                break
        
        if not existing_stable:
            # print(f"\nNot all prims are stable ({prim_i}) at elapsed_time: {elapsed_time:.5f}, "
            #       f"stop simulation...")
            timeline.stop()

            return traced_data_all
        
        if init:
            init = False
        # print(f"\relapsed_time: {elapsed_time:.5f}", end="")
        # Step the simulation
        simulation_app.update()
        # Increment the elapsed time
        elapsed_time += dt

    # Stop the simulation
    timeline.stop()

    return traced_data_all

# Step 3: Start simulation and trace position, orientation, and speed of the object
def start_simulation_and_track_past_n_eval_added(prims, prim_paths, duration=1.0, dt=1.0 / 60.0, past_n=10):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    global simulation_app
    # Define a list to store the traced data
    traced_data_all = {}
    init_data = {}

    # Get the timeline and start the simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Initialize variables for tracking the previous position for speed calculation
    prev_position = None
    elapsed_time = 0.0
    init = True

    while elapsed_time < duration:

        # Get the current time code
        current_time_code = Usd.TimeCode.Default()

        # Get current position and orientation
        traced_data_frame_prims = []
        for prim in prims:
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(current_time_code)
            traced_data_frame_prim = extract_position_orientation(transform)
            traced_data_frame_prims.append(traced_data_frame_prim)

        for prim_i, (position, orientation) in enumerate(traced_data_frame_prims):
            # Calculate speed if previous position is available

            prim_path = prim_paths[prim_i]
            
            traced_data = traced_data_all.get(prim_path, [])


            if init:
                init_data[prim_path] = {}
                init_data[prim_path]["position"] = [position[0], position[1], position[2]]
                init_data[prim_path]["orientation"] = [orientation.GetReal(),
                                                         orientation.GetImaginary()[0],
                                                         orientation.GetImaginary()[1],
                                                         orientation.GetImaginary()[2]
                                                         ]
                relative_position = 0.
                relative_orientation = 0.
                
                position_cur = np.array([init_data[prim_path]["position"][0],
                                          init_data[prim_path]["position"][1],
                                          init_data[prim_path]["position"][2]])
                
                orientation_cur = np.array([init_data[prim_path]["orientation"][0],
                                             init_data[prim_path]["orientation"][1],
                                             init_data[prim_path]["orientation"][2],
                                             init_data[prim_path]["orientation"][3]
                                             ])

            else:
                position_cur = np.array([position[0], position[1], position[2]])
                position_init = np.array([init_data[prim_path]["position"][0],
                                          init_data[prim_path]["position"][1],
                                          init_data[prim_path]["position"][2]])

                orientation_cur = np.array([orientation.GetReal(),
                                            orientation.GetImaginary()[0],
                                            orientation.GetImaginary()[1],
                                            orientation.GetImaginary()[2]
                                            ])
                orientation_init = np.array([init_data[prim_path]["orientation"][0],
                                             init_data[prim_path]["orientation"][1],
                                             init_data[prim_path]["orientation"][2],
                                             init_data[prim_path]["orientation"][3]
                                             ])
                
                position_last = traced_data[0]["position_last"]
                orientation_last = traced_data[0]["orientation_last"]
                
                relative_position_last = position_cur - position_last
                relative_orientation_last = quaternion_angle(orientation_cur, orientation_last)
                
                relative_position_last = float(np.linalg.norm(relative_position_last))
                relative_orientation_last = float(relative_orientation_last)
                
                relative_position = position_cur - position_init
                relative_orientation = quaternion_angle(orientation_cur, orientation_init)

                relative_position = float(np.linalg.norm(relative_position))
                relative_orientation = float(relative_orientation)

            
            traced_data.append({
                "time": elapsed_time,
                "position": position_cur.copy(),
                "orientation": orientation_cur.copy(),
                "d_position": relative_position,
                "d_orientation": relative_orientation,
                "position_last": position_cur.copy(),
                "orientation_last": orientation_cur.copy(),
            })
            
            if not init:
                traced_data[-1]["relative_position_last"] = relative_position_last
                traced_data[-1]["relative_orientation_last"] = relative_orientation_last
                if relative_position_last < 1e-3 and relative_orientation_last < 1e-3:
                    traced_data[-1]["stable"] = True
                else:
                    traced_data[-1]["stable"] = False
                    
                # if prim_i == len(prims) - 1:
                #     print(f"prim_path: {prim_path}, relative_position_last: {relative_position_last:.5f}, "
                #           f"relative_orientation_last: {relative_orientation_last:.5f}, "
                #           f"stable: {traced_data[-1]['stable']}, "
                #           f"len(traced_data): {len(traced_data)}")
                    
                                  
            if len(traced_data) > past_n:
                traced_data.pop(0)
                
                longterm_stable = True
                for trace_item in traced_data:
                    longterm_stable = longterm_stable and trace_item["stable"]
                    
                traced_data[-1]["longterm_stable"] = longterm_stable
            else:
                traced_data[-1]["longterm_stable"] = False
            traced_data_all[prim_path] = traced_data
        
        all_longterm_stable = True
                    
        for prim_i, traced_data in traced_data_all.items():
            all_longterm_stable = all_longterm_stable and traced_data[-1]["longterm_stable"]
            # print(f"prim_path: {prim_i}, longterm_stable: {traced_data[-1]['longterm_stable']}")

        if all_longterm_stable:
            print(f"\nAll prims are longterm stable at elapsed_time: {elapsed_time:.5f}")
            timeline.stop()

            return traced_data_all
        
        existing_stable = True
        stable_rotation_limit = 8.0
        stable_position_limit = 0.2
        
        for prim_i, traced_data in traced_data_all.items():
            if int(prim_i.replace("/obj_", "")) == len(prims) - 1:
                continue
            
            if traced_data[-1]["d_position"] > stable_position_limit or \
               traced_data[-1]["d_orientation"] > stable_rotation_limit:
                existing_stable = False
                print(f"prim_path: {prim_i}, d_position: {traced_data[-1]['d_position']:.5f}, "
                      f"d_orientation: {traced_data[-1]['d_orientation']:.5f}, "
                      f"stable: {traced_data[-1]['stable']}")
                break
        
        if not existing_stable:
            print(f"\nNot all prims are stable ({prim_i}) at elapsed_time: {elapsed_time:.5f}, "
                  f"stop simulation...")
            timeline.stop()

            return traced_data_all
        
        if init:
            init = False
        print(f"\relapsed_time: {elapsed_time:.5f}", end="")
        # Step the simulation
        simulation_app.update()
        # Increment the elapsed time
        elapsed_time += dt

    # Stop the simulation
    timeline.stop()

    return traced_data_all



def sim_scene(mesh_dict_list):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils

    stage = Usd.Stage.CreateInMemory()

    collision_approximation = "sdf"

    for mesh_idx in mesh_dict_list:
        usd_internal_path = f"/obj_{mesh_idx}"
        usd_internal_path = usd_internal_path.replace("-", "_")
        mesh_dict = mesh_dict_list[mesh_idx]
        mesh_obj_i = mesh_dict['mesh']
        static = mesh_dict['static']
        
        stage = convert_mesh_to_usd(stage, usd_internal_path,
                                    mesh_obj_i.vertices, mesh_obj_i.faces,
                                    collision_approximation, static, physics_iter=(16, 1))

    cache = UsdUtils.StageCache.Get()
    stage_id = cache.Insert(stage).ToLongInt()
    omni.usd.get_context().attach_stage_with_callback(stage_id)
    
    prims = []
    prim_paths = []

    # Get the prim of the object
    for mesh_idx in mesh_dict_list:
        usd_prim_path = f"/obj_{mesh_idx}"
        prim_paths.append(usd_prim_path)
        prim = get_prim(usd_prim_path)
        if prim is None:
            assert False, f"Failed to get prim at path {usd_prim_path}"
        prims.append(prim)

    # Start the simulation and trace the object
    traced_data = start_simulation_and_trace(prims, prim_paths, duration=10000.0)
    

def sim_scene_eval(mesh_dict_list, mesh_idx_to_object_id, duration=100.0):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils

    stage = Usd.Stage.CreateInMemory()

    collision_approximation = "sdf"

    for mesh_idx in mesh_dict_list:
        usd_internal_path = f"/obj_{mesh_idx}"
        usd_internal_path = usd_internal_path.replace("-", "_")
        mesh_dict = mesh_dict_list[mesh_idx]
        mesh_obj_i = mesh_dict['mesh']
        static = mesh_dict['static']
        
        stage = convert_mesh_to_usd(stage, usd_internal_path,
                                    mesh_obj_i.vertices, mesh_obj_i.faces,
                                    collision_approximation, static, physics_iter=(16, 1))


    cache = UsdUtils.StageCache.Get()
    stage_id = cache.Insert(stage).ToLongInt()
    omni.usd.get_context().attach_stage_with_callback(stage_id)
    
    prims = []
    prim_paths = []

    # Get the prim of the object
    for mesh_idx in mesh_dict_list:
        usd_prim_path = f"/obj_{mesh_idx}"
        prim_paths.append(usd_prim_path)
        prim = get_prim(usd_prim_path)
        if prim is None:
            assert False, f"Failed to get prim at path {usd_prim_path}"
        prims.append(prim)

    # Start the simulation and trace the object
    traced_data = start_simulation_and_track_past_n(prims, prim_paths, duration=duration)

    traced_data_updated_key = {}
    for prim_path, traced_data_item in traced_data.items():
        traced_data_item_last = traced_data_item[-1]
        for key, value in traced_data_item_last.items():
            if isinstance(value, np.ndarray):
                traced_data_item_last[key] = value.tolist()

        # traced_data_item_last = {}
        # traced_data_item_last["stable"] = traced_data_item[-1]["d_stable"]

        traced_data_updated_key[mesh_idx_to_object_id[int(prim_path.replace("/obj_", ""))]] = traced_data_item_last

    return traced_data_updated_key



def sim_scene_eval_added(mesh_dict_list, duration=1.0):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils

    stage = Usd.Stage.CreateInMemory()

    collision_approximation = "sdf"

    for mesh_idx in mesh_dict_list:
        usd_internal_path = f"/obj_{mesh_idx}"
        usd_internal_path = usd_internal_path.replace("-", "_")
        mesh_dict = mesh_dict_list[mesh_idx]
        mesh_obj_i = mesh_dict['mesh']
        static = mesh_dict['static']
        
        stage = convert_mesh_to_usd(stage, usd_internal_path,
                                    mesh_obj_i.vertices, mesh_obj_i.faces,
                                    collision_approximation, static, physics_iter=(16, 1))

    cache = UsdUtils.StageCache.Get()
    stage_id = cache.Insert(stage).ToLongInt()
    omni.usd.get_context().attach_stage_with_callback(stage_id)
    
    prims = []
    prim_paths = []

    # Get the prim of the object
    for mesh_idx in mesh_dict_list:
        usd_prim_path = f"/obj_{mesh_idx}"
        prim_paths.append(usd_prim_path)
        prim = get_prim(usd_prim_path)
        if prim is None:
            assert False, f"Failed to get prim at path {usd_prim_path}"
        prims.append(prim)

    # Start the simulation and trace the object
    traced_data = start_simulation_and_track_past_n_eval_added(prims, prim_paths, duration=duration)
    return traced_data