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
from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils, Sdf, UsdShade
import numpy as np
import trimesh
import sys

def AddTranslate(top, offset):
    top.AddTranslateOp().Set(value=offset)

def convert_mesh_to_usd(stage, usd_internal_path, verts, faces, collision_approximation, static, articulation, 
                        physics_iter=(16, 1), mass=None, apply_debug_torque=False, debug_torque_value=50.0, 
                        texture=None, usd_internal_art_reference_path="/World",
                        add_damping=False):
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
    mesh.CreateExtentAttr([(-100, -100, -100), (100, 100, 100)])

    # tilt = mesh.AddRotateXOp(opSuffix='tilt')
    # tilt.Set(value=-90)
    # AddTranslate(mesh, center)

    prim = stage.GetPrimAtPath(usd_internal_path)

    if texture is not None:
        vts = texture["vts"]
        fts = texture["fts"]
        texture_map_path = texture["texture_map_path"]
        tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)

        texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", 
                                    Sdf.ValueTypeNames.TexCoord2fArray, 
                                    UsdGeom.Tokens.faceVarying)
        texCoords.Set(Vt.Vec2fArray.FromNumpy(tex_coords))
        
        usd_mat_path = usd_internal_path+"_mat"
        material = UsdShade.Material.Define(stage, usd_mat_path)
        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')
        
        pbrShader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/PBRShader")
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        if "pbr_parameters" in texture:
            pbr_parameters = texture["pbr_parameters"]
            roughness = pbr_parameters.get("roughness", 1.0)
            metallic = pbr_parameters.get("metallic", 0.0)
        else:
            roughness = 1.0
            metallic = 0.0
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        pbrShader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Bool).Set(True)

        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

        # create texture coordinate reader 
        stReader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/stReader")
        stReader.CreateIdAttr('UsdPrimvarReader_float2')
        # Note here we are connecting the shader's input to the material's 
        # "public interface" attribute. This allows users to change the primvar name
        # on the material itself without drilling inside to examine shader nodes.
        stReader.CreateInput('varname',Sdf.ValueTypeNames.Token).ConnectToSource(stInput)

        # diffuse texture
        diffuseTextureSampler = UsdShade.Shader.Define(stage, f"{usd_mat_path}/diffuseTexture")
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_map_path)
        diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')

        # Now bind the Material to the card
        mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(mesh).Bind(material)


    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    if not static:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        if mass is not None:
            mass_api.CreateMassAttr(mass)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        ps_rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rigid_body.CreateSolverPositionIterationCountAttr(physics_iter[0])
        physx_rigid_body.CreateSolverVelocityIterationCountAttr(physics_iter[1])

    if articulation is not None:
        articulation_api = UsdPhysics.ArticulationRootAPI.Apply(prim)
        # Add revolute joint articulation
        rotate_axis_point_lower, rotate_axis_point_upper = articulation
        
        # Calculate the rotation axis vector
        axis_vector = np.array(rotate_axis_point_upper) - np.array(rotate_axis_point_lower)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)  # Normalize
        # Create a revolute joint
        joint_path = usd_internal_path + "_joint"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        # Set the joint axis (in local space)
        joint.CreateAxisAttr("Z")  # Default to Z-axis, we'll transform to match our axis
        # Set the joint bodies - this connects the joint to the rigid body
        joint.CreateBody0Rel().SetTargets([usd_internal_path])
        joint.CreateBody1Rel().SetTargets([usd_internal_art_reference_path])
        # For a single body joint (attached to world), we don't set body1
        # Create joint position (midpoint of the axis)
        joint_pos = (np.array(rotate_axis_point_lower) + np.array(rotate_axis_point_upper)) / 2
        # Apply transform to position the joint at the rotation axis
        joint_prim = stage.GetPrimAtPath(joint_path)
        joint_xform = UsdGeom.Xformable(joint_prim)
        # Set the joint position using physics:localPos0 and physics:localPos1
        # These define the connection points on each body
        joint.CreateLocalPos0Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        
        # # Also set the transform position for visualization/debugging
        # translate_op = joint_xform.AddTranslateOp()
        # translate_op.Set(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        # If the rotation axis is not along Z, we need to rotate the joint
        if not np.allclose(axis_vector, [0, 0, 1]):
            # Calculate rotation to align Z-axis with our desired axis
            z_axis = np.array([0, 0, 1])
            # Use cross product to find rotation axis
            rotation_axis = np.cross(z_axis, axis_vector)
            if np.linalg.norm(rotation_axis) > 1e-6:  # Not parallel
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                # Calculate angle between vectors
                angle = np.arccos(np.clip(np.dot(z_axis, axis_vector), -1.0, 1.0))
                # Convert to degrees
                angle_degrees = np.degrees(angle)
                
                # Create rotation quaternion
                sin_half = np.sin(angle / 2)
                cos_half = np.cos(angle / 2)
                quat = Gf.Quatf(cos_half, sin_half * rotation_axis[0], 
                               sin_half * rotation_axis[1], sin_half * rotation_axis[2])
                
                # Apply rotation using physics:localRot0 and physics:localRot1
                joint.CreateLocalRot0Attr(quat)
                joint.CreateLocalRot1Attr(quat)
        # Optional: Set joint limits if needed
        # joint.CreateLowerLimitAttr(-180.0)  # -180 degrees
        # joint.CreateUpperLimitAttr(180.0)   # +180 degrees
        
        # Apply debug torque if requested (for testing joint functionality)
        if apply_debug_torque:
            print(f"Applying debug torque: {debug_torque_value}")
            # Apply DriveAPI to the joint
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            # Set drive type to velocity control
            drive_api.CreateTypeAttr("force")
            # Set target velocity to make the joint rotate
            drive_api.CreateTargetVelocityAttr(debug_torque_value)  # degrees per second
            # Set drive stiffness and damping
            drive_api.CreateStiffnessAttr(0.0)  # No position control
            drive_api.CreateDampingAttr(1e4)   # High damping for velocity control
            # Set max force
            drive_api.CreateMaxForceAttr(1000.0)  # Maximum force for the drive
            print("Debug torque applied - joint should rotate")
        
        # Apply PhysX-specific joint properties for better simulation
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        # Note: Break force/torque attributes may not be available for all joint types
        # physx_joint.CreateBreakForceAttr(1e10)  # Very large value - effectively never break
        # physx_joint.CreateBreakTorqueAttr(1e10)  # Very large value - effectively never break
    UsdPhysics.CollisionAPI.Apply(prim)
    ps_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    ps_collision_api.CreateContactOffsetAttr(0.005)
    ps_collision_api.CreateRestOffsetAttr(0.001)
    ps_collision_api.CreateTorsionalPatchRadiusAttr(0.01)

    physx_rigid_body.CreateLinearDampingAttr(10.0)
    physx_rigid_body.CreateAngularDampingAttr(10.0)

    physx_rigid_body.CreateMaxLinearVelocityAttr(0.5)
    physx_rigid_body.CreateMaxAngularVelocityAttr(0.5)
    physx_rigid_body.CreateMaxDepenetrationVelocityAttr(50.0)

    # physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(prim)
    # physxSceneAPI.CreateGpuTempBufferCapacityAttr(16 * 1024 * 1024 * 2)
    # physxSceneAPI.CreateGpuHeapCapacityAttr(64 * 1024 * 1024 * 2)

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
    mat.CreateDynamicFrictionAttr(1e20)
    mat.CreateStaticFrictionAttr(1e20)
    # mat.CreateDynamicFrictionAttr(2.0)  # Increased from 0.4 for better grasping
    # mat.CreateStaticFrictionAttr(2.0)   # Increased from 0.4 for better grasping

    return stage


def convert_mesh_to_usd_simple(stage, usd_internal_path, verts, faces, collision_approximation, static, articulation, 
                        physics_iter=(255, 255), apply_debug_torque=False, debug_torque_value=50.0, texture=None):
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
    mesh.CreateExtentAttr([(-100, -100, -100), (100, 100, 100)])

    # tilt = mesh.AddRotateXOp(opSuffix='tilt')
    # tilt.Set(value=-90)
    # AddTranslate(mesh, center)

    prim = stage.GetPrimAtPath(usd_internal_path)

    if texture is not None:
        vts = texture["vts"]
        fts = texture["fts"]
        texture_map_path = texture["texture_map_path"]
        tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)

        texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", 
                                    Sdf.ValueTypeNames.TexCoord2fArray, 
                                    UsdGeom.Tokens.faceVarying)
        texCoords.Set(Vt.Vec2fArray.FromNumpy(tex_coords))
        
        usd_mat_path = usd_internal_path+"_mat"
        material = UsdShade.Material.Define(stage, usd_mat_path)
        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')
        
        pbrShader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/PBRShader")
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        pbrShader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Bool).Set(True)

        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

        # create texture coordinate reader 
        stReader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/stReader")
        stReader.CreateIdAttr('UsdPrimvarReader_float2')
        # Note here we are connecting the shader's input to the material's 
        # "public interface" attribute. This allows users to change the primvar name
        # on the material itself without drilling inside to examine shader nodes.
        stReader.CreateInput('varname',Sdf.ValueTypeNames.Token).ConnectToSource(stInput)

        # diffuse texture
        diffuseTextureSampler = UsdShade.Shader.Define(stage, f"{usd_mat_path}/diffuseTexture")
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_map_path)
        diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')

        # Now bind the Material to the card
        mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(mesh).Bind(material)


    if not static:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        # ps_rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        # ps_rigid_api.CreateSolverPositionIterationCountAttr(physics_iter[0])
        # ps_rigid_api.CreateSolverVelocityIterationCountAttr(physics_iter[1])
        # ps_rigid_api.CreateEnableCCDAttr(True)
        # ps_rigid_api.CreateEnableSpeculativeCCDAttr(True)

    if articulation is not None:
        # Add revolute joint articulation
        rotate_axis_point_lower, rotate_axis_point_upper = articulation
        
        # Calculate the rotation axis vector
        axis_vector = np.array(rotate_axis_point_upper) - np.array(rotate_axis_point_lower)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)  # Normalize
        # Create a revolute joint
        joint_path = usd_internal_path + "_joint"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        # Set the joint axis (in local space)
        joint.CreateAxisAttr("Z")  # Default to Z-axis, we'll transform to match our axis
        # Set the joint bodies - this connects the joint to the rigid body
        joint.CreateBody0Rel().SetTargets([usd_internal_path])
        # For a single body joint (attached to world), we don't set body1
        # Create joint position (midpoint of the axis)
        joint_pos = (np.array(rotate_axis_point_lower) + np.array(rotate_axis_point_upper)) / 2
        # Apply transform to position the joint at the rotation axis
        joint_prim = stage.GetPrimAtPath(joint_path)
        joint_xform = UsdGeom.Xformable(joint_prim)
        # Set the joint position using physics:localPos0 and physics:localPos1
        # These define the connection points on each body
        joint.CreateLocalPos0Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        
        # # Also set the transform position for visualization/debugging
        # translate_op = joint_xform.AddTranslateOp()
        # translate_op.Set(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        # If the rotation axis is not along Z, we need to rotate the joint
        if not np.allclose(axis_vector, [0, 0, 1]):
            # Calculate rotation to align Z-axis with our desired axis
            z_axis = np.array([0, 0, 1])
            # Use cross product to find rotation axis
            rotation_axis = np.cross(z_axis, axis_vector)
            if np.linalg.norm(rotation_axis) > 1e-6:  # Not parallel
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                # Calculate angle between vectors
                angle = np.arccos(np.clip(np.dot(z_axis, axis_vector), -1.0, 1.0))
                # Convert to degrees
                angle_degrees = np.degrees(angle)
                
                # Create rotation quaternion
                sin_half = np.sin(angle / 2)
                cos_half = np.cos(angle / 2)
                quat = Gf.Quatf(cos_half, sin_half * rotation_axis[0], 
                               sin_half * rotation_axis[1], sin_half * rotation_axis[2])
                
                # Apply rotation using physics:localRot0 and physics:localRot1
                joint.CreateLocalRot0Attr(quat)
                joint.CreateLocalRot1Attr(quat)
        # Optional: Set joint limits if needed
        # joint.CreateLowerLimitAttr(-180.0)  # -180 degrees
        # joint.CreateUpperLimitAttr(180.0)   # +180 degrees
        
        # Apply debug torque if requested (for testing joint functionality)
        # if apply_debug_torque:
        #     print(f"Applying debug torque: {debug_torque_value}")
        #     # Apply DriveAPI to the joint
        #     drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        #     # Set drive type to velocity control
        #     drive_api.CreateTypeAttr("force")
        #     # Set target velocity to make the joint rotate
        #     drive_api.CreateTargetVelocityAttr(debug_torque_value)  # degrees per second
        #     # Set drive stiffness and damping
        #     drive_api.CreateStiffnessAttr(0.0)  # No position control
        #     drive_api.CreateDampingAttr(1e4)   # High damping for velocity control
        #     # Set max force
        #     drive_api.CreateMaxForceAttr(1000.0)  # Maximum force for the drive
        #     print("Debug torque applied - joint should rotate")
        
        # Apply PhysX-specific joint properties for better simulation
        # physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        # Note: Break force/torque attributes may not be available for all joint types
        # physx_joint.CreateBreakForceAttr(1e10)  # Very large value - effectively never break
        # physx_joint.CreateBreakTorqueAttr(1e10)  # Very large value - effectively never break
    UsdPhysics.CollisionAPI.Apply(prim)
    # ps_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    # ps_collision_api.CreateContactOffsetAttr(0.4)
    # ps_collision_api.CreateRestOffsetAttr(0.)
    # collider.CreateApproximationAttr("convexDecomposition")
    # physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    # physx_rigid_body.CreateLinearDampingAttr(50.0)
    # physx_rigid_body.CreateAngularDampingAttr(200.0)
    # physx_rigid_body.CreateLinearDampingAttr(2.0)
    # physx_rigid_body.CreateAngularDampingAttr(2.0)

    # physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(prim)
    # physxSceneAPI.CreateGpuTempBufferCapacityAttr(16 * 1024 * 1024 * 2)
    # physxSceneAPI.CreateGpuHeapCapacityAttr(64 * 1024 * 1024 * 2)

    # if collision_approximation == "sdf":
    #     physx_sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    #     physx_sdf.CreateSdfResolutionAttr(256)
    #     collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
    #     collider.CreateApproximationAttr("sdf")
    # elif collision_approximation == "convexDecomposition":
    #     convexdecomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    #     collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
    #     collider.CreateApproximationAttr("convexDecomposition")

    mat = UsdPhysics.MaterialAPI.Apply(prim)
    # mat.CreateDynamicFrictionAttr(1e20)
    # mat.CreateStaticFrictionAttr(1e20)
    mat.CreateDynamicFrictionAttr(2.0)  # Increased from 0.4 for better grasping
    mat.CreateStaticFrictionAttr(2.0)   # Increased from 0.4 for better grasping

    return stage


def door_frame_to_usd(
    stage,
    usd_internal_path_door,
    usd_internal_path_door_frame,
    mesh_obj_door,
    mesh_obj_door_frame,
    articulation_door,
    texture_door,
    texture_door_frame,
    apply_debug_torque=False,
    debug_torque_value=50.0
):
    """
    Create door and door frame USD objects with a revolute joint between them.
    
    Args:
        stage: USD stage
        usd_internal_path_door: USD path for the door
        usd_internal_path_door_frame: USD path for the door frame
        mesh_obj_door: Trimesh object for the door
        mesh_obj_door_frame: Trimesh object for the door frame
        articulation_door: Tuple of (rotate_axis_point_lower, rotate_axis_point_upper) in world coordinates
        texture_door: Texture info for door (can be None)
        texture_door_frame: Texture info for door frame (can be None)
        apply_debug_torque: Whether to apply debug torque for testing joint functionality
        debug_torque_value: Target velocity for debug torque (degrees per second)
    """
    
    # Extract vertices and faces from mesh objects
    door_verts = np.array(mesh_obj_door.vertices)
    door_faces = np.array(mesh_obj_door.faces)
    frame_verts = np.array(mesh_obj_door_frame.vertices)
    frame_faces = np.array(mesh_obj_door_frame.faces)
    
    # Create the door frame first (this will be the static parent)
    # Door frame is static and acts as the base for the joint
    n_frame_verts = frame_verts.shape[0]
    n_frame_faces = frame_faces.shape[0]
    frame_vertex_counts = np.ones(n_frame_faces).astype(np.int32) * 3
    
    # Create door frame mesh
    frame_mesh = UsdGeom.Mesh.Define(stage, usd_internal_path_door_frame)
    frame_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(frame_verts))
    frame_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(frame_vertex_counts))
    frame_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(frame_faces))
    frame_mesh.CreateExtentAttr([(-100, -100, -100), (100, 100, 100)])
    
    frame_prim = stage.GetPrimAtPath(usd_internal_path_door_frame)
    
    # Apply texture to door frame if provided
    if texture_door_frame is not None:
        vts = texture_door_frame["vts"]
        fts = texture_door_frame["fts"]
        texture_map_path = texture_door_frame["texture_map_path"]
        tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)

        texCoords = UsdGeom.PrimvarsAPI(frame_mesh).CreatePrimvar("st", 
                                    Sdf.ValueTypeNames.TexCoord2fArray, 
                                    UsdGeom.Tokens.faceVarying)
        texCoords.Set(Vt.Vec2fArray.FromNumpy(tex_coords))
        
        usd_mat_path = usd_internal_path_door_frame + "_mat"
        material = UsdShade.Material.Define(stage, usd_mat_path)
        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')
        
        pbrShader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/PBRShader")
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        pbrShader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Bool).Set(True)

        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

        # create texture coordinate reader 
        stReader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/stReader")
        stReader.CreateIdAttr('UsdPrimvarReader_float2')
        stReader.CreateInput('varname',Sdf.ValueTypeNames.Token).ConnectToSource(stInput)

        # diffuse texture
        diffuseTextureSampler = UsdShade.Shader.Define(stage, f"{usd_mat_path}/diffuseTexture")
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_map_path)
        diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')

        # Bind material to door frame
        frame_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(frame_mesh).Bind(material)
    
    # Set up door frame physics (static)
    # UsdPhysics.CollisionAPI.Apply(frame_prim)
    # UsdPhysics.RigidBodyAPI.Apply(frame_prim)
    
    # Apply physics material to door frame
    frame_mat = UsdPhysics.MaterialAPI.Apply(frame_prim)
    frame_mat.CreateDynamicFrictionAttr(2.0)
    frame_mat.CreateStaticFrictionAttr(2.0)
    
    # Create the door (this will be the moving part)
    n_door_verts = door_verts.shape[0]
    n_door_faces = door_faces.shape[0]
    door_vertex_counts = np.ones(n_door_faces).astype(np.int32) * 3
    
    # Create door mesh
    door_mesh = UsdGeom.Mesh.Define(stage, usd_internal_path_door)
    door_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(door_verts))
    door_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(door_vertex_counts))
    door_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(door_faces))
    door_mesh.CreateExtentAttr([(-100, -100, -100), (100, 100, 100)])
    
    door_prim = stage.GetPrimAtPath(usd_internal_path_door)
    
    # Apply texture to door if provided
    if texture_door is not None:
        vts = texture_door["vts"]
        fts = texture_door["fts"]
        texture_map_path = texture_door["texture_map_path"]
        tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)

        texCoords = UsdGeom.PrimvarsAPI(door_mesh).CreatePrimvar("st", 
                                    Sdf.ValueTypeNames.TexCoord2fArray, 
                                    UsdGeom.Tokens.faceVarying)
        texCoords.Set(Vt.Vec2fArray.FromNumpy(tex_coords))
        
        usd_mat_path = usd_internal_path_door + "_mat"
        material = UsdShade.Material.Define(stage, usd_mat_path)
        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')
        
        pbrShader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/PBRShader")
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        pbrShader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Bool).Set(True)

        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

        # create texture coordinate reader 
        stReader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/stReader")
        stReader.CreateIdAttr('UsdPrimvarReader_float2')
        stReader.CreateInput('varname',Sdf.ValueTypeNames.Token).ConnectToSource(stInput)

        # diffuse texture
        diffuseTextureSampler = UsdShade.Shader.Define(stage, f"{usd_mat_path}/diffuseTexture")
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_map_path)
        diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')

        # Bind material to door
        door_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(door_mesh).Bind(material)
    
    # Set up door physics (dynamic)
    UsdPhysics.CollisionAPI.Apply(door_prim)
    mass_api = UsdPhysics.MassAPI.Apply(door_prim)
    mass_api.CreateMassAttr(10.0)  # Set door mass to 10kg
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(door_prim)
    
    # Apply PhysX rigid body properties for better simulation
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(door_prim)
    physx_rigid_body.CreateSolverPositionIterationCountAttr(255)
    physx_rigid_body.CreateSolverVelocityIterationCountAttr(255)
    physx_rigid_body.CreateLinearDampingAttr(10.0)
    physx_rigid_body.CreateAngularDampingAttr(10.0)
    physx_rigid_body.CreateMaxLinearVelocityAttr(0.5)
    physx_rigid_body.CreateMaxAngularVelocityAttr(0.5)
    physx_rigid_body.CreateMaxDepenetrationVelocityAttr(50.0)
    
    # # Apply collision properties
    # ps_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(door_prim)
    # ps_collision_api.CreateContactOffsetAttr(0.005)
    # ps_collision_api.CreateRestOffsetAttr(0.001)
    # ps_collision_api.CreateTorsionalPatchRadiusAttr(0.01)

    physx_sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(door_prim)
    physx_sdf.CreateSdfResolutionAttr(256)
    collider = UsdPhysics.MeshCollisionAPI.Apply(door_prim)
    collider.CreateApproximationAttr("sdf")
    
    # Apply physics material to door
    door_mat = UsdPhysics.MaterialAPI.Apply(door_prim)
    door_mat.CreateDynamicFrictionAttr(2.0)
    door_mat.CreateStaticFrictionAttr(2.0)
    
    # Create the revolute joint between door and door frame
    if articulation_door is not None:
        rotate_axis_point_lower, rotate_axis_point_upper = articulation_door
        
        # Calculate the rotation axis vector
        axis_vector = np.array(rotate_axis_point_upper) - np.array(rotate_axis_point_lower)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)  # Normalize
        
        # Create a revolute joint
        joint_path = usd_internal_path_door + "_hinge_joint"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        
        # Set the joint axis (in local space)
        joint.CreateAxisAttr("Z")  # Default to Z-axis, we'll transform to match our axis
        
        # Set the joint bodies - door rotates relative to door frame
        joint.CreateBody0Rel().SetTargets([usd_internal_path_door])      # Moving body (door)
        joint.CreateBody1Rel().SetTargets([usd_internal_path_door_frame]) # Static body (door frame)
        
        # Create joint position (midpoint of the axis)
        joint_pos = (np.array(rotate_axis_point_lower) + np.array(rotate_axis_point_upper)) / 2
        
        # Set the joint position using physics:localPos0 and physics:localPos1
        # These define the connection points on each body
        joint.CreateLocalPos0Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        
        # If the rotation axis is not along Z, we need to rotate the joint
        if not np.allclose(axis_vector, [0, 0, 1]):
            # Calculate rotation to align Z-axis with our desired axis
            z_axis = np.array([0, 0, 1])
            # Use cross product to find rotation axis
            rotation_axis = np.cross(z_axis, axis_vector)
            if np.linalg.norm(rotation_axis) > 1e-6:  # Not parallel
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                # Calculate angle between vectors
                angle = np.arccos(np.clip(np.dot(z_axis, axis_vector), -1.0, 1.0))
                
                # Create rotation quaternion
                sin_half = np.sin(angle / 2)
                cos_half = np.cos(angle / 2)
                quat = Gf.Quatf(cos_half, sin_half * rotation_axis[0], 
                               sin_half * rotation_axis[1], sin_half * rotation_axis[2])
                
                # Apply rotation using physics:localRot0 and physics:localRot1
                joint.CreateLocalRot0Attr(quat)
                joint.CreateLocalRot1Attr(quat)
        
        # Set joint limits for a typical door (0 to 120 degrees)
        joint.CreateLowerLimitAttr(-120.0)    # 0 degrees (closed)
        joint.CreateUpperLimitAttr(120.0)  # 120 degrees (open)
        
        # Apply PhysX-specific joint properties for better simulation
        joint_prim = stage.GetPrimAtPath(joint_path)
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        
        # Apply debug torque if requested (for testing joint functionality)
        if apply_debug_torque:
            print(f"Applying debug torque to door joint: {debug_torque_value}")
            # Apply DriveAPI to the joint
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            # Set drive type to velocity control
            drive_api.CreateTypeAttr("force")
            # Set target velocity to make the door rotate
            drive_api.CreateTargetVelocityAttr(debug_torque_value)  # degrees per second
            # Set drive stiffness and damping
            drive_api.CreateStiffnessAttr(0.0)  # No position control
            drive_api.CreateDampingAttr(100)   # High damping for velocity control
            # Set max force
            drive_api.CreateMaxForceAttr(1000.0)  # Maximum force for the drive
            print("Debug torque applied - door should rotate")
        else:
            # Add some damping to make the door movement more realistic without active torque
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive_api.CreateTypeAttr("force")
            drive_api.CreateStiffnessAttr(0.0)  # No position control
            drive_api.CreateDampingAttr(100.0)  # Add damping for realistic movement
            drive_api.CreateMaxForceAttr(1000.0)  # Maximum force for the drive
    
    return stage