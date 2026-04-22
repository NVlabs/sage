# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from sympy import true
import torch
import numpy as np
from curobo.geom.types import Cuboid, WorldConfig
from curobo.util.usd_helper import UsdHelper

from typing import Dict, List, Optional, Union
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from curobo.geom.types import (
    Mesh,
    WorldConfig,
)
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.core.utils.prims as prim_utils
from isaaclab.curobo_tools.curobo_ik_planner import IKPlanner
import omni
from omni.isaac.lab.assets import RigidObject, Articulation


def get_prim_world_pose(cache: UsdGeom.XformCache,
                        prim: Usd.Prim,
                        inverse: bool = False):
    world_transform: Gf.Matrix4d = cache.GetLocalToWorldTransform(prim)
    # get scale:
    scale: Gf.Vec3d = Gf.Vec3d(
        *(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    scale = list(scale)
    t_mat = world_transform.RemoveScaleShear()
    if inverse:
        t_mat = t_mat.GetInverse()

    translation: Gf.Vec3d = t_mat.ExtractTranslation()
    rotation: Gf.Rotation = t_mat.ExtractRotation()
    q = rotation.GetQuaternion()
    orientation = [q.GetReal()] + list(q.GetImaginary())
    t_mat = (Pose.from_list(
        list(translation) + orientation,
        TensorDeviceType()).get_matrix().view(4, 4).cpu().numpy())

    return t_mat, scale


def get_mesh_attrs(
    prim,
    cache=None,
    apply_trasform=False,
) -> Mesh:
    # read cube information
    # scale = prim.GetAttribute("size").Get()
    try:
        points = list(prim.GetAttribute("points").Get())
        points = [np.ravel(x) for x in points]

        faces = list(prim.GetAttribute("faceVertexIndices").Get())

        face_count = list(prim.GetAttribute("faceVertexCounts").Get())

        faces = np.array(faces).reshape(-1, 3)
        if prim.GetAttribute("xformOp:scale").IsValid():
            scale = list(prim.GetAttribute("xformOp:scale").Get())
        else:
            scale = [1.0, 1.0, 1.0]
        size = prim.GetAttribute("size").Get()
        if size is None:
            size = 1
        scale = [s * size for s in scale]

        points = np.array(points)

        if apply_trasform:  # kitchen can be transformed since unmoved in the scene

            mat, scale = get_prim_world_pose(cache, prim)

            # # compute position and orientation on cuda:
            # tensor_mat = torch.as_tensor(mat)

            # tensor_mat[:3, :3] *= torch.as_tensor(scale)
            # transformed_points = (tensor_mat[:3, :3] @ points.T).T + tensor_mat[:3,
            #                                                                     3]
            w = prim.GetAttribute("xformOp:orient").Get().GetReal()
            x, y, z = prim.GetAttribute("xformOp:orient").Get().GetImaginary()

            pos = torch.as_tensor(prim.GetAttribute("xformOp:translate").Get())

            orientation = torch.as_tensor([w, x, y, z])
            transformed_points = math_utils.transform_points(
                torch.as_tensor(points) * scale[0], pos, orientation)

        else:
            mat, scale = get_prim_world_pose(cache, prim)
            transformed_points = points * scale

        return [str(prim.GetPath()), transformed_points, faces, scale]
    except:

        return None


class MotionPlanner:

    def __init__(
        self,
        env=None,
        robot_file="franka.yml",
        world_file="collision_table.yml",
        collision_checker=False,
        only_paths=None,
        reference_prim_path=None,
        ignore_substring=None,
        collision_avoidance_distance=0.01,
    ):
        # init sim env
        self.env = env

        self.only_paths = only_paths
        self.reference_prim_path = reference_prim_path
        self.ignore_substring = ignore_substring

        # self.device = device
        self.tensor_args = TensorDeviceType()

        # # mod this later
        n_obstacle_cuboids = 30
        n_obstacle_mesh = 100

        robot_file = join_path(get_robot_configs_path(), robot_file)
        world_file = join_path(get_world_configs_path(), world_file)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            world_file,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={
                "obb": n_obstacle_cuboids,
                "mesh": n_obstacle_mesh
            },
            interpolation_dt=0.015,
            position_threshold=0.1,
            rotation_threshold=0.5,
            # store_debug_in_result=True,
            num_ik_seeds=32,
            # cspace_threshold=0.1,
            collision_activation_distance=collision_avoidance_distance,
            # velocity_scale=[10.0],      # 10x higher velocity limits
            # acceleration_scale=[10.0],  # 10x higher acceleration limits  
            # jerk_scale=[10.0],         # 10x higher jerk limits
            self_collision_check=True,
            # store_ik_debug=True,
            # ik_opt_iters=50,  # Default is usually 100+
            # # Try different optimization settings
            # use_gradient_descent=True,  # Instead of L-BFGS
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(enable_graph=True)

        joint_limits = self.motion_gen.kinematics.get_joint_limits()
        print("Position limits:", joint_limits.position)

        robot_cfg = load_yaml(join_path(get_robot_configs_path(),
                                        robot_file))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg, self.tensor_args)
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

        self.retract_cfg = self.motion_gen.get_retract_config()
        self.device = env.device

        self.world = WorldConfig()
        self.motion_gen.update_world(self.world)
        self.curobo_ik = IKPlanner(env, device=self.device)

        # init usd helper
        self.usd_help = UsdHelper()
        self.usd_help.load_stage(env.scene.stage)
        self.usd_help.add_world_to_stage(self.world, base_frame="/World")

        # init collision checker
        self.collision_checker = collision_checker
        if self.collision_checker:
            self.init_collision_mesh()

    def plan_batch_motion(self,
                          qpos,
                          target_position,
                          target_quat,
                          return_ee_pose=True,
                          jpos_dim=7):

        start = JointState.from_position(qpos[..., :jpos_dim])

        goal = Pose(target_position, target_quat)

        result = self.motion_gen.plan_batch(start,
                                            goal,
                                            plan_config=MotionGenPlanConfig(
                                                enable_graph=False,
                                                max_attempts=2,
                                                enable_finetune_trajopt=True))

    def plan_motion(
        self,
        qpos,
        target_position,
        target_quat,
        return_ee_pose=True,
        jpos_dim=7,
        max_attempts=10
    ):

        start = JointState.from_position(qpos[..., :jpos_dim])

        
        

        # self.curobo_ik.ik_solver.fk(qpos[..., :jpos_dim])

        goal = Pose(target_position, target_quat)

        result = self.motion_gen.plan_single(
            start, goal, MotionGenPlanConfig(
                max_attempts=max_attempts,
                # ik_fail_return=10,
                # enable_graph=True,
                # enable_opt=True,
                # enable_finetune_trajopt=True,
                # enable_graph_attempt=10,
            ))
        # get all the keys of result.debug_info['ik_result']
        # print("js_solution", result.debug_info['ik_result'].js_solution)
        # print("goal_pose", result.debug_info['ik_result'].goal_pose)
        # print("solution", result.debug_info['ik_result'].solution)
        # print("seed", result.debug_info['ik_result'].seed)
        # print("success", result.debug_info['ik_result'].success)
        # print("position_error", result.debug_info['ik_result'].position_error)
        # print("rotation_error", result.debug_info['ik_result'].rotation_error)
        # print("error", result.debug_info['ik_result'].error)
        # print("goalset_index", result.debug_info['ik_result'].goalset_index)

        

        # print("result.debug_info['ik_result'].solution.shape", result.debug_info['ik_result'].solution.shape)


        # metrics = self.motion_gen.check_constraints(
        #     JointState(
        #         position=result.debug_info['ik_result'].solution[0, 0:1],
        #         joint_names=self.motion_gen.joint_names[:7],
        #     )
        # )
        # print(f"solution: {result.debug_info['ik_result'].solution[0, 0:1]}")
        # print(f"Feasible: {metrics.feasible.item()}")
        # print(f"Constraint value: {metrics.constraint.item()}")

        # print("Robot config file:", self.motion_gen.robot_cfg) 
        # print("Robot config path:", self.motion_gen.robot_cfg.kinematics.kinematics_config)

        # position_limits = self.motion_gen.kinematics.get_joint_limits().position[:7]
        # print(f"position_limits: {position_limits}")

        # print("Joint violations:")
        # for i, (val, lower, upper) in enumerate(zip(result.debug_info['ik_result'].solution[0, 0], position_limits[0], position_limits[1])):
        #     violation = max(0, lower - val) + max(0, val - upper)
        #     print(f"Joint {i+1}: {val:.4f} in [{lower:.4f}, {upper:.4f}], violation: {violation:.4f}")

        # print(f"result: {result.debug_info['ik_result'].debug_info['solver'].keys()}")
        # print(f"result steps length: {len(result.debug_info['ik_result'].debug_info['solver']['steps'])}")
        # print(f"result steps 0 length: {len(result.debug_info['ik_result'].debug_info['solver']['steps'][0])}")
        # print(f"result steps 0, 0: {result.debug_info['ik_result'].debug_info['solver']['steps'][0][0].shape}")
        # for i in range(len(result.debug_info['ik_result'].debug_info['solver']['steps'][0])):
        #     print(f"result steps 0, {i} start: {result.debug_info['ik_result'].debug_info['solver']['steps'][0][i][0]}")
        #     print(f"result steps 0, {i} end: {result.debug_info['ik_result'].debug_info['solver']['steps'][0][i][-1]}")
        # print(f"result steps 0, 0 start: {result.debug_info['ik_result'].debug_info['solver']['steps'][0][0][0]}")
        # print(f"result steps 0, 0 end: {result.debug_info['ik_result'].debug_info['solver']['steps'][0][0][-1]}")
        # print(f"result steps 0, 0: {result.debug_info['ik_result'].debug_info['solver']['steps'][0][1].shape}")
        # print(f"result steps 0, 0: {result.debug_info['ik_result'].debug_info['solver']['steps'][0][2].shape}")
        # print(f"result steps 0, 0: {result.debug_info['ik_result'].debug_info['solver']['steps'][0][3].shape}")
        # print(f"result steps 1 length: {len(result.debug_info['ik_result'].debug_info['solver']['steps'][1])}")
        # print(f"result steps 1 0 length: {len(result.debug_info['ik_result'].debug_info['solver']['steps'][1][0])}")
        # print(f"result cost length: {len(result.debug_info['ik_result'].debug_info['solver']['cost'])}")
        # print(f"result cost 0 length: {len(result.debug_info['ik_result'].debug_info['solver']['cost'][0])}")
        # print(f"result cost 0 0 length: {len(result.debug_info['ik_result'].debug_info['solver']['cost'][0][0])}")
        # print(f"result cost 0 0 shape: {result.debug_info['ik_result'].debug_info['solver']['cost'][0][0].shape}")
        if result.success.item():
            traj = result.get_interpolated_plan()

        else:

            traj = None

        if traj is not None:
            print(
                f"Trajectory Generated: success {result.success.item()} | len {len(traj)} | optimized_dt {result.optimized_dt.item()}"
            )

        # replace joint position with ee pose
        ee_pose = None

        if return_ee_pose and traj is not None:
            ee_pose = self.kin_model.get_state(traj.position)

        return ee_pose, traj

    def update_world(self, cuboids):

        self.world = WorldConfig(cuboid=cuboids)

        self.motion_gen.update_world(self.world)

    def attach_obstacle(self, name, qpos):
        obstacle = name

        start_state = JointState.from_position(
            torch.as_tensor(qpos, dtype=torch.float32, device=self.device))
        
        self.motion_gen.attach_objects_to_robot(start_state, [obstacle], 
            surface_sphere_radius=0.1,
            # world_objects_pose_offset=Pose(
            #     position=pose[:, :3],
            #     quaternion=pose[:, 3:7]
            # )
        )

    def detach_obstacle(self) -> None:
        self.motion_gen.detach_object_from_robot()

    def add_obstacle(self, plan_grasp=False, target_object_name=None) -> None:

        obstacles = {}
        obstacles["mesh"] = []
        robot = self.env.scene["robot"]
        robot_root_state = robot._data.root_state_w[0, :7]


        for key in self.obstacles_mesh.keys():

            meta_data = self.obstacles_mesh[key]
            prim_path = meta_data[0]

            # if "visuals" not in prim_path:

            prim_name = prim_path.split("/")[4]
            

            if plan_grasp:
                if target_object_name == key:
                    print(f"target_object_name: {target_object_name}; prim_name: {prim_name}")
                    # vertices = meta_data[1]
                    # get the bbox of the object
                    # vertices = vertices.copy()
                    # bbox_min = np.min(vertices, axis=0)
                    # bbox_max = np.max(vertices, axis=0)
                    # print(f"bbox_min: {bbox_min}; bbox_max: {bbox_max}")
                    # assert False
                    continue
            if isinstance(self.env.scene[prim_name], RigidObject):
                object = self.env.scene[prim_name]

                object_root_state = object._data.root_state_w[0, :7]
            elif isinstance(self.env.scene[prim_name], Articulation):
                articulated_object = self.env.scene[prim_name]

                id, _ = articulated_object.find_bodies(prim_path.split("/")[5])
                print(f"articulated_object: {prim_path.split('/')[5]}")
                object_root_state = articulated_object._data.body_state_w[
                    0, id[0], :7]
            else:
                object_root_state = torch.zeros(7, device=self.device)
                object_root_state[3:4] = torch.ones(1, device=self.device)
                # continue
                # assert False

            # elif "collision" in prim_path:

            #     sub_prim = self.env.scene.stage.GetPrimAtPath(prim_path)

            #     # true_prim_name = '_'.join(prim_name.lower().split('_')[:2])
            #     # rigid_collections = self.env.scene[true_prim_name]
            #     # rigid_bodies_id, _ = rigid_collections.find_objects(prim_name)
            #     # object_root_state = rigid_collections._data.object_state_w[
            #     #     0, rigid_bodies_id[0], :7]
            #     world_transform = self.usd_help._xform_cache.GetLocalToWorldTransform(
            #         sub_prim)
            #     transform_matrix = torch.as_tensor(world_transform)
            #     transform_quat = math_utils.quat_from_matrix(
            #         transform_matrix[:3, :3])
            #     object_root_state = torch.cat(
            #         [transform_matrix[3, :3], transform_quat]).to(self.device)

            # else:

            #     continue

            robot2object_pos, robot2object_quat = math_utils.subtract_frame_transforms(
                robot_root_state[:3], robot_root_state[3:7],
                object_root_state[:3], object_root_state[3:7])

            m_data = Mesh(name=meta_data[0],
                          vertices=meta_data[1].tolist(),
                          faces=meta_data[2].tolist(),
                          pose=torch.cat([robot2object_pos,
                                          robot2object_quat]).tolist(),
                          scale=[1, 1, 1])
            obstacles["mesh"].append(m_data)
            # print(f"add obstacle: {meta_data[0]}")

        world_model = WorldConfig(**obstacles)

        print(f"world_model.cuboid: {len(world_model.cuboid)}")
        print(f"world_model.mesh: {len(world_model.mesh)}")
        obstacles = world_model.get_collision_check_world()

        self.motion_gen.update_world(obstacles)

        obstacles.save_world_as_mesh("isaaclab/test_obstacles/obstacles.ply")

    def init_collision_mesh(
        self,
        timecode: float = 0,
    ):

        self.usd_help._xform_cache.Clear()
        self.usd_help._xform_cache.SetTime(timecode)

        all_items = self.usd_help.stage.Traverse()

        self.obstacles_mesh = {}

        obstacles = {}
        obstacles["mesh"] = []

        for x in all_items:

            if self.only_paths is not None:
                if not any(
                    [str(x.GetPath()).startswith(k) for k in self.only_paths]):
                    continue

            if self.ignore_substring is not None:
                if any([k in str(x.GetPath()) for k in self.ignore_substring]):
                    continue

            if x.IsA(UsdGeom.Mesh) and "collision" not in str(
                    x.GetPath().pathString) and "visuals" not in str(
                        x.GetPath().pathString):

                m_data = get_mesh_attrs(
                    x,
                    cache=self.usd_help._xform_cache,
                )

                if m_data is not None:
                    # print(f"add: ", x.GetPath().pathString.split("/"), "   |||  ", x.GetPath().pathString.split("/")
                    #                     [-2])
                    self.obstacles_mesh[x.GetPath().pathString.split("/")
                                        [-2]] = m_data
            elif "collision" in str(x.GetPath().pathString):
                # print(f"x in 1: {x.GetPath().pathString}")
                sub_mesh = x.GetChildren()

                if sub_mesh is None:

                    if x.IsA(UsdGeom.Mesh):
                        m_data = get_mesh_attrs(
                            x,
                            cache=self.usd_help._xform_cache,
                            apply_trasform=True,
                        )

                        self.obstacles_mesh[x.GetPath().pathString.split("/")
                                            [-2]] = m_data
                else:
                    points_buffer = []
                    faces_buffer = []
                    faces_count = 0
                    for mesh in sub_mesh:
                        if mesh.IsA(UsdGeom.Mesh):

                            m_data = get_mesh_attrs(
                                mesh,
                                cache=self.usd_help._xform_cache,
                                apply_trasform=True,
                            )
                            if m_data is not None:
                                points_buffer.append(m_data[1])
                                faces_buffer.append(m_data[2] + faces_count)
                                faces_count += len(m_data[2])
                                name = m_data[0]
                                scale = m_data[3]

                    if len(points_buffer) > 0:
                        final_m_data = [
                            name,
                            np.concatenate(points_buffer),
                            np.concatenate(faces_buffer), scale
                        ]

                        self.obstacles_mesh[x.GetPath().pathString.split("/")
                                            [-2]] = final_m_data

        try:

            for name in self.obstacles_mesh.keys():
                if "visuals" in name:
                    self.obstacles_mesh.pop(name, None)
                    # print(f"pop name: {name}")
            self.obstacles_mesh.pop("collisions", None)
            self.obstacles_mesh.pop("visuals", None)

        except:
            pass

    def clear_obstacles(self):
        self.motion_gen.clear_world_cache()

    def remove_obstacle(self, str):
        self.world.remove_obstacle(str)
        #self.motion_gen.update_world(self.world)
