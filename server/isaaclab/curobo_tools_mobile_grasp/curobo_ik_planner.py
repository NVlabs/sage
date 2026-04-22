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
import time
import copy
from typing import Dict, List, Optional, Union
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from curobo.geom.types import (
    Capsule,
    Cuboid,
    Cylinder,
    Material,
    Mesh,
    Obstacle,
    Sphere,
    WorldConfig,
)
import omni.isaac.lab.utils.math as math_utils



def get_pose_grid(n_x, n_y, n_z, max_x, max_y, max_z, min_x, min_y, min_z):
    x = np.linspace(min_x, max_x, n_x)
    y = np.linspace(min_y, max_y, n_y)
    z = np.linspace(min_z, max_z, n_z)
    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    position_arr = np.zeros((n_x * n_y * n_z, 3))
    position_arr[:, 0] = x.flatten()
    position_arr[:, 1] = y.flatten()
    position_arr[:, 2] = z.flatten()
    return position_arr


class IKPlanner:

    def __init__(
        self,
        env=None,
        device=None,
        robot_file="omron_franka_grasp.yml",
        world_file="collision_table.yml",
        only_paths=None,
        reference_prim_path=None,
        ignore_substring=None,
    ):
        robot_file = join_path(get_robot_configs_path(), robot_file)
        world_file = join_path(get_world_configs_path(), world_file)

        self.device = device
        self.tensor_args = TensorDeviceType()

        robot_cfg = load_yaml(join_path(get_robot_configs_path(),
                                        robot_file))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg, self.tensor_args)

        n_obstacle_cuboids = 30
        n_obstacle_mesh = 10
        from curobo.geom.types import Cuboid, WorldConfig
        self.ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            WorldConfig(),
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={
                "obb": n_obstacle_cuboids,
                "mesh": n_obstacle_mesh
            },
            collision_activation_distance=0.00
            # use_fixed_samples=True,
        )
        self.ik_solver = IKSolver(self.ik_config)
        self.usd_help = UsdHelper()
        if env is not None:
            self.usd_help.load_stage(env.scene.stage)
            from curobo.geom.types import Cuboid, WorldConfig
            self.world = WorldConfig()
            self.usd_help.add_world_to_stage(self.world, base_frame="/World")

            self.only_paths = only_paths
            self.reference_prim_path = reference_prim_path
            self.ignore_substring = ignore_substring

    def plan_motion(
        self,
        target_position,
        target_quat,
        # default_joints,
    ):

        goal = Pose(target_position, target_quat)

        result = self.ik_solver.solve_batch(goal)

        torch.cuda.empty_cache()
        return result.js_solution.position

    def init_reachability(self, offset=None, grid_size=5):
        self.position_grid_offset = self.tensor_args.to_device(
            get_pose_grid(grid_size, grid_size, grid_size, offset[0],
                          offset[1], offset[2], offset[3], offset[4],
                          offset[5]))
        # read current ik pose and warmup?

        fk_state = self.ik_solver.fk(self.ik_solver.get_retract_config().view(
            1, -1))
        goal_pose = fk_state.ee_pose
        goal_pose = goal_pose.repeat(self.position_grid_offset.shape[0])
        goal_pose.position = self.position_grid_offset

        result = self.ik_solver.solve_batch(goal_pose)
        self.draw_goal_pose = goal_pose
        self.draw_success = result.success

        return torch.sum(result.success.int())

    def draw_points(self):
        # Third Party
        from omni.isaac.debug_draw import _debug_draw
        pose = self.draw_goal_pose
        success = self.draw_success

        draw = _debug_draw.acquire_debug_draw_interface()
        N = 100
        # if draw.get_num_points() > 0:
        draw.clear_points()
        cpu_pos = pose.position.cpu().numpy()
        b, _ = cpu_pos.shape
        point_list = []
        colors = []
        for i in range(b):
            # get list of points:
            point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
            if success[i].item():
                colors += [(0, 1, 0, 0.25)]
            else:
                colors += [(1, 0, 0, 0.25)]
        sizes = [40.0 for _ in range(b)]

        draw.draw_points(point_list, colors, sizes)
