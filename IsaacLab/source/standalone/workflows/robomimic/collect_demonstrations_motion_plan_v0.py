import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Single-Obj-Scene-Franka-IK-Rel-v0", help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--num_demos", type=int, default=1, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import torch
import numpy as np

from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.data_collector import RobomimicDataCollector
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def reinitialize_env(env):
    robot = env.unwrapped.scene["robot"]
    base_pos = (7.59, 6.76, 0.9)
    sim = env.unwrapped.sim

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    # print(f"joint_pos: {joint_pos.shape}; {joint_pos}")
    # print(f"joint_vel: {joint_vel.shape}; {joint_vel}")
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # write_root_link_state_to_sim
    base_w = torch.zeros(1, 13).to(sim.device)
    random_pos = np.random.uniform(-0.05, 0.05, 3)
    robot_base = torch.tensor([
        base_pos[0]+random_pos[0], 
        base_pos[1]+random_pos[1], 
        base_pos[2]+random_pos[2], 
    ]).to(sim.device)
    base_w[..., :3] = robot_base
    base_w[..., 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(sim.device)

    robot.write_root_link_state_to_sim(base_w)
    robot.reset()

    print("Env reinitialized.")


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    # assert (
    #     args_cli.task == "Isaac-Lift-Cube-Franka-IK-Rel-v0"
    # ), "Only 'Isaac-Lift-Cube-Franka-IK-Rel-v0' is supported currently."
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # modify configuration such that the environment runs indefinitely
    # until goal is reached
    env_cfg.terminations.time_out = None
    # set the resampling time range to large number to avoid resampling
    # env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # add termination condition for reaching the goal otherwise the environment won't reset
    # env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # # create controller
    # if args_cli.teleop_device.lower() == "keyboard":
    #     teleop_interface = Se3Keyboard(pos_sensitivity=0.04, rot_sensitivity=0.08)
    # elif args_cli.teleop_device.lower() == "spacemouse":
    #     teleop_interface = Se3SpaceMouse(pos_sensitivity=0.05, rot_sensitivity=0.005)
    # else:
    #     raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # # add teleoperation key for env reset
    # teleop_interface.add_callback("L", env.reset)
    # # print helper
    # print(teleop_interface)

    # specify directory for logging experiments
    log_dir = os.path.join("./logs/robomimic", args_cli.task)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
        flush_freq=args_cli.num_envs,
        # env_config={"teleop_device": args_cli.teleop_device},
    )

    # reset environment
    obs_dict, _ = env.reset()

    # reset interfaces
    # teleop_interface.reset()
    collector_interface.reset()

    iteration = 0

    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while not collector_interface.is_stopped():
            # # get keyboard command
            # delta_pose, gripper_command = teleop_interface.advance()
            # # convert to torch
            # delta_pose = torch.tensor(delta_pose, dtype=torch.float, device="cuda").repeat(args_cli.num_envs, 1)
            # # compute actions based on environment
            # actions = pre_process_actions(delta_pose, gripper_command)

            if iteration % 100 == 0:
                reinitialize_env(env)

            # TODO: Get actions from motion planner
            actions = torch.zeros((args_cli.num_envs, 7), dtype=torch.float, device="cuda")


            # TODO: Deal with the case when reset is triggered by teleoperation device.
            #   The observations need to be recollected.
            # store signals before stepping
            # -- obs
            for key, value in obs_dict["policy"].items():
                # print("obs", key, value.shape, value)
                collector_interface.add(f"obs/{key}", value)
            # -- actions
            # print("actions", actions.shape, actions)
            collector_interface.add("actions", actions)

            # perform action on environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated
            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

            # robomimic only cares about policy observations
            # store signals from the environment
            # -- next_obs
            for key, value in obs_dict["policy"].items():
                # print("next_obs", key, value.shape, value)
                collector_interface.add(f"next_obs/{key}", value)
            # -- rewards
            # print("rewards", rewards)
            collector_interface.add("rewards", rewards)
            # -- dones
            # print("dones", dones)
            collector_interface.add("dones", dones)

            # -- is success label
            # collector_interface.add("success", env.unwrapped.termination_manager.get_term("object_reached_goal"))

            # flush data from collector for successful environments
            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            collector_interface.flush(reset_env_ids)

            # print("env scene: ", env.unwrapped.scene)
            # print("env scene robot: ", env.unwrapped.scene["robot"])
            # print("env scene rigid_object_collection object_com_state_w: ", 
            # env.unwrapped.scene["rigid_object_collection"].data.object_com_state_w)

            # check if enough data is collected
            if collector_interface.is_stopped():
                break

            iteration += 1

    # close the simulator
    collector_interface.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
