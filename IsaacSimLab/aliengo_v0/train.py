"""Script to train RL agent with SKRL.

    conda activate isaacenv_
    cd
    cd IsaacLab_
    ./isaaclab.sh -p ~/RL_Dog/IsaacSimLab/aliengo_v0/train.py --task Isaac-Cartpole-v0 --num_envs 32 --headless --enable_cameras --video

Launch Isaac Sim Simulator first.
"""

import argparse

from omni.isaac.lab.app import AppLauncher # type: ignore

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with SKRL.")
parser.add_argument("--video",          action="store_true",    default=False,  help="Record videos during training.")
parser.add_argument("--video_length",   type=int,               default=400,    help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int,               default=2000,   help="Interval between video recordings (in steps).")
parser.add_argument("--cpu",            action="store_true",    default=False,  help="Use CPU pipeline.")
parser.add_argument("--disable_fabric", action="store_true",    default=False,  help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs",       type=int,               default=None,   help="Number of environments to simulate.")
parser.add_argument("--task",           type=str,               default=None,   help="Name of the task.")
parser.add_argument("--seed",           type=int,               default=None,   help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int,               default=None,   help="RL Policy training iterations.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import numpy as np
import os
from datetime import datetime

from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml


def main():

    """Train SKRL-PPO agent."""

    ############### Train Configs ###############
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")  # NAME OF THE TASK

    # override configuration with command line arguments
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    ############### LOGS ###############
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    dump_yaml(  os.path.join(log_dir, "params", "env.yaml"),    env_cfg)
    dump_yaml(  os.path.join(log_dir, "params", "agent.yaml"),  agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"),     env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"),   agent_cfg)


    ########
    #######
    ######
    ####
    ###
    ##

    return None



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()