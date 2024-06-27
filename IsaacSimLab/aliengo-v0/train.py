"""Script to train RL agent with SKRL.

    conda activate isaacenv_
    cd
    cd IsaacLab_
    ./isaaclab.sh -p ~/RL_Dog/IsaacSimLab/aliengo-v0/train.py --task Isaac-Cartpole-v0 --num_envs 32 --headless --enable_cameras --video

Launch Isaac Sim Simulator first.
"""

import argparse

from omni.isaac.lab.app import AppLauncher # type: ignore

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video",          action="store_true",    default=False,  help="Record videos during training.")
parser.add_argument("--video_length",   type=int,               default=200,    help="Length of the recorded video (in steps).")
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


def main():

    return None



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()