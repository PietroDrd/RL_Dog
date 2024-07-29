"""
This script demonstrates the environment for a quadruped robot AlienGo.

    In this example, we use a locomotion policy to control the robot. The robot is commanded to
    move forward at a constant velocity. (?????)

    conda activate isaacenv_
    cd
    cd IsaacLab_
    ./isaaclab.sh -p /home/rl_sim/RL_Dog/IsaacSimLab/aliengo_v3/aliengo_main.py --num_envs 256

Launch Isaac Sim Simulator first.
"""

from omni.isaac.lab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser(description='AlienGo_v1 Environment Configuration')
parser.add_argument('--num_envs',       type=int,   default=128,            help='Number of environments')
parser.add_argument('--env_spacing',    type=float, default=2.5,           help='Environment spacing')
parser.add_argument('--walk',           type=int,   default=0,             help='ask to Walk or not (1,0)')
parser.add_argument("--task",           type=str,   default="AlienGo-v0",  help="Name of the task.")

#parser.add_argument("--headless",       action="store_true",    default=False,  help="GUI or not GUI.")
#parser.add_argument("--video",          action="store_true",    default=False,  help="Record videos during training.")
#parser.add_argument("--video_length",   type=int,               default=200,    help="Length of the recorded video (in steps).")
#parser.add_argument("--video_interval", type=int,               default=2000,   help="Interval between video recordings (in steps).")
#parser.add_argument("--device",         type=str,               default="cpu",  help="cpu or cuda.")
args = parser.parse_args()

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.lab.envs     import ManagerBasedRLEnv
from omni.isaac.lab.envs     import ManagerBasedRLEnvCfg

from aliengo_ppo import PPO_v1

import os
import torch
import gymnasium as gym

from colorama import Fore, Style

import tensorboard
"""
cmd -->     tensorboard --logdir = "/home/rl_sim/RL_Dog/runs     
            http://localhost:6006
"""

MODE = 1

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"

    if not MODE:    # FULL SENSORS (IDEAL config)
        from aliengo_env import AliengoEnvCfg
        env_cfg = AliengoEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
    elif MODE:     # REAL SENSORS (REAL config)
        from aliengo_env_real import AliengoEnvCfg
        env_cfg = AliengoEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs

    gym.register(
        id=args_cli.task,
        entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
        kwargs={'cfg': AliengoEnvCfg}
    )

    env = ManagerBasedRLEnv(cfg=env_cfg)
    agent = PPO_v1(env=env, device=device, verbose=1)
    print(Fore.YELLOW + '[INFO-AlienGo] Start trianing' + Style.RESET_ALL)

    agent.train_sequential(timesteps=20000, headless=False)
    #agent.train_parallel(timesteps=20000, headless=False)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()