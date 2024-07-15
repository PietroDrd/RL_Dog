"""
This script demonstrates the environment for a quadruped robot AlienGo.

    In this example, we use a locomotion policy to control the robot. The robot is commanded to
    move forward at a constant velocity. (?????)

    conda activate isaacenv_
    cd
    cd IsaacLab_
    ./isaaclab.sh -p /home/rl_sim/RL_Dog/IsaacSimLab/aliengo_v1/aliengo_main.py --num_envs 32

Launch Isaac Sim Simulator first.
"""

from omni.isaac.lab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser(description='AlienGo_v1 Environment Configuration')
parser.add_argument('--num_envs',       type=int,   default=16,            help='Number of environments')
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

from aliengo_env import AliengoEnvCfg
from aliengo_ppo import PPO_v1

import torch
import gymnasium as gym


""" 
    MODE:
    0 -> just check the environment, no training, no policy
    1 -> Wrap, Register and Make the environment
    2 -> Train the Policy
"""
MODE = 1



def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = AliengoEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    gym.register(
        id=args_cli.task,
        entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
        kwargs={'cfg': AliengoEnvCfg}
    )
    
    match MODE:
        case 0: mode0_aka_check(env_cfg)
        case 1: mode1_aka_doEnv(env_cfg, device)
        case 2: mode2_aka_train(env_cfg)
    print("[INFO]: Simulation END ---> Closing IsaacSim")


def mode0_aka_check(env_cfg: ManagerBasedRLEnvCfg):
    env = ManagerBasedRLEnv(cfg=env_cfg)
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1200 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            joint_efforts = torch.randn_like(env.action_manager.action)
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            count += 1
    env.close()

from omni.isaac.lab_tasks.utils import parse_env_cfg
def mode1_aka_doEnv(env_cfg: ManagerBasedRLEnvCfg, device):
    
    env = gym.make(args_cli.task, env_cfg=env_cfg)
    # agent = PPO_v1(env=env, device=device, verbose=1)

    # count = 0
    # while simulation_app.is_running():
    #     with torch.inference_mode():
    #         # reset
    #         if count % 1200 == 0:
    #             count = 0
    #             env.reset()
    #             print("-" * 80)
    #             print("[INFO]: Resetting environment...")
    #         #joint_efforts = torch.randn_like(env.action_manager.action)
    #         #obs, rew, terminated, truncated, info = env.step(joint_efforts)
    #         count += 1
    env.close()

def mode2_aka_train(env_cfg: ManagerBasedRLEnvCfg):
    # TODO: Implement mode 2 functionality
    pass

if __name__ == "__main__":
    main()
    simulation_app.close()
