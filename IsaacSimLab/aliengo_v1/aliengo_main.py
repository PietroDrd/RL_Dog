"""
This script demonstrates the environment for a quadruped robot AlienGo.

    In this example, we use a locomotion policy to control the robot. The robot is commanded to
    move forward at a constant velocity. (?????)

    

Launch Isaac Sim Simulator first.
"""

from omni.isaac.lab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser(description='AlienGo_v1 Environment Configuration')
parser.add_argument('--num_envs',       type=int,               default=16,     help='Number of environments')
parser.add_argument('--env_spacing',    type=float,             default=2.5,    help='Environment spacing')
parser.add_argument('--walk',           type=int,               default=0,      help='ask to Walk or not (1,0)')

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

from omni.isaac.lab.envs     import ManagerBasedRLEnv, ManagerBasedEnv
from aliengo_env import AliengoEnvCfg
from aliengo_ppo import PPO_v1

import torch
import gymnasium as gym

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = AliengoEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    #gym.register("alienv")
    #env = gym.make("alienv", cfg=env_cfg)
    
    agent = PPO_v1(env=env, device=device, verbose=1) # doesn't crash
    
    #obs, _extras_ = env.reset()   # gives problems !!
    # try with predefined skrl trained 
    
    #agent.train_mine_easy()
    #agent.train_sequential()
    #agent.train_parallel()

    import time
    time.sleep(5)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
