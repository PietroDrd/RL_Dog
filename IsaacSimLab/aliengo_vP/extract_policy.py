import torch
import os

path = "/home/rluser/RL_Dog/runs/AlienGo_vP_stoptry_29_08_FULL_STATE_v2/checkpoints/best_agent.pt"
policy_nn = torch.load(path)['policy']
torch.save(policy_nn, "/home/rluser/RL_Dog/models/FULL_STATE__NN_v2.pt")

