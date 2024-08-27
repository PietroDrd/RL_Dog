import torch
import os

path = "/home/rluser/RL_Dog/runs/AlienGo_vP_stoptry_22_08_FULL_STATE/checkpoints/agent_21000.pt"
policy_nn = torch.load(path)['policy']
torch.save(policy_nn, "/home/rluser/RL_Dog/models/policy_nn.pt")

