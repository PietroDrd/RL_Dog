import torch
import os

path = "/home/rluser/RL_Dog/runs/AlienGo_vP_stoptry_22_08_FULL_STATE/checkpoints/agent_21000.pt"
something = torch.load(path)
state_dict = torch.load(path)['policy']
torch.save(something, "/home/rluser/RL_Dog/models/something.pt")

