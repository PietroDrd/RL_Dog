
## BODY order:
['base', 'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip', 'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']

## JOINT order:
['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']

## To Extract Policy:
```
path = "/home/rluser/RL_Dog/runs/AlienGo_vP_stoptry_**EXPERIMENT_NAME**/checkpoints/best_agent.pt"
policy_nn = torch.load(path)['policy']
torch.save(policy_nn, "/home/rluser/RL_Dog/models/policy_nn_NAME.pt")
```

### Observation Space: 37

NOTE:  JOINT's Position and Velocities are <u>RELATIVE</u> to the default ones
```
# e.g: Jpos_1 = Jpos1_Actual - Jpos1_Default

asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
```

- **`Pose`**: x, y, z 
- **`Quat`**: w, x, y, z
- **`Lvel`**: vx, vy, vz
- **`Avel`**: r, p, y
- **`Jpos`**: 'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
 - **`Jvel`**: 'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'


### Action Space: 12
- **`Jpos`**
