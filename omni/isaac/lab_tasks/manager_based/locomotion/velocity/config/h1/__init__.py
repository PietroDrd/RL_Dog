# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Velocity-Rough-H1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.H1RoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.H1RoughPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Rough-H1-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.H1RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.H1RoughPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Flat-H1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.H1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.H1FlatPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Flat-H1-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.H1FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.H1FlatPPORunnerCfg,
    },
)
