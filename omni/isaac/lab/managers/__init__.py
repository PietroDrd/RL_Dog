# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for environment managers.

The managers are used to handle various aspects of the environment such as randomization events, curriculum,
and observations. Each manager implements a specific functionality for the environment. The managers are
designed to be modular and can be easily extended to support new functionality.
"""

from .action_manager import ActionManager, ActionTerm
from .command_manager import CommandManager, CommandTerm
from .curriculum_manager import CurriculumManager
from .event_manager import EventManager, RandomizationManager
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import (
    ActionTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ManagerTermBaseCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RandomizationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from .observation_manager import ObservationManager
from .reward_manager import RewardManager
from .scene_entity_cfg import SceneEntityCfg
from .termination_manager import TerminationManager
