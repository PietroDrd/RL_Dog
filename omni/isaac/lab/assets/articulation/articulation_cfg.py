# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.actuators import ActuatorBaseCfg
from omni.isaac.lab.utils import configclass

from ..rigid_object import RigidObjectCfg
from .articulation import Articulation


@configclass
class ArticulationCfg(RigidObjectCfg):
    """Configuration parameters for an articulation."""

    class_type: type = Articulation

    @configclass
    class InitialStateCfg(RigidObjectCfg.InitialStateCfg):
        """Initial state of the articulation."""

        # root position
        joint_pos: dict[str, float] = {".*": 0.0}
        """Joint positions of the joints. Defaults to 0.0 for all joints."""
        joint_vel: dict[str, float] = {".*": 0.0}
        """Joint velocities of the joints. Defaults to 0.0 for all joints."""

    ##
    # Initialize configurations.
    ##

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero joint state."""
    soft_joint_pos_limit_factor: float = 1.0
    """Fraction specifying the range of DOF position limits (parsed from the asset) to use.
    Defaults to 1.0."""
    actuators: dict[str, ActuatorBaseCfg] = MISSING
    """Actuators for the robot with corresponding joint names."""
