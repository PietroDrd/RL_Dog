# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers.config import CONTACT_SENSOR_MARKER_CFG
from omni.isaac.lab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .contact_sensor import ContactSensor


@configclass
class ContactSensorCfg(SensorBaseCfg):
    """Configuration for the contact sensor."""

    class_type: type = ContactSensor

    track_pose: bool = False
    """Whether to track the pose of the sensor's origin. Defaults to False."""

    track_air_time: bool = False
    """Whether to track the air/contact time of the bodies (time between contacts). Defaults to False."""

    force_threshold: float = 1.0
    """The threshold on the norm of the contact force that determines whether two bodies are in collision or not.

    This value is only used for tracking the mode duration (the time in contact or in air),
    if :attr:`track_air_time` is True.
    """

    filter_prim_paths_expr: list[str] = list()
    """The list of primitive paths (or expressions) to filter contacts with. Defaults to an empty list, in which case
    no filtering is applied.

    The contact sensor allows reporting contacts between the primitive specified with :attr:`prim_path` and
    other primitives in the scene. For instance, in a scene containing a robot, a ground plane and an object,
    you can obtain individual contact reports of the base of the robot with the ground plane and the object.

    .. note::
        The expression in the list can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Object`` will be replaced with ``/World/envs/env_.*/Object``.

    """

    visualizer_cfg: VisualizationMarkersCfg = CONTACT_SENSOR_MARKER_CFG.replace(prim_path="/Visuals/ContactSensor")
    """The configuration object for the visualization markers. Defaults to CONTACT_SENSOR_MARKER_CFG.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """
