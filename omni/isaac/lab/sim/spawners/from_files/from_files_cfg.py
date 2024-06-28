# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from omni.isaac.lab.sim import converters, schemas
from omni.isaac.lab.sim.spawners import materials
from omni.isaac.lab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg, SpawnerCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from . import from_files


@configclass
class FileCfg(RigidObjectSpawnerCfg):
    """Configuration parameters for spawning an asset from a file.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.
    """

    scale: tuple[float, float, float] | None = None
    """Scale of the asset. Defaults to None, in which case the scale is not modified."""

    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None
    """Properties to apply to the articulation root."""

    fixed_tendons_props: schemas.FixedTendonsPropertiesCfg | None = None
    """Properties to apply to the fixed tendons (if any)."""

    joint_drive_props: schemas.JointDrivePropertiesCfg | None = None
    """Properties to apply to a joint."""

    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties to override the visual material properties in the URDF file.

    Note:
        If None, then no visual material will be added.
    """


@configclass
class UsdFileCfg(FileCfg):
    """USD file to spawn asset from.

    See :meth:`spawn_from_usd` for more information.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.
    """

    func: Callable = from_files.spawn_from_usd

    usd_path: str = MISSING
    """Path to the USD file to spawn asset from."""

    variants: object | dict[str, str] | None = None
    """Variants to select from in the input USD file. Defaults to None, in which case no variants are applied.

    This can either be a configclass object, in which case each attribute is used as a variant set name and its specified value,
    or a dictionary mapping between the two. Please check the :meth:`~omni.isaac.lab.sim.utils.select_usd_variants` function
    for more information.
    """


@configclass
class UrdfFileCfg(FileCfg, converters.UrdfConverterCfg):
    """URDF file to spawn asset from.

    It uses the :class:`UrdfConverter` class to create a USD file from URDF and spawns the imported
    USD file. See :meth:`spawn_from_urdf` for more information.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.
    """

    func: Callable = from_files.spawn_from_urdf


"""
Spawning ground plane.
"""


@configclass
class GroundPlaneCfg(SpawnerCfg):
    """Create a ground plane prim.

    This uses the USD for the standard grid-world ground plane from Isaac Sim by default.
    """

    func: Callable = from_files.spawn_ground_plane

    usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"
    """Path to the USD file to spawn asset from. Defaults to the grid-world ground plane."""

    color: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    """The color of the ground plane. Defaults to (0.0, 0.0, 0.0).

    If None, then the color remains unchanged.
    """

    size: tuple[float, float] = (100.0, 100.0)
    """The size of the ground plane. Defaults to 100 m x 100 m."""

    physics_material: materials.RigidBodyMaterialCfg = materials.RigidBodyMaterialCfg()
    """Physics material properties. Defaults to the default rigid body material."""
