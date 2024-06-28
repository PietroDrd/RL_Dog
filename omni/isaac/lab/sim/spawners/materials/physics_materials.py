# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

from omni.isaac.lab.sim.utils import clone, safe_set_attribute_on_usd_schema

if TYPE_CHECKING:
    from . import physics_materials_cfg


@clone
def spawn_rigid_body_material(prim_path: str, cfg: physics_materials_cfg.RigidBodyMaterialCfg) -> Usd.Prim:
    """Create material with rigid-body physics properties.

    Rigid body materials are used to define the physical properties to meshes of a rigid body. These
    include the friction, restitution, and their respective combination modes. For more information on
    rigid body material, please refer to the `documentation on PxMaterial <https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/_build/physx/latest/class_px_material.html>`_.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration for the physics material.

    Returns:
        The spawned rigid body material prim.

    Raises:
        ValueError:  When a prim already exists at the specified prim path and is not a material.
    """
    # create material prim if no prim exists
    if not prim_utils.is_prim_path_valid(prim_path):
        _ = UsdShade.Material.Define(stage_utils.get_current_stage(), prim_path)

    # obtain prim
    prim = prim_utils.get_prim_at_path(prim_path)
    # check if prim is a material
    if not prim.IsA(UsdShade.Material):
        raise ValueError(f"A prim already exists at path: '{prim_path}' but is not a material.")
    # retrieve the USD rigid-body api
    usd_physics_material_api = UsdPhysics.MaterialAPI(prim)
    if not usd_physics_material_api:
        usd_physics_material_api = UsdPhysics.MaterialAPI.Apply(prim)
    # retrieve the collision api
    physx_material_api = PhysxSchema.PhysxMaterialAPI(prim)
    if not physx_material_api:
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)

    # convert to dict
    cfg = cfg.to_dict()
    del cfg["func"]
    # set into USD API
    for attr_name in ["static_friction", "dynamic_friction", "restitution"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_physics_material_api, attr_name, value, camel_case=True)
    # set into PhysX API
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_material_api, attr_name, value, camel_case=True)
    # return the prim
    return prim
