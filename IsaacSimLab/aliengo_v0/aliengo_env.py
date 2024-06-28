# here i will create the custom environment, wrapped by skrl
"""
            ### NOT SURE IT IS NECESSARY ###

import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""

import math
import torch

import omni.isaac.lab.sim        as sim_utils
import omni.isaac.lab.envs.mdp   as mdp

from omni.isaac.lab.envs     import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.assets   import ArticulationCfg, AssetBaseCfg

from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.utils.noise   import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.terrains.config.rough   import ROUGH_TERRAINS_CFG
from unitree                                import UNITREE_AlienGo_CFG


######### SCENE #########

@configclass
class BaseSceneCfg(InteractiveSceneCfg):

    #############################################

    def __init__(self, rough_terrain=0):

        self.ROUGH_TERRAIN = rough_terrain
        self.terrain = self._setup_terrain()
        self.robot = self._setup_robot()
        self.light = self._setup_light()
        self.dome_light = self._setup_dome_light()

    #############################################

    ### GROUND - TERRAIN ###
    def _setup_terrain(self):
        if self.ROUGH_TERRAIN:
            return TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=ROUGH_TERRAINS_CFG,
                max_init_terrain_level=5,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                debug_vis=False,
            )
        else:
            return AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
            )
    
    ### ROBOT ###
    def _setup_robot(self):
        return UNITREE_AlienGo_CFG
    
    ### LIGHTS ###
    def _setup_light(self):
        return AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
    def _setup_dome_light(self):
        return AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=400.0),
        )

######### MDP - RL #########


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)





### TO CONTINUE 