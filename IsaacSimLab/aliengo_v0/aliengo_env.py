# here i will create the custom environment, wrapped by skrl
"""
            ### NOT SURE IT IS NECESSARY ###

from omni.isaac.lab.app import AppLauncher


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Quadruped Environment Configuration')
    parser.add_argument('--num_envs', type=int, default=16, help='Number of environments')
    parser.add_argument('--env_spacing', type=float, default=2.5, help='Environment spacing')
    args = parser.parse_args()
    return args

import math
import torch

import omni.isaac.lab.sim        as sim_utils
import omni.isaac.lab.envs.mdp   as mdp

from omni.isaac.lab.envs     import ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
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


"""
    ALIENGO_ENV.PY script STRUCTURE:
    - scene         --> GUI and World appearance 
    - mdp-rl        --> Actions, commmands, observations (policy i defined in ppo.py)
    - environment   --> encapsules ALL objs and cfgs from: SCENE, ACTIONS, COMMANDS, OSERVATIONS

    you will have to create an AliengoEnvCfg object in your main_script.py
    it will contains almost everything, usefull to pass infos and configs to the other functions

"""


######### SCENE #########

@configclass
class BaseSceneCfg(InteractiveSceneCfg):

    #############################################

    def __init__(self, rough_terrain=0):

        self.ROUGH_TERRAIN = rough_terrain
        self.terrain = self.terrain_()
        self.robot = self.robot_()
        self.light = self.light_()
        self.dome_light = self.dome_light_()

    #############################################

    ### GROUND - TERRAIN ###
    def terrain_(self):
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
    def robot_(self):
        return UNITREE_AlienGo_CFG
    
    ### LIGHTS ###
    def light_(self):
        return AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
    def dome_light_(self):
        return AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=400.0),
        )


######### MDP - RL #########

### ACTIONS - COMMANDS ###
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

class CommandsCfg:
    """Command terms for the MDP. HIGH LEVEL GOALS --> e.g., velocity, task to do"""

    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        self.velocity_cmd = self._initialize_velocity_cmd()

    def _initialize_velocity_cmd(self):
        # Create a tensor with the shape (num_envs, 3) and fill it with the initial velocity command
        return torch.tensor([[1, 0, 0]], device=self.env.device).repeat(self.env.num_envs, 1)


### OBSERVATIONS ###
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        def __init__(self, env: ManagerBasedRLEnv):
            self.cmnd = CommandsCfg(env)                            # noise -> Added noise 
            self.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
            self.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            self.projected_gravity = ObsTerm(
                func=mdp.projected_gravity,
                noise=Unoise(n_min=-0.05, n_max=0.05),
            )

            self.velocity_commands = ObsTerm(func=lambda: self.cmnd.velocity_cmd)  # wrap it in a function
            """
                If you didn't use a lambda function and directly passed the tensor, like this:
                    self.velocity_commands = ObsTerm(func=self.cmnd.velocity_cmd)
                - This would set func to the current value of self.cmnd.velocity_cmd at the time of assignment.
                - If self.cmnd.velocity_cmd changes later, ObsTerm would not see the updated value 
                    because it holds the initial tensor value, not a function to fetch the latest value.
            """

            self.joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            self.joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
            self.floor_dis = ObsTerm(func=mdp.base_pos_z,    noise=Unoise(n_min=-0.02, n_max=0.02))
            self.actions   = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True   # IDK
                self.concatenate_terms = True   # IDK

    # observation groups
    policy: PolicyCfg = PolicyCfg()  #CONFIGURATIONS FOR THE POLICY (PPO) --> check ppo.py

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep body raised from the floor
    body_height = RewTerm(
        func=mdp.base_pos_z,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["trunk"]), "target": 0.35},
    )
    # (4) Shaping tasks: Keep body almost horizontal
    # orientation_stability = RewTerm(  #  NOT an MDP feature, i have to built it by myself
    #     func=mdp.base_orientation,
    #     weight=0.2,
    #     params={"target": [0, 0, 0, 1]}  # Horizontal Body -> x // to ground
    # )
    # (5) Shaping tasks: Foot contact
    # foot_contact = RewTerm(           # TO CHECK or TO BUILD IT
    #     func=mdp.foot_contact,
    #     weight=0.1,
    #     params={"contact_points": [".*_calf_joint"]}
    # )
    # (6) Walk

### EVENTS ###
@configclass
class EventCfg:
    """Configuration for events."""

    def __init__(self):
        self.reset_scene = self.reset_scene_()
    def reset_scene_(self):
        return EventTerm(func=mdp.reset_scene_to_default, mode="reset")

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    def __init__(self):
        self.time_out = self.time_out_()
    def time_out_(self):
        return DoneTerm(func=mdp.time_out, time_out=True)
    

######### ENVIRONMENT #########

class AliengoEnvCfg(ManagerBasedEnvCfg):   #MBEnv --> _init_, _del_, load_managers(), reset(), step(), seed(), close(), 
    """Configuration for the locomotion velocity-tracking environment."""

    def __init__(self):
        args = parse_args()
        
        self.scene          = BaseSceneCfg(num_envs=args.num_envs, env_spacing=args.env_spacing)
        self.num_envs       = args.num_envs     # needed in PPO, for the env its already specified above in SCENE
        self.observations   = ObservationsCfg()
        self.actions        = ActionsCfg()
        self.events         = EventCfg()
        self.terminator     = TerminationsCfg()

    def __post_init__(self):

        self.decimation = 4  # env decimation -> 50 Hz control
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics

        if self.scene.ROUGH_TERRAIN:
            self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # tick all the sensors based on the smallest update period (physics update period)

