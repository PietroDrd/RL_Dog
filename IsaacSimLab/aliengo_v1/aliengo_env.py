
import math
import torch

import omni.isaac.lab.sim        as sim_utils
import omni.isaac.lab.envs.mdp   as mdp

##
#########   import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp    #  !!!!!!!!!!!
##

from omni.isaac.lab.envs     import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.assets   import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.assets   import Articulation

from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.utils.noise   import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.terrains      import TerrainImporterCfg
from omni.isaac.lab.terrains.config.rough   import ROUGH_TERRAINS_CFG
from omni.isaac.lab_assets.unitree          import AliengoCFG_Color, AliengoCFG_Black #modified in IsaacLab_
#from unitree import AliengoCFG_Black, AliengoCFG_Color

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

    rough_terrain = 0

    # GROUND - TERRAIN
    if rough_terrain:
        terrain = TerrainImporterCfg(
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
        terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        )

    # ROBOT
    robot: ArticulationCfg = UNITREE_AlienGo_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Alternatively, if using the commented code from the previous example
    # robot_cfg = UNITREE_AlienGo_CFG.copy()
    # robot_cfg.prim_path = "/World/envs/env_.*/Robot"
    # robot = Articulation(cfg=robot_cfg)

    # LIGHTS
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=400.0),
    )

######### MDP - RL #########

### ACTIONS ###
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


### COMANDS ###

def constant_commands(env: ManagerBasedEnv, walk=False) -> torch.Tensor:
    """The generated command from the command generator."""
    dir = [1, 0, 0] if walk else [0, 0, 0]
    return torch.tensor([dir], device=env.device).repeat(env.num_envs, 1)

class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


### OBSERVATIONS ###
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
            
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        #velocity_commands = ObsTerm(func=constant_commands(walk=False))  # wrap it in a function
        """
            If you didn't use a lambda function and directly passed the tensor, like this:
                self.velocity_commands = ObsTerm(func=self.cmnd.velocity_cmd)
            - This would set func to the current value of self.cmnd.velocity_cmd at the time of assignment.
            - If self.cmnd.velocity_cmd changes later, ObsTerm would not see the updated value 
                because it holds the initial tensor value, not a function to fetch the latest value.
        """

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        floor_dis = ObsTerm(func=mdp.base_pos_z,    noise=Unoise(n_min=-0.02, n_max=0.02))
        actions   = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True   # IDK
            self.concatenate_terms = True   # IDK

    policy: PolicyCfg = PolicyCfg()

### EVENTS ###
@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


### REWARDS ###
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
        weight=1.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["trunk"]), "target": 0.35},
    )
    
    # (4) Shaping tasks: Keep body almost horizontal
    horiz_orient = RewTerm(
        func=mdp.root_quat_w,
        weight=0.8,
        params={"target": [1, 0, 0, 0]} # Robot's BaseLink x_axis // target ---> be straight
    )
    # another sol is to have the projected_grav as [0, 0, -g] --> [0, 0, -1] --> q = [0.707, 0, 0, -0.707]

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    #other
    
@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""
    pass
    

######### ENVIRONMENT #########
@configclass
class AliengoEnvCfg(ManagerBasedRLEnvCfg):   #MBEnv --> _init_, _del_, load_managers(), reset(), step(), seed(), close(), 
    """Configuration for the locomotion velocity-tracking environment."""

    scene : BaseSceneCfg            = BaseSceneCfg(num_envs=16, env_spacing=2.5)
    actions : ActionsCfg            = ActionsCfg()
    commands : CommandsCfg          = CommandsCfg() 
    observations : ObservationsCfg  = ObservationsCfg()
 
    events : EventCfg               = EventCfg()
    rewards : RewardsCfg            = RewardsCfg()
    terminations : TerminationsCfg  = TerminationsCfg()
    curriculum : CurriculumCfg      = CurriculumCfg()


    def __post_init__(self):
        """Initialize additional environment settings."""
        self.decimation = 4  # env decimation -> 50 Hz control
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.sim.render_interval = self.decimation

        # viewer settings
        self.viewer.eye = (6.0, 0.0, 4.5)