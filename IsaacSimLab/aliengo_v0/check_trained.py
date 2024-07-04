# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the environment for a quadruped robot with height-scan sensor.

In this example, we use a locomotion policy to control the robot. The robot is commanded to
move forward at a constant velocity. The height-scan sensor is used to detect the height of
the terrain.

    conda activate isaacenv_
    cd
    cd IsaacLab_
    ./isaaclab.sh -p /home/rl_sim/RL_Dog/IsaacSimLab/aliengo_v0/check_trained.py --num_envs 32

Launch Isaac Sim Simulator first.
"""

################################## AFTER TRAINING ONLY ##################################

import argparse

from omni.isaac.lab.app import AppLauncher # type: ignore
parser = argparse.ArgumentParser(description="Tutorial on creating a quadruped base environment.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import  omni.isaac.lab.envs.mdp      as mdp # type: ignore
import  omni.isaac.lab.sim           as sim_utils # type: ignore
from    omni.isaac.lab.assets        import ArticulationCfg, AssetBaseCfg # type: ignore
from    omni.isaac.lab.envs          import ManagerBasedEnv, ManagerBasedEnvCfg # type: ignore
from    omni.isaac.lab.managers      import EventTermCfg as EventTerm # type: ignore
from    omni.isaac.lab.managers      import ObservationGroupCfg as ObsGroup # type: ignore
from    omni.isaac.lab.managers      import ObservationTermCfg as ObsTerm # type: ignore
from    omni.isaac.lab.managers      import SceneEntityCfg # type: ignore
from    omni.isaac.lab.scene         import InteractiveSceneCfg # type: ignore
from    omni.isaac.lab.sensors       import RayCasterCfg, patterns # type: ignore
from    omni.isaac.lab.terrains      import TerrainImporterCfg # type: ignore
from    omni.isaac.lab.utils         import configclass # type: ignore
from    omni.isaac.lab.utils.assets  import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file # type: ignore
from    omni.isaac.lab.utils.noise   import AdditiveUniformNoiseCfg as Unoise # type: ignore


##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # type: ignore # isort: skip
from omni.isaac.lab_assets.anymal   import ANYMAL_C_CFG  # type: ignore # isort: skip
from omni.isaac.lab_assets.unitree  import UNITREE_A1_CFG
from unitree import UNITREE_AlienGo_CFG


ROUGH_TERRAIN = 0

##
# Custom observation terms
##


def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # add terrain
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    if ROUGH_TERRAIN:
        terrain= TerrainImporterCfg(
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

    # add robot
    robot: ArticulationCfg = UNITREE_AlienGo_CFG #.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Sensor implemented in Anymal, idk if ALIENGO HAS IT
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=400.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions   = ObsTerm(func=mdp.last_action)
        
        ## AlienGo does not has it,   TO CHECK !!
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        # HERE I WANT TO CALCULATE THE HEIGHT FROM FLOOR, since we do not have the scan
        """
        Idea:   distance between robot base and the mid point from the vertical interception between the lines 
                that are connecting the opposite leg tips (ends)

                RWD: + if 25< h <60              25 and 60 are just two random values,
                RWD: - if h<= 25 or h>=60        change them with the favourite/optimal ones

        """

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


##
# Environment configuration
##


@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        if ROUGH_TERRAIN:
            self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz


def main():
    """Main function."""
    # setup base environment
    env_cfg = QuadrupedEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    ### RL POLICY ###
    # policy_path = ISAACLAB_NUCLEUS_DIR + "/Policies/ANYmal-C/HeightScan/policy.pt"
    # # check if policy file exists
    # if not check_file_path(policy_path):
    #     raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
    # file_bytes = read_file(policy_path)
    # # jit load the policy
    # policy = torch.jit.load(file_bytes).to(env.device).eval()

    # simulate physics
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # infer action
            action = None    ################################## PUT HERE THE TRAINED POLICY !!!!!
            # step env
            obs, _ = env.step(action)
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()