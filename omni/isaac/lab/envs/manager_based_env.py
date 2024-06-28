# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import torch
import warnings
from collections.abc import Sequence
from typing import Any

import carb
import omni.isaac.core.utils.torch as torch_utils

from omni.isaac.lab.managers import ActionManager, EventManager, ObservationManager
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.timer import Timer

from .common import VecEnvObs
from .manager_based_env_cfg import ManagerBasedEnvCfg
from .ui import ViewportCameraController


class ManagerBasedEnv:
    """The base environment encapsulates the simulation scene and the environment managers for the manager-based workflow.

    While a simulation scene or world comprises of different components such as the robots, objects,
    and sensors (cameras, lidars, etc.), the environment is a higher level abstraction
    that provides an interface for interacting with the simulation. The environment is comprised of
    the following components:

    * **Scene**: The scene manager that creates and manages the virtual world in which the robot operates.
      This includes defining the robot, static and dynamic objects, sensors, etc.
    * **Observation Manager**: The observation manager that generates observations from the current simulation
      state and the data gathered from the sensors. These observations may include privileged information
      that is not available to the robot in the real world. Additionally, user-defined terms can be added
      to process the observations and generate custom observations. For example, using a network to embed
      high-dimensional observations into a lower-dimensional space.
    * **Action Manager**: The action manager that processes the raw actions sent to the environment and
      converts them to low-level commands that are sent to the simulation. It can be configured to accept
      raw actions at different levels of abstraction. For example, in case of a robotic arm, the raw actions
      can be joint torques, joint positions, or end-effector poses. Similarly for a mobile base, it can be
      the joint torques, or the desired velocity of the floating base.
    * **Event Manager**: The event manager orchestrates operations triggered based on simulation events.
      This includes resetting the scene to a default state, applying random pushes to the robot at different intervals
      of time, or randomizing properties such as mass and friction coefficients. This is useful for training
      and evaluating the robot in a variety of scenarios.

    The environment provides a unified interface for interacting with the simulation. However, it does not
    include task-specific quantities such as the reward function, or the termination conditions. These
    quantities are often specific to defining Markov Decision Processes (MDPs) while the base environment
    is agnostic to the MDP definition.

    The environment steps forward in time at a fixed time-step. The physics simulation is decimated at a
    lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
    independently using the :attr:`ManagerBasedEnvCfg.decimation` (number of simulation steps per environment step)
    and the :attr:`ManagerBasedEnvCfg.sim.dt` (physics time-step) parameters. Based on these parameters, the
    environment time-step is computed as the product of the two. The two time-steps can be obtained by
    querying the :attr:`physics_dt` and the :attr:`step_dt` properties respectively.
    """

    def __init__(self, cfg: ManagerBasedEnvCfg):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment.

        Raises:
            RuntimeError: If a simulation context already exists. The environment must always create one
                since it configures the simulation context and controls the simulation.
        """
        # store inputs to class
        self.cfg = cfg
        # initialize internal variables
        self._is_closed = False

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            # the type-annotation is required to avoid a type-checking error
            # since it gets confused with Isaac Sim's SimulationContext class
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")
        print(f"\tPhysics GPU pipeline  : {self.cfg.sim.use_gpu_pipeline}")
        print(f"\tPhysics GPU simulation: {self.cfg.sim.physx.use_gpu}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple multiple render calls will happen for each environment step. "
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            carb.log_warn(msg)

        # counter for simulation steps
        self._sim_step_counter = 0

        # generate scene
        with Timer("[INFO]: Time taken for scene creation"):
            self.scene = InteractiveScene(self.cfg.scene)
        print("[INFO]: Scene manager: ", self.scene)

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start"):
                self.sim.reset()
            # add timeline event to load managers
            self.load_managers()

        # make sure torch is running on the correct device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            # if no window, then we don't need to store the window
            self._window = None

        # allocate dictionary to store metrics
        self.extras = {}

    def __del__(self):
        """Cleanup for the environment."""
        self.close()

    """
    Properties.
    """

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return self.scene.num_envs

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation is happening.
        """
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def device(self):
        """The device on which the environment is running."""
        return self.sim.device

    """
    Operations - Setup.
    """

    def load_managers(self):
        """Load the managers for the environment.

        This function is responsible for creating the various managers (action, observation,
        events, etc.) for the environment. Since the managers require access to physics handles,
        they can only be created after the simulator is reset (i.e. played for the first time).

        .. note::
            In case of standalone application (when running simulator from Python), the function is called
            automatically when the class is initialized.

            However, in case of extension mode, the user must call this function manually after the simulator
            is reset. This is because the simulator is only reset when the user calls
            :meth:`SimulationContext.reset_async` and it isn't possible to call async functions in the constructor.

        """
        # check the configs
        if self.cfg.randomization is not None:
            msg = (
                "The 'randomization' attribute is deprecated and will be removed in a future release. "
                "Please use the 'events' attribute to configure the randomization settings."
            )
            warnings.warn(msg, category=DeprecationWarning)
            carb.log_warn(msg)
            # set the randomization as events (for backward compatibility)
            self.cfg.events = self.cfg.randomization

        # prepare the managers
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)
        # -- event manager
        self.event_manager = EventManager(self.cfg.events, self)
        print("[INFO] Event Manager: ", self.event_manager)

    """
    Operations - MDP.
    """

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Resets all the environments and returns observations.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        # set the seed
        if seed is not None:
            self.seed(seed)
        # reset state of scene
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)
        # return observations
        return self.observation_manager.compute(), self.extras

    def step(self, action: torch.Tensor) -> tuple[VecEnvObs, dict]:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is
        decimated at a lower time-step. This is to ensure that the simulation is stable. These two
        time-steps can be configured independently using the :attr:`ManagerBasedEnvCfg.decimation` (number of
        simulation steps per environment step) and the :attr:`ManagerBasedEnvCfg.sim.dt` (physics time-step).
        Based on these parameters, the environment time-step is computed as the product of the two.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations and extras.
        """
        # process actions
        self.action_manager.process_action(action)
        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            render = self._sim_step_counter % self.cfg.sim.render_interval == 0 and (
                self.sim.has_gui() or self.sim.has_rtx_sensors()
            )
            # simulate
            self.sim.step(render=render)
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step: step interval event
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # return observations and extras
        return self.observation_manager.compute(), self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return torch_utils.set_seed(seed)

    def close(self):
        """Cleanup for the environment."""
        if not self._is_closed:
            # destructor is order-sensitive
            del self.viewport_camera_controller
            del self.action_manager
            del self.observation_manager
            del self.event_manager
            del self.scene
            # clear callbacks and instance
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            # destroy the window
            if self._window is not None:
                self._window = None
            # update closing status
            self._is_closed = True

    """
    Helper functions.
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(env_ids=env_ids, mode="reset")

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
