# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package with the utility class to configure the :class:`omni.isaac.kit.SimulationApp`.

The :class:`AppLauncher` parses environment variables and input CLI arguments to launch the simulator in
various different modes. This includes with or without GUI and switching between different Omniverse remote
clients. Some of these require the extensions to be loaded in a specific order, otherwise a segmentation
fault occurs. The launched :class:`omni.isaac.kit.SimulationApp` instance is accessible via the
:attr:`AppLauncher.app` property.
"""

import argparse
import contextlib
import faulthandler
import os
import re
import signal
import sys
from typing import Any, Literal

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401

from omni.isaac.kit import SimulationApp


class AppLauncher:
    """A utility class to launch Isaac Sim application based on command-line arguments and environment variables.

    The class resolves the simulation app settings that appear through environments variables,
    command-line arguments (CLI) or as input keyword arguments. Based on these settings, it launches the
    simulation app and configures the extensions to load (as a part of post-launch setup).

    The input arguments provided to the class are given higher priority than the values set
    from the corresponding environment variables. This provides flexibility to deal with different
    users' preferences.

    .. note::
        Explicitly defined arguments are only given priority when their value is set to something outside
        their default configuration. For example, the ``livestream`` argument is -1 by default. It only
        overrides the ``LIVESTREAM`` environment variable when ``livestream`` argument is set to a
        value >-1. In other words, if ``livestream=-1``, then the value from the environment variable
        ``LIVESTREAM`` is used.

    """

    def __init__(self, launcher_args: argparse.Namespace | dict | None = None, **kwargs):
        """Create a `SimulationApp`_ instance based on the input settings.

        Args:
            launcher_args: Input arguments to parse using the AppLauncher and set into the SimulationApp.
                Defaults to None, which is equivalent to passing an empty dictionary. A detailed description of
                the possible arguments is available in the `SimulationApp`_ documentation.
            **kwargs : Additional keyword arguments that will be merged into :attr:`launcher_args`.
                They serve as a convenience for those who want to pass some arguments using the argparse
                interface and others directly into the AppLauncher. Duplicated arguments with
                the :attr:`launcher_args` will raise a ValueError.

        Raises:
            ValueError: If there are common/duplicated arguments between ``launcher_args`` and ``kwargs``.
            ValueError: If combination of ``launcher_args`` and ``kwargs`` are missing the necessary arguments
                that are needed by the AppLauncher to resolve the desired app configuration.
            ValueError: If incompatible or undefined values are assigned to relevant environment values,
                such as ``LIVESTREAM``.

        .. _argparse.Namespace: https://docs.python.org/3/library/argparse.html?highlight=namespace#argparse.Namespace
        .. _SimulationApp: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html
        """
        # Enable call-stack on crash
        faulthandler.enable()

        # We allow users to pass either a dict or an argparse.Namespace into
        # __init__, anticipating that these will be all of the argparse arguments
        # used by the calling script. Those which we appended via add_app_launcher_args
        # will be used to control extension loading logic. Additional arguments are allowed,
        # and will be passed directly to the SimulationApp initialization.
        #
        # We could potentially require users to enter each argument they want passed here
        # as a kwarg, but this would require them to pass livestream, headless, and
        # any other options we choose to add here explicitly, and with the correct keywords.
        #
        # @hunter: I feel that this is cumbersome and could introduce error, and would prefer to do
        # some sanity checking in the add_app_launcher_args function
        if launcher_args is None:
            launcher_args = {}
        elif isinstance(launcher_args, argparse.Namespace):
            launcher_args = launcher_args.__dict__

        # Check that arguments are unique
        if len(kwargs) > 0:
            if not set(kwargs.keys()).isdisjoint(launcher_args.keys()):
                overlapping_args = set(kwargs.keys()).intersection(launcher_args.keys())
                raise ValueError(
                    f"Input `launcher_args` and `kwargs` both provided common attributes: {overlapping_args}."
                    " Please ensure that each argument is supplied to only one of them, as the AppLauncher cannot"
                    " discern priority between them."
                )
            launcher_args.update(kwargs)

        # Define config members that are read from env-vars or keyword args
        self._headless: bool  # 0: GUI, 1: Headless
        self._livestream: Literal[0, 1, 2]  # 0: Disabled, 1: Native, 2: WebRTC
        self._offscreen_render: bool  # 0: Disabled, 1: Enabled
        self._sim_experience_file: str  # Experience file to load

        # Exposed to train scripts
        self.device_id: int  # device ID for GPU simulation (defaults to 0)
        self.local_rank: int  # local rank of GPUs in the current node
        self.global_rank: int  # global rank for multi-node training

        # Integrate env-vars and input keyword args into simulation app config
        self._config_resolution(launcher_args)
        # Create SimulationApp, passing the resolved self._config to it for initialization
        self._create_app()
        # Load IsaacSim extensions
        self._load_extensions()
        # Hide the stop button in the toolbar
        self._hide_stop_button()

        # Set up signal handlers for graceful shutdown
        # -- during interrupts
        signal.signal(signal.SIGINT, self._interrupt_signal_handle_callback)
        # -- during explicit `kill` commands
        signal.signal(signal.SIGTERM, self._abort_signal_handle_callback)
        # -- during segfaults
        signal.signal(signal.SIGABRT, self._abort_signal_handle_callback)
        signal.signal(signal.SIGSEGV, self._abort_signal_handle_callback)

    """
    Properties.
    """

    @property
    def app(self) -> SimulationApp:
        """The launched SimulationApp."""
        if self._app is not None:
            return self._app
        else:
            raise RuntimeError("The `AppLauncher.app` member cannot be retrieved until the class is initialized.")

    """
    Operations.
    """

    @staticmethod
    def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
        """Utility function to configure AppLauncher arguments with an existing argument parser object.

        This function takes an ``argparse.ArgumentParser`` object and does some sanity checking on the existing
        arguments for ingestion by the SimulationApp. It then appends custom command-line arguments relevant
        to the SimulationApp to the input :class:`argparse.ArgumentParser` instance. This allows overriding the
        environment variables using command-line arguments.

        Currently, it adds the following parameters to the argparser object:

        * ``headless`` (bool): If True, the app will be launched in headless (no-gui) mode. The values map the same
          as that for the ``HEADLESS`` environment variable. If False, then headless mode is determined by the
          ``HEADLESS`` environment variable.
        * ``livestream`` (int): If one of {0, 1, 2}, then livestreaming and headless mode is enabled. The values
          map the same as that for the ``LIVESTREAM`` environment variable. If :obj:`-1`, then livestreaming is
          determined by the ``LIVESTREAM`` environment variable.
        * ``enable_cameras`` (bool): If True, the app will enable camera sensors and render them, even when in
          headless mode. This flag must be set to True if the environments contains any camera sensors.
          The values map the same as that for the ``ENABLE_CAMERAS`` environment variable.
          If False, then enable_cameras mode is determined by the ``ENABLE_CAMERAS`` environment variable.
        * ``device_id`` (int): If specified, simulation will run on the specified GPU device.
        * ``experience`` (str): The experience file to load when launching the SimulationApp. If a relative path
          is provided, it is resolved relative to the ``apps`` folder in Isaac Sim and Isaac Lab (in that order).

          If provided as an empty string, the experience file is determined based on the headless flag:

          * If headless and enable_cameras are True, the experience file is set to ``isaaclab.python.headless.rendering.kit``.
          * If headless is False and enable_cameras is True, the experience file is set to ``isaaclab.python.rendering.kit``.
          * If headless is False and enable_cameras is False, the experience file is set to ``isaaclab.python.kit``.
          * If headless is True and enable_cameras is False, the experience file is set to ``isaaclab.python.headless.kit``.

        Args:
            parser: An argument parser instance to be extended with the AppLauncher specific options.
        """
        # If the passed parser has an existing _HelpAction when passed,
        # we here remove the options which would invoke it,
        # to be added back after the additional AppLauncher args
        # have been added. This is equivalent to
        # initially constructing the ArgParser with add_help=False,
        # but this means we don't have to require that behavior
        # in users and can handle it on our end.
        # We do this because calling parse_known_args() will handle
        # any -h/--help options being passed and then exit immediately,
        # before the additional arguments can be added to the help readout.
        parser_help = None
        if len(parser._actions) > 0 and isinstance(parser._actions[0], argparse._HelpAction):  # type: ignore
            parser_help = parser._actions[0]
            parser._option_string_actions.pop("-h")
            parser._option_string_actions.pop("--help")

        # Parse known args for potential name collisions/type mismatches
        # between the config fields SimulationApp expects and the ArgParse
        # arguments that the user passed.
        known, _ = parser.parse_known_args()
        config = vars(known)
        if len(config) == 0:
            print(
                "[WARN][AppLauncher]: There are no arguments attached to the ArgumentParser object."
                " If you have your own arguments, please load your own arguments before calling the"
                " `AppLauncher.add_app_launcher_args` method. This allows the method to check the validity"
                " of the arguments and perform checks for argument names."
            )
        else:
            AppLauncher._check_argparser_config_params(config)

        # Add custom arguments to the parser
        arg_group = parser.add_argument_group(
            "app_launcher arguments",
            description="Arguments for the AppLauncher. For more details, please check the documentation.",
        )
        arg_group.add_argument(
            "--headless",
            action="store_true",
            default=AppLauncher._APPLAUNCHER_CFG_INFO["headless"][1],
            help="Force display off at all times.",
        )
        arg_group.add_argument(
            "--livestream",
            type=int,
            default=AppLauncher._APPLAUNCHER_CFG_INFO["livestream"][1],
            choices={0, 1, 2},
            help="Force enable livestreaming. Mapping corresponds to that for the `LIVESTREAM` environment variable.",
        )
        arg_group.add_argument(
            "--enable_cameras",
            action="store_true",
            default=AppLauncher._APPLAUNCHER_CFG_INFO["enable_cameras"][1],
            help="Enable camera sensors and relevant extension dependencies.",
        )
        arg_group.add_argument(
            "--device_id",
            type=int,
            default=AppLauncher._APPLAUNCHER_CFG_INFO["device_id"][1],
            help="GPU device ID used for running simulation.",
        )
        arg_group.add_argument(
            "--verbose",  # Note: This is read by SimulationApp through sys.argv
            action="store_true",
            help="Enable verbose terminal output from the SimulationApp.",
        )
        arg_group.add_argument(
            "--experience",
            type=str,
            default="",
            help=(
                "The experience file to load when launching the SimulationApp. If an empty string is provided,"
                " the experience file is determined based on the headless flag. If a relative path is provided,"
                " it is resolved relative to the `apps` folder in Isaac Sim and Isaac Lab (in that order)."
            ),
        )

        # Corresponding to the beginning of the function,
        # if we have removed -h/--help handling, we add it back.
        if parser_help is not None:
            parser._option_string_actions["-h"] = parser_help
            parser._option_string_actions["--help"] = parser_help

    """
    Internal functions.
    """

    _APPLAUNCHER_CFG_INFO: dict[str, tuple[list[type], Any]] = {
        "headless": ([bool], False),
        "livestream": ([int], -1),
        "enable_cameras": ([bool], False),
        "device_id": ([int], 0),
        "experience": ([str], ""),
    }
    """A dictionary of arguments added manually by the :meth:`AppLauncher.add_app_launcher_args` method.

    The values are a tuple of the expected type and default value. This is used to check against name collisions
    for arguments passed to the :class:`AppLauncher` class as well as for type checking.

    They have corresponding environment variables as detailed in the documentation.
    """

    # TODO: Find some internally managed NVIDIA list of these types.
    # SimulationApp.DEFAULT_LAUNCHER_CONFIG almost works, except that
    # it is ambiguous where the default types are None
    _SIM_APP_CFG_TYPES: dict[str, list[type]] = {
        "headless": [bool],
        "hide_ui": [bool, type(None)],
        "active_gpu": [int, type(None)],
        "physics_gpu": [int],
        "multi_gpu": [bool],
        "sync_loads": [bool],
        "width": [int],
        "height": [int],
        "window_width": [int],
        "window_height": [int],
        "display_options": [int],
        "subdiv_refinement_level": [int],
        "renderer": [str],
        "anti_aliasing": [int],
        "samples_per_pixel_per_frame": [int],
        "denoiser": [bool],
        "max_bounces": [int],
        "max_specular_transmission_bounces": [int],
        "max_volume_bounces": [int],
        "open_usd": [str, type(None)],
        "livesync_usd": [str, type(None)],
        "fast_shutdown": [bool],
        "experience": [str],
    }
    """A dictionary containing the type of arguments passed to SimulationApp.

    This is used to check against name collisions for arguments passed to the :class:`AppLauncher` class
    as well as for type checking. It corresponds closely to the :attr:`SimulationApp.DEFAULT_LAUNCHER_CONFIG`,
    but specifically denotes where None types are allowed.
    """

    @staticmethod
    def _check_argparser_config_params(config: dict) -> None:
        """Checks that input argparser object has parameters with valid settings with no name conflicts.

        First, we inspect the dictionary to ensure that the passed ArgParser object is not attempting to add arguments
        which should be assigned by calling :meth:`AppLauncher.add_app_launcher_args`.

        Then, we check that if the key corresponds to a config setting expected by SimulationApp, then the type of
        that key's value corresponds to the type expected by the SimulationApp. If it passes the check, the function
        prints out that the setting with be passed to the SimulationApp. Otherwise, we raise a ValueError exception.

        Args:
            config: A configuration parameters which will be passed to the SimulationApp constructor.

        Raises:
            ValueError: If a key is an already existing field in the configuration parameters but
                should be added by calling the :meth:`AppLauncher.add_app_launcher_args.
            ValueError: If keys corresponding to those used to initialize SimulationApp
                (as found in :attr:`_SIM_APP_CFG_TYPES`) are of the wrong value type.
        """
        # check that no config key conflicts with AppLauncher config names
        applauncher_keys = set(AppLauncher._APPLAUNCHER_CFG_INFO.keys())
        for key, value in config.items():
            if key in applauncher_keys:
                raise ValueError(
                    f"The passed ArgParser object already has the field '{key}'. This field will be added by"
                    " `AppLauncher.add_app_launcher_args()`, and should not be added directly. Please remove the"
                    " argument or rename it to a non-conflicting name."
                )
        # check that type of the passed keys are valid
        simulationapp_keys = set(AppLauncher._SIM_APP_CFG_TYPES.keys())
        for key, value in config.items():
            if key in simulationapp_keys:
                given_type = type(value)
                expected_types = AppLauncher._SIM_APP_CFG_TYPES[key]
                if type(value) not in set(expected_types):
                    raise ValueError(
                        f"Invalid value type for the argument '{key}': {given_type}. Expected one of {expected_types},"
                        " if intended to be ingested by the SimulationApp object. Please change the type if this"
                        " intended for the SimulationApp or change the name of the argument to avoid name conflicts."
                    )
                # Print out values which will be used
                print(f"[INFO][AppLauncher]: The argument '{key}' will be used to configure the SimulationApp.")

    def _config_resolution(self, launcher_args: dict):
        """Resolve the input arguments and environment variables.

        Args:
            launcher_args: A dictionary of all input arguments passed to the class object.
        """
        # Handle all control logic resolution

        # --LIVESTREAM logic--
        #
        livestream_env = int(os.environ.get("LIVESTREAM", 0))
        livestream_arg = launcher_args.pop("livestream", AppLauncher._APPLAUNCHER_CFG_INFO["livestream"][1])
        livestream_valid_vals = {0, 1, 2}
        # Value checking on LIVESTREAM
        if livestream_env not in livestream_valid_vals:
            raise ValueError(
                f"Invalid value for environment variable `LIVESTREAM`: {livestream_env} ."
                f" Expected: {livestream_valid_vals}."
            )
        # We allow livestream kwarg to supersede LIVESTREAM envvar
        if livestream_arg >= 0:
            if livestream_arg in livestream_valid_vals:
                self._livestream = livestream_arg
                # print info that we overrode the env-var
                print(
                    f"[INFO][AppLauncher]: Input keyword argument `livestream={livestream_arg}` has overridden"
                    f" the environment variable `LIVESTREAM={livestream_env}`."
                )
            else:
                raise ValueError(
                    f"Invalid value for input keyword argument `livestream`: {livestream_arg} ."
                    f" Expected: {livestream_valid_vals}."
                )
        else:
            self._livestream = livestream_env

        # --HEADLESS logic--
        #
        # Resolve headless execution of simulation app
        # HEADLESS is initially passed as an int instead of
        # the bool of headless_arg to avoid messy string processing,
        headless_env = int(os.environ.get("HEADLESS", 0))
        headless_arg = launcher_args.pop("headless", AppLauncher._APPLAUNCHER_CFG_INFO["headless"][1])
        headless_valid_vals = {0, 1}
        # Value checking on HEADLESS
        if headless_env not in headless_valid_vals:
            raise ValueError(
                f"Invalid value for environment variable `HEADLESS`: {headless_env} . Expected: {headless_valid_vals}."
            )
        # We allow headless kwarg to supersede HEADLESS envvar if headless_arg does not have the default value
        # Note: Headless is always true when livestreaming
        if headless_arg is True:
            self._headless = headless_arg
        elif self._livestream in {1, 2}:
            # we are always headless on the host machine
            self._headless = True
            # inform who has toggled the headless flag
            if self._livestream == livestream_arg:
                print(
                    f"[INFO][AppLauncher]: Input keyword argument `livestream={self._livestream}` has implicitly"
                    f" overridden the environment variable `HEADLESS={headless_env}` to True."
                )
            elif self._livestream == livestream_env:
                print(
                    f"[INFO][AppLauncher]: Environment variable `LIVESTREAM={self._livestream}` has implicitly"
                    f" overridden the environment variable `HEADLESS={headless_env}` to True."
                )
        else:
            # Headless needs to be a bool to be ingested by SimulationApp
            self._headless = bool(headless_env)
        # Headless needs to be passed to the SimulationApp so we keep it here
        launcher_args["headless"] = self._headless

        # --enable_cameras logic--
        #
        enable_cameras_env = int(os.environ.get("ENABLE_CAMERAS", 0))
        enable_cameras_arg = launcher_args.pop("enable_cameras", AppLauncher._APPLAUNCHER_CFG_INFO["enable_cameras"][1])
        enable_cameras_valid_vals = {0, 1}
        if enable_cameras_env not in enable_cameras_valid_vals:
            raise ValueError(
                f"Invalid value for environment variable `ENABLE_CAMERAS`: {enable_cameras_env} ."
                f"Expected: {enable_cameras_valid_vals} ."
            )
        # We allow enable_cameras kwarg to supersede ENABLE_CAMERAS envvar
        if enable_cameras_arg is True:
            self._enable_cameras = enable_cameras_arg
        else:
            self._enable_cameras = bool(enable_cameras_env)
        self._offscreen_render = False
        if self._enable_cameras and self._headless:
            self._offscreen_render = True
        # hide_ui flag
        launcher_args["hide_ui"] = False
        if self._headless and not self._livestream:
            launcher_args["hide_ui"] = True

        # --simulation GPU device logic --
        self.device_id = launcher_args.pop("device_id", AppLauncher._APPLAUNCHER_CFG_INFO["device_id"][1])
        if "distributed" in launcher_args:
            distributed_train = launcher_args["distributed"]
            # local rank (GPU id) in a current multi-gpu mode
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank (GPU id) in multi-gpu multi-node mode
            self.global_rank = int(os.getenv("RANK", "0"))
            if distributed_train:
                self.device_id = self.local_rank
                launcher_args["multi_gpu"] = False
        # set physics and rendering device
        launcher_args["physics_gpu"] = self.device_id
        launcher_args["active_gpu"] = self.device_id

        # Check if input keywords contain an 'experience' file setting
        # Note: since experience is taken as a separate argument by Simulation App, we store it separately
        self._sim_experience_file = launcher_args.pop("experience", "")

        # If nothing is provided resolve the experience file based on the headless flag
        kit_app_exp_path = os.environ["EXP_PATH"]
        isaaclab_app_exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 6, "apps")
        if self._sim_experience_file == "":
            # check if the headless flag is setS
            if self._enable_cameras:
                if self._headless and not self._livestream:
                    self._sim_experience_file = os.path.join(
                        isaaclab_app_exp_path, "isaaclab.python.headless.rendering.kit"
                    )
                else:
                    self._sim_experience_file = os.path.join(isaaclab_app_exp_path, "isaaclab.python.rendering.kit")
            elif self._headless and not self._livestream:
                self._sim_experience_file = os.path.join(isaaclab_app_exp_path, "isaaclab.python.headless.kit")
            else:
                self._sim_experience_file = os.path.join(isaaclab_app_exp_path, "isaaclab.python.kit")
        elif not os.path.isabs(self._sim_experience_file):
            option_1_app_exp_path = os.path.join(kit_app_exp_path, self._sim_experience_file)
            option_2_app_exp_path = os.path.join(isaaclab_app_exp_path, self._sim_experience_file)
            if os.path.exists(option_1_app_exp_path):
                self._sim_experience_file = option_1_app_exp_path
            elif os.path.exists(option_2_app_exp_path):
                self._sim_experience_file = option_2_app_exp_path
            else:
                raise FileNotFoundError(
                    f"Invalid value for input keyword argument `experience`: {self._sim_experience_file}."
                    "\n No such file exists in either the Kit or Isaac Lab experience paths. Checked paths:"
                    f"\n\t [1]: {option_1_app_exp_path}"
                    f"\n\t [2]: {option_2_app_exp_path}"
                )
        elif not os.path.exists(self._sim_experience_file):
            raise FileNotFoundError(
                f"Invalid value for input keyword argument `experience`: {self._sim_experience_file}."
                " The file does not exist."
            )

        print(f"[INFO][AppLauncher]: Loading experience file: {self._sim_experience_file}")
        # Remove all values from input keyword args which are not meant for SimulationApp
        # Assign all the passed settings to a dictionary for the simulation app
        self._sim_app_config = {
            key: launcher_args[key] for key in set(AppLauncher._SIM_APP_CFG_TYPES.keys()) & set(launcher_args.keys())
        }

    def _create_app(self):
        """Launch and create the SimulationApp based on the parsed simulation config."""
        # Initialize SimulationApp
        # hack sys module to make sure that the SimulationApp is initialized correctly
        # this is to avoid the warnings from the simulation app about not ok modules
        r = re.compile(".*lab.*")
        found_modules = list(filter(r.match, list(sys.modules.keys())))
        found_modules += ["omni.isaac.kit.app_framework"]
        # remove Isaac Lab modules from sys.modules
        hacked_modules = dict()
        for key in found_modules:
            hacked_modules[key] = sys.modules[key]
            del sys.modules[key]
        # launch simulation app
        self._app = SimulationApp(self._sim_app_config, experience=self._sim_experience_file)
        # add Isaac Lab modules back to sys.modules
        for key, value in hacked_modules.items():
            sys.modules[key] = value

    def _rendering_enabled(self):
        # Indicates whether rendering is required by the app.
        # Extensions required for rendering bring startup and simulation costs, so we do not enable them if not required.
        return not self._headless or self._livestream >= 1 or self._enable_cameras

    def _load_extensions(self):
        """Load correct extensions based on AppLauncher's resolved config member variables."""
        # These have to be loaded after SimulationApp is initialized
        import carb
        import omni.physx.bindings._physx as physx_impl
        from omni.isaac.core.utils.extensions import enable_extension

        # Retrieve carb settings for modification
        carb_settings_iface = carb.settings.get_settings()

        if self._livestream >= 1:
            # Ensure that a viewport exists in case an experience has been
            # loaded which does not load it by default
            enable_extension("omni.kit.viewport.window")
            # Set carb settings to allow for livestreaming
            carb_settings_iface.set_bool("/app/livestream/enabled", True)
            carb_settings_iface.set_bool("/app/window/drawMouse", True)
            carb_settings_iface.set_bool("/ngx/enabled", False)
            carb_settings_iface.set_string("/app/livestream/proto", "ws")
            carb_settings_iface.set_int("/app/livestream/websocket/framerate_limit", 120)
            # Note: Only one livestream extension can be enabled at a time
            if self._livestream == 1:
                # Enable Native Livestream extension
                # Default App: Streaming Client from the Omniverse Launcher
                enable_extension("omni.kit.streamsdk.plugins-3.2.1")
                enable_extension("omni.kit.livestream.core-3.2.0")
                enable_extension("omni.kit.livestream.native-4.1.0")
            elif self._livestream == 2:
                # Enable WebRTC Livestream extension
                # Default URL: http://localhost:8211/streaming/webrtc-client/
                enable_extension("omni.services.streamclient.webrtc")
            else:
                raise ValueError(f"Invalid value for livestream: {self._livestream}. Expected: 1, 2 .")
        else:
            carb_settings_iface.set_bool("/app/livestream/enabled", False)

        # set carb setting to indicate Isaac Lab's offscreen_render pipeline should be enabled
        # this flag is used by the SimulationContext class to enable the offscreen_render pipeline
        # when the render() method is called.
        carb_settings_iface.set_bool("/isaaclab/render/offscreen", self._offscreen_render)

        # set carb setting to indicate no RTX sensors are used
        # this flag is set to True when an RTX-rendering related sensor is created
        # for example: the `Camera` sensor class
        carb_settings_iface.set_bool("/isaaclab/render/rtx_sensors", False)

        # set fabric update flag to disable updating transforms when rendering is disabled
        carb_settings_iface.set_bool("/physics/fabricUpdateTransformations", self._rendering_enabled())

        # set the nucleus directory manually to the latest published Nucleus
        # note: this is done to ensure prior versions of Isaac Sim still use the latest assets
        assets_path = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0"
        carb_settings_iface.set_string("/persistent/isaac/asset_root/default", assets_path)
        carb_settings_iface.set_string("/persistent/isaac/asset_root/cloud", assets_path)
        carb_settings_iface.set_string("/persistent/isaac/asset_root/nvidia", assets_path)

        # disable physics backwards compatibility check
        carb_settings_iface.set_int(physx_impl.SETTING_BACKWARD_COMPATIBILITY, 0)

    def _hide_stop_button(self):
        """Hide the stop button in the toolbar.

        For standalone executions, having a stop button is confusing since it invalidates the whole simulation.
        Thus, we hide the button so that users don't accidentally click it.
        """
        # when we are truly headless, then we can't import the widget toolbar
        # thus, we only hide the stop button when we are not headless (i.e. GUI is enabled)
        if self._livestream >= 1 or not self._headless:
            import omni.kit.widget.toolbar

            # grey out the stop button because we don't want to stop the simulation manually in standalone mode
            toolbar = omni.kit.widget.toolbar.get_instance()
            play_button_group = toolbar._builtin_tools._play_button_group  # type: ignore
            if play_button_group is not None:
                play_button_group._stop_button.visible = False  # type: ignore
                play_button_group._stop_button.enabled = False  # type: ignore
                play_button_group._stop_button = None  # type: ignore

    def _interrupt_signal_handle_callback(self, signal, frame):
        """Handle the interrupt signal from the keyboard."""
        # close the app
        self._app.close()
        # raise the error for keyboard interrupt
        raise KeyboardInterrupt

    def _abort_signal_handle_callback(self, signal, frame):
        """Handle the abort/segmentation/kill signals."""
        # close the app
        self._app.close()
