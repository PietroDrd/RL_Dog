### GENERAL ###

import torch
import torch.nn as nn

### SKRL-RL STUFF ###

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import Wrapper, wrap_env
        # equal to --> from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.memories.torch import RandomMemory

from skrl.agents.torch import Agent
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

from skrl.trainers.torch import Trainer, SequentialTrainer, ParallelTrainer, StepTrainer 
from skrl.utils import set_seed

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

### ISAACLAB CLASSES ###
from omni.isaac.lab.envs     import ManagerBasedRLEnv

############# CHECK IF TO-DO ###############
# omni.isaac.lab_tasks.utils.wrappers.skrl.process_skrl_cfg(cfg: dict) → dict:
"""
from omni.isaac.lab_tasks.utils.wrappers.skrl import procss_skrl_cfg
process_skrl_cfg(ALIENGO_ENV_CFG)   # IF ANY (must adapt it)
"""

# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        """
        clip:   Clipping actions means restricting the action values to be within the bounds defined by the action space. 
                This ensures that the actions taken by the agent are valid within the defined environment's action space.
                Clipping the log_of_STD ensures that values for the log STD do not goes below or above specified thresholds.

        reduction: The reduction method specifies how to aggregate the log probability densities when computing the total log probability.
        """

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256), # activ fcns were ELU
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU())

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}


class PPO_v1:
    def __init__(self, env: ManagerBasedRLEnv, config=PPO_DEFAULT_CONFIG, device = "cpu", verbose=0):
        self.env = wrap_env(env, verbose=verbose)    # SKRL: wrapper = "auto", by default,  otherwise--> "isaaclab"
        self.config = config
        self.device = device
        self.num_envs = env.num_envs   #needed for MEMORY of PPO, num_envs comes from "args" of the env object
        self.agent = self._create_agent()

    def _create_agent(self):

        #Create the model: In --> feedbacks/states , Out --> Actions/Actuators
        model_nn_ = {}
        model_nn_["policy"] = Shared(self.env.observation_space, self.env.action_space, self.device)
        model_nn_["value"] = model_nn_["policy"]

        # Adjusts Learning Rate
        # self.config["learning_rate_scheduler"] = KLAdaptiveRL   # Has problems with "verbose" param --> commented
        # self.config["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}

        # Standardize the input data by removing the mean and scaling
        self.config["state_preprocessor"] = RunningStandardScaler
        self.config["state_preprocessor_kwargs"] = {"size": self.env.observation_space, "device": self.device}
        self.config["value_preprocessor"] = RunningStandardScaler
        self.config["value_preprocessor_kwargs"] = {"size": 1, "device": self.device}
        
        # instantiate a memory as rollout buffer (any memory can be used for this)
        mem_size = 24
        batch_dim = 6
        memory_rndm_ = RandomMemory(memory_size=mem_size, num_envs=self.num_envs, device=self.device)
        self.config["rollouts"] = mem_size
        self.config["mini_batches"] = min(mem_size * batch_dim / 48, 2 )# 48Gb VRAM of the RTX A6000

        agent = PPO(
            models=model_nn_,
            memory=memory_rndm_,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            cfg=self.config,
            device=self.device,
        )
        return agent
    
    def train_mine_easy(self, num_episodes=1000):
        for ep in range(num_episodes):
            obs = self.env.reset()
            done = False
            tot_reward = 0
            cnt = 0
            while not done:
                action = self.agent.act(obs)  # act if a skrl.agent.torch.ppo --> PPO's method
                """ Process the environment's states to make a decision (actions) using the main policy """
                obs, reward, done, info = self.env.step(actions=action)
                tot_reward += reward 
                cnt += 1
                if cnt % 10 == 0: print(f"Episode: {ep}, Total Reward: {tot_reward}")

    def train_sequential(self, timesteps=20000, headless=False):
        cfg_trainer= {"timesteps": timesteps, "headless": headless}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent)
        trainer.train()

    def train_parallel(self, timesteps=20000, headless=False):
        cfg_trainer = {"timesteps": timesteps, "headless": headless}
        trainer = ParallelTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent)
        trainer.train() 

#########################################################################################


# just to have a look about the values, this is ignored by the code since its after 
PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}

# ADD TRAIN 

# ppo 1
# https://skrl.readthedocs.io/en/latest/_downloads/17f299c7b73f8d5f2c56b336c693da94/torch_velocity_anymal_c_ppo.py

# ppo2  --> EXAMPLE_anymal_PPO.py
# https://skrl.readthedocs.io/en/latest/_downloads/7f665f3e3ea3a391c065747e4d1ef288/torch_anymal_ppo.py

# skrl exampl
# https://skrl.readthedocs.io/en/latest/intro/examples.html#nvidia-isaac-lab