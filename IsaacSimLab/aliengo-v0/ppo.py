import torch

##
#   SKRL-RL STUFF
##
from skrl.agents.torch.ppo import PPO
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.envs.wrappers.torch import Wrapper, wrap_env
from skrl.agents.torch import Agent
from skrl.trainers.torch import Trainer, StepTrainer, ParallelTrainer 


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

def train_ppo(env_, agent=None):
    # Wrap the custom environment
    wrapped_env = wrap_env(env_)

    # Define the model
    models = {}
    models["policy"] = torch.nn.Sequential(
        torch.nn.Linear(wrapped_env.observation_space.shape[0], 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, wrapped_env.action_space.shape[0])
    )

    # Preprocessors
    preprocessors = {"observation_scaler": RunningStandardScaler(shape=wrapped_env.observation_space.shape)}

    # Create PPO agent
    if agent == None:
        cfg_agent = PPO_DEFAULT_CONFIG.copy()
        agent = PPO(
            models=models,
            preprocessors=preprocessors,
            memory=None,
            cfg=cfg_agent,
            env=wrapped_env,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Configure PPO
        agent.configure({
            "rollouts": 1024,
            "learning_epochs": 10,
            "mini_batches": 32,
            "discount_factor": 0.99,
            "lambda": 0.95,
            "learning_rate": 0.0003,
            "learning_rate_scheduler": "linear",
            "learning_rate_scheduler_initial": 1.0,
            "learning_rate_scheduler_final": 0.0,
            "entropy_loss_scale": 0.01,
            "value_loss_scale": 0.5,
            "gradient_clipping": 0.5
        })

    # Training loop
    for epoch in range(1000):
        agent.train()
        rewards = 0
        obs = wrapped_env.reset()
        done = False
        
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = wrapped_env.step(action)
            rewards += reward

        print(f"Epoch: {epoch}, Reward: {rewards}")

#if __name__ == "__main__":
    #CALL THE FOLLOWING FUNCTION IN THE DESIRED (ENV) SCRIPT
    #train_ppo()
