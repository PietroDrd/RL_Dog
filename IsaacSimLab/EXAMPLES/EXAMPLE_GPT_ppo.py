import torch
from skrl.agents.torch.ppo import PPO
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.envs.wrappers.torch import wrap_env

PPO_DEFAULT_CONFIG = {
    "rollouts": 16,
    "learning_epochs": 8,
    "mini_batches": 2,
    "discount_factor": 0.99,
    "lambda": 0.95,
    "learning_rate": 1e-3,
    "learning_rate_scheduler": None,
    "learning_rate_scheduler_kwargs": {},
    "state_preprocessor": None,
    "state_preprocessor_kwargs": {},
    "value_preprocessor": None,
    "value_preprocessor_kwargs": {},
    "random_timesteps": 0,
    "learning_starts": 0,
    "grad_norm_clip": 0.5,
    "ratio_clip": 0.2,
    "value_clip": 0.2,
    "clip_predicted_values": False,
    "entropy_loss_scale": 0.0,
    "value_loss_scale": 1.0,
    "kl_threshold": 0,
    "rewards_shaper": None,
    "time_limit_bootstrap": False,
    "experiment": {
        "directory": "",
        "experiment_name": "",
        "write_interval": "auto",
        "checkpoint_interval": "auto",
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {}
    }
}

class PPOAgent:
    def __init__(self, env, config=PPO_DEFAULT_CONFIG):
        self.env = wrap_env(env, wrapper="isaaclab")
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent = self._create_agent()

    def _create_agent(self):
        models = self._create_models()
        preprocessors = self._create_preprocessors()
        
        agent = PPO(
            models=models,
            preprocessors=preprocessors,
            memory=None,  # Add memory handling if needed
            cfg=self.config,
            env=self.env,
            device=self.device
        )

        # Further configuration
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

        return agent

    def _create_models(self):
        return {
            "policy": torch.nn.Sequential(
                torch.nn.Linear(self.env.observation_space.shape[0], 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.env.action_space.shape[0])
            )
        }

    def _create_preprocessors(self):
        return {
            "observation_scaler": RunningStandardScaler(shape=self.env.observation_space.shape)
        }

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.agent.act(obs)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
            print(f"Episode: {episode}, Total Reward: {total_reward}")

    def save(self, filepath):
        self.agent.save(filepath)

    def load(self, filepath):
        self.agent.load(filepath)

