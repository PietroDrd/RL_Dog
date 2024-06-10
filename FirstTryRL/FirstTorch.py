import torch    # v2.2.2 - pytorch
import skrl     # v1.1.0 - pypi
from skrl.agents.torch import PPO
from skrl.envs.torch import wrap_env

# Activate your Conda environment
!conda activate isaacenv

# Wrap the environment for skrl
env = wrap_env(env)


# Define the policy model
class Policy(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.fc1 = torch.nn.Linear(observation_space.shape[0], 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, action_space.n)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)
    


# Initialize the policy model
observation_space = env.observation_space
action_space = env.action_space
policy = Policy(observation_space, action_space)

# Define the PPO agent
agent = PPO(env, policy)

# Configure the agent (example configuration, customize as needed)
agent.configure({
    "learning_rate": 3e-4,
    "discount_factor": 0.99,
    "gae_lambda": 0.95,
    "batch_size": 64,
    "ppo_clip": 0.2,
    "entropy_coefficient": 0.01
})

# Training loop
num_episodes = 100      #1000 or more 
for episode in range(num_episodes):
    observation = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.act(observation)
        next_observation, reward, done, _ = env.step(action)
        agent.record(observation, action, reward, done)
        observation = next_observation
        episode_reward += reward

    agent.learn()
    print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward}")

# Save the trained model
#torch.save(agent.policy.state_dict(), "ppo_policy.pth")