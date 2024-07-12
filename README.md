# RL_Dog 
Reinforced-Learning for autonomous walking and suddenly-stopping of Legged Robot (AlienGO by Unitree)

Project by Pietro Dardano, advised by prof. [A. Del Prete](https://andreadelprete.github.io/) - UniTn - Summer 2024

## Methodology

Our methodology integrates several advanced technologies and approaches:

- **Proximal Policy Optimization (PPO)** and **Constraints as Terminations (CAT)**: For detailed information on these methods, refer to the [research paper](https://arxiv.org/pdf/2403.18765).
- **Architecture Inspired by ANYmal (ETH-RSL)**: We modeled our architecture based on the principles outlined in the [ANYmal paper](https://www.science.org/doi/epdf/10.1126/scirobotics.aau5872).
- **SKRL**: We utilized the SKRL library to streamline our reinforcement learning implementations. More details can be found in the SKRL [documentation](https://skrl.readthedocs.io/en/latest/intro/getting_started.html).
- **Python + PyTorch**: Our primary programming languages and framework for development and deep learning.
  
### Workstation Setup

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)

- **Isaac Lab**: NVIDIA's [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) provided the high-performance simulation environment necessary for training our models. Refer to the [Orbit](https://isaac-orbit.github.io/) and [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) pages for more information. <br>

### Laptop Setup (lower grade simulations)
**Remark**: for the time being i am mainly working on the Isaac Sim+Lab version for a more complete and realistic simulation.<br>
However i'll try my best to implement it on Gymnasium and with it's code structures and characteristics.

- **OpenAI Gymnasium**: Since Isaac Sim is almost not suitable for being installed on **laptops**, I opted for the lightweight Gymnasium as a simulation environment. It supports Python 3.10 and Ubuntu 22.04 and it's well documented. Obviously, it is far from a realistic simulation, as Isaac Sim is, but for quick tests and trainings, I consider it a good trade-off considering my hardware limitations. For more details on Gymnasium, visit the [official documentation](https://gymnasium.farama.org/).
- `Why not Isaac Gym?`: It requires Ubuntu 20.04 or earlier and Python 3.8 or 3.9. Having installed Ubuntu 22.04, I excluded this option.


## Understanding the Project

For a comprehensive understanding of the principles and techniques used in this project, refer to the following resources:
- A detailed review of related methodologies can be found in [reference 1](https://journals.sagepub.com/doi/full/10.1177/17298814211007305).
- Insights into recent advancements are discussed in [reference 2](https://arxiv.org/html/2308.12517v2).

## Project Structure

 === TBD ===

## Installation

To set up the project, follow these steps:
1. Setup your OS and Environment
    Instructions in the file: [IsaacSim-Setup_Guide](https://github.com/PietroDrd/RL_Dog/blob/main/SETUP_GUIDE.md) or TODO_Gymnasium-Setup_Guide
1. Clone the repository:
   ```
   git clone https://github.com/PietroDrd/RL_Dog.git
   ```


