# RL_Dog 
Reinforced-Learning for Walk and Falls Recover of autonomous Legged Robot (AlienGO by Unitree)

Project initiated by Pietro Dardano, advised by prof. A. Del Prete - UniTn - Summer 2024

## Methodology

Our methodology integrates several advanced technologies and approaches:

- **Proximal Policy Optimization (PPO)** and **Constraints as Terminations (CAT)**: For detailed information on these methods, refer to the [research paper](https://arxiv.org/pdf/2403.18765).
- **Architecture Inspired by ANYmal (ETH-RSL)**: We modeled our architecture based on the principles outlined in the [ANYmal paper](https://www.science.org/doi/epdf/10.1126/scirobotics.aau5872).
- **SKRL**: We utilized the SKRL library to streamline our reinforcement learning implementations. More details can be found in the SKRL [documentation](https://skrl.readthedocs.io/en/latest/intro/getting_started.html).
- **ISAAC GYM**: NVIDIA's ISAAC GYM provided the high-performance simulation environment necessary for training our models. Refer to the ISAAC GYM [paper](https://arxiv.org/abs/2108.10470) for more information.
- **Python + PyTorch**: Our primary programming languages and framework for development and deep learning.

## Understanding the Project

For a comprehensive understanding of the principles and techniques used in this project, refer to the following resources:
- A detailed review of related methodologies can be found in [reference 1](https://journals.sagepub.com/doi/full/10.1177/17298814211007305).
- Insights into recent advancements are discussed in [reference 2](https://arxiv.org/html/2308.12517v2).

## Project Structure

- **src/**: Contains the source code for the project.
- **data/**: Includes datasets used for training and evaluation.
- **models/**: Pretrained models and architectures.
- **notebooks/**: Jupyter notebooks for experiments and visualizations.
- **docs/**: Documentation and references.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourproject.git
