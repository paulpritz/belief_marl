from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import numpy as np
import random


def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed}")


@dataclass
class Args:
    # Required arguments
    agent_type: str
    obs_shape: Tuple[int, ...]
    state_shape: Tuple[int, ...]
    num_actions: int
    num_agents: int
    seed: float = 42

    # Optional arguments with default values
    epsilon: float = 0.1
    episodes: int = 10_000
    max_steps_per_episode: int = 100
    memory_length: int = 10_000
    batch_size: int = 32
    gamma: float = 0.99
    tau: float = 0.005

    # Learning rates (optional)
    q_lr: float = 0.001
    qss_lr: Optional[float] = None
    f_lr: Optional[float] = None
    belief_lr: Optional[float] = None

    # Belief-specific parameters (optional)
    state_shape_unobserved: Tuple[int, ...] = None
    hidden_dim: Optional[int] = None
    latent_dim: Optional[int] = None
    num_belief_samples: Optional[int] = None
    belief_temperature: Optional[float] = None
    lambda_f: Optional[float] = None
    belief_dim: Optional[int] = None
