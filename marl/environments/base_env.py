import gymnasium as gym
from typing import Optional


class BaseEnv(gym.Env):
    def __init__(self, num_agents: int):
        super().__init__()
        self.num_agents = num_agents

    def step(self, actions: dict) -> tuple[dict, dict, dict, bool, bool, dict]:
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        pass

    def render(self):
        pass
