import gymnasium as gym
from marl.environments.base_env import BaseEnv
import numpy as np
from typing import Optional
import random
import matplotlib.pyplot as plt
from gymnasium.spaces import Discrete, Box


class MaSphinx(gym.Env):
    def __init__(self, num_agents=2):
        super().__init__()
        self.num_agents = num_agents

        # Initialize continuous positions
        self.agent_positions = {i: (0.0, 0.0) for i in range(num_agents)}
        self.box_positions = [(0.1, 0.1), (0.9, 0.9), (0.1, 0.9)]
        self.correct_box_pos = random.choice(self.box_positions)

        self.sphinx_position = (0.9, 0.1)
        self.sphinx_asked = False
        self.sphinx_just_asked = False
        self.sphinx_proximity_threshold = 0.1

        self.step_count = 0

        self.actions = np.array(
            [
                [0, 1],  # UP
                [1, 0],  # RIGHT
                [1, 1],  # UP-RIGHT
                [-1, 0],  # LEFT
                [0, -1],  # DOWN
                [-1, -1],  # DOWN-LEFT
                [1, -1],  # RIGHT-DOWN
                [-1, 1],  # LEFT-UP
                [0, 0],  # STAY
            ]
        )

        self.step_length = 0.05
        self.step_penalty = -0.01
        self.action_space = Discrete(len(self.actions))
        self.state_space = Box(
            -np.inf, np.inf, shape=(2 * num_agents + 2 * len(self.box_positions) + 2,)
        )
        self.state_space_unobserved = Box(-np.inf, np.inf, shape=(2,))
        self.observation_space = Box(
            -np.inf, np.inf, shape=(2 * num_agents + 2 * len(self.box_positions) + 2,)
        )

    def reset(self, random_start=False, unobserved=False):
        self.agent_positions = {
            i: (
                (random.uniform(0, 1) if random_start else 0.5),
                (random.uniform(0, 1) if random_start else 0.5),
            )
            for i in range(self.num_agents)
        }
        self.correct_box_pos = random.choice(self.box_positions)
        self.sphinx_asked = False
        self.sphinx_just_asked = False
        self.step_count = 0
        return (
            self._make_states(unobserved=unobserved),
            self._make_observations(),
        )

    def step(self, actions, unobserved=False):
        total_reward = self.step_penalty
        done = False
        truncated = False

        self.sphinx_just_asked = False

        for agent, action in actions.items():
            move = self.actions[action]
            if self.is_valid(agent, move):
                x, y = self.agent_positions[agent]
                new_x = x + move[0] * self.step_length
                new_y = y + move[1] * self.step_length
                self.agent_positions[agent] = (new_x, new_y)

            if not self.sphinx_asked and self.in_region(
                self.agent_positions[agent], self.sphinx_position
            ):
                self.sphinx_asked = True
                self.sphinx_just_asked = True
                total_reward += 0.1

        for agent in range(self.num_agents):
            if self.in_region(self.agent_positions[agent], self.correct_box_pos):
                total_reward += 1.0
                done = True
                break

        self.step_count += 1

        # Assign same reward to all agents
        rewards = {i: total_reward for i in range(self.num_agents)}

        return (
            self._make_states(unobserved=unobserved),
            self._make_observations(),
            rewards,
            done,
            truncated,
            {},
        )

    def is_valid(self, agent, move):
        x, y = self.agent_positions[agent]
        new_x = x + move[0] * self.step_length
        new_y = y + move[1] * self.step_length
        return 0 <= new_x <= 1 and 0 <= new_y <= 1

    def in_region(self, pos, target, threshold=None):
        if threshold is None:
            threshold = self.sphinx_proximity_threshold
        x, y = pos
        tx, ty = target
        return np.sqrt((x - tx) ** 2 + (y - ty) ** 2) < threshold

    def _make_observations(self):
        obs = []
        for agent in range(self.num_agents):
            r_a, c_a = self.agent_positions[agent]
            obs.append(r_a)
            obs.append(c_a)

        for box_pos in self.box_positions:
            obs.extend(box_pos)

        if self.sphinx_just_asked:
            box_x, box_y = self.correct_box_pos
            obs.extend([box_x, box_y])
        else:
            obs.extend([-1.0, -1.0])

        observations = {i: np.array(obs) for i in range(self.num_agents)}
        return observations

    def _make_states(self, unobserved=False):
        state = []
        if not unobserved:
            for agent in range(self.num_agents):
                r_a, c_a = self.agent_positions[agent]
                state.append(r_a)
                state.append(c_a)

            for box_pos in self.box_positions:
                state.extend(box_pos)
        box_x, box_y = self.correct_box_pos
        state.extend([box_x, box_y])

        states = {i: np.array(state) for i in range(self.num_agents)}
        return states


if __name__ == "__main__":
    env = MaSphinx(num_agents=2)
    env.reset()
    counter = 30
    for _ in range(20):
        # Take random actions for both agents
        actions = {0: np.random.randint(0, 9), 1: np.random.randint(0, 9)}
        state, obs, rewards, done, truncated, info = env.step(actions)
        print("actions: ", actions)
        print("rewards: ", rewards)
        print("state: ", state)
        print("observation: ", obs)
        if env.sphinx_asked:
            print("Sphinx asked")
            counter = 2
        counter -= 1
        print("--------------------------------")
        if done or counter == 0:
            if counter == 0:
                print("Sphinx was asked")
            break
