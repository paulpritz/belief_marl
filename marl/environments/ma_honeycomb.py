import gymnasium as gym
from marl.environments.base_env import BaseEnv
import numpy as np
from typing import Optional
import random

# Constants
PAYOFF_LOCATIONS = [(-3, -6), (3, -3), (6, 3), (3, 6), (-3, 3), (-6, -3)]
START_POSITION = (0, 0)

# Actions
ACTIONS = [
    (0, -1),  # UP
    (0, 1),  # DOWN
    (-1, -1),  # UP_LEFT
    (1, 0),  # UP_RIGHT
    (-1, 0),  # DOWN_LEFT
    (1, 1),  # DOWN_RIGHT
    (0, 0),
]  # PASS


class MaHoneyComb(BaseEnv):
    def __init__(
        self, num_uninformed: int = 8, num_informed: int = 2, max_steps: int = 30
    ):
        num_agents = num_uninformed + num_informed
        super().__init__(num_agents)

        self.num_uninformed = num_uninformed
        self.num_informed = num_informed
        self.max_steps = max_steps

        # Track positions and special payoff field
        self.agent_positions = {i: START_POSITION for i in range(self.num_agents)}
        self.special_payoff_field = None

        # State dimensions
        obs_dim = self.num_agents * 2 + len(PAYOFF_LOCATIONS) * 2 + 2 + 1

        # State space includes positions of all agents and payoff fields
        self.state_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Add state space for unobserved elements (special payoff field only)
        self.state_space_unobserved = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        # Observation space varies between informed and uninformed agents
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Actions: UP, DOWN, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT, PASS
        self.action_space = gym.spaces.Discrete(7)
        self.action_moves = ACTIONS

        # Episode parameters
        self.step_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        random_start: bool = False,
        unobserved: bool = False,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0

        # Reset agent positions and storage
        if random_start:
            valid_positions = [
                (i, j)
                for i in range(-5, 6)
                for j in range(-5, 6)
                if self._sNorm(i, j) <= 5
            ]

            for agent_id in range(self.num_agents):
                if valid_positions:
                    pos_idx = np.random.randint(0, len(valid_positions))
                    self.agent_positions[agent_id] = valid_positions.pop(pos_idx)
                else:
                    self.agent_positions[agent_id] = START_POSITION
        else:
            self.agent_positions = {i: START_POSITION for i in range(self.num_agents)}

        self.special_payoff_field = random.sample(PAYOFF_LOCATIONS, 1)[0]

        states = self._make_states(unobserved=unobserved)
        observations = self._make_observations()
        return states, observations

    def step(
        self, actions: dict, unobserved: bool = False
    ) -> tuple[dict, dict, dict, bool, bool, dict]:
        self.step_count += 1

        for agent_id, action in actions.items():
            move = self.action_moves[action]
            self._submit_move_for_agent(agent_id, move)

        done = self.check_all_arrived() or self.step_count >= self.max_steps

        rewards = {i: 0.0 for i in range(self.num_agents)}
        if done:
            rewards = self._compute_rewards()

        states = self._make_states(unobserved=unobserved)
        observations = self._make_observations()
        return states, observations, rewards, done, False, {}

    def _submit_move_for_agent(self, agent_id, move):
        current_pos = self.agent_positions[agent_id]

        if current_pos in PAYOFF_LOCATIONS:
            return

        new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

        if self._sNorm(new_pos[0], new_pos[1]) <= 5 or new_pos in PAYOFF_LOCATIONS:
            self.agent_positions[agent_id] = new_pos

    def _sNorm(self, i: int, j: int) -> int:
        if i < 0:
            i = -i
            j = -j
        if j < 0:
            return i - j
        elif i < j:
            return j
        else:
            return i

    def _make_states(self, unobserved: bool = False):
        if unobserved:
            max_coord = 6
            state = np.array(
                [
                    self.special_payoff_field[0] / max_coord,
                    self.special_payoff_field[1] / max_coord,
                ],
                dtype=np.float32,
            )

            return {agent: state for agent in range(self.num_agents)}

        states = {}
        max_coord = 6

        for agent in range(self.num_agents):
            state = []

            for pos in self.agent_positions.values():
                state.extend([pos[0] / max_coord, pos[1] / max_coord])

            for pos in PAYOFF_LOCATIONS:
                state.extend([pos[0] / max_coord, pos[1] / max_coord])

            state.append(self.step_count / self.max_steps)

            state.extend(
                [
                    self.special_payoff_field[0] / max_coord,
                    self.special_payoff_field[1] / max_coord,
                ]
            )

            states[agent] = np.array(state, dtype=np.float32)

        return states

    def _make_observations(self):
        observations = {}

        max_coord = 6

        for agent in range(self.num_agents):
            obs = []

            for pos in self.agent_positions.values():
                obs.extend([pos[0] / max_coord, pos[1] / max_coord])

            for pos in PAYOFF_LOCATIONS:
                obs.extend([pos[0] / max_coord, pos[1] / max_coord])

            obs.append(self.step_count / self.max_steps)

            if agent < self.num_informed:
                obs.extend(
                    [
                        self.special_payoff_field[0] / max_coord,
                        self.special_payoff_field[1] / max_coord,
                    ]
                )
            else:
                obs.extend([0, 0])

            observations[agent] = np.array(obs, dtype=np.float32)

        return observations

    def _compute_rewards(self):
        total_reward = 0.0

        rewarded_fields = {}
        for agent in range(self.num_agents):
            pos = self.agent_positions[agent]
            if pos in PAYOFF_LOCATIONS:
                rewarded_fields[pos] = rewarded_fields.get(pos, 0.0) + 1.0

        for pos, count in rewarded_fields.items():
            if pos == self.special_payoff_field:
                total_reward += count * 2.0
            else:
                total_reward += count

        # Distribute the same total reward to all agents
        rewards = {i: total_reward for i in range(self.num_agents)}
        return rewards

    def check_all_arrived(self) -> bool:
        return all(pos in PAYOFF_LOCATIONS for pos in self.agent_positions.values())


if __name__ == "__main__":
    env = MaHoneyComb(num_uninformed=8, num_informed=2, max_steps=15)
    obs, _ = env.reset()

    for _ in range(10):
        actions = {i: np.random.randint(0, 7) for i in range(env.num_agents)}
        state, obs, rewards, done, _, _ = env.step(actions, unobserved=True)

        print("\nActions:", actions)
        print("Rewards:", rewards)
        print("payoff_fields:", PAYOFF_LOCATIONS)
        print("special_payoff_field:", env.special_payoff_field)
        print("\nUnobserved state (special payoff field only):")
        print(state[0])
        print("\nPositions:")
        for agent_id, pos in env.agent_positions.items():
            print(f"Agent {agent_id}: {pos}")

        if done:
            break
