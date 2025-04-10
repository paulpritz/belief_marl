import gymnasium as gym
from marl.environments.base_env import BaseEnv
import numpy as np
from typing import Optional


class MaEscapeRoom(BaseEnv):
    def __init__(
        self,
        num_agents: int,
        grid_size: tuple[int, int],
        vision_radius: int = 2,
        num_keys: int = 2,
    ):
        super().__init__(num_agents)

        self.grid_size = grid_size
        self.vision_radius = vision_radius
        self.num_keys = num_keys

        # Track explored areas and key positions
        self.explored_areas = np.zeros(grid_size, dtype=bool)
        self.grid_elements = np.zeros(grid_size, dtype=np.float32)
        self.key_positions = []
        self.collected_keys = set()
        self.exit_position = None

        # Rewards
        self.explore_reward = 0.0
        self.key_reward = 0
        self.escape_reward = 5.0
        self.step_penalty = -0.01
        self.collision_penalty = -0.1

        # Episode parameters
        self.max_steps = 100
        self.step_count = 0

        # State space includes flattened grid plus all agent coordinates
        self.state_space = gym.spaces.Box(
            low=-1,
            high=4,
            shape=(self.grid_size[0] * self.grid_size[1] + 2 * self.num_agents,),
            dtype=np.float32,
        )

        # Observation space same as state space
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=4,
            shape=(self.grid_size[0] * self.grid_size[1] + 2 * self.num_agents,),
            dtype=np.float32,
        )

        # Actions: Right, Left, Up, Down, Stay
        self.action_space = gym.spaces.Discrete(5)
        self.action_moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]

        # Initialize agent positions
        self.agent_positions = {i: (0, 0) for i in range(self.num_agents)}

        # Add state space for unobserved elements (key and exit positions)
        self.state_space_unobserved = gym.spaces.Box(
            low=-1,
            high=4,
            shape=(2 * (self.num_keys + 1),),
            dtype=np.float32,
        )

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
        self.collected_keys = set()
        self.explored_areas = np.zeros(self.grid_size, dtype=bool)
        self.grid_elements = np.zeros(self.grid_size, dtype=np.float32)

        if random_start:
            self._reset_random_starting_locations()
        else:
            self._reset_starting_locations()

        self._place_keys_and_exit()

        self._update_explored_areas()

        states = self._make_states(unobserved=unobserved)
        observations = self._make_observations()
        return states, observations

    def step(
        self, actions: dict, unobserved: bool = False
    ) -> tuple[dict, dict, dict, bool, bool, dict]:
        self.step_count += 1
        total_reward = 0.0

        prev_explored = self.explored_areas.copy()

        new_positions = {}
        for agent_id, action in actions.items():
            move = self.action_moves[action]
            if self._is_valid_move(agent_id, move):
                new_pos = (
                    self.agent_positions[agent_id][0] + move[0],
                    self.agent_positions[agent_id][1] + move[1],
                )
                new_positions[agent_id] = new_pos
                if action != 4:  # Apply step penalty only for movement actions
                    total_reward += self.step_penalty
            else:
                new_positions[agent_id] = self.agent_positions[agent_id]
                total_reward += self.collision_penalty

        # Resolve collisions
        self.agent_positions = self._resolve_collisions(new_positions)

        # Update explored areas and handle key collection
        self._update_explored_areas()
        total_reward += self._handle_key_collection()

        # Calculate exploration rewards
        new_explored = np.sum(self.explored_areas) - np.sum(prev_explored)
        if new_explored > 0:
            total_reward += self.explore_reward * new_explored

        # Check escape condition
        escaped = self._check_escape()
        if escaped:
            total_reward += self.escape_reward

        # Check if episode is done
        done = escaped or self.step_count >= self.max_steps

        # Distribute the same total reward to all agents
        rewards = {i: total_reward for i in range(self.num_agents)}

        states = self._make_states(unobserved=unobserved)
        observations = self._make_observations()
        return states, observations, rewards, done, False, {}

    def _reset_starting_locations(self):
        center_x = self.grid_size[0] // 2
        center_y = self.grid_size[1] // 2

        for agent in range(self.num_agents):
            self.agent_positions[agent] = (center_x, center_y)

    def _place_keys_and_exit(self):
        self.key_positions = []
        available_positions = [
            (i, j)
            for i in range(1, self.grid_size[0] - 1)
            for j in range(1, self.grid_size[1] - 1)
            if (i, j) not in self.agent_positions.values()
        ]

        # Place keys
        for _ in range(self.num_keys):
            if available_positions:
                idx = np.random.randint(0, len(available_positions))
                pos = available_positions.pop(idx)
                self.key_positions.append(pos)
                self.grid_elements[pos] = 2

        bottom_positions = [
            (self.grid_size[0] - 1, j) for j in range(self.grid_size[1])
        ]
        self.exit_position = bottom_positions[
            np.random.randint(0, len(bottom_positions))
        ]
        self.grid_elements[self.exit_position] = 3

    def _update_explored_areas(self):
        for agent_pos in self.agent_positions.values():
            for i in range(-self.vision_radius, self.vision_radius + 1):
                for j in range(-self.vision_radius, self.vision_radius + 1):
                    new_pos = (agent_pos[0] + i, agent_pos[1] + j)
                    if (
                        0 <= new_pos[0] < self.grid_size[0]
                        and 0 <= new_pos[1] < self.grid_size[1]
                    ):
                        self.explored_areas[new_pos] = True

    def _make_observations(self):
        observations = {}

        for agent_id, agent_pos in self.agent_positions.items():
            # Create a grid with unexplored areas as 0
            grid_obs = np.zeros(self.grid_size, dtype=np.float32)

            # Only reveal areas within the current vision radius
            for i in range(-self.vision_radius, self.vision_radius + 1):
                for j in range(-self.vision_radius, self.vision_radius + 1):
                    obs_pos = (agent_pos[0] + i, agent_pos[1] + j)
                    if (
                        0 <= obs_pos[0] < self.grid_size[0]
                        and 0 <= obs_pos[1] < self.grid_size[1]
                    ):
                        # Mark this position as visible and show its content
                        grid_obs[obs_pos] = self.grid_elements[obs_pos]

            # Add all agents with value 4
            for other_agent_id, other_agent_pos in self.agent_positions.items():
                if (
                    abs(other_agent_pos[0] - agent_pos[0]) <= self.vision_radius
                    and abs(other_agent_pos[1] - agent_pos[1]) <= self.vision_radius
                ):
                    grid_obs[other_agent_pos] = 4

            # Create agent coordinates array
            agent_coords = []
            for i in range(self.num_agents):
                agent_coords.extend(self.agent_positions[i])
            agent_coords = np.array(agent_coords, dtype=np.float32)

            # Flatten grid and append agent's coordinates
            obs = grid_obs.flatten()
            observations[agent_id] = np.concatenate([obs, agent_coords])

        return observations

    def _make_states(self, unobserved: bool = False):
        if unobserved:
            state = []
            for key_pos in self.key_positions:
                state.extend([key_pos[0], key_pos[1]])
            state.extend([self.exit_position[0], self.exit_position[1]])
            state = np.array(state, dtype=np.float32)
            return {i: state for i in range(self.num_agents)}

        grid_state = self.grid_elements.copy()

        for agent_pos in self.agent_positions.values():
            grid_state[agent_pos] = 4

        # Create agent coordinates array once
        agent_coords = []
        for i in range(self.num_agents):
            agent_coords.extend(self.agent_positions[i])
        agent_coords = np.array(agent_coords, dtype=np.float32)

        # Create states for each agent
        states = {}
        for agent_id in range(self.num_agents):
            state = grid_state.flatten()
            states[agent_id] = np.concatenate([state, agent_coords])

        return states

    def _handle_key_collection(self):
        reward = 0.0
        for agent_pos in self.agent_positions.values():
            if agent_pos in self.key_positions and agent_pos not in self.collected_keys:
                self.collected_keys.add(agent_pos)
                self.grid_elements[agent_pos] = 1
                reward += self.key_reward
        return reward

    def _check_escape(self):
        if len(self.collected_keys) == self.num_keys:
            return any(
                pos == self.exit_position for pos in self.agent_positions.values()
            )
        return False

    def _is_valid_move(self, agent, move):
        r, c = self.agent_positions[agent]
        new_r, new_c = r + move[0], c + move[1]
        return 0 <= new_r < self.grid_size[0] and 0 <= new_c < self.grid_size[1]

    def _resolve_collisions(self, new_positions):
        position_count = {}
        for pos in new_positions.values():
            position_count[pos] = position_count.get(pos, 0) + 1

        final_positions = {}
        for agent_id, pos in new_positions.items():
            if position_count[pos] > 1:
                final_positions[agent_id] = self.agent_positions[agent_id]
            else:
                final_positions[agent_id] = pos

        return final_positions

    def _reset_random_starting_locations(self):
        available_positions = [
            (i, j) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])
        ]
        for agent in range(self.num_agents):
            if available_positions:
                idx = np.random.randint(0, len(available_positions))
                self.agent_positions[agent] = available_positions.pop(idx)
            else:
                self.agent_positions[agent] = (0, 0)


if __name__ == "__main__":
    env = MaEscapeRoom(num_agents=2, grid_size=(10, 10), vision_radius=1, num_keys=2)
    obs, _ = env.reset(random_start=True)

    for _ in range(10):
        actions = {i: np.random.randint(0, 5) for i in range(env.num_agents)}
        state, obs, rewards, done, _, _ = env.step(actions, unobserved=False)

        print("\nActions:", actions)
        print("Rewards:", rewards)
        print("\nGrid state:")
        print(state[0])

        print("\nLocal observations:")
        for agent_id, agent_obs in obs.items():
            print(f"Agent {agent_id} observation:")
            print(agent_obs)

        if done:
            break
