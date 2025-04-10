import gymnasium as gym
from marl.environments.base_env import BaseEnv
import numpy as np
from typing import Optional


class MaGathering(BaseEnv):
    def __init__(
        self, num_agents: int, grid_size: tuple[int, int], num_resources: int = 3
    ):
        super().__init__(num_agents)

        self.grid_size = grid_size
        self.num_resources = num_resources
        self.resource_positions = []

        # Rewards and penalties
        self.gather_reward = 3.0
        self.step_penalty = -0.01
        self.collision_penalty = -0.05

        # Episode parameters
        self.max_steps = 100
        self.step_count = 0

        # State space includes flattened grid
        self.state_space = gym.spaces.Box(
            low=-1,
            high=2,
            shape=(self.grid_size[0] * self.grid_size[1],),
            dtype=np.float32,
        )

        # Observation space is flattened 3x3 grid for each agent (shared vision)
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=2,
            shape=(9 * self.num_agents,),
            dtype=np.float32,
        )

        # Add state space for unobserved elements (resource positions and agent positions)
        # Using fixed size with padding
        self.state_space_unobserved = gym.spaces.Box(
            low=-1,
            high=max(grid_size),
            shape=(2 * num_resources + 2 * num_agents,),
            dtype=np.float32,
        )

        # Actions: Right, Left, Up, Down, Stay
        self.action_space = gym.spaces.Discrete(5)
        self.action_moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]

        # Initialize agent positions
        self.agent_positions = {i: (0, 0) for i in range(self.num_agents)}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        unobserved: bool = False,
        **kwargs,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self._reset_starting_locations()
        self._place_resources()

        states = self._make_states(unobserved=unobserved)
        observations = self._make_observations()
        return states, observations

    def step(
        self, actions: dict, unobserved: bool = False
    ) -> tuple[dict, dict, dict, bool, bool, dict]:
        # Initialize shared reward
        total_reward = 0
        self.step_count += 1

        # Move agents
        new_positions = {}
        for agent_id, action in actions.items():
            move = self.action_moves[action]
            if self._is_valid_move(agent_id, move):
                new_pos = (
                    self.agent_positions[agent_id][0] + move[0],
                    self.agent_positions[agent_id][1] + move[1],
                )
                new_positions[agent_id] = new_pos

                # Apply step penalty if agent moved
                if action != 4:
                    total_reward += self.step_penalty
            else:
                new_positions[agent_id] = self.agent_positions[agent_id]

        # Check for collisions and resolve them
        final_positions = self._resolve_collisions(new_positions, total_reward)
        self.agent_positions = final_positions

        # Check for resource gathering and update shared reward
        total_reward = self._handle_resource_gathering(total_reward)

        # Distribute the same reward to all agents
        rewards = {i: total_reward for i in range(self.num_agents)}

        # Check if episode is done
        done = self.step_count >= self.max_steps or len(self.resource_positions) == 0

        states = self._make_states(unobserved=unobserved)
        observations = self._make_observations()
        return states, observations, rewards, done, False, {}

    def _reset_starting_locations(self):
        for agent in range(self.num_agents):
            while True:
                pos = (
                    np.random.randint(0, self.grid_size[0]),
                    np.random.randint(0, self.grid_size[1]),
                )
                if pos not in self.agent_positions.values():
                    self.agent_positions[agent] = pos
                    break

    def _place_resources(self):
        self.resource_positions = []
        available_positions = [
            (i, j)
            for i in range(self.grid_size[0])
            for j in range(self.grid_size[1])
            if (i, j) not in self.agent_positions.values()
        ]

        positions = np.random.choice(
            len(available_positions),
            size=min(self.num_resources, len(available_positions)),
            replace=False,
        )

        self.resource_positions = [available_positions[i] for i in positions]

    def _make_observations(self):
        individual_observations = {}

        for agent_id in range(self.num_agents):
            # Create 3x3 grid centered on agent
            obs = np.zeros((3, 3), dtype=np.float32)
            agent_pos = self.agent_positions[agent_id]

            # Iterate through 3x3 grid around agent
            for i in range(3):
                for j in range(3):
                    world_pos = (
                        agent_pos[0] + (i - 1),
                        agent_pos[1] + (j - 1),
                    )

                    # Check if position is outside grid (wall)
                    if (
                        world_pos[0] < 0
                        or world_pos[0] >= self.grid_size[0]
                        or world_pos[1] < 0
                        or world_pos[1] >= self.grid_size[1]
                    ):
                        obs[i, j] = -1  # Wall
                        continue

                    # Check for resources
                    if world_pos in self.resource_positions:
                        obs[i, j] = 1  # Resource
                        continue

                    # Check for other agents
                    for other_agent_id, other_pos in self.agent_positions.items():
                        if other_agent_id != agent_id and other_pos == world_pos:
                            obs[i, j] = 2  # Other agent
                            break

            individual_observations[agent_id] = obs.flatten()
        combined_observations = {}
        for agent_id in range(self.num_agents):
            all_obs = []
            for i in range(self.num_agents):
                all_obs.append(individual_observations[i])

            combined_observations[agent_id] = np.concatenate(all_obs)

        return combined_observations

    def _is_valid_move(self, agent, move):
        r, c = self.agent_positions[agent]
        new_r, new_c = r + move[0], c + move[1]

        # Check grid bounds
        if (
            new_r < 0
            or new_r >= self.grid_size[0]
            or new_c < 0
            or new_c >= self.grid_size[1]
        ):
            return False

        return True

    def _resolve_collisions(self, new_positions, shared_reward):
        position_count = {}
        for agent_id, pos in new_positions.items():
            position_count[pos] = position_count.get(pos, 0) + 1

        final_positions = {}
        for agent_id, pos in new_positions.items():
            if position_count[pos] > 1:
                # Collision occurred, keep agent in original position
                final_positions[agent_id] = self.agent_positions[agent_id]
                shared_reward += self.collision_penalty
            else:
                final_positions[agent_id] = pos

        return final_positions

    def _handle_resource_gathering(self, total_reward):
        resources_to_remove = []
        for resource_pos in self.resource_positions:
            for agent_pos in self.agent_positions.values():
                if agent_pos == resource_pos:
                    total_reward += self.gather_reward
                    resources_to_remove.append(resource_pos)
                    break

        # Remove gathered resources
        for pos in resources_to_remove:
            self.resource_positions.remove(pos)

        return total_reward

    def _make_states(self, unobserved: bool = False):

        if unobserved:
            # Include resource positions (x,y coordinates) and agent positions
            state = np.full(
                2 * self.num_resources + 2 * self.num_agents, -1, dtype=np.float32
            )

            # Add resource positions (with padding if needed)
            for i, pos in enumerate(self.resource_positions):
                if i < self.num_resources:  # Only include up to num_resources
                    state[2 * i] = pos[0]
                    state[2 * i + 1] = pos[1]

            # Add agent positions
            offset = 2 * self.num_resources
            for agent_id in range(self.num_agents):
                state[offset + 2 * agent_id] = self.agent_positions[agent_id][0]
                state[offset + 2 * agent_id + 1] = self.agent_positions[agent_id][1]

            return {i: state for i in range(self.num_agents)}

        # Create base grid state
        grid_state = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=np.float32)

        # Add resources
        for pos in self.resource_positions:
            grid_state[pos[0], pos[1]] = 1

        # Add agents
        for agent_pos in self.agent_positions.values():
            grid_state[agent_pos[0], agent_pos[1]] = 2

        # Create flattened state dictionary for all agents
        states = {i: grid_state.flatten() for i in range(self.num_agents)}
        return states


if __name__ == "__main__":
    env = MaGathering(num_agents=3, grid_size=(20, 20), num_resources=3)
    obs, _ = env.reset()
    env.render()

    for _ in range(10):
        # Take random actions for both agents
        actions = {i: np.random.randint(0, 5) for i in range(env.num_agents)}
        state, obs, rewards, done, _, _ = env.step(actions, unobserved=True)

        print("\nActions:", actions)
        print("Rewards:", rewards)
        print("\nGlobal state:")
        env.render()
        print("Unobserved state (resource positions):")
        print(state[0])
        print(state[0].shape)

        print("\nShared observations:")
        for agent_id, agent_obs in obs.items():
            print(f"Agent {agent_id} observation (includes all agents' views):")
            print(agent_obs)
            print(agent_obs.shape)

        if done:
            break
