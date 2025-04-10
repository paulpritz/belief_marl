import random
from collections import namedtuple, deque
import torch

EpisodeBelief = namedtuple(
    "EpisodeBelief",
    (
        "states",
        "observations",
        "actions",
        "next_states",
        "next_observations",
        "rewards",
        "dones",
        "mask",  # Single mask per episode for length
        "belief_samples",  # Belief samples for current state
        "next_belief_samples",  # Belief samples for next state
    ),
)


class EpisodeReplayMemoryBelief:
    def __init__(self, capacity, num_agents):
        self.num_agents = num_agents
        # Store completed episodes
        self.memories = {i: deque([], maxlen=capacity) for i in range(num_agents)}
        # Buffer for building current episode
        self.current_episodes = {
            i: {
                "states": [],
                "observations": [],
                "actions": [],
                "next_states": [],
                "next_observations": [],
                "rewards": [],
                "dones": [],
                "belief_samples": [],
                "next_belief_samples": [],
            }
            for i in range(num_agents)
        }

    def push(
        self,
        agent_id,
        state,
        observation,
        action,
        next_state,
        next_observation,
        reward,
        done,
        belief_samples,
        next_belief_samples,
    ):
        self.current_episodes[agent_id]["states"].append(state)
        self.current_episodes[agent_id]["observations"].append(observation)
        self.current_episodes[agent_id]["actions"].append(action)
        self.current_episodes[agent_id]["next_states"].append(next_state)
        self.current_episodes[agent_id]["next_observations"].append(next_observation)
        self.current_episodes[agent_id]["rewards"].append(reward)
        self.current_episodes[agent_id]["dones"].append(done)
        self.current_episodes[agent_id]["belief_samples"].append(belief_samples)
        self.current_episodes[agent_id]["next_belief_samples"].append(
            next_belief_samples
        )

    def finish_episode(self, agent_id):
        if len(self.current_episodes[agent_id]["states"]) == 0:
            return  # No transitions to save

        # Get next belief samples and shift them
        next_belief_samples = self.current_episodes[agent_id]["next_belief_samples"]

        # Create zero tensor matching shape of belief samples
        zero_belief = torch.zeros_like(next_belief_samples[-1])

        # Shift next_belief_samples and prepend zero tensor
        shifted_next_beliefs = next_belief_samples[:-1] + [zero_belief]

        # Convert lists to tensors
        episode = EpisodeBelief(
            states=torch.cat(self.current_episodes[agent_id]["states"]).float(),
            observations=torch.cat(
                self.current_episodes[agent_id]["observations"]
            ).float(),
            actions=torch.cat(self.current_episodes[agent_id]["actions"]),
            next_states=torch.cat(
                self.current_episodes[agent_id]["next_states"]
            ).float(),
            next_observations=torch.cat(
                self.current_episodes[agent_id]["next_observations"]
            ).float(),
            rewards=torch.tensor(
                self.current_episodes[agent_id]["rewards"], dtype=torch.float32
            ),
            dones=torch.tensor(
                self.current_episodes[agent_id]["dones"], dtype=torch.float32
            ),
            mask=len(self.current_episodes[agent_id]["states"]),  # Store episode length
            belief_samples=torch.stack(
                self.current_episodes[agent_id]["belief_samples"]
            ).float(),
            next_belief_samples=torch.stack(shifted_next_beliefs).float(),
        )

        # Store the completed episode
        self.memories[agent_id].append(episode)

        # Clear the current episode buffer
        self.current_episodes[agent_id] = {
            "states": [],
            "observations": [],
            "actions": [],
            "next_states": [],
            "next_observations": [],
            "rewards": [],
            "dones": [],
            "belief_samples": [],
            "next_belief_samples": [],
        }

    def pad_sequence(self, sequences, max_len=None):
        if max_len is None:
            max_len = max(seq.size(0) for seq in sequences)

        rest_shape = sequences[0].shape[1:]

        pad_shape = (max_len,) + rest_shape

        padded_seqs = []
        for seq in sequences:
            padded = torch.zeros(pad_shape, dtype=seq.dtype)
            padded[: seq.size(0), ...] = seq
            padded_seqs.append(padded)

        return torch.stack(padded_seqs)

    def sample_episodes(self, batch_size):
        batch = {}
        for i in range(self.num_agents):
            episodes = random.sample(self.memories[i], batch_size)

            # Get max episode length
            max_len = max(ep.mask for ep in episodes)

            # Pad all sequences to the same length
            states = self.pad_sequence([ep.states for ep in episodes], max_len)
            obs = self.pad_sequence([ep.observations for ep in episodes], max_len)
            actions = self.pad_sequence([ep.actions for ep in episodes], max_len)
            next_states = self.pad_sequence(
                [ep.next_states for ep in episodes], max_len
            )
            next_obs = self.pad_sequence(
                [ep.next_observations for ep in episodes], max_len
            )
            rewards = self.pad_sequence(
                [ep.rewards.unsqueeze(-1) for ep in episodes], max_len
            )
            dones = self.pad_sequence(
                [ep.dones.unsqueeze(-1) for ep in episodes], max_len
            )
            belief_samples = self.pad_sequence(
                [ep.belief_samples for ep in episodes], max_len
            )
            next_belief_samples = self.pad_sequence(
                [ep.next_belief_samples for ep in episodes], max_len
            )

            # Store episode lengths
            lengths = torch.tensor([ep.mask for ep in episodes])

            batch[i] = EpisodeBelief(
                states=states,  # [batch_size, max_ep_len, state_dim]
                observations=obs,  # [batch_size, max_ep_len, obs_dim]
                actions=actions,  # [batch_size, max_ep_len, action_dim]
                next_states=next_states,  # [batch_size, max_ep_len, state_dim]
                next_observations=next_obs,  # [batch_size, max_ep_len, obs_dim]
                rewards=rewards.squeeze(-1),  # [batch_size, max_ep_len]
                dones=dones.squeeze(-1),  # [batch_size, max_ep_len]
                mask=lengths,  # [batch_size] containing episode lengths
                belief_samples=belief_samples,  # [batch_size, max_ep_len, num_samples, belief_dim]
                next_belief_samples=next_belief_samples,  # [batch_size, max_ep_len, num_samples, belief_dim]
            )
        return batch

    def sample_transitions(self, batch_size):
        transitions = {}

        for agent_id in range(self.num_agents):
            episodes = random.sample(
                self.memories[agent_id], min(batch_size, len(self.memories[agent_id]))
            )

            sampled_states = []
            sampled_obs = []
            sampled_actions = []
            sampled_next_states = []
            sampled_next_obs = []
            sampled_rewards = []
            sampled_dones = []
            sampled_belief_samples = []
            sampled_next_belief_samples = []

            while len(sampled_states) < batch_size:
                episode = random.choice(episodes)
                t = random.randint(0, episode.mask - 1)  # mask contains episode length

                # Add the transition
                sampled_states.append(episode.states[t])
                sampled_obs.append(episode.observations[t])
                sampled_actions.append(episode.actions[t])
                sampled_next_states.append(episode.next_states[t])
                sampled_next_obs.append(episode.next_observations[t])
                sampled_rewards.append(episode.rewards[t])
                sampled_dones.append(episode.dones[t])
                sampled_belief_samples.append(episode.belief_samples[t])
                sampled_next_belief_samples.append(episode.next_belief_samples[t])

            # Stack all the sampled transitions
            transitions[agent_id] = EpisodeBelief(
                states=torch.stack(sampled_states),
                observations=torch.stack(sampled_obs),
                actions=torch.stack(sampled_actions),
                next_states=torch.stack(sampled_next_states),
                next_observations=torch.stack(sampled_next_obs),
                rewards=torch.stack(sampled_rewards),
                dones=torch.stack(sampled_dones),
                mask=batch_size,  # All transitions are valid
                belief_samples=torch.stack(sampled_belief_samples),
                next_belief_samples=torch.stack(sampled_next_belief_samples),
            )

        return transitions

    def __len__(self):
        return len(self.memories[0])
