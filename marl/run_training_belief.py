import torch
import numpy as np
import os
import json
import datetime
from matplotlib import pyplot as plt
import pickle

from marl.utils.args_class import Args, set_seed
from marl.agents import BeliefAgents
from marl.utils.replay_memory_belief import EpisodeReplayMemoryBelief
from marl.environments.ma_sphinx import MaSphinx


def pretrain_belief_models(args, env, agents):
    print("Pre-training belief models...")

    # Create separate buffer for pre-training data
    pretrain_buffer = EpisodeReplayMemoryBelief(args.memory_length, args.num_agents)
    pretrain_episodes = 100  # 1000

    for episode in range(pretrain_episodes):
        states, observations = env.reset(random_start=True, unobserved=True)
        agents.reset()

        cur_states = {
            i: torch.tensor(states[i], dtype=torch.float32).unsqueeze(0)
            for i in states.keys()
        }
        cur_observations = {
            i: torch.tensor(observations[i], dtype=torch.float32).unsqueeze(0)
            for i in observations.keys()
        }

        terminal = False
        steps_in_episode = 0

        while not terminal:
            actions, belief_samples = agents.act_random(cur_observations)

            states, observations, rewards, done, truncated, _ = env.step(
                actions, unobserved=True
            )

            terminal = done or steps_in_episode == args.max_steps_per_episode

            next_states = {
                i: torch.tensor(states[i], dtype=torch.float32).unsqueeze(0)
                for i in states.keys()
            }
            next_observations = {
                i: torch.tensor(observations[i], dtype=torch.float32).unsqueeze(0)
                for i in observations.keys()
            }

            for i in range(args.num_agents):
                pretrain_buffer.push(
                    i,
                    cur_states[i],
                    cur_observations[i],
                    actions[i].unsqueeze(0),
                    next_states[i],
                    next_observations[i],
                    torch.tensor(rewards[i]).unsqueeze(0),
                    terminal,
                    belief_samples[i],
                    belief_samples[i],
                )

            cur_states = next_states
            cur_observations = next_observations
            steps_in_episode += 1

        for agent in range(args.num_agents):
            pretrain_buffer.finish_episode(agent)

        if episode % 100 == 0:
            print(f"Collected {episode}/{pretrain_episodes} episodes for pre-training")

    pretrain_epochs = 10  # 150
    batch_size = 32  # 256

    for epoch in range(pretrain_epochs):
        total_loss = 0
        num_batches = 0

        for _ in range(10):  # 10
            if len(pretrain_buffer) > batch_size:
                batch = pretrain_buffer.sample_episodes(batch_size=batch_size)
                loss = agents.pretrain(batch)
                total_loss += loss
                num_batches += 1

        if epoch % 10 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(
                f"Pre-training epoch {epoch}/{pretrain_epochs}, Average loss: {avg_loss:.4f}"
            )


def training_run(env=None, args=None, parent_folder=None):
    if args is None:
        args = Args(
            agent_type="QSSBeliefVAE",
            obs_shape=(1, 4),
            state_shape=(1, 4),
            state_shape_unobserved=(1, 2),
            hidden_dim=64,
            latent_dim=4,
            num_belief_samples=2,
            belief_dim=4,
            belief_temperature=0.5,
            num_actions=9,
            num_agents=2,
            q_lr=0.001,
            belief_lr=0.001,
            qss_lr=0.001,
            f_lr=0.001,
            lambda_f=0.1,
            tau=0.005,
            episodes=10_000,
            memory_length=10_000,
            gamma=0.99,
            max_steps_per_episode=40,
            batch_size=128,
            epsilon=0.6,
            seed=42,
        )

    set_seed(args.seed)

    if env is None:
        env = MaSphinx(num_agents=args.num_agents)

    print("env state", env.state_space.shape)
    print("env observation", env.observation_space.shape)
    print("env action", env.action_space.n)

    args.obs_shape = (1, env.observation_space.shape[0])
    args.state_shape = (1, env.state_space.shape[0])
    args.state_shape_unobserved = (1, env.state_space_unobserved.shape[0])
    args.num_actions = env.action_space.n
    print("env unobserved", env.state_space_unobserved.shape)

    buffer = EpisodeReplayMemoryBelief(args.memory_length, args.num_agents)
    agents = BeliefAgents(args)
    ########################## Logging ##########################
    # Create logging directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = env.__class__.__name__

    # Create base data directory
    data_dir = "data"
    if parent_folder:
        data_dir = os.path.join(data_dir, parent_folder)

    os.makedirs(data_dir, exist_ok=True)

    log_dir = os.path.join(data_dir, f"{env_name}_{args.agent_type}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_dict = {}
    for key, value in vars(args).items():
        if isinstance(value, tuple):
            config_dict[key] = list(value)
        elif isinstance(value, (int, float, str, bool, type(None))):
            config_dict[key] = value
        else:
            config_dict[key] = str(value)

    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    print(f"Logging data to: {log_dir}")

    ########################## Pre-training ##########################

    # Pre-train belief models
    pretrain_belief_models(args, env, agents)

    episode_returns = {i: [] for i in range(args.num_agents)}
    episode_states = {i: [] for i in range(args.num_agents)}
    episode_beliefs = {i: [] for i in range(args.num_agents)}

    ########################## Training ##########################

    for episode in range(args.episodes):
        states, observations = env.reset()
        agents.reset()

        cur_states = {
            i: torch.tensor(states[i], dtype=torch.float32).unsqueeze(0)
            for i in states.keys()
        }
        cur_observations = {
            i: torch.tensor(observations[i], dtype=torch.float32).unsqueeze(0)
            for i in observations.keys()
        }

        terminal = False
        steps_in_episode = 0

        for i in range(args.num_agents):
            episode_returns[i].append(0)

        while not terminal:
            actions, belief_samples = agents.act(cur_observations)

            if steps_in_episode == 0 and episode % 100 == 0:
                for i in range(args.num_agents):
                    episode_beliefs[i].append(
                        [belief_samples[i].cpu().numpy().tolist()]
                    )

            states, observations, rewards, done, truncated, _ = env.step(actions)

            terminal = done or steps_in_episode == args.max_steps_per_episode

            next_states = {
                i: torch.tensor(states[i], dtype=torch.float32).unsqueeze(0)
                for i in states.keys()
            }
            next_observations = {
                i: torch.tensor(observations[i], dtype=torch.float32).unsqueeze(0)
                for i in observations.keys()
            }

            for i in range(args.num_agents):
                buffer.push(
                    i,
                    cur_states[i],
                    cur_observations[i],
                    actions[i].unsqueeze(0),
                    next_states[i],
                    next_observations[i],
                    torch.tensor(rewards[i]).unsqueeze(0),
                    terminal,
                    belief_samples[i],
                    belief_samples[i],
                )
                episode_returns[i][-1] += rewards[i]

                if episode % 100 == 0:
                    episode_states[i].append(
                        cur_states[i].squeeze(0).cpu().numpy().tolist()
                    )
                    episode_beliefs[i].append(belief_samples[i].cpu().numpy().tolist())

            cur_states = next_states
            cur_observations = next_observations
            steps_in_episode += 1

        for agent in range(args.num_agents):
            buffer.finish_episode(agent)

        if len(buffer) > args.batch_size:
            for _ in range(10):
                batch = buffer.sample_transitions(batch_size=args.batch_size)
                agents.update(batch)

        # Print progress and save logs
        if episode % 100 == 0:
            avg_rewards = np.mean(episode_returns[0][-100:])
            print(f"Episode {episode}, Avg. Rewards: {avg_rewards}")

            returns_path = os.path.join(log_dir, "episode_returns.json")
            with open(returns_path, "w") as f:
                json.dump(episode_returns, f)

            states_path = os.path.join(checkpoint_dir, f"episode_states_{episode}.json")
            with open(states_path, "w") as f:
                json.dump(episode_states, f)

            beliefs_path = os.path.join(
                checkpoint_dir, f"episode_beliefs_{episode}.json"
            )
            with open(beliefs_path, "w") as f:
                json.dump(episode_beliefs, f)

            if episode < args.episodes - 100:
                episode_states = {i: [] for i in range(args.num_agents)}
                episode_beliefs = {i: [] for i in range(args.num_agents)}

    # Save all agents in a single file
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "agents.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(agents, f)


if __name__ == "__main__":
    training_run()
