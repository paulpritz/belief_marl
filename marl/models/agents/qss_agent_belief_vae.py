import torch
import torch.nn as nn
import random

from marl.models.q_policy_model import QPolicyModel
from marl.models.qss_model import QssCritic
from marl.models.next_state_predictor import StatePredictor
from marl.models.belief_vae import BeliefVAE
from marl.utils.args_class import Args


class QssBeliefAgent:
    def __init__(self, args: Args, agent_id: int) -> None:
        self.args = args
        self.agent_id = agent_id
        self.gamma = args.gamma

        self.steps_taken = 0

        self.belief_hidden = None
        self.last_action = None
        self.current_history = None

        self.belief_vae = BeliefVAE(
            obs_dim=args.obs_shape[1],
            action_dim=1,
            state_dim=args.state_shape_unobserved[1],
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_samples=args.num_belief_samples,
            temperature=args.belief_temperature,
        )

        # Q-network with belief input
        input_dim = args.obs_shape[1] + args.belief_dim
        self.q = QPolicyModel(args, input_length=input_dim)
        self.q_target = QPolicyModel(args, input_length=input_dim)
        self.q_target.load_state_dict(self.q.state_dict())

        # QSS and f networks (working in obs + belief encoding space)
        self.qss = QssCritic(args)
        self.f = StatePredictor(args, input_length=input_dim)
        self.qss_target = QssCritic(args)
        self.qss_target.load_state_dict(self.qss.state_dict())

        # Optimizers
        self.q_optim = torch.optim.Adam(
            self.q.parameters(),
            args.q_lr,
        )
        self.qss_optim = torch.optim.Adam(self.qss.parameters(), args.qss_lr)
        self.f_optim = torch.optim.Adam(self.f.parameters(), args.f_lr)
        self.belief_optim = torch.optim.Adam(
            self.belief_vae.parameters(),
            args.belief_lr,
        )

    def reset(self):
        self.belief_hidden = None
        self.last_action = None
        self.current_history = None

    def get_belief_samples(self, observation, action):
        belief_samples, new_hidden, history_encoding = (
            self.belief_vae.get_belief_samples(observation, action, self.belief_hidden)
        )
        return belief_samples

    def encode_belief_samples(self, belief_samples):
        res = belief_samples.mean(dim=1)
        return res

    def step(self, observations) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # Get belief samples and history encoding
            belief_samples, new_hidden, history_encoding = (
                self.belief_vae.get_belief_samples(
                    observations, self.last_action, self.belief_hidden
                )
            )
            # Update hidden state and history encoding
            self.belief_hidden = new_hidden
            self.current_history = history_encoding

            # Encode belief samples into single belief state and add batch dimension
            belief_encoding = self.encode_belief_samples(belief_samples.unsqueeze(0))

            # Combine observation and belief for Q-network input
            combined_input = torch.cat([observations, belief_encoding], dim=-1)

            # Get Q-values and select action
            q_values = self.q(combined_input)
            action_index = torch.argmax(q_values, dim=-1)

            fraction_done = self.steps_taken / (
                self.args.episodes * self.args.max_steps_per_episode
            )
            epsilon = self.args.epsilon * (1 - fraction_done)
            if random.random() < epsilon:
                action_index = torch.randint(0, self.args.num_actions - 1, (1,))

            self.last_action = action_index

            self.steps_taken += 1
            return action_index, belief_samples

    def random_step(self, observations) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            belief_samples, new_hidden, history_encoding = (
                self.belief_vae.get_belief_samples(
                    observations, self.last_action, self.belief_hidden
                )
            )
            self.belief_hidden = new_hidden
            self.current_history = history_encoding

            action_index = torch.randint(0, self.args.num_actions - 1, (1,))
            self.last_action = action_index

            return action_index, belief_samples

    def compute_belief_loss(self, agent_batch):
        return self.belief_vae.compute_loss(
            agent_batch.states,
            agent_batch.observations,
            agent_batch.actions,
            agent_batch.mask,
        )

    def pretrain_belief_model(self, batch):
        belief_loss = self.compute_belief_loss(batch)
        self.belief_optim.zero_grad()
        belief_loss.backward()
        self.belief_optim.step()
        return belief_loss.item()

    def optimisation_step(self, agent_batch):
        loss_f = self.compute_f_loss(agent_batch)
        self.f_optim.zero_grad()
        loss_f.backward()
        self.f_optim.step()

        loss_qss = self.compute_qss_loss(agent_batch)
        self.qss_optim.zero_grad()
        loss_qss.backward()
        self.qss_optim.step()

        loss_q = self.compute_q_loss(agent_batch)
        self.q_optim.zero_grad()
        loss_q.backward()
        self.q_optim.step()

        # Update target networks
        self.update_target_net(self.q, self.q_target)
        self.update_target_net(self.qss, self.qss_target)

    def update_target_net(self, training_net, target_net):
        target_net_state_dict = target_net.state_dict()
        training_net_state_dict = training_net.state_dict()
        for key in training_net_state_dict:
            target_net_state_dict[key] = training_net_state_dict[
                key
            ] * self.args.tau + target_net_state_dict[key] * (1 - self.args.tau)
        target_net.load_state_dict(target_net_state_dict)

    def compute_f_loss(self, agent_batch):
        observations = agent_batch.observations
        belief_samples = agent_batch.belief_samples
        actions = agent_batch.actions
        next_observations = agent_batch.next_observations

        belief_encoding = self.encode_belief_samples(belief_samples)
        combined_input = torch.cat([observations, belief_encoding], dim=-1)

        pred_opt_next = self.f(combined_input, actions)
        with torch.no_grad():
            q_ss_values = self.qss(observations, pred_opt_next)

        criterion = nn.MSELoss(reduction="mean")
        loss = -torch.mean(self.args.lambda_f * q_ss_values) + criterion(
            next_observations, pred_opt_next
        )
        return loss

    def compute_q_loss(self, agent_batch):
        observations = agent_batch.observations
        actions = agent_batch.actions
        next_observations = agent_batch.next_observations
        belief_samples = agent_batch.belief_samples
        next_belief_samples = agent_batch.next_belief_samples

        # Input beliefs
        belief_encoding = self.encode_belief_samples(belief_samples)
        combined_input = torch.cat([observations, belief_encoding], dim=-1)

        state_action_values = self.q(combined_input).gather(1, actions)

        with torch.no_grad():
            pred_observation_next = self.f(combined_input, actions)
            Qss_pred_next_observation = self.qss_target(
                observations, pred_observation_next
            )
            Qss_obs_next_observations = self.qss_target(observations, next_observations)
            Q_target = torch.maximum(
                Qss_obs_next_observations, Qss_pred_next_observation
            )

        criterion = nn.MSELoss(reduction="mean")
        loss = criterion(state_action_values, Q_target)
        return loss

    def compute_qss_loss(self, agent_batch):
        observations = agent_batch.observations
        belief_samples = agent_batch.belief_samples
        next_observations = agent_batch.next_observations
        next_belief_samples = agent_batch.next_belief_samples
        rewards = agent_batch.rewards

        non_final_mask = torch.tensor(
            tuple(map(lambda d: d is not True, agent_batch.dones)),
            dtype=torch.bool,
        )
        non_final_next_observations = next_observations[non_final_mask]
        non_final_next_beliefs = next_belief_samples[non_final_mask]

        state_values = self.qss(observations, next_observations).squeeze()

        next_state_values = torch.zeros(observations.size(0))
        with torch.no_grad():
            non_final_belief_encoding = self.encode_belief_samples(
                non_final_next_beliefs
            )
            # Combine next observations with belief encodings
            non_final_combined_input = torch.cat(
                [non_final_next_observations, non_final_belief_encoding], dim=-1
            )

            # Get next actions using combined input
            next_action_probs = self.q(non_final_combined_input)
            next_actions = torch.argmax(next_action_probs, dim=-1).unsqueeze(-1)

            next_next_observation = self.f(non_final_combined_input, next_actions)
            next_state_values[non_final_mask] = self.qss_target(
                non_final_next_observations, next_next_observation
            ).squeeze()

        expected_state_value = (next_state_values * self.gamma) + rewards
        criterion = nn.MSELoss(reduction="mean")
        loss = criterion(state_values, expected_state_value)
        return loss
