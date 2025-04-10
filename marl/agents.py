import torch

from marl.utils.args_class import Args
from marl.models.agents.qss_agent_belief_vae import QssBeliefAgent


class BeliefAgents:
    def __init__(self, args: Args) -> None:
        self.num_agents = args.num_agents
        if args.agent_type == "QSSBeliefVAE":
            self.agents = {i: QssBeliefAgent(args, i) for i in range(self.num_agents)}
        else:
            raise ValueError(f"Unknown belief agent type: {args.agent_type}")

    def act(self, observations):
        actions_and_beliefs = {
            i: self.agents[i].step(observations[i]) for i in range(self.num_agents)
        }
        actions = {i: action for i, (action, _) in actions_and_beliefs.items()}
        beliefs = {i: belief for i, (_, belief) in actions_and_beliefs.items()}
        return actions, beliefs

    def act_random(self, observations):
        actions_and_beliefs = {
            i: self.agents[i].random_step(observations[i])
            for i in range(self.num_agents)
        }
        actions = {i: action for i, (action, _) in actions_and_beliefs.items()}
        beliefs = {i: belief for i, (_, belief) in actions_and_beliefs.items()}
        return actions, beliefs

    def pretrain(self, batch):
        total_loss = 0
        for i in range(self.num_agents):
            loss = self.agents[i].pretrain_belief_model(batch[i])
            total_loss += loss
        return total_loss / self.num_agents

    def reset(self):
        for agent in self.agents.values():
            agent.reset()

    def update(self, batch):
        for i in range(self.num_agents):
            self.agents[i].optimisation_step(batch[i])
