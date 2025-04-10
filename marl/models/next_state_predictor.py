# Model to predict optimal next state

import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.utils.args_class import Args


class StatePredictor(nn.Module):
    def __init__(self, args: Args, input_length: int = None) -> None:
        super(StatePredictor, self).__init__()
        self.args = args
        self.representation_length = (
            input_length if input_length is not None else args.obs_shape[1]
        )

        # Add 1 for action
        self.fc1 = nn.Linear(self.representation_length + 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.args.obs_shape[1])

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
