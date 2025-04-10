# Class to accomodate the state-state Q-function


# Class to accomodate the basic Q-function model

import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.utils.args_class import Args


class QssCritic(nn.Module):
    def __init__(self, args: Args, input_length: int = None) -> None:
        super(QssCritic, self).__init__()
        self.args = args
        self.input_length = (
            input_length if input_length is not None else args.obs_shape[1] * 2
        )

        self.fc1 = nn.Linear(self.input_length, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    # Input two states and output Q value
    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
