import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.utils.args_class import Args


class QPolicyModel(nn.Module):
    def __init__(self, args: Args, input_length: int = None) -> None:
        super(QPolicyModel, self).__init__()
        self.args = args
        self.input_length = (
            input_length if input_length is not None else args.obs_shape[1]
        )
        self.fc1 = nn.Linear(self.input_length, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, args.num_actions)

    def forward(self, observation):
        x = self.fc1(observation)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
