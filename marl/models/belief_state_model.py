import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder-Decoder CVAE model to generate belief states (per-agent)
class HistoryEncoder(nn.Module):
    def __init__(self, args):
        super(HistoryEncoder, self).__init__()
        self.args = args
        self.input_dim = args.obs_shape[1]
        self.memory_size = 128

        self.rnn_1 = nn.GRUCell(self.input_dim, self.memory_size)
        self.rnn_2 = nn.GRUCell(self.memory_size, self.memory_size)
        self.fc1 = nn.Linear(self.memory_size + self.input_dim, 128)
        self.fc_out = nn.Linear(128, self.args.history_embedding_size)

    def forward(self, obs, memory):
        memory_1, memory_2 = (
            memory[:, : self.memory_size],
            memory[:, self.memory_size :],
        )
        memory_1 = self.rnn_1(obs, memory_1)
        memory_2 = self.rnn_2(F.relu(memory_1), memory_2)

        x = torch.cat([memory_2, obs], dim=-1)
        x = self.fc1(x)
        x = self.fc_out(F.relu(x))

        memory = torch.cat([memory_1, memory_2], dim=-1)
        return x, memory


class BeliefStateModel(nn.Module):
    def __init__(self, args):
        super(BeliefStateModel, self).__init__()
        self.args = args

        self.history_model = HistoryEncoder(args)

        self.embedding_size = self.args.history_embedding_size

        self.obs_dim = self.args.obs_shape[1]
        self.vae_latent_dim = self.args.vae_latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim + self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * self.vae_latent_dim),
        )

        # Generative model is also probabilistic. Hence output is dim. 2 * obs_dim for mean and cov
        self.decoder = nn.Sequential(
            nn.Linear(self.vae_latent_dim + self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * self.obs_dim),
        )

    def forward(self, obs, memory):
        encoding, memory = self.history_model(obs, memory)
        return encoding, memory

    def encoder_dist(self, obs, context):
        input = torch.cat([obs, context], dim=-1)
        out = self.encoder(input)
        mean = [out[:, : self.vae_latent_dim]]
        # Restrict the std to be between 1 and -1
        std = F.softplus([out[:, self.vae_latent_dim :]], threshold=1) + 1e-1
        return mean, std

    def decoder_dist(self, z, context):
        input = torch.cat([z, context], dim=-1)
        out = self.decoder(input)
        mean = [out[:, : self.obs_dim]]
        # Restrict the std to be between 1 and -1
        std = F.softplus([out[:, self.obs_dim :]], threshold=1) + 1e-1
        return mean, std
