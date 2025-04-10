import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim=1, hidden_dim=64):
        super().__init__()
        self.action_dim = action_dim
        self.rnn = nn.GRU(
            input_size=obs_dim + action_dim, hidden_size=hidden_dim, batch_first=True
        )

    def forward(self, obs, action=None, hidden_state=None):
        # Add sequence dimension if processing single step
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)
            if action is not None:
                action = action.unsqueeze(1)
                if len(action.shape) == 2:
                    action = action.unsqueeze(0)

        # Handle actions
        if action is None:
            action = torch.zeros(*obs.shape[:-1], self.action_dim, device=obs.device)

        # Concatenate observations and actions
        x = torch.cat([obs, action], dim=-1)

        # Process through RNN
        output, new_hidden = self.rnn(x, hidden_state)

        if x.shape[1] == 1:
            output = output.squeeze(1)
        return output, new_hidden


class BeliefVAE(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        state_dim,
        latent_dim,
        hidden_dim=64,
        num_samples=10,
        temperature=1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.temperature = temperature
        self.action_dim = action_dim

        # Single history encoder
        self.history_encoder = HistoryEncoder(obs_dim, action_dim, hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.dec_mu = nn.Linear(hidden_dim, state_dim)
        self.dec_logvar = nn.Linear(hidden_dim, state_dim)

    def encode(self, state, history_encoding):
        x = torch.cat([state, history_encoding], dim=-1)
        x = self.encoder(x)
        return self.enc_mu(x), self.enc_logvar(x)

    def decode(self, z, history_encoding):
        x = torch.cat([z, history_encoding], dim=-1)
        x = self.decoder(x)
        return self.dec_mu(x), self.dec_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_sequence(self, state_sequence, obs_sequence, action_sequence=None):
        if action_sequence is not None:
            history_input = torch.cat([obs_sequence, action_sequence], dim=-1)
        else:
            zeros = torch.zeros_like(obs_sequence[..., : self.action_dim])
            history_input = torch.cat([obs_sequence, zeros], dim=-1)

        h_seq, _ = self.history_encoder.rnn(
            history_input
        )  # [batch_size, seq_len, hidden_dim]

        # Encode
        z_mu, z_logvar = self.encode(state_sequence, h_seq)
        z = self.reparameterize(z_mu, z_logvar)

        # Decode
        dec_mu, dec_logvar = self.decode(z, h_seq)

        return dec_mu, dec_logvar, z_mu, z_logvar, h_seq

    def compute_loss(
        self, state_sequence, obs_sequence, action_sequence=None, mask=None
    ):
        dec_mu, dec_logvar, z_mu, z_logvar, h_seq = self.forward_sequence(
            state_sequence, obs_sequence, action_sequence
        )

        # Create mask for valid timesteps [batch_size, seq_len, 1]
        if mask is not None:
            seq_mask = (
                torch.arange(state_sequence.size(1), device=state_sequence.device)[
                    None, :
                ]
                < mask[:, None]
            )
            seq_mask = seq_mask.unsqueeze(-1).float()
        else:
            seq_mask = torch.ones_like(state_sequence[..., :1])

        # Reconstruction loss (masked)
        seq_mask_recon = seq_mask.expand_as(dec_logvar)
        recon_loss = (
            0.5
            * torch.sum(
                seq_mask_recon
                * (dec_logvar + (state_sequence - dec_mu).pow(2) / dec_logvar.exp())
            )
            / torch.sum(seq_mask_recon)
        )

        # KL divergence (masked)
        seq_mask_KL = seq_mask.expand_as(z_logvar)
        kl_loss = -0.5 * torch.sum(
            seq_mask_KL * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        )

        return recon_loss + 0.5 * kl_loss

    def get_belief_samples_sequence(self, obs, action=None, hidden_state=None):
        with torch.no_grad():
            h, new_hidden = self.history_encoder(obs, action, hidden_state)
            print("h", h.shape)

            # Expand h to match samples: [batch_size, num_samples, seq_len, hidden_dim]
            h_expanded = h.unsqueeze(1).expand(-1, self.num_samples, -1, -1)
            # Sample from prior with temperature
            z = (
                torch.randn(
                    h_expanded.shape[0],
                    self.num_samples,
                    h_expanded.shape[2],
                    self.latent_dim,
                    device=h_expanded.device,
                )
                * self.temperature
            )
            # Decode to get state samples
            dec_mu, logvar = self.decode(z, h_expanded)
            # Return mu and logvar
            samples = torch.cat([dec_mu, logvar], dim=-1)
            return (
                samples,
                new_hidden,
                h,
            )

    def get_belief_samples(self, obs, action=None, hidden_state=None):
        with torch.no_grad():
            h, new_hidden = self.history_encoder(obs, action, hidden_state)

            # Expand h to match samples: [batch_size, num_samples, hidden_dim]
            h_expanded = h.expand(self.num_samples, -1)
            # Sample from prior with temperature
            z = (
                torch.randn(self.num_samples, self.latent_dim, device=h.device)
                * self.temperature
            )
            # Decode to get state samples
            dec_mu, dec_logvar = self.decode(z, h_expanded)
            # Concatenate mu and logvar
            samples = torch.cat([dec_mu, dec_logvar], dim=-1)
            return (
                samples,
                new_hidden,
                h,
            )


class BeliefEncoder(nn.Module):
    def __init__(self, state_dim, belief_dim=32):
        super().__init__()

        self.W_agg = nn.Sequential(
            nn.Linear(state_dim * 2, belief_dim),
            nn.ReLU(),
            nn.Linear(belief_dim, belief_dim),
        )

    def forward(self, belief_samples):

        # Average pooling over samples for each batch
        x = belief_samples.mean(dim=1)  # [batch_size, belief_dim]

        return self.W_agg(x)  # [batch_size, belief_dim]
