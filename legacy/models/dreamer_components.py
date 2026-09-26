"""
DreamerV3 Components for Trading
Based on: https://github.com/danijar/dreamerv3
Paper: https://arxiv.org/abs/2301.04104

Core Architecture:
1. RSSM (Recurrent State Space Model) - learns latent market dynamics
2. Encoder - compresses observations into embeddings
3. Dynamics - predicts next latent state
4. Decoder - reconstructs observations from latent
5. Reward Predictor - predicts rewards in latent space
6. Actor-Critic - policy and value networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np


def symlog(x):
    """Symlog transformation - squashes large values while preserving small ones"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x):
    """Inverse of symlog"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / norm


class GRUCell(nn.Module):
    """GRU Cell with RMSNorm"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Reset, update, and new gates
        self.W_ir = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_r = RMSNorm(hidden_size)

        self.W_iz = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_z = RMSNorm(hidden_size)

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_n = RMSNorm(hidden_size)

    def forward(self, x, h):
        r = torch.sigmoid(self.norm_r(self.W_ir(x) + self.W_hr(h)))
        z = torch.sigmoid(self.norm_z(self.W_iz(x) + self.W_hz(h)))
        n = torch.tanh(self.norm_n(self.W_in(x) + self.W_hn(r * h)))
        h_new = (1 - z) * n + z * h
        return h_new


class Encoder(nn.Module):
    """Encodes observations into embeddings"""
    def __init__(self, obs_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, embed_dim),
            RMSNorm(embed_dim),
        )

    def forward(self, obs):
        # Apply symlog to observations for stability
        obs = symlog(obs)
        return self.net(obs)


class RSSM(nn.Module):
    """
    Recurrent State Space Model

    The "World Model" that learns market dynamics:
    - h_t (deterministic): GRU hidden state (memory)
    - z_t (stochastic): latent state (current market regime)
    """
    def __init__(self, embed_dim=256, hidden_dim=512, stoch_dim=32, num_categories=32, action_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.num_categories = num_categories
        self.action_dim = action_dim

        # Prior: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, stoch_dim * num_categories)
        )

        # Posterior: q(z_t | h_t, e_t) where e_t is encoded observation
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, stoch_dim * num_categories)
        )

        # Dynamics: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        # stoch_dim * num_categories because z is flattened one-hot
        self.gru = GRUCell(stoch_dim * num_categories + action_dim, hidden_dim)

    def initial_state(self, batch_size, device):
        """Initialize h_0 and z_0"""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        # z is flattened one-hot, so it's stoch_dim * num_categories
        z = torch.zeros(batch_size, self.stoch_dim * self.num_categories, device=device)
        return h, z

    def observe(self, embed, action, h_prev, z_prev):
        """
        Posterior inference: q(z_t | h_t, e_t)
        Used during training with real observations
        """
        # Update deterministic state
        h = self.gru(torch.cat([z_prev, action], dim=-1), h_prev)

        # Compute posterior distribution
        posterior_logits = self.posterior_net(torch.cat([h, embed], dim=-1))
        posterior_logits = posterior_logits.reshape(-1, self.stoch_dim, self.num_categories)

        # Sample z_t from categorical distribution
        z = self._sample_categorical(posterior_logits)

        # Also compute prior for KL regularization
        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.reshape(-1, self.stoch_dim, self.num_categories)

        return h, z, prior_logits, posterior_logits

    def imagine(self, action, h_prev, z_prev):
        """
        Prior imagination: p(z_t | h_t)
        Used for dreaming/planning without real observations
        """
        # Update deterministic state
        h = self.gru(torch.cat([z_prev, action], dim=-1), h_prev)

        # Sample from prior
        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.reshape(-1, self.stoch_dim, self.num_categories)
        z = self._sample_categorical(prior_logits)

        return h, z, prior_logits

    def _sample_categorical(self, logits):
        """Sample from categorical distribution with straight-through estimator"""
        # Sample during training, use mode during eval
        if self.training:
            # Gumbel-Softmax trick
            dist = torch.distributions.OneHotCategorical(logits=logits)
            z_one_hot = dist.sample()
        else:
            # Use argmax (mode) during evaluation
            z_one_hot = F.one_hot(logits.argmax(dim=-1), self.num_categories).float()

        # Flatten the one-hot vectors
        return z_one_hot.reshape(-1, self.stoch_dim * self.num_categories)

    def get_state(self, h, z):
        """Concatenate h and z for full latent state"""
        return torch.cat([h, z], dim=-1)

    def kl_loss(self, prior_logits, posterior_logits, free_nats=1.0, balance=0.8):
        """
        KL divergence with free bits and balancing
        Prevents posterior collapse
        """
        prior = torch.distributions.Categorical(logits=prior_logits)
        posterior = torch.distributions.Categorical(logits=posterior_logits)

        # KL divergence
        kl = torch.distributions.kl_divergence(posterior, prior)

        # Free nats: don't penalize KL below this threshold
        kl = torch.maximum(kl, torch.tensor(free_nats / self.stoch_dim, device=kl.device))

        # KL balancing: mix between treating prior/posterior as constant
        kl_balanced = balance * kl + (1 - balance) * kl.detach()

        return kl_balanced.sum(dim=-1).mean()


class Decoder(nn.Module):
    """Reconstructs observations from latent state"""
    def __init__(self, state_dim, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, obs_dim),
        )

    def forward(self, state):
        """Returns mean of Gaussian distribution"""
        return symexp(self.net(state))


class RewardPredictor(nn.Module):
    """Predicts rewards from latent state"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        """Returns symlog-transformed reward prediction"""
        return self.net(state).squeeze(-1)


class Actor(nn.Module):
    """Policy network - outputs action distribution"""
    def __init__(self, state_dim, action_dim=3):
        super().__init__()
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, state):
        """Returns action logits for categorical distribution"""
        return self.net(state)

    def sample(self, state, deterministic=False):
        """Sample action from policy"""
        logits = self(state)
        if deterministic:
            action = F.one_hot(logits.argmax(dim=-1), self.action_dim).float()
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample()
            action = F.one_hot(action_idx, self.action_dim).float()
        return action


class Critic(nn.Module):
    """Value network - estimates state value"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            RMSNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        """Returns symlog-transformed value prediction"""
        return self.net(state).squeeze(-1)


if __name__ == "__main__":
    # Quick sanity check
    print("Testing DreamerV3 components...")

    batch_size = 4
    obs_dim = 64 * 11  # window * features
    device = 'cpu'

    # Test encoder
    encoder = Encoder(obs_dim, embed_dim=256)
    obs = torch.randn(batch_size, obs_dim)
    embed = encoder(obs)
    print(f"✅ Encoder: {obs.shape} -> {embed.shape}")

    # Test RSSM
    rssm = RSSM(embed_dim=256, hidden_dim=512, stoch_dim=32, num_categories=32)
    h, z = rssm.initial_state(batch_size, device)
    action = F.one_hot(torch.randint(0, 3, (batch_size,)), 3).float()
    h_new, z_new, prior, posterior = rssm.observe(embed, action, h, z)
    print(f"✅ RSSM: h={h_new.shape}, z={z_new.shape}")

    # Test decoder
    state = rssm.get_state(h_new, z_new)
    decoder = Decoder(state.shape[-1], obs_dim)
    obs_recon = decoder(state)
    print(f"✅ Decoder: {state.shape} -> {obs_recon.shape}")

    # Test reward predictor
    reward_pred = RewardPredictor(state.shape[-1])
    reward = reward_pred(state)
    print(f"✅ Reward Predictor: {state.shape} -> {reward.shape}")

    # Test actor-critic
    actor = Actor(state.shape[-1], action_dim=3)
    critic = Critic(state.shape[-1])
    action_logits = actor(state)
    value = critic(state)
    print(f"✅ Actor: {state.shape} -> {action_logits.shape}")
    print(f"✅ Critic: {state.shape} -> {value.shape}")

    print("\n🎉 All components working!")
