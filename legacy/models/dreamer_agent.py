"""
DreamerV3 Agent for Trading

This implements the full DreamerV3 algorithm:
1. World Model Learning (representation + dynamics + reward prediction)
2. Behavior Learning (actor-critic in imagination)

Based on: https://arxiv.org/abs/2301.04104
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam  # Use Adam instead of AdamW to avoid transformers import issue
import numpy as np
from collections import deque

from models.dreamer_components import (
    Encoder, RSSM, Decoder, RewardPredictor, Actor, Critic,
    symlog, symexp
)


class ReplayBuffer:
    """Experience replay buffer for sequences"""
    def __init__(self, capacity=100_000, seq_len=64):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, done):
        """Add a single transition"""
        self.buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': done
        })

    def sample(self, batch_size):
        """Sample sequences of length seq_len"""
        if len(self.buffer) < self.seq_len + 1:
            return None

        sequences = []
        for _ in range(batch_size):
            # Sample random starting point
            start_idx = np.random.randint(0, len(self.buffer) - self.seq_len)

            # Extract sequence
            seq_obs = []
            seq_action = []
            seq_reward = []
            seq_done = []

            for i in range(self.seq_len):
                step = self.buffer[start_idx + i]
                seq_obs.append(step['obs'])
                seq_action.append(step['action'])
                seq_reward.append(step['reward'])
                seq_done.append(step['done'])

            sequences.append({
                'obs': np.stack(seq_obs),
                'action': np.stack(seq_action),
                'reward': np.array(seq_reward, dtype=np.float32),
                'done': np.array(seq_done, dtype=np.float32)
            })

        # Stack into batch
        batch = {
            'obs': torch.FloatTensor(np.stack([s['obs'] for s in sequences])),
            'action': torch.FloatTensor(np.stack([s['action'] for s in sequences])),
            'reward': torch.FloatTensor(np.stack([s['reward'] for s in sequences])),
            'done': torch.FloatTensor(np.stack([s['done'] for s in sequences]))
        }

        return batch

    def __len__(self):
        return len(self.buffer)


class DreamerV3Agent:
    """
    DreamerV3 Agent for Trading

    The agent learns a World Model of the market, then uses it to
    imagine trajectories and improve its policy.
    """
    def __init__(
        self,
        obs_dim,
        action_dim=3,  # [flat, long, short] or [flat, long] for long-only
        device='cpu',
        # Architecture
        embed_dim=256,
        hidden_dim=512,
        stoch_dim=32,
        num_categories=32,
        # Optimization
        lr_world_model=3e-4,
        lr_actor=1e-4,
        lr_critic=3e-4,
        # Hyperparameters
        gamma=0.99,
        lambda_=0.95,  # GAE lambda
        horizon=15,  # imagination horizon
        free_nats=1.0,
        kl_balance=0.8,
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_ = lambda_
        self.horizon = horizon

        # Build networks
        self.encoder = Encoder(obs_dim, embed_dim).to(device)
        self.rssm = RSSM(embed_dim, hidden_dim, stoch_dim, num_categories, action_dim).to(device)
        self.decoder = Decoder(hidden_dim + stoch_dim * num_categories, obs_dim).to(device)
        self.reward_predictor = RewardPredictor(hidden_dim + stoch_dim * num_categories).to(device)
        self.actor = Actor(hidden_dim + stoch_dim * num_categories, action_dim).to(device)
        self.critic = Critic(hidden_dim + stoch_dim * num_categories).to(device)

        # Optimizers
        world_model_params = (
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_predictor.parameters())
        )
        self.optimizer_world_model = Adam(world_model_params, lr=lr_world_model)
        self.optimizer_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100_000, seq_len=64)

        # Hyperparameters
        self.free_nats = free_nats
        self.kl_balance = kl_balance

        # Tracking
        self.training_step = 0

    def act(self, obs, h=None, z=None, deterministic=False):
        """
        Select action given observation

        Args:
            obs: observation (np.array)
            h, z: previous latent state (None if first step)
            deterministic: use greedy action

        Returns:
            action (np.array), new (h, z)
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Encode observation
            embed = self.encoder(obs_t)

            # Initialize or update latent state
            if h is None or z is None:
                h, z = self.rssm.initial_state(1, self.device)
                # For first step, use zero action
                action = torch.zeros(1, self.action_dim, device=self.device)
            else:
                # Use previous action (stored in z)
                action = self.prev_action

            # Update latent state
            h, z, _, _ = self.rssm.observe(embed, action, h, z)

            # Get state
            state = self.rssm.get_state(h, z)

            # Sample action
            action = self.actor.sample(state, deterministic=deterministic)

            # Store for next step
            self.prev_action = action

            # Detach before converting to numpy
            return action.detach().cpu().numpy()[0], (h, z)

    def train_step(self, batch_size=16):
        """
        Single training step

        1. Train World Model (representation + dynamics + reward)
        2. Imagine trajectories in learned world model
        3. Train Actor-Critic on imagined trajectories
        """
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        if batch is None:
            return None

        obs = batch['obs'].to(self.device)  # (B, T, obs_dim)
        action = batch['action'].to(self.device)  # (B, T, action_dim)
        reward = batch['reward'].to(self.device)  # (B, T)

        B, T = obs.shape[0], obs.shape[1]

        # ==================== PHASE 1: Train World Model ====================
        self.optimizer_world_model.zero_grad()

        # Encode observations
        embed = self.encoder(obs.reshape(B * T, -1)).reshape(B, T, -1)

        # Initialize state
        h, z = self.rssm.initial_state(B, self.device)

        # Storage for losses
        recon_losses = []
        reward_losses = []
        kl_losses = []

        # Unroll sequence
        for t in range(T):
            # Posterior inference
            h, z, prior_logits, posterior_logits = self.rssm.observe(
                embed[:, t], action[:, t], h, z
            )

            # Get state
            state = self.rssm.get_state(h, z)

            # Reconstruction loss
            obs_pred = self.decoder(state)
            recon_loss = F.mse_loss(obs_pred, obs[:, t])
            recon_losses.append(recon_loss)

            # Reward prediction loss (symlog space)
            reward_pred = self.reward_predictor(state)
            reward_target = symlog(reward[:, t])
            reward_loss = F.mse_loss(reward_pred, reward_target)
            reward_losses.append(reward_loss)

            # KL divergence loss
            kl_loss = self.rssm.kl_loss(prior_logits, posterior_logits, self.free_nats, self.kl_balance)
            kl_losses.append(kl_loss)

        # Total world model loss
        recon_loss_total = torch.stack(recon_losses).mean()
        reward_loss_total = torch.stack(reward_losses).mean()
        kl_loss_total = torch.stack(kl_losses).mean()

        world_model_loss = recon_loss_total + reward_loss_total + kl_loss_total

        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_predictor.parameters()),
            max_norm=100.0
        )
        self.optimizer_world_model.step()

        # ==================== PHASE 2: Imagine & Train Actor-Critic ====================

        # Start from random states in the batch
        with torch.no_grad():
            # Use final states from world model rollout
            h_start = h.detach()
            z_start = z.detach()

        # Imagine trajectories
        imagined_states, imagined_rewards = self._imagine_trajectory(
            h_start, z_start, self.horizon
        )

        # Train critic
        self.optimizer_critic.zero_grad()
        value_loss = self._compute_value_loss(imagined_states.detach(), imagined_rewards.detach())
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=100.0)
        self.optimizer_critic.step()

        # Train actor (re-imagine to get fresh gradients)
        imagined_states_actor, imagined_rewards_actor = self._imagine_trajectory(
            h_start, z_start, self.horizon
        )
        self.optimizer_actor.zero_grad()
        policy_loss = self._compute_policy_loss(imagined_states_actor, imagined_rewards_actor)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=100.0)
        self.optimizer_actor.step()

        self.training_step += 1

        return {
            'world_model_loss': world_model_loss.item(),
            'recon_loss': recon_loss_total.item(),
            'reward_loss': reward_loss_total.item(),
            'kl_loss': kl_loss_total.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
        }

    def _imagine_trajectory(self, h, z, horizon):
        """
        Imagine trajectory by rolling out policy in world model

        Args:
            h, z: starting latent state (B, h_dim), (B, z_dim)
            horizon: number of steps to imagine

        Returns:
            states: (B, H, state_dim)
            rewards: (B, H)
        """
        states = []
        rewards = []

        for t in range(horizon):
            state = self.rssm.get_state(h, z)
            states.append(state)

            # Predict reward
            reward_pred = self.reward_predictor(state)
            rewards.append(symexp(reward_pred))

            # Sample action from policy
            action = self.actor.sample(state, deterministic=False)

            # Imagine next state
            h, z, _ = self.rssm.imagine(action, h, z)

        states = torch.stack(states, dim=1)  # (B, H, state_dim)
        rewards = torch.stack(rewards, dim=1)  # (B, H)

        return states, rewards

    def _compute_value_loss(self, states, rewards):
        """
        Train critic using lambda-returns

        Args:
            states: (B, H, state_dim)
            rewards: (B, H)
        """
        B, H = states.shape[0], states.shape[1]

        # Predict values
        values = self.critic(states.reshape(B * H, -1)).reshape(B, H)

        # Compute lambda-returns (bootstrapped from value function)
        with torch.no_grad():
            # Bootstrap from final value
            next_values = torch.cat([values[:, 1:], torch.zeros(B, 1, device=self.device)], dim=1)

            # TD targets
            td_targets = rewards + self.gamma * next_values

            # Lambda-returns (GAE-style)
            returns = torch.zeros_like(rewards)
            returns[:, -1] = td_targets[:, -1]
            for t in reversed(range(H - 1)):
                returns[:, t] = td_targets[:, t] + self.gamma * self.lambda_ * (returns[:, t + 1] - next_values[:, t])

        # Value loss
        value_loss = F.mse_loss(values, returns)
        return value_loss

    def _compute_policy_loss(self, states, rewards):
        """
        Train actor to maximize expected return

        Args:
            states: (B, H, state_dim)
            rewards: (B, H)
        """
        B, H = states.shape[0], states.shape[1]

        # Get action logits from actor
        action_logits = self.actor(states.reshape(B * H, -1)).reshape(B, H, -1)

        # Sample actions (for policy gradient)
        dist = torch.distributions.Categorical(logits=action_logits.reshape(B * H, -1))

        # Compute returns for policy gradient
        with torch.no_grad():
            values = self.critic(states.reshape(B * H, -1)).reshape(B, H)
            next_values = torch.cat([values[:, 1:], torch.zeros(B, 1, device=self.device)], dim=1)

            # Advantages
            td_targets = rewards + self.gamma * next_values
            advantages = td_targets - values

        # Policy gradient loss (REINFORCE with baseline)
        # Maximize log prob weighted by advantages
        sampled_actions = dist.sample()
        log_probs = dist.log_prob(sampled_actions).reshape(B, H)

        policy_loss = -(log_probs * advantages).mean()

        return policy_loss

    def save(self, path):
        """Save agent"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'rssm': self.rssm.state_dict(),
            'decoder': self.decoder.state_dict(),
            'reward_predictor': self.reward_predictor.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'training_step': self.training_step,
        }, path)

    def load(self, path):
        """Load agent"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.reward_predictor.load_state_dict(checkpoint['reward_predictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.training_step = checkpoint.get('training_step', 0)
        print(f"Loaded checkpoint from step {self.training_step}")


if __name__ == "__main__":
    # Quick test
    print("Testing DreamerV3 Agent...")

    obs_dim = 64 * 11  # window * features
    agent = DreamerV3Agent(obs_dim, action_dim=3, device='cpu')

    # Test act
    obs = np.random.randn(obs_dim)
    action, (h, z) = agent.act(obs)
    print(f"✅ Action: {action.shape}")

    # Test training
    # Add some dummy data
    for _ in range(100):
        obs = np.random.randn(obs_dim)
        action = np.random.randn(3)
        reward = np.random.randn()
        done = False
        agent.replay_buffer.add(obs, action, reward, done)

    # Train step
    losses = agent.train_step(batch_size=4)
    if losses:
        print(f"✅ Training step completed")
        print(f"   World Model Loss: {losses['world_model_loss']:.4f}")
        print(f"   Value Loss: {losses['value_loss']:.4f}")
        print(f"   Policy Loss: {losses['policy_loss']:.4f}")

    print("\n🎉 DreamerV3 Agent working!")
