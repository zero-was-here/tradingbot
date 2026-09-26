"""
Monte Carlo Tree Search (MCTS) for DreamerV3

This implements the "Stockfish" lookahead capability:
Before each trade, simulate N possible futures and choose the best action.

Phase 3 of PROJECT GOD MODE

Based on:
- AlphaZero/MuZero MCTS
- Adapted for continuous trading with world models
"""

import torch
import numpy as np
from collections import defaultdict
import math


class MCTSNode:
    """
    Node in the Monte Carlo Search Tree

    Represents a state-action pair during planning
    """
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state  # (h, z) latent state
        self.parent = parent
        self.action = action  # action that led to this node
        self.prior = prior  # P(a|s) from policy network

        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0  # immediate reward from taking action

    @property
    def value(self):
        """Average value (Q-value)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self):
        """Check if node has been expanded"""
        return len(self.children) > 0

    def select_child(self, c_puct=1.0):
        """
        Select child using UCB formula (like AlphaZero)

        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        best_score = -float('inf')
        best_child = None

        for action, child in self.children.items():
            # Q-value
            q_value = child.value

            # Exploration bonus
            u_value = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, actions, priors, agent):
        """
        Expand node by creating children for all possible actions

        Args:
            actions: list of possible actions (one-hot encoded)
            priors: action probabilities from policy network
            agent: DreamerV3Agent (for world model)
        """
        h, z = self.state

        for i, action in enumerate(actions):
            # Imagine next state using world model
            with torch.no_grad():
                action_t = torch.FloatTensor(action).unsqueeze(0).to(agent.device)
                h_next, z_next, _ = agent.rssm.imagine(action_t, h, z)

                # Predict immediate reward
                state_next = agent.rssm.get_state(h_next, z_next)
                from models.dreamer_components import symexp
                reward = symexp(agent.reward_predictor(state_next)).item()

            # Create child node
            child = MCTSNode(
                state=(h_next, z_next),
                parent=self,
                action=action,
                prior=priors[i]
            )
            child.reward = reward

            self.children[i] = child

    def backup(self, value):
        """
        Backpropagate value up the tree

        Updates visit counts and value sums for all ancestors
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent


class MCTS:
    """
    Monte Carlo Tree Search for Trading

    Usage:
        mcts = MCTS(agent, num_simulations=100)
        best_action = mcts.search(h, z)
    """
    def __init__(
        self,
        agent,
        num_simulations=100,
        c_puct=1.0,  # exploration constant
        gamma=0.99,  # discount factor
    ):
        self.agent = agent
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.gamma = gamma

        # Possible actions
        self.actions = [
            np.array([1.0, 0.0], dtype=np.float32),  # flat
            np.array([0.0, 1.0], dtype=np.float32),  # long
            # For long/short: add [0, 0, 1] for short
        ]

    def search(self, h, z):
        """
        Run MCTS to find best action

        Args:
            h, z: current latent state

        Returns:
            best_action (one-hot encoded)
        """
        # Create root node
        root = MCTSNode(state=(h, z))

        # Run simulations
        for _ in range(self.num_simulations):
            # 1. Selection: traverse tree to leaf
            node = root
            search_path = [node]

            while node.expanded():
                node = node.select_child(self.c_puct)
                search_path.append(node)

            # 2. Expansion: expand leaf node
            h_leaf, z_leaf = node.state

            # Get policy priors from actor
            with torch.no_grad():
                state = self.agent.rssm.get_state(h_leaf, z_leaf)
                action_logits = self.agent.actor(state)
                priors = torch.softmax(action_logits, dim=-1).cpu().numpy()[0]

            # Expand node
            node.expand(self.actions, priors, self.agent)

            # 3. Simulation: evaluate leaf using critic
            with torch.no_grad():
                value = self.agent.critic(state).item()

            # 4. Backpropagation: update all nodes in search path
            for node in reversed(search_path):
                node.backup(value)
                value = node.reward + self.gamma * value

        # Select action with highest visit count (most promising)
        visit_counts = {action: child.visit_count for action, child in root.children.items()}
        best_action_idx = max(visit_counts, key=visit_counts.get)

        return self.actions[best_action_idx]

    def search_with_stats(self, h, z):
        """
        Run MCTS and return statistics for analysis

        Returns:
            best_action, stats_dict
        """
        root = MCTSNode(state=(h, z))

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            while node.expanded():
                node = node.select_child(self.c_puct)
                search_path.append(node)

            h_leaf, z_leaf = node.state

            with torch.no_grad():
                state = self.agent.rssm.get_state(h_leaf, z_leaf)
                action_logits = self.agent.actor(state)
                priors = torch.softmax(action_logits, dim=-1).cpu().numpy()[0]

            node.expand(self.actions, priors, self.agent)

            with torch.no_grad():
                value = self.agent.critic(state).item()

            for node in reversed(search_path):
                node.backup(value)
                value = node.reward + self.gamma * value

        # Collect statistics
        visit_counts = {action: child.visit_count for action, child in root.children.items()}
        q_values = {action: child.value for action, child in root.children.items()}
        expected_rewards = {action: child.reward for action, child in root.children.items()}

        best_action_idx = max(visit_counts, key=visit_counts.get)

        stats = {
            'visit_counts': visit_counts,
            'q_values': q_values,
            'expected_rewards': expected_rewards,
            'best_action_idx': best_action_idx,
        }

        return self.actions[best_action_idx], stats


class DreamerMCTSAgent:
    """
    DreamerV3 Agent with MCTS Planning

    This is the "Stockfish for Markets" - uses world model + search
    """
    def __init__(self, dreamer_agent, num_simulations=50, c_puct=1.0):
        self.dreamer = dreamer_agent
        self.mcts = MCTS(dreamer_agent, num_simulations, c_puct)

    def act(self, obs, h=None, z=None, use_mcts=True):
        """
        Select action with optional MCTS planning

        Args:
            obs: observation
            h, z: latent state (None if first step)
            use_mcts: whether to use MCTS (slower but better)

        Returns:
            action, (h, z)
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.dreamer.device)
            embed = self.dreamer.encoder(obs_t)

            if h is None or z is None:
                h, z = self.dreamer.rssm.initial_state(1, self.dreamer.device)
                action = torch.zeros(1, 2, device=self.dreamer.device)
            else:
                action = self.prev_action

            # Update latent state
            h, z, _, _ = self.dreamer.rssm.observe(embed, action, h, z)

            if use_mcts:
                # Use MCTS for action selection
                action = self.mcts.search(h, z)
                action = torch.FloatTensor(action).unsqueeze(0).to(self.dreamer.device)
            else:
                # Use actor directly (faster)
                state = self.dreamer.rssm.get_state(h, z)
                action = self.dreamer.actor.sample(state, deterministic=True)

            self.prev_action = action

            return action.cpu().numpy()[0], (h, z)


if __name__ == "__main__":
    print("Testing MCTS implementation...")

    # Create dummy agent
    import sys
    sys.path.insert(0, '..')
    from models.dreamer_agent import DreamerV3Agent

    obs_dim = 64 * 11
    agent = DreamerV3Agent(obs_dim, action_dim=2, device='cpu')

    # Create MCTS
    mcts = MCTS(agent, num_simulations=10)

    # Test search
    h, z = agent.rssm.initial_state(1, 'cpu')

    print("Running 10 MCTS simulations...")
    best_action, stats = mcts.search_with_stats(h, z)

    print(f"\n✅ MCTS Search Complete!")
    print(f"   Best Action: {np.argmax(best_action)}")
    print(f"   Visit Counts: {stats['visit_counts']}")
    print(f"   Q-Values: {stats['q_values']}")
    print(f"   Expected Rewards: {stats['expected_rewards']}")

    # Test DreamerMCTS agent
    print("\nTesting DreamerMCTS Agent...")
    dreamer_mcts = DreamerMCTSAgent(agent, num_simulations=10)

    obs = np.random.randn(obs_dim).astype(np.float32)
    action, (h, z) = dreamer_mcts.act(obs, use_mcts=True)

    print(f"\n✅ DreamerMCTS Agent working!")
    print(f"   Action: {action}")

    print("\n🎉 MCTS implementation ready for trading!")
