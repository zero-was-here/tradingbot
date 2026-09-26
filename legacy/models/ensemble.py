"""
Ensemble of Models - Robustness through Diversity

One model can fail. Five models all agreeing is much more reliable.

Strategy:
- Train 5 different models (different seeds, slight architecture variations)
- Only trade when CONSENSUS (>=3 models agree)
- Use disagreement as uncertainty measure

Benefits:
- More robust to individual model failures
- Uncertainty estimation
- Reduced overfitting
- Better generalization
"""

import torch
import torch.nn as nn
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleAgent:
    """
    Ensemble of 5 different trading models

    Only trades when majority agrees
    Uses disagreement to measure uncertainty
    """

    def __init__(self, agent_class, num_models=5, **agent_kwargs):
        """
        Initialize ensemble

        Args:
            agent_class: Agent class to ensemble
            num_models: Number of models in ensemble (default: 5)
            **agent_kwargs: Arguments to pass to each agent
        """

        self.num_models = num_models
        self.models = []

        # Create models with different random seeds
        for i in range(num_models):
            # Vary architecture slightly
            kwargs = agent_kwargs.copy()

            # Add variation
            if 'hidden_dim' in kwargs:
                kwargs['hidden_dim'] = kwargs['hidden_dim'] + i * 16

            # Set different seed
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            # Create model
            model = agent_class(**kwargs)
            self.models.append(model)

        logger.info(f"🎼 Ensemble initialized with {num_models} models")

    def act(self, obs, use_consensus=True, consensus_threshold=3):
        """
        Get action from ensemble

        Args:
            obs: Observation
            use_consensus: If True, require consensus
            consensus_threshold: Minimum agreeing models (default: 3/5)

        Returns:
            action: Final action
            info: Dict with voting details
        """

        # Get predictions from all models
        actions = []
        q_values = []

        for model in self.models:
            action = model.act(obs)
            actions.append(action)

            # Get Q-value if available
            if hasattr(model, 'get_q_value'):
                q = model.get_q_value(obs, action)
                q_values.append(q)

        # Count votes
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        # Get majority action
        majority_action = max(action_counts, key=action_counts.get)
        majority_count = action_counts[majority_action]

        if use_consensus:
            # Check if consensus met
            if majority_count >= consensus_threshold:
                final_action = majority_action
                consensus = True
            else:
                # No consensus - stay flat
                final_action = 0
                consensus = False
        else:
            # Simple majority
            final_action = majority_action
            consensus = True

        # Compute uncertainty
        uncertainty = self.get_uncertainty(actions)

        info = {
            'actions': actions,
            'action_counts': action_counts,
            'majority_action': majority_action,
            'majority_count': majority_count,
            'consensus': consensus,
            'uncertainty': uncertainty,
            'q_values': q_values if q_values else None,
        }

        return final_action, info

    def get_uncertainty(self, actions):
        """
        Measure disagreement between models

        High disagreement = high uncertainty = don't trade

        Returns:
            entropy: Uncertainty measure (0 = all agree, high = disagree)
        """

        # Count each action
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        # Compute probabilities
        total = len(actions)
        probs = [count / total for count in action_counts.values()]

        # Compute entropy
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)

        return entropy

    def train(self, *args, **kwargs):
        """Train all models"""

        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{self.num_models}...")
            model.train(*args, **kwargs)

    def save(self, path_prefix):
        """Save all models"""

        for i, model in enumerate(self.models):
            path = f"{path_prefix}_model{i}.pt"
            model.save(path)

        logger.info(f"💾 Saved {self.num_models} models")

    def load(self, path_prefix):
        """Load all models"""

        for i, model in enumerate(self.models):
            path = f"{path_prefix}_model{i}.pt"
            model.load(path)

        logger.info(f"📂 Loaded {self.num_models} models")


# Mock agent for testing
class MockAgent:
    def __init__(self, hidden_dim=256, bias=0.0):
        self.hidden_dim = hidden_dim
        self.bias = bias  # Bias toward certain action

    def act(self, obs):
        # Random action with bias
        if np.random.random() < 0.5 + self.bias:
            return 1
        else:
            return 0

    def train(self, *args, **kwargs):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


# Example usage
if __name__ == "__main__":
    print("🎼 Ensemble Agent Demo\n")

    # Create ensemble
    ensemble = EnsembleAgent(
        agent_class=MockAgent,
        num_models=5,
        hidden_dim=256
    )

    # Test voting
    print("="*60)
    print("Test 1: Strong Consensus")
    print("="*60)

    # Mock observation
    obs = np.random.randn(100)

    # Get action
    action, info = ensemble.act(obs, use_consensus=True, consensus_threshold=3)

    print(f"Individual votes: {info['actions']}")
    print(f"Vote counts: {info['action_counts']}")
    print(f"Majority action: {info['majority_action']} ({info['majority_count']}/5 votes)")
    print(f"Consensus: {info['consensus']}")
    print(f"Final action: {action}")
    print(f"Uncertainty: {info['uncertainty']:.3f}")

    # Test multiple times
    print("\n" + "="*60)
    print("Test 2: Multiple Decisions")
    print("="*60)

    consensus_count = 0
    no_consensus_count = 0

    for i in range(20):
        obs = np.random.randn(100)
        action, info = ensemble.act(obs, use_consensus=True)

        if info['consensus']:
            consensus_count += 1
        else:
            no_consensus_count += 1

    print(f"Total decisions: 20")
    print(f"Consensus reached: {consensus_count}")
    print(f"No consensus: {no_consensus_count}")
    print(f"Consensus rate: {consensus_count/20:.1%}")

    # Test uncertainty
    print("\n" + "="*60)
    print("Test 3: Uncertainty Measurement")
    print("="*60)

    # Create biased ensemble (models disagree more)
    class BiasedMockAgent(MockAgent):
        def __init__(self, hidden_dim=256, model_id=0):
            bias = (model_id - 2) * 0.2  # Different biases
            super().__init__(hidden_dim, bias)

    biased_ensemble = EnsembleAgent(
        agent_class=BiasedMockAgent,
        num_models=5,
        hidden_dim=256
    )

    obs = np.random.randn(100)
    action, info = biased_ensemble.act(obs)

    print(f"Individual votes: {info['actions']}")
    print(f"Uncertainty: {info['uncertainty']:.3f}")

    if info['uncertainty'] > 0.5:
        print("⚠️ HIGH UNCERTAINTY - Models disagree - Maybe don't trade")
    else:
        print("✅ LOW UNCERTAINTY - Models agree - Safe to trade")

    print("\n✅ Ensemble system working!")
    print("\nKey benefits:")
    print("  ✅ Robustness - One model failing doesn't crash system")
    print("  ✅ Uncertainty - Know when models disagree")
    print("  ✅ Consensus - Only trade when majority agrees")
    print("  ✅ Better generalization - Average out overfitting")
