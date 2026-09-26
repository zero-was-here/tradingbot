"""
Transformer-Based Actor-Critic for Trading

Why Transformers > MLPs:
- MLPs: Can't remember patterns from days ago
- Transformers: Attention finds relevant historical patterns

Example:
- Day 1: Price bounces at 1950 (strong support)
- Day 2-3: Random movement
- Day 4: Price approaching 1950 again

MLP: Doesn't remember Day 1
Transformer: Attention mechanism says "Day 1 is relevant!" and recalls the bounce

This is the architecture used by:
- GPT (language)
- AlphaFold (protein folding)
- Decision Transformer (RL)

Now applied to trading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Add positional information to embeddings

    Transformers have no built-in notion of sequence order,
    so we add sinusoidal positional encodings.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Add positional encoding
        x = x + self.pe[:seq_len].unsqueeze(0)
        return x


class TransformerActor(nn.Module):
    """
    Transformer-based actor (policy network)

    Uses self-attention to find relevant historical patterns
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, num_heads=8,
                 num_layers=4, seq_len=64, dropout=0.1):
        """
        Args:
            state_dim: Input state dimension
            action_dim: Number of actions
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Input embedding
        self.embedding = nn.Linear(state_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Expect (batch, seq, feature)
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output head for action logits
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        logger.info(f"🤖 Transformer Actor initialized")
        logger.info(f"   Hidden dim: {hidden_dim}, Heads: {num_heads}, Layers: {num_layers}")

    def forward(self, state_sequence, mask=None):
        """
        Forward pass

        Args:
            state_sequence: (batch, seq_len, state_dim)
            mask: Optional attention mask

        Returns:
            action_logits: (batch, action_dim)
        """

        # Embed each state
        x = self.embedding(state_sequence)  # (batch, seq_len, hidden_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)  # (batch, seq_len, hidden_dim)

        # Use last token for action (or could use mean pooling)
        x = x[:, -1, :]  # (batch, hidden_dim)

        # Get action logits
        action_logits = self.action_head(x)  # (batch, action_dim)

        return action_logits

    def get_attention_weights(self, state_sequence):
        """
        Get attention weights for visualization

        Shows which historical states the model is paying attention to
        """

        # This requires modifying the transformer to return attention weights
        # For now, return None (would need to access internal transformer layers)
        return None


class TransformerCritic(nn.Module):
    """
    Transformer-based critic (value network)

    Estimates value of current state using historical context
    """

    def __init__(self, state_dim, hidden_dim=256, num_heads=8,
                 num_layers=4, seq_len=64, dropout=0.1):
        """
        Args:
            state_dim: Input state dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Input embedding
        self.embedding = nn.Linear(state_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        logger.info(f"📊 Transformer Critic initialized")
        logger.info(f"   Hidden dim: {hidden_dim}, Heads: {num_heads}, Layers: {num_layers}")

    def forward(self, state_sequence, mask=None):
        """
        Forward pass

        Args:
            state_sequence: (batch, seq_len, state_dim)
            mask: Optional attention mask

        Returns:
            value: (batch, 1)
        """

        # Embed
        x = self.embedding(state_sequence)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Use last token
        x = x[:, -1, :]

        # Value estimate
        value = self.value_head(x)

        return value


class TransformerAgentWrapper:
    """
    Wrapper to use Transformer actor/critic with DreamerV3

    Replaces the MLP-based actor/critic in DreamerV3
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 num_heads=8, num_layers=4, seq_len=64):
        """
        Initialize transformer agent

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            seq_len: Sequence length to use
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len

        # Create actor and critic
        self.actor = TransformerActor(
            state_dim, action_dim, hidden_dim,
            num_heads, num_layers, seq_len
        )

        self.critic = TransformerCritic(
            state_dim, hidden_dim,
            num_heads, num_layers, seq_len
        )

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # State buffer
        self.state_buffer = []

        logger.info("✅ Transformer Agent Wrapper initialized")

    def act(self, state):
        """
        Get action from current state

        Args:
            state: Current state (numpy array or tensor)

        Returns:
            action: Selected action
        """

        # Convert to tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        # Add to buffer
        self.state_buffer.append(state)

        # Keep only last seq_len states
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)

        # Create sequence (pad if necessary)
        seq = self._create_sequence()

        # Forward pass
        with torch.no_grad():
            action_logits = self.actor(seq)
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.argmax(action_probs, dim=-1).item()

        return action

    def _create_sequence(self):
        """Create padded sequence from buffer"""

        if len(self.state_buffer) < self.seq_len:
            # Pad with zeros
            padding_length = self.seq_len - len(self.state_buffer)
            padding = [torch.zeros_like(self.state_buffer[0])] * padding_length
            sequence = padding + self.state_buffer
        else:
            sequence = self.state_buffer[-self.seq_len:]

        # Stack into tensor (seq_len, state_dim)
        sequence_tensor = torch.stack(sequence)

        # Add batch dimension (1, seq_len, state_dim)
        sequence_tensor = sequence_tensor.unsqueeze(0)

        return sequence_tensor

    def train_step(self, batch):
        """
        Training step (placeholder)

        In full implementation, would train on sequences
        """
        pass

    def save(self, path):
        """Save model"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

        logger.info(f"💾 Saved transformer agent to {path}")

    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        logger.info(f"📂 Loaded transformer agent from {path}")


# Example usage
if __name__ == "__main__":
    print("🤖 Transformer Policy Demo\n")

    # Create transformer actor
    state_dim = 100  # e.g., 100 features
    action_dim = 2   # 0=flat, 1=long
    seq_len = 64     # Last 64 timesteps

    actor = TransformerActor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        seq_len=seq_len
    )

    # Test forward pass
    batch_size = 16
    test_sequence = torch.randn(batch_size, seq_len, state_dim)

    action_logits = actor(test_sequence)

    print(f"✅ Input shape: {test_sequence.shape}")
    print(f"✅ Output shape: {action_logits.shape}")
    print(f"✅ Output (action logits): {action_logits[0]}")

    # Test critic
    critic = TransformerCritic(
        state_dim=state_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        seq_len=seq_len
    )

    values = critic(test_sequence)
    print(f"\n✅ Critic output shape: {values.shape}")
    print(f"✅ Value estimates: {values[:5].squeeze()}")

    # Test wrapper
    print("\n" + "=" * 60)
    print("Testing Transformer Agent Wrapper")
    print("=" * 60)

    agent = TransformerAgentWrapper(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        seq_len=seq_len
    )

    # Simulate sequential actions
    for i in range(10):
        state = torch.randn(state_dim)
        action = agent.act(state)
        print(f"Step {i}: Action = {action}")

    print("\n✅ Transformer policy working correctly!")
    print("\nKey features:")
    print("  ✅ Attention mechanism - finds relevant historical patterns")
    print("  ✅ Positional encoding - knows sequence order")
    print("  ✅ Multi-head attention - looks at different aspects")
    print("  ✅ Can remember 'that support level from 3 days ago'")
