# 🧠 DreamerV3 Implementation Guide - The World Model Revolution

## 🎯 What We Built

You now have a **Model-Based Reinforcement Learning** system for trading - the foundation of "God Mode" trading AI. Unlike PPO (which learns from trial and error), DreamerV3 learns a **World Model** of the market and uses it to imagine future scenarios before making decisions.

### Key Difference: PPO vs DreamerV3

**PPO (Model-Free RL):**
- Learns: "If I see X, do Y"
- Limitation: Must experience every situation to learn
- Training: Requires millions of real trades

**DreamerV3 (Model-Based RL):**
- Learns: "The market works like THIS (world model)"
- Capability: Can simulate 10,000 trades in its head in 1 second
- Training: Learns from fewer samples by dreaming/planning

## 🏗 Architecture Overview

### The 6 Core Components

1. **Encoder** - Compresses observations (price, volume, indicators) into compact embeddings
2. **RSSM (World Model)** - Learns the "physics" of the market
   - `h_t` (deterministic state): Memory of past events
   - `z_t` (stochastic state): Current market regime (trending, ranging, volatile, etc.)
3. **Decoder** - Reconstructs observations from latent state (checks if world model is accurate)
4. **Reward Predictor** - Predicts future rewards without executing trades
5. **Actor** - Policy that selects actions (flat, long, short)
6. **Critic** - Estimates value of states to guide policy improvement

### How It Works

```
Real Market
    ↓
Encoder → Embed
    ↓
RSSM → (h_t, z_t)  ← World Model learns market dynamics
    ↓
┌─────────────────────────────────┐
│ Imagination Phase               │
│ Actor imagines 15 steps ahead   │
│ using the learned World Model   │
└─────────────────────────────────┘
    ↓
Action Selection (flat, long, short)
```

## 🚀 Training Your DreamerV3 Agent

### Step 1: Ensure Dependencies

```bash
pip install torch>=2.0.0 tqdm gymnasium
```

### Step 2: Run Training

```bash
python train/train_dreamer.py
```

**What happens during training:**

1. **Phase 1: Prefill (5,000 steps)**
   - Random exploration to fill replay buffer
   - Gathers diverse experiences

2. **Phase 2: Training (100,000 steps)**
   - **World Model Learning**: The agent learns to predict what happens next
   - **Imagination Training**: The agent imagines trajectories and improves policy
   - Checkpoints saved every 10,000 steps

3. **Phase 3: Evaluation**
   - Tests on unseen data (post-2022)
   - Reports final equity and statistics

### Training Progress

You'll see output like:
```
Step 1000:
  World Model Loss: 0.4523
  - Recon: 0.2100  ← How well it reconstructs observations
  - Reward: 0.1200  ← How well it predicts rewards
  - KL: 0.1223      ← Regularization (prevents overfitting)
  Value Loss: 0.0312
  Policy Loss: -0.0145
```

**Good signs:**
- World Model Loss decreasing (model learning market dynamics)
- KL Loss stable around 1.0-2.0 (not collapsing)
- Value/Policy Loss converging

## 📊 Using the Trained Model

### Load and Evaluate

```python
from models.dreamer_agent import DreamerV3Agent

# Create agent
agent = DreamerV3Agent(obs_dim=704, action_dim=2, device='cpu')

# Load checkpoint
agent.load("train/dreamer/dreamer_xauusd_final.pt")

# Use for trading
obs = env.reset()
h, z = None, None

while True:
    action, (h, z) = agent.act(obs, h, z, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

## 🔥 Key Features Implemented

### 1. Symlog Transformation
- Handles extreme price movements without exploding gradients
- Used in decoder, reward predictor, and critic

### 2. Categorical Latent Variables
- Instead of continuous Gaussians, uses 32 categorical distributions
- More stable and expressive for discrete market regimes

### 3. KL Balancing
- Prevents posterior collapse (common issue in VAEs)
- Uses "free nats" to allow some KL divergence

### 4. Lambda Returns
- Combines TD-learning with Monte Carlo for better value estimation
- Controlled by `lambda_` parameter (0.95)

### 5. Imagination Horizon
- Agent imagines 15 steps ahead before selecting action
- Longer horizon = more planning, slower execution

## ⚙️ Hyperparameter Tuning

### Critical Parameters

```python
# Architecture
embed_dim=256          # Embedding size (higher = more capacity, slower)
hidden_dim=512         # RSSM hidden state size
stoch_dim=32           # Number of categorical variables
num_categories=32      # Categories per variable (32×32 = 1024 discrete states)

# Learning Rates
lr_world_model=3e-4    # World model learns faster
lr_actor=1e-4          # Actor learns slower (stability)
lr_critic=3e-4         # Critic matches world model

# Planning
horizon=15             # Imagination steps (15-50 typical)
gamma=0.99             # Discount factor
lambda_=0.95           # GAE parameter

# Regularization
free_nats=1.0          # KL free bits
kl_balance=0.8         # KL balancing coefficient
```

### When to Adjust

- **Model underfitting** (poor performance):
  - Increase `embed_dim`, `hidden_dim`, `stoch_dim`
  - Decrease `free_nats` (less regularization)

- **Model overfitting** (good train, bad test):
  - Increase `free_nats` (more regularization)
  - Add dropout to networks

- **Unstable training** (loss exploding):
  - Decrease learning rates
  - Increase gradient clipping (currently 100.0)

- **Poor long-term planning**:
  - Increase `horizon`
  - Adjust `gamma` (higher = values future more)

## 🎮 Next Steps: MCTS Integration (Phase 3)

To achieve true "Stockfish for Markets", integrate **Monte Carlo Tree Search**:

```
Before each trade:
1. Current state: (h, z)
2. Expand tree: Try 3 actions (flat, long, short)
3. For each action:
   - Simulate 100 trajectories using World Model
   - Evaluate average return
4. Select action with highest expected return
```

**Implementation outline:**
```python
class MCTS:
    def search(self, agent, h, z, num_simulations=100):
        # For each possible action
        for action in [flat, long, short]:
            returns = []
            for _ in range(num_simulations):
                # Imagine trajectory
                h_sim, z_sim = h.clone(), z.clone()
                total_reward = 0

                for t in range(horizon):
                    h_sim, z_sim, _ = agent.rssm.imagine(action, h_sim, z_sim)
                    state = agent.rssm.get_state(h_sim, z_sim)
                    reward = agent.reward_predictor(state)
                    total_reward += reward

                returns.append(total_reward)

            # Average return for this action
            action_values[action] = mean(returns)

        # Select best action
        return argmax(action_values)
```

## 📈 Expected Performance

### Training Time
- MacBook (CPU): ~2-4 hours for 100k steps
- With GPU: ~30-60 minutes

### Sample Efficiency
- DreamerV3: Needs ~50k-100k steps to converge
- PPO: Needs ~500k-1M steps for same performance

### Memory Requirements
- Model size: ~10-20 MB
- Replay buffer: ~500 MB (100k transitions)

## 🐛 Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` (16 → 8)
- Reduce `hidden_dim` or `embed_dim`
- Use CPU: `device='cpu'`

### "Loss is NaN"
- Check for inf/nan in data (run validation)
- Reduce learning rates
- Increase gradient clipping

### "KL loss too high/low"
- High (>5.0): Increase `free_nats`
- Low (<0.1): Posterior collapse, decrease `free_nats`

### "Agent only takes one action"
- Increase exploration during prefill
- Check action space is correct
- Verify reward signal isn't too sparse

## 📚 References & Sources

This implementation is based on:

1. **DreamerV3 Paper**: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
2. **Official Implementation**: [danijar/dreamerv3](https://github.com/danijar/dreamerv3)
3. **PyTorch Implementations**:
   - [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)
   - [burchim/DreamerV3-PyTorch](https://github.com/burchim/DreamerV3-PyTorch)

## 🏆 What Makes This "God Mode"

Traditional trading bots react to what they see. **This bot simulates futures.**

```
Traditional Bot:        DreamerV3 Bot:
"Price is up"    →      "I will imagine 100 scenarios:"
"Buy"                   Scenario 1: Buy → Price drops → Loss
                        Scenario 2: Wait → Flag forms → Buy → Profit ✓
                        Scenario 3: Short → Trend continues → Loss

                        Decision: Wait (Scenario 2 wins)
```

By learning the physics of the market (world model), the agent can:
- Simulate trades without risk
- Plan multiple steps ahead
- Detect patterns that emerge over time
- Avoid traps it's seen in simulation

**This is the foundation of superhuman trading.**

---

## 🎯 Current Status

✅ **Phase 1 Complete**: DreamerV3 World Model + Basic Policy
- World model learns market dynamics
- Actor-critic trained via imagination
- Macro data integration ready (DXY, SPX, US10Y)

🔄 **Phase 2 (In Progress)**: Data Nexus
- [x] Macro features (DXY, SPX, US10Y)
- [ ] Economic calendar integration
- [ ] Volatility regime detection

⏳ **Phase 3 (Pending)**: MCTS Integration
- [ ] Implement tree search
- [ ] Integrate with world model
- [ ] 500ms "thinking time" before trades

⏳ **Phase 4 (Future)**: Adversarial Training
- [ ] Train "Market Maker" agent
- [ ] Self-play training loop
- [ ] Trap detection system

---

**"We do not predict price. We simulate the future and pick the timeline where we win."**
