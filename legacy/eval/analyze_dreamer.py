"""
Analyze DreamerV3 World Model

This script helps you understand what the World Model learned:
1. Reconstruction quality (can it predict the next candle?)
2. Latent space visualization (what market regimes did it discover?)
3. Reward prediction accuracy
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.make_features import make_features
from models.dreamer_agent import DreamerV3Agent


def analyze_reconstruction(agent, env, num_steps=1000):
    """
    Test how well the world model reconstructs observations

    Good reconstruction = World model understands the market
    """
    print("\n" + "="*60)
    print("ANALYSIS 1: Observation Reconstruction")
    print("="*60)

    obs = env.reset()
    h, z = agent.rssm.initial_state(1, agent.device)

    reconstruction_errors = []

    for _ in tqdm(range(num_steps), desc="Testing reconstruction"):
        # Get action
        action, (h, z) = agent.act(obs, h, z, deterministic=True)

        # Step environment
        next_obs, reward, done, info = env.step(action)

        if done:
            break

        # Encode observation
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        embed = agent.encoder(obs_t)

        # Get current state
        state = agent.rssm.get_state(h, z)

        # Reconstruct observation
        with torch.no_grad():
            obs_recon = agent.decoder(state).cpu().numpy()[0]

        # Compute error
        error = np.mean((obs - obs_recon) ** 2)
        reconstruction_errors.append(error)

        obs = next_obs

    avg_error = np.mean(reconstruction_errors)
    print(f"\n📊 Average Reconstruction Error: {avg_error:.6f}")
    print(f"   (Lower is better - should be < 0.1 for good world model)")

    return reconstruction_errors


def analyze_reward_prediction(agent, env, num_steps=1000):
    """
    Test how well the world model predicts rewards

    Good prediction = Can anticipate profitable scenarios
    """
    print("\n" + "="*60)
    print("ANALYSIS 2: Reward Prediction Accuracy")
    print("="*60)

    obs = env.reset()
    h, z = agent.rssm.initial_state(1, agent.device)

    true_rewards = []
    predicted_rewards = []

    for _ in tqdm(range(num_steps), desc="Testing reward prediction"):
        # Get state
        state = agent.rssm.get_state(h, z)

        # Predict reward
        with torch.no_grad():
            from models.dreamer_components import symexp
            reward_pred = symexp(agent.reward_predictor(state)).cpu().numpy()[0]

        # Take action and observe true reward
        action, (h, z) = agent.act(obs, h, z, deterministic=True)
        next_obs, reward, done, info = env.step(action)

        true_rewards.append(reward)
        predicted_rewards.append(reward_pred)

        if done:
            break

        obs = next_obs

    true_rewards = np.array(true_rewards)
    predicted_rewards = np.array(predicted_rewards)

    # Compute correlation
    correlation = np.corrcoef(true_rewards, predicted_rewards)[0, 1]

    print(f"\n📊 Reward Prediction Correlation: {correlation:.4f}")
    print(f"   (1.0 = perfect, >0.5 = good, <0 = broken)")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(true_rewards[:100], label='True Rewards', alpha=0.7)
    plt.plot(predicted_rewards[:100], label='Predicted Rewards', alpha=0.7)
    plt.legend()
    plt.title('Reward Prediction (First 100 Steps)')
    plt.xlabel('Step')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.scatter(true_rewards, predicted_rewards, alpha=0.3)
    plt.plot([true_rewards.min(), true_rewards.max()],
             [true_rewards.min(), true_rewards.max()],
             'r--', label='Perfect Prediction')
    plt.xlabel('True Reward')
    plt.ylabel('Predicted Reward')
    plt.title(f'Correlation: {correlation:.3f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig('eval/dreamer_reward_prediction.png', dpi=150)
    print(f"\n📊 Plot saved: eval/dreamer_reward_prediction.png")

    return correlation


def analyze_latent_space(agent, env, num_steps=1000):
    """
    Analyze the latent space to understand what market regimes the model discovered

    The stochastic state (z) should cluster into different market regimes
    """
    print("\n" + "="*60)
    print("ANALYSIS 3: Latent Space (Market Regimes)")
    print("="*60)

    obs = env.reset()
    h, z = agent.rssm.initial_state(1, agent.device)

    latent_states = []
    rewards = []
    positions = []

    for _ in tqdm(range(num_steps), desc="Collecting latent states"):
        # Store current latent state
        latent_states.append(z.cpu().numpy()[0])

        # Take action
        action, (h, z) = agent.act(obs, h, z, deterministic=True)
        next_obs, reward, done, info = env.step(action)

        rewards.append(reward)
        positions.append(info['pos'])

        if done:
            break

        obs = next_obs

    latent_states = np.array(latent_states)
    rewards = np.array(rewards)
    positions = np.array(positions)

    print(f"\n📊 Latent State Statistics:")
    print(f"   Shape: {latent_states.shape}")
    print(f"   Mean: {latent_states.mean():.4f}")
    print(f"   Std: {latent_states.std():.4f}")

    # PCA for visualization
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_states)

    print(f"   PCA Explained Variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                         c=positions, cmap='coolwarm', alpha=0.5)
    plt.colorbar(scatter, label='Position (0=flat, 1=long)')
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.title('Latent Space colored by Position')

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                         c=rewards, cmap='RdYlGn', alpha=0.5, vmin=-0.001, vmax=0.001)
    plt.colorbar(scatter, label='Reward')
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.title('Latent Space colored by Reward')

    plt.tight_layout()
    plt.savefig('eval/dreamer_latent_space.png', dpi=150)
    print(f"\n📊 Plot saved: eval/dreamer_latent_space.png")


def compare_with_random(agent, env, num_episodes=10):
    """
    Compare DreamerV3 performance with random agent

    This validates that the agent learned something useful
    """
    print("\n" + "="*60)
    print("ANALYSIS 4: DreamerV3 vs Random Agent")
    print("="*60)

    def run_episode(use_agent=True):
        obs = env.reset()
        h, z = None, None
        total_reward = 0
        steps = 0

        while True:
            if use_agent:
                action, (h, z) = agent.act(obs, h, z, deterministic=True)
            else:
                # Random action
                action = np.zeros(2, dtype=np.float32)
                action[np.random.randint(0, 2)] = 1.0

            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                return total_reward, info['equity'], steps

    # DreamerV3 performance
    dreamer_rewards = []
    dreamer_equities = []
    for _ in tqdm(range(num_episodes), desc="DreamerV3"):
        total_reward, equity, steps = run_episode(use_agent=True)
        dreamer_rewards.append(total_reward)
        dreamer_equities.append(equity)

    # Random performance
    random_rewards = []
    random_equities = []
    for _ in tqdm(range(num_episodes), desc="Random"):
        total_reward, equity, steps = run_episode(use_agent=False)
        random_rewards.append(total_reward)
        random_equities.append(equity)

    print(f"\n📊 Performance Comparison:")
    print(f"   DreamerV3:")
    print(f"      Avg Equity: {np.mean(dreamer_equities):.4f} ± {np.std(dreamer_equities):.4f}")
    print(f"      Avg Return: {(np.mean(dreamer_equities) - 1) * 100:.2f}%")
    print(f"   Random:")
    print(f"      Avg Equity: {np.mean(random_equities):.4f} ± {np.std(random_equities):.4f}")
    print(f"      Avg Return: {(np.mean(random_equities) - 1) * 100:.2f}%")

    improvement = (np.mean(dreamer_equities) - np.mean(random_equities)) / np.mean(random_equities) * 100
    print(f"\n   📈 Improvement over Random: {improvement:+.2f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='train/dreamer/dreamer_xauusd_final.pt')
    parser.add_argument('--num_steps', type=int, default=1000)
    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print(f"   Train the model first: python train/train_dreamer.py")
        return

    # Load data
    print("Loading data...")
    if os.path.exists("data/xauusd_1h_macro.csv"):
        from features.make_features import make_features
        df, X, r = make_features("data/xauusd_1h_macro.csv", window=64)
    else:
        print("❌ Macro data not found. Run: python data/merge_macro.py")
        return

    # Use test set
    train_end = np.searchsorted(df["time"].to_numpy(), np.datetime64("2022-01-01"))
    X_test, r_test = X[train_end:], r[train_end:]

    # Create environment
    from train.train_dreamer import TradingEnvironment
    env = TradingEnvironment(X_test, r_test, window=64, cost_per_trade=0.0001)

    # Load agent
    print(f"Loading checkpoint: {args.checkpoint}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    obs_dim = env._get_obs().shape[0]

    agent = DreamerV3Agent(obs_dim=obs_dim, action_dim=2, device=device)
    agent.load(args.checkpoint)

    # Run analyses
    try:
        from sklearn.decomposition import PCA
        has_sklearn = True
    except ImportError:
        print("\n⚠️  sklearn not found. Skipping latent space analysis.")
        print("   Install: pip install scikit-learn matplotlib")
        has_sklearn = False

    # Analysis 1: Reconstruction
    recon_errors = analyze_reconstruction(agent, env, args.num_steps)

    # Analysis 2: Reward Prediction
    try:
        reward_corr = analyze_reward_prediction(agent, env, args.num_steps)
    except ImportError:
        print("\n⚠️  matplotlib not found. Skipping reward prediction plot.")

    # Analysis 3: Latent Space
    if has_sklearn:
        try:
            analyze_latent_space(agent, env, args.num_steps)
        except Exception as e:
            print(f"\n⚠️  Error in latent space analysis: {e}")

    # Analysis 4: Compare with random
    # compare_with_random(agent, env, num_episodes=5)

    print("\n" + "="*60)
    print("✅ Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
