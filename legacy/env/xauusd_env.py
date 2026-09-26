# env/xauusd_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class XAUUSDTradingEnv(gym.Env):
    """
    Discrete long-only trading env.

    Actions:
      0 = Flat
      1 = Long

    Position is applied on the NEXT step to avoid look-ahead.
    Reward = pnl - trade_cost - turnover_penalty - flat_penalty + hold_bonus
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,             # (T, F)
        returns: np.ndarray,              # (T,)
        window: int = 64,
        cost_per_trade: float = 0.0001,   # cost per unit position change
        turnover_coef: float = 0.0002,     # extra penalty for changing position
        flat_penalty: float = 0.00002,    # tiny penalty for being flat (nudges to stay exposed)
        hold_bonus: float = 0.00002,      # tiny bonus for NOT changing position (nudges stability)
        max_episode_steps: int | None = None,
    ):
        super().__init__()
        assert features.ndim == 2
        assert returns.ndim == 1
        assert len(features) == len(returns)

        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)

        self.window = int(window)
        self.cost = float(cost_per_trade)
        self.turnover_coef = float(turnover_coef)
        self.flat_penalty = float(flat_penalty)
        self.hold_bonus = float(hold_bonus)

        self.T = len(self.r)
        self.max_episode_steps = max_episode_steps

        obs_dim = self.window * self.X.shape[1] + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Actions: 0=Flat, 1=Long
        self.action_space = spaces.Discrete(2)

        self._reset_state()

    def _reset_state(self):
        self.t = self.window
        self.pos = 0  # 0 or 1
        self.steps = 0
        self.equity = 1.0

    def _get_obs(self):
        w = self.X[self.t - self.window : self.t]  # (window, F)
        obs = np.concatenate([w.reshape(-1), np.array([self.pos], dtype=np.float32)])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        action = int(action)
        new_pos = 1 if action == 1 else 0

        # position change magnitude (0 or 1)
        delta = abs(new_pos - self.pos)

        # costs/penalties for changing position
        trade_cost = self.cost * delta
        turnover_penalty = self.turnover_coef * delta

        # pnl from holding PREVIOUS position over this bar
        pnl = self.pos * self.r[self.t]

        # penalize being flat (nudges agent to stay exposed in drift markets)
        flat_pen = self.flat_penalty if new_pos == 0 else 0.0

        # bonus for holding the same position (nudges stability, reduces flip-flop)
        hold_bonus = self.hold_bonus if delta == 0 else 0.0

        reward = pnl - trade_cost - turnover_penalty - flat_pen + hold_bonus

        # track equity
        self.equity *= (1.0 + reward)

        # update position after reward (avoid look-ahead)
        self.pos = new_pos

        # advance time
        self.t += 1
        self.steps += 1

        terminated = self.t >= self.T
        truncated = False
        if self.max_episode_steps is not None and self.steps >= self.max_episode_steps:
            truncated = True

        info = {
            "equity": float(self.equity),
            "pos": int(self.pos),
            "trade_cost": float(trade_cost),
        }
        return self._get_obs(), float(reward), terminated, truncated, info
