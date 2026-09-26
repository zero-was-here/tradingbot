import numpy as np
import gymnasium as gym
from gymnasium import spaces


class XAUUSDTradingEnvAggressive(gym.Env):
    """
    "Smart Aggressive" trading env.
    
    Philosophy:
    - No "participation trophies" (no flat penalty).
    - No "fake math" (no profit multiplier).
    - Realism: You pay the spread/commissions.
    
    This forces the agent to only take trades where the expected return > cost.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,             # (T, F)
        returns: np.ndarray,              # (T,)
        window: int = 128,                # WIDER WINDOW: 2 Hours of vision
        cost_per_trade: float = 0.0002,   # 2bps
        turnover_coef: float = 0.0,       
        flat_penalty: float = 0.0,        
        hold_bonus: float = 0.0,          
        leverage: float = 1.0,            
        stop_loss_pct: float = 0.001,     
        max_episode_steps: int | None = None,
        **kwargs
    ):
        super().__init__()
        
        # Input validation
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
        self.leverage = float(leverage)
        self.stop_loss_pct = float(stop_loss_pct)

        self.T = len(self.r)
        self.max_episode_steps = max_episode_steps

        # Observation: window features + current position (-1, 0, or 1)
        obs_dim = self.window * self.X.shape[1] + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Actions: 0=Short, 1=Flat, 2=Long
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    def _reset_state(self):
        self.t = self.window
        self.pos = 0  # -1 (short), 0 (flat), or 1 (long)
        self.entry_price = 1.0 # Virtual entry price
        self.steps = 0
        self.equity = 1.0

    def _get_obs(self):
        # Slice window
        w = self.X[self.t - self.window : self.t]
        
        # Flatten and append position
        obs = np.concatenate([w.reshape(-1), np.array([self.pos], dtype=np.float32)])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        # 1. Map action (0,1,2) -> position (-1, 0, 1)
        new_pos = int(action) - 1

        # 2. Calculate costs
        delta = abs(new_pos - self.pos)
        trade_cost = self.cost * delta
        turnover_pen = self.turnover_coef * delta

        # 3. Calculate PnL
        # r[t] is log return (~pct change). 
        # approximate price change = r[t]
        raw_pnl = self.pos * self.r[self.t]
        pnl = raw_pnl * self.leverage

        # 4. Check Stop Loss
        # If we are in a position, check if the single bar return killed us
        # (Simplified SL: if bar return < -SL, we die)
        sl_penalty = 0.0
        truncated = False
        
        if self.pos != 0:
            # If long and price dropped > SL
            if self.pos == 1 and self.r[self.t] < -self.stop_loss_pct:
                sl_penalty = -0.05 # Heavy penalty for hitting SL
                new_pos = 0 # Force Close
                truncated = True # End episode (Game Over logic teaches safety)

            # If short and price rose > SL
            elif self.pos == -1 and self.r[self.t] > self.stop_loss_pct:
                sl_penalty = -0.05
                new_pos = 0
                truncated = True

        # 5. Auxiliary penalties/bonuses
        flat_pen = self.flat_penalty if new_pos == 0 else 0.0
        hold_bon = self.hold_bonus if delta == 0 else 0.0
        
        # 6. Total Reward
        reward = pnl - trade_cost - turnover_pen - flat_pen + hold_bon + sl_penalty

        # 7. Update State
        self.equity *= (1.0 + reward)
        self.pos = new_pos
        self.t += 1
        self.steps += 1

        terminated = self.t >= self.T
        if self.max_episode_steps is not None and self.steps >= self.max_episode_steps:
            truncated = True

        info = {
            "equity": self.equity,
            "pos": int(self.pos),
            "trade_cost": trade_cost,
            "pnl": raw_pnl
        }

        return self._get_obs(), float(reward), terminated, truncated, info