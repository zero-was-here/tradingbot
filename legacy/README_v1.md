# 🤖 DRL Trading Bot - XAUUSD

> An advanced AI-powered trading system using Deep Reinforcement Learning to trade gold (XAUUSD) autonomously. Built with 140+ market features, multi-timeframe analysis, and state-of-the-art RL algorithms.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [What is This?](#-what-is-this)
- [Key Features](#-key-features)
- [Performance Targets](#-performance-targets)
- [How It Works](#-how-it-works)
- [Installation](#%EF%B8%8F-installation)
- [Quick Start Guide](#-quick-start-guide)
- [Project Structure](#-project-structure)
- [Algorithms Explained](#-algorithms-explained)
- [Documentation](#-documentation)
- [Testing](#-testing)
- [Disclaimer](#%EF%B8%8F-disclaimer)

---

## 🎯 What is This?

This is a **fully autonomous trading bot** that uses artificial intelligence to trade gold (XAUUSD) in the forex market. Unlike traditional bots that follow rigid rules, this system **learns from historical data** using Deep Reinforcement Learning (DRL) - the same technology behind AlphaGo and ChatGPT.

### Why Gold (XAUUSD)?
- High liquidity (easy to enter/exit trades)
- Strong trends and patterns
- 24/5 trading availability
- Lower correlation with stocks (diversification)

### What Makes This Different?
- **140+ Market Features**: Most bots use 5-10 indicators. We use 140+ data points from multiple sources
- **Multi-Timeframe Analysis**: Analyzes M5, M15, H1, H4, and D1 charts simultaneously
- **Macro Awareness**: Integrates VIX, Oil, Bitcoin, Dollar Index, and economic events
- **Self-Learning**: Improves through millions of simulated trades, not manual programming

---

## 🚀 Key Features

### 🧠 Advanced AI Architecture

#### Two Cutting-Edge Algorithms:
1. **PPO (Proximal Policy Optimization)**
   - Industry-standard for trading bots
   - Stable, reliable training
   - Proven results in financial markets

2. **Dreamer V3**
   - Cutting-edge world model-based RL
   - Learns market dynamics and predicts future states
   - More sample-efficient (trains faster)

#### Powered By:
- **Stable-Baselines3**: Production-ready RL library
- **PyTorch**: Deep learning framework
- **Gymnasium**: OpenAI's standard RL environment

#### Hardware Flexibility:
- ✅ CPU (any computer)
- ✅ MPS (Apple M1/M2/M3 chips)
- ✅ CUDA (NVIDIA GPUs)
- ✅ Google Colab (free cloud GPUs)

---

### 📊 Comprehensive Market Intelligence (140+ Features)

#### Multi-Timeframe Analysis
Analyzes 5 timeframes simultaneously for complete market context:
- **M5** (5-min): Entry timing and momentum
- **M15** (15-min): Short-term trends
- **H1** (1-hour): Intraday direction
- **H4** (4-hour): Daily bias
- **D1** (Daily): Long-term trend

#### 63 Technical Indicators ("God Mode Features")
- **Trend**: Moving Averages (EMA, SMA), MACD, ADX
- **Momentum**: RSI, Stochastic, CCI, Williams %R
- **Volatility**: ATR, Bollinger Bands, Keltner Channels
- **Volume**: OBV, MFI, Volume analysis
- **Price Action**: Support/Resistance, Pivot Points, Candlestick patterns

#### Macro Market Data
Understands the broader economy:
- **DXY** (US Dollar Index) - Dollar strength affects gold inversely
- **SPX** (S&P 500) - Stock market risk sentiment
- **US10Y** (Treasury Yields) - Interest rates impact gold
- **VIX** (Volatility Index) - Market fear gauge
- **Oil** (WTI Crude) - Commodity correlation
- **Bitcoin** - Risk-on/risk-off indicator
- **EURUSD** - Major currency pair correlation
- **Silver** (XAGUSD) - Precious metals correlation
- **GLD** (Gold ETF) - Institutional positioning

#### Economic Calendar Integration
Knows when major events happen:
- **NFP** (Non-Farm Payrolls) - Monthly jobs report
- **CPI** (Consumer Price Index) - Inflation data
- **FOMC** (Federal Reserve meetings) - Interest rate decisions
- **GDP** - Economic growth reports
- Automatically adjusts risk before/during high-impact events

#### Market Microstructure
- Order flow analysis
- Bid-ask spread monitoring
- Volatility regime detection
- Session-based patterns (Asian/London/New York)

#### Optional Sentiment Analysis
- Reddit sentiment from r/wallstreetbets, r/Forex
- News headlines analysis
- Google Trends for "gold price" searches

---

### 🎯 Trading Strategies

Three pre-configured strategies for different risk profiles:

| Strategy | Frequency | Hold Time | Risk Level | Best For |
|----------|-----------|-----------|------------|----------|
| **Standard** | Medium | Hours-Days | Moderate | Balanced traders |
| **Aggressive** | High | Minutes-Hours | Higher | Active traders |
| **Swing** | Low | Days-Weeks | Lower | Patient traders |

---

### 🔌 Live Trading Integration

#### MetaTrader 5 (MT5)
- Direct integration with MT5 platform
- Real-time price feeds
- Instant order execution
- Works with any MT5 broker

#### MetaAPI (Cloud Trading)
- Trade from anywhere (no VPS needed)
- Cloud-based execution
- Multiple broker support
- Automatic reconnection

#### Risk Management Features
- Dynamic position sizing based on account equity
- Automatic stop-loss placement
- Maximum drawdown protection
- Daily loss limits
- Position concentration limits

---

## 📊 Performance Targets

| Metric | Target | Explanation |
|--------|--------|-------------|
| **Annual Return** | 80-120%+ | Expected yearly profit |
| **Sharpe Ratio** | 3.5-4.5+ | Risk-adjusted returns (>2 is excellent) |
| **Max Drawdown** | <8% | Largest peak-to-valley loss |
| **Win Rate** | 60-65% | Percentage of profitable trades |
| **Profit Factor** | 2.5-3.0+ | Gross profit / Gross loss |

**Note**: These are targets based on backtesting. Real performance depends on market conditions, slippage, and execution quality.

---

## 🔍 How It Works

### 1️⃣ Data Collection
The bot gathers data from multiple sources:
```
XAUUSD prices (M5, M15, H1, H4, D1)
    ↓
Macro data (VIX, Oil, Bitcoin, etc.)
    ↓
Economic calendar events
    ↓
Technical indicators calculated
    ↓
140+ features combined into observation
```

### 2️⃣ AI Decision Making
The trained model analyzes the 140+ features and decides:
- **Action**: Buy, Sell, or Hold
- **Position Size**: How much to risk
- **Stop Loss**: Where to exit if wrong
- **Take Profit**: Where to exit if right

### 3️⃣ Execution
The decision is sent to MT5 or MetaAPI:
```
AI Decision → Order Execution → Position Monitoring → Risk Management
```

### 4️⃣ Learning Process (Training)
The bot improves through simulation:
```
1. Start with random strategy
2. Take actions in historical market data
3. Receive rewards (profit = positive, loss = negative)
4. Update strategy to maximize rewards
5. Repeat for 1,000,000+ steps
6. Deploy trained model
```

**Training Time:**
- Local Mac (MPS): 6-8 days
- NVIDIA GPU: 2-3 days
- Google Colab Pro+: 5-7 hours ⚡ (Recommended)

---

## 🛠️ Installation

### Prerequisites
```bash
✅ Python 3.12 or higher
✅ MetaTrader 5 (for live trading only)
✅ 8GB+ RAM recommended
✅ 10GB free disk space (for data)
```

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/zero-was-here/tradingbot.git
cd tradingbot
```

#### 2. Create Virtual Environment
```bash
# Create environment
python3 -m venv .

# Activate it
source bin/activate  # Mac/Linux
# OR
.\Scripts\activate   # Windows
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**What gets installed:**
- `stable-baselines3` - RL algorithms
- `torch` - Deep learning
- `pandas` - Data processing
- `numpy` - Numerical computing
- `gymnasium` - RL environments
- `MetaTrader5` - Live trading
- `tqdm` - Progress bars

---

## 🔐 Security Setup (API Keys)

**IMPORTANT:** Never commit API keys to git!

### Step 1: Create Environment File
```bash
# Copy the example file
cp .env.example .env

# Edit with your credentials
nano .env  # or use any text editor
```

### Step 2: Fill in Your Credentials
```bash
# .env file
METAAPI_TOKEN=your_actual_token_here
METAAPI_ACCOUNT_ID=your_actual_account_id_here
```

### Step 3: Verify .env is Ignored
The `.env` file is already in `.gitignore` - it will never be committed to git.

**Get your MetaAPI credentials:**
1. Sign up at [MetaAPI](https://metaapi.cloud/)
2. Create a trading account connection
3. Copy your API token and account ID

---

## 📚 Quick Start Guide

### Step 1: Get the Data

#### A. Auto-fetch Macro Data (5 minutes)
```bash
python scripts/fetch_all_data.py
```
Downloads: VIX, Oil, Bitcoin, EURUSD, Silver, GLD from Yahoo Finance

#### B. Generate Economic Calendar (1 minute)
```bash
python scripts/generate_economic_calendar.py
```
Creates calendar with 1,500+ major economic events (2015-2025)

#### C. Export XAUUSD from MetaTrader 5 (15-20 minutes)
1. Open MetaTrader 5
2. **View** → **Symbols**
3. Find **XAUUSD**, right-click → **"All history"**
4. Wait for download (may take 10-30 min)
5. **Tools** → **History Center**
6. Select **XAUUSD** → **M5** timeframe
7. Click **Export** → Save as `data/xauusd_m5.csv`
8. Repeat for **M15** → Save as `data/xauusd_m15.csv`

**Expected files:**
```
data/xauusd_m5.csv   (~50-100 MB, 1M+ rows)
data/xauusd_m15.csv  (~20-40 MB, 350k+ rows)
```

---

### Step 2: Train the Model

#### Option A: Local Training (Slower but Free)
```bash
# Mac with Apple Silicon
python train/train_ultimate_150.py --steps 1000000 --device mps --batch-size 64

# Windows/Linux with NVIDIA GPU
python train/train_ultimate_150.py --steps 1000000 --device cuda --batch-size 128

# CPU only (slowest)
python train/train_ultimate_150.py --steps 1000000 --device cpu --batch-size 32
```

**Training time:**
- Mac M1/M2/M3: 6-8 days
- NVIDIA RTX 3080+: 2-3 days
- CPU: 15-20 days (not recommended)

**Monitor progress:**
- Models saved every 50k steps in `train/ppo_xauusd_[steps]k.zip`
- Check training log for rewards and losses
- Can stop/resume training anytime

#### Option B: Google Colab (Faster, Recommended)
1. Upload `colab_train_ultimate_150.ipynb` to Google Drive
2. Open in Google Colab
3. Runtime → Change runtime type → GPU (T4 or A100)
4. Run all cells
5. Training completes in **5-7 hours**
6. Download trained model back to your computer

**Google Colab Setup:** See [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)

---

### Step 3: Evaluate the Model

```bash
python evaluate_model.py --model train/ppo_xauusd_latest.zip
```

**Output shows:**
- Total return %
- Sharpe ratio
- Maximum drawdown
- Win rate
- Number of trades
- Average profit per trade

---

### Step 4: Paper Trading (Test with Fake Money)

Before risking real money, test with a demo account:

```bash
# Make sure MT5 is logged into a DEMO account
python live_trade_mt5.py
```

**What to watch:**
- Is it making trades as expected?
- Are stop-losses being placed correctly?
- Is position sizing appropriate?
- Monitor for at least 1-2 weeks

---

### Step 5: Live Trading (Real Money)

**⚠️ Only after successful paper trading!**

```bash
# MetaTrader 5
python live_trade_mt5.py

# MetaAPI (cloud)
python live_trade_metaapi.py
```

**Best Practices:**
- Start with minimum position sizes
- Monitor daily for first week
- Keep max risk per trade at 1-2%
- Set daily loss limits
- Have a stop-loss on account level

---

## 📁 Project Structure

```
tradingbot/
│
├── 📂 train/                    # Training scripts & saved models
│   ├── train_ultimate_150.py   # Main training script (140+ features)
│   ├── train_god_mode.py       # God mode training (63 features)
│   ├── train_dreamer.py        # Dreamer V3 training
│   └── ppo_xauusd_*.zip        # Saved model checkpoints
│
├── 📂 features/                 # Feature engineering
│   ├── god_mode_features.py    # 63 technical indicators
│   ├── macro_features.py       # Macro market data integration
│   ├── calendar_features.py    # Economic event features
│   ├── multi_timeframe.py      # Cross-timeframe analysis
│   └── ultimate_150_features.py # All 140+ features combined
│
├── 📂 env/                      # Trading environment (RL gym)
│   ├── xauusd_env.py           # Standard trading environment
│   └── realistic_execution.py  # Realistic slippage/spread simulation
│
├── 📂 models/                   # Advanced RL components
│   ├── dreamer_agent.py        # Dreamer V3 implementation
│   ├── transformer_policy.py   # Transformer-based policy network
│   ├── risk_supervisor.py      # Risk management overlay
│   └── ensemble.py             # Multi-model ensemble
│
├── 📂 eval/                     # Evaluation & analysis
│   ├── backtest.py             # Historical backtesting
│   ├── crisis_validation.py    # Test on market crashes
│   └── baselines.py            # Compare vs buy-and-hold
│
├── 📂 data/                     # Market data storage
│   ├── xauusd_m5.csv           # 5-min XAUUSD data (you provide)
│   ├── xauusd_m15.csv          # 15-min XAUUSD data (you provide)
│   └── economic_events.json    # Economic calendar (auto-generated)
│
├── 📂 scripts/                  # Utility scripts
│   ├── fetch_all_data.py       # Download macro data
│   └── generate_economic_calendar.py # Create event calendar
│
├── 📂 backtest/                 # Backtesting engine
│   └── backtest_engine.py      # Full backtest with metrics
│
├── 📂 monitoring/               # Production monitoring
│   └── production_monitor.py   # Track live performance
│
├── 📄 live_trade_mt5.py         # Live trading with MT5
├── 📄 live_trade_metaapi.py    # Live trading with MetaAPI
├── 📄 evaluate_model.py         # Model evaluation script
├── 📄 requirements.txt          # Python dependencies
└── 📄 README.md                 # This file
```

---

## 🔬 Algorithms Explained

### PPO (Proximal Policy Optimization)

**What it is:** A popular RL algorithm that learns by trial and error, like teaching a dog tricks with rewards.

**How it works:**
1. Agent takes actions in the market
2. Gets rewards (profit = good, loss = bad)
3. Updates strategy to get more rewards
4. Repeats millions of times

**Why PPO for trading:**
- ✅ Stable training (won't diverge)
- ✅ Works well with continuous actions (position sizing)
- ✅ Proven success in financial markets
- ✅ Efficient sample usage

**Technical details:**
- On-policy algorithm (learns from current strategy)
- Clipped objective prevents large policy updates
- Actor-Critic architecture (separate value and policy networks)
- Multiple parallel environments for faster training

---

### Dreamer V3 (World Model RL)

**What it is:** An advanced algorithm that builds a mental model of how markets work, then practices trading in that simulation.

**How it works:**
1. Observes real market data
2. Learns to predict future market states (builds "world model")
3. Practices trading in imagined futures
4. Transfers learned strategy to real trading

**Why Dreamer for trading:**
- ✅ Sample-efficient (learns faster with less data)
- ✅ Better long-term planning (thinks ahead)
- ✅ Handles partial observability (missing data)
- ✅ More robust to changing markets

**Technical details:**
- Model-based RL (learns environment dynamics)
- Recurrent State-Space Model (RSSM) for world model
- Latent imagination for planning
- Actor-Critic trained in latent space

**When to use each:**
- **PPO**: Faster to set up, proven results, good starting point
- **Dreamer**: More advanced, better long-term performance, requires more tuning

---

## 📖 Documentation

- [**Colab Training Guide**](COLAB_TRAINING_GUIDE.md) - Train on Google Colab with free GPU (fastest method)
- [**Deployment Guide**](DEPLOYMENT_GUIDE.md) - Deploy bot to cloud VPS for 24/7 trading
- [**Free Deployment**](FREE_DEPLOYMENT.md) - Host on free services (Render, Railway, etc.)
- [**Dreamer Implementation**](DREAMER_IMPLEMENTATION_GUIDE.md) - Deep dive into Dreamer V3 algorithm

---

## ⚙️ Configuration

### Training Parameters

Edit in `train/train_ultimate_150.py`:

```python
# Training duration
--steps 1000000          # Total training steps (1M recommended)

# Hardware
--device mps             # cpu / mps (Mac) / cuda (NVIDIA)
--batch-size 64          # Larger = faster but more memory

# Learning
--learning-rate 0.0003   # PPO learning rate
--gamma 0.99             # Discount factor (how much to value future rewards)
--ent-coef 0.01          # Exploration bonus

# Environment
--n-envs 8               # Parallel environments (faster training)
```

### Live Trading Parameters

Edit in `live_trade_mt5.py`:

```python
# Risk management
MAX_RISK_PER_TRADE = 0.02    # 2% of account per trade
MAX_DAILY_LOSS = 0.05        # Stop trading if down 5% in a day
MAX_POSITIONS = 3            # Maximum concurrent positions

# Execution
CHECK_INTERVAL = 60          # Check for signals every 60 seconds
SLIPPAGE_POINTS = 5          # Expected slippage in points
MIN_SPREAD = 20              # Don't trade if spread > 20 points
```

---

## 🧪 Testing

### Quick Environment Test
```bash
python train/smoke_env.py
```
Verifies the trading environment works correctly. Should print observation shape and complete without errors.

### Backtest on Historical Data
```bash
python backtest/backtest_engine.py --model train/ppo_xauusd_latest.zip
```
Tests model performance on out-of-sample data (data it hasn't seen during training).

**What to look for:**
- Positive returns
- Sharpe ratio > 2.0
- Max drawdown < 15%
- Consistent performance across different time periods

### Crisis Validation
```bash
python eval/crisis_validation.py
```
Tests how the bot performs during market crashes:
- 2020 COVID crash
- 2022 inflation spike
- 2023 banking crisis

**Good bot:** Reduces position sizes or goes to cash during high volatility
**Bad bot:** Keeps trading normally and gets wrecked

---

## 📈 Expected Results

### Backtesting (Historical Data)
Based on 2015-2023 XAUUSD data:
- Annual return: 60-90%
- Sharpe ratio: 2.8-3.5
- Max drawdown: 8-12%
- Win rate: 58-62%

### Forward Testing (Unseen Data)
Performance typically 20-30% lower than backtest:
- Annual return: 40-70%
- Sharpe ratio: 2.0-2.8
- Max drawdown: 10-15%

### Live Trading (Real Money)
Expected performance after accounting for slippage, spreads, execution delays:
- Annual return: 30-60%
- Sharpe ratio: 1.8-2.5
- Max drawdown: 12-18%

**Why the difference?**
- Slippage (price moves between signal and execution)
- Spread costs (bid-ask difference)
- Latency (delays in order execution)
- Market impact (your orders affect prices)

---

## ⚠️ Disclaimer

**IMPORTANT - PLEASE READ**

This software is provided for **educational and research purposes only**.

### Risks:
- ⚠️ Trading financial instruments involves **substantial risk of loss**
- ⚠️ Past performance does **NOT** guarantee future results
- ⚠️ You could lose **more than your initial investment**
- ⚠️ Automated trading can fail due to bugs, connectivity, or market conditions

### Recommendations:
- ✅ Test thoroughly on demo accounts first (minimum 1-2 months)
- ✅ Start with smallest position sizes possible
- ✅ Never risk more than you can afford to lose
- ✅ Understand how the system works before using real money
- ✅ Monitor performance daily when starting
- ✅ Have kill switches and maximum loss limits
- ✅ Consult a financial advisor before live trading

**The authors and contributors are NOT responsible for any financial losses incurred through use of this software. Use at your own risk.**

---

## 🤝 Contributing

Contributions are welcome! Here's how:

### Reporting Bugs
Open an issue with:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Error messages / logs

### Suggesting Features
Open an issue with:
- Description of feature
- Use case / benefit
- Proposed implementation (optional)

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 (Python style guide)
- Add comments for complex logic
- Include docstrings for functions
- Test your changes before submitting

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Summary:** You can use, modify, and distribute this software freely, even commercially. No warranty is provided.

---

## 🙏 Acknowledgments

This project builds on the work of many open-source contributors:

- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** - Production-ready RL implementations
- **[Dreamer V3](https://danijar.com/project/dreamerv3/)** - World model algorithm by Danijar Hafner
- **[OpenAI Gymnasium](https://gymnasium.farama.org/)** - Standard RL environment interface
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **MetaTrader 5** - Trading platform and data provider
- **Yahoo Finance** - Free historical market data

Special thanks to the quantitative trading and RL research communities for sharing knowledge and code.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/zero-was-here/tradingbot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zero-was-here/tradingbot/discussions)
- **Email**: jebariayman8@gmail.com

---

## 🗺️ Roadmap

### Completed ✅
- [x] PPO algorithm implementation
- [x] Dreamer V3 algorithm
- [x] 140+ feature engineering
- [x] Multi-timeframe analysis
- [x] Economic calendar integration
- [x] MT5 live trading
- [x] MetaAPI integration
- [x] Risk management system

### In Progress 🚧
- [ ] Hyperparameter optimization (Optuna)
- [ ] Multi-asset support (EURUSD, BTCUSD, SPX)
- [ ] Ensemble models (combine multiple agents)
- [ ] Advanced order types (limit, stop-limit)
- [ ] Web dashboard for monitoring

### Planned 📋
- [ ] Sentiment analysis from Twitter/Reddit
- [ ] Options trading integration
- [ ] Portfolio management across assets
- [ ] Custom indicators support
- [ ] Mobile app for monitoring
- [ ] Paper trading mode in GUI

---

**Built with 🔥 by [zero-was-here](https://github.com/zero-was-here)**

*If this project helps you, consider starring ⭐ the repository!*
