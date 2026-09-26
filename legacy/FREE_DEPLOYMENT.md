# Deploy Trading Bot for FREE on Google Cloud

## What You Get (FREE Forever):
- 1 e2-micro VM instance (0.25-1 vCPU, 1 GB RAM)
- 30 GB standard persistent disk
- Runs 24/7 completely FREE
- No credit card charges (stays within free tier)

---

## Step-by-Step Setup (15 minutes)

### 1. Create Google Cloud Account

1. Go to: https://cloud.google.com/free
2. Click **"Get started for free"**
3. Sign in with Google account
4. Fill in billing info (required but won't be charged if you stay in free tier)
5. Accept terms and click **"Start my free trial"**
   - You get $300 credits for 90 days PLUS always-free tier

---

### 2. Create Your Free VM

1. Go to: https://console.cloud.google.com/
2. Click **"Activate Cloud Shell"** (icon at top right: `>_`)
3. In the Cloud Shell, paste this command:

```bash
gcloud compute instances create trading-bot \
  --machine-type=e2-micro \
  --zone=us-central1-a \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-standard \
  --tags=http-server
```

4. Wait ~1 minute for VM to be created
5. You'll see output like: `Created [https://www.googleapis.com/compute/v1/projects/...]`

---

### 3. Connect to Your VM

In Cloud Shell, run:
```bash
gcloud compute ssh trading-bot --zone=us-central1-a
```

**Important:** First time it asks to create SSH keys, type `Y` and press Enter (no passphrase needed)

You're now connected to your free server! 🎉

---

### 4. Install Dependencies

Copy and paste this entire block into your SSH session:

```bash
# Update system
sudo apt-get update

# Install Python 3.12
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip

# Install system dependencies
sudo apt-get install -y build-essential git

# Create project directory
mkdir -p ~/trading-bot
cd ~/trading-bot

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python packages (this takes ~5 minutes)
pip install numpy pandas stable-baselines3 metaapi-cloud-sdk scikit-learn gymnasium

echo ""
echo "✅ Dependencies installed!"
```

Wait for installation to complete (~5 minutes).

---

### 5. Upload Your Trading Bot Code

**Open a NEW terminal on your Mac** (don't close the SSH session) and run:

```bash
# First, create a tarball of your code
cd ~/Desktop/trading/drl-trading
tar -czf trading-bot.tar.gz \
  live_trade_metaapi.py \
  features/ \
  train/ppo_xauusd_latest.zip

# Upload to Google Cloud
gcloud compute scp trading-bot.tar.gz trading-bot:~/trading-bot/ --zone=us-central1-a
```

**Note:** First time you run `gcloud`, it will ask you to authenticate. Follow the prompts.

If you don't have `gcloud` installed on your Mac:
```bash
# Install Google Cloud SDK
brew install --cask google-cloud-sdk

# Initialize
gcloud init
```

---

### 6. Extract and Test Your Bot

**Back in your SSH session** (the first terminal):

```bash
cd ~/trading-bot
tar -xzf trading-bot.tar.gz
ls -la

# Test run your bot
source venv/bin/activate
PYTHONPATH=. python3 live_trade_metaapi.py
```

You should see your bot connect and start trading! Press `Ctrl+C` to stop.

---

### 7. Set Up Auto-Start (Run 24/7)

Create a systemd service so your bot runs automatically and restarts if it crashes:

```bash
# Create service file
cat > ~/trading-bot.service << 'EOF'
[Unit]
Description=DRL Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/trading-bot
Environment="PYTHONPATH=/home/$USER/trading-bot"
ExecStart=/home/$USER/trading-bot/venv/bin/python /home/$USER/trading-bot/live_trade_metaapi.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Replace $USER with actual username
sed -i "s/\$USER/$USER/g" ~/trading-bot.service

# Install and start service
sudo cp ~/trading-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot
```

You should see: `Active: active (running)`

---

## Managing Your Bot

### View Live Logs
```bash
sudo journalctl -u trading-bot -f
```
Press `Ctrl+C` to exit

### Check Status
```bash
sudo systemctl status trading-bot
```

### Stop Bot
```bash
sudo systemctl stop trading-bot
```

### Start Bot
```bash
sudo systemctl start trading-bot
```

### Restart Bot
```bash
sudo systemctl restart trading-bot
```

### View Last 100 Lines of Logs
```bash
sudo journalctl -u trading-bot -n 100
```

---

## Disconnect and Reconnect

### To Disconnect
Just close the terminal or type:
```bash
exit
```

Your bot keeps running!

### To Reconnect Later
```bash
gcloud compute ssh trading-bot --zone=us-central1-a
```

---

## Update Your Bot Code

When you make changes to your local code:

```bash
# On your Mac:
cd ~/Desktop/trading/drl-trading
tar -czf trading-bot.tar.gz \
  live_trade_metaapi.py \
  features/ \
  train/ppo_xauusd_latest.zip

gcloud compute scp trading-bot.tar.gz trading-bot:~/trading-bot/ --zone=us-central1-a

# On the server (SSH):
cd ~/trading-bot
tar -xzf trading-bot.tar.gz
sudo systemctl restart trading-bot
```

---

## Monitor Your VM

### Check VM Status
Go to: https://console.cloud.google.com/compute/instances

You'll see:
- VM status (running/stopped)
- CPU usage
- Network traffic
- Uptime

### Check Free Tier Usage
Go to: https://console.cloud.google.com/billing/

Under "Free tier usage", make sure you're within limits:
- ✅ e2-micro instance in us-central1, us-west1, or us-east1
- ✅ 30 GB standard persistent disk
- ✅ Network egress (within 1 GB/month)

---

## Important Notes

### Keep It FREE:
1. **Use only e2-micro** in us-central1, us-west1, or us-east1
2. Don't upgrade machine type
3. Don't add external IP charges (included in free tier)
4. Don't exceed 1 GB network egress per month (your bot uses ~100 MB/month)

### If You Want to Stop VM (to save free tier):
```bash
gcloud compute instances stop trading-bot --zone=us-central1-a
```

### To Start Again:
```bash
gcloud compute instances start trading-bot --zone=us-central1-a
```

Your bot will auto-start when VM boots up!

---

## Troubleshooting

### Bot Not Running?
```bash
# Check status
sudo systemctl status trading-bot

# View errors
sudo journalctl -u trading-bot -n 50

# Restart
sudo systemctl restart trading-bot
```

### Can't Connect to VM?
```bash
# Make sure VM is running
gcloud compute instances list

# If stopped, start it
gcloud compute instances start trading-bot --zone=us-central1-a
```

### Out of Disk Space?
```bash
# Check disk usage
df -h

# Clean up
sudo apt-get autoremove
sudo apt-get clean
```

---

## Cost Monitoring

Set up billing alerts:
1. Go to: https://console.cloud.google.com/billing/
2. Click "Budgets & alerts"
3. Create budget: $0.50/month
4. You'll get email if you accidentally exceed free tier

---

## Summary

Your trading bot is now:
- ✅ Running 24/7 on Google Cloud
- ✅ Completely FREE (e2-micro always-free tier)
- ✅ Auto-restarts if it crashes
- ✅ Auto-starts when VM reboots
- ✅ Accessible from anywhere

**Your bot will run forever for free as long as you stay within the free tier limits!**

Need help? Just SSH back in and check the logs! 🚀
