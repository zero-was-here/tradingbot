# Trading Bot Deployment Guide

## Option 1: AWS Lightsail (Recommended - $3.50/month)

### Step 1: Create Lightsail Instance
1. Go to https://lightsail.aws.amazon.com/
2. Click "Create instance"
3. Select:
   - Platform: Linux/Unix
   - Blueprint: Ubuntu 22.04 LTS
   - Plan: $3.50/month (512 MB RAM, 1 vCPU)
4. Name it "trading-bot"
5. Click "Create instance"

### Step 2: Connect and Setup
```bash
# SSH into your server (use Lightsail web console or SSH)
ssh ubuntu@YOUR_SERVER_IP

# Upload setup script
# On your local machine:
scp deploy_setup.sh ubuntu@YOUR_SERVER_IP:~/
scp trading-bot.service ubuntu@YOUR_SERVER_IP:~/

# On server, run setup:
chmod +x deploy_setup.sh
./deploy_setup.sh
```

### Step 3: Upload Your Code
```bash
# On your local machine:
cd ~/Desktop/trading
scp -r drl-trading ubuntu@YOUR_SERVER_IP:~/trading-bot/
```

### Step 4: Setup Auto-Start Service
```bash
# On server:
sudo cp ~/trading-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Check status:
sudo systemctl status trading-bot

# View logs:
sudo journalctl -u trading-bot -f
```

### Step 5: Manage Your Bot
```bash
# Stop bot
sudo systemctl stop trading-bot

# Start bot
sudo systemctl start trading-bot

# Restart bot
sudo systemctl restart trading-bot

# View logs (live)
sudo journalctl -u trading-bot -f

# View last 100 lines
sudo journalctl -u trading-bot -n 100
```

---

## Option 2: DigitalOcean ($4/month)

### Step 1: Create Droplet
1. Go to https://www.digitalocean.com/
2. Click "Create" → "Droplets"
3. Choose:
   - Ubuntu 22.04 LTS
   - Basic plan: $4/month (512 MB / 1 CPU)
   - Datacenter: Closest to you
4. Add SSH key or use password
5. Click "Create Droplet"

### Step 2: Follow same steps as AWS Lightsail above

---

## Option 3: Google Cloud Free Tier

### Create VM Instance
1. Go to https://console.cloud.google.com/
2. Compute Engine → VM instances
3. Create instance:
   - Machine type: e2-micro (free tier eligible)
   - Boot disk: Ubuntu 22.04 LTS
   - Allow HTTP/HTTPS traffic
4. SSH and follow setup steps above

---

## Option 4: Run on Your Mac (Simple but computer must stay on)

### Using screen (keeps running when you close terminal):
```bash
# Install screen if needed
brew install screen

# Start screen session
screen -S trading-bot

# Run your bot
cd ~/Desktop/trading/drl-trading
PYTHONPATH=. bin/python live_trade_metaapi.py

# Detach: Press Ctrl+A then D
# Reattach: screen -r trading-bot
# Kill session: screen -X -S trading-bot quit
```

### Using nohup (simpler):
```bash
cd ~/Desktop/trading/drl-trading
nohup PYTHONPATH=. bin/python live_trade_metaapi.py > trading.log 2>&1 &

# View logs:
tail -f trading.log

# Find process ID:
ps aux | grep live_trade_metaapi

# Stop bot (replace PID with actual process ID):
kill PID
```

---

## Monitoring Your Bot

### Check if bot is running:
```bash
# If using systemd:
sudo systemctl status trading-bot

# If using nohup/screen:
ps aux | grep live_trade_metaapi
```

### View logs:
```bash
# Systemd:
sudo journalctl -u trading-bot -f

# Nohup:
tail -f ~/trading.log

# Screen:
screen -r trading-bot
```

---

## Cost Comparison

| Provider | Cost/Month | RAM | CPU | Notes |
|----------|------------|-----|-----|-------|
| AWS Lightsail | $3.50 | 512 MB | 1 vCPU | Easy setup |
| DigitalOcean | $4.00 | 512 MB | 1 vCPU | User-friendly |
| Google Cloud | Free* | 614 MB | Shared | Free tier limits apply |
| Heroku | $7.00 | 512 MB | 1 vCPU | Easiest deployment |
| Your Mac | $0 | N/A | N/A | Must keep computer on |

*Google Cloud free tier: 720 hours/month (enough for 1 VM running 24/7)

---

## Recommended: AWS Lightsail

For your use case, I recommend **AWS Lightsail** because:
- Only $3.50/month
- Very reliable (99.99% uptime)
- Easy to set up
- Simple billing (no surprises)
- Can easily upgrade if needed
- Built-in firewall and monitoring

The 512 MB RAM instance is more than enough for your trading bot since it only:
- Fetches data every 10 seconds
- Runs one ML model prediction
- Makes occasional API calls

Your bot uses minimal resources!
