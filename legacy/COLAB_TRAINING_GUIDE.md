# 🚀 Google Colab Training Guide - 1 Million Steps

## 🎯 Overview

Train your DreamerV3 God Mode trading AI on Google Colab GPUs for **FREE** or accelerated with Colab Pro/Pro+.

**Time Estimates:**
- **Colab Free (T4)**: 20-30 hours (2-3 session resumes needed)
- **Colab Pro (V100)**: 12-18 hours (1-2 resumes)
- **Colab Pro+ (A100)**: 5-7 hours (one continuous session)

---

## 📋 Prerequisites

1. Google account
2. Your `drl-trading` project folder
3. Google Drive with ~2GB free space

---

## 🔧 Setup Steps

### Step 1: Prepare Your Project for Upload

```bash
# On your Mac, create a compressed archive
cd /Users/mac/Desktop/trading
tar -czf drl-trading.tar.gz drl-trading/

# This creates: drl-trading.tar.gz (~500MB-1GB)
```

**OR** zip the entire folder:
```bash
zip -r drl-trading.zip drl-trading/
```

---

### Step 2: Upload to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create a folder called `AI-Trading` (or any name)
3. Upload **ONE OF:**
   - `drl-trading.tar.gz` (compressed - faster)
   - Entire `drl-trading` folder (easier to edit)

**Recommended**: Upload the folder directly for easier debugging

---

### Step 3: Open Colab Notebook

1. Upload `colab_train_dreamer.ipynb` to Google Drive
2. Double-click it → "Open with Google Colaboratory"
3. **IMPORTANT**: Enable GPU
   - Menu: `Runtime` → `Change runtime type`
   - Hardware accelerator: `GPU`
   - GPU type: `T4` (free) or `A100` (Pro+)
   - Click `Save`

---

### Step 4: Run Training

**Execute cells in order:**

1. **Cell 1**: Mount Google Drive ✅
2. **Cell 2**: Load project (adjust path if needed)
3. **Cell 3**: Install dependencies
4. **Cell 4**: Verify data loaded correctly
5. **Cell 5**: Configure training parameters
6. **Cell 6**: 🔥 **START TRAINING** (main cell)

**Cell 6 will run for hours.** This is normal.

---

## 📊 Monitoring Progress

### While Training Runs:

**Option 1: Check in Colab**
- Run **Cell 7** anytime to see checkpoint progress
- Look for saved files in `train/dreamer/`

**Option 2: Check Output**
- Cell 6 shows live progress bars
- Updates every step

**Option 3: Training Logs**
```python
# Add this cell to tail logs
!tail -f training.log
```

---

## ⚠️ Handling Session Disconnects

Colab Free disconnects after 12 hours. **Don't panic!**

### To Resume:

1. **Re-run Cell 1** (mount Drive)
2. **Re-run Cell 2** (load project)
3. **Re-run Cell 6** (training auto-resumes from last checkpoint)

The training script automatically detects and loads the latest checkpoint.

---

## 💾 Saving Your Model

### Automatic Backup (Recommended)
After training completes, run **Cell 9**:
- Backs up to Google Drive automatically
- Timestamped folder
- Safe from Colab session deletion

### Manual Download
Run **Cell 8**:
- Downloads `trained_model_1m.tar.gz`
- Extract on your Mac:
  ```bash
  tar -xzf trained_model_1m.tar.gz
  ```

---

## 🔥 Optimization Tips

### For Free Tier Users:

1. **Keep browser tab open**
   - Colab disconnects idle sessions
   - Use browser extension to keep alive

2. **Monitor GPU usage**
   ```python
   !nvidia-smi
   ```

3. **Train in stages**
   - 100k steps → save → 200k → save
   - Less painful if disconnects

### For Pro+ Users:

1. **Increase batch size** (Cell 5):
   ```python
   BATCH_SIZE = 256  # A100 can handle it
   ```

2. **Train multiple models in parallel**
   - Open 3 Colab tabs
   - Train 3 models with different seeds
   - Create ensemble

---

## 🐛 Troubleshooting

### "GPU not available"
**Solution:**
- Runtime → Change runtime type → GPU
- Restart runtime
- Re-run Cell 3 to verify

### "Data file not found"
**Solution:**
- Check path in Cell 2
- Verify upload to Drive
- Run Cell 4 to diagnose

### "Out of Memory"
**Solution:**
- Reduce `BATCH_SIZE` to 64 or 32 (Cell 5)
- Runtime → Restart runtime
- Re-run from Cell 1

### "Session disconnected"
**Solution:**
- Normal for Colab Free after 12 hours
- Re-run Cells 1, 2, 6 to resume
- Training continues from checkpoint

### Training is very slow
**Check:**
1. GPU enabled? (Run Cell 1, check nvidia-smi)
2. Using CPU by mistake? (Cell 5 should say `device: cuda`)
3. Colab might throttle after long usage

**Solution:** Upgrade to Pro+ for consistent A100 access

---

## 📈 Expected Training Timeline

### Colab Free (T4):
```
Hours 0-12:   Steps 0 → 400k (then disconnects)
Resume:       Steps 400k → 800k (then disconnects)
Final Resume: Steps 800k → 1M ✅

Total: 24-30 hours over 2-3 sessions
```

### Colab Pro+ (A100):
```
Continuous:   Steps 0 → 1M ✅

Total: 5-7 hours, one session
```

---

## 🎯 After Training Completes

### 1. Download Model
Run Cell 8 or Cell 9

### 2. Transfer to Mac
```bash
# Extract downloaded archive
tar -xzf trained_model_1m.tar.gz

# Copy to your project
cp -r train/dreamer /Users/mac/Desktop/trading/drl-trading/train/
```

### 3. Validate
```bash
cd /Users/mac/Desktop/trading/drl-trading
python eval/crisis_validation.py
```

### 4. Deploy to Demo
Use the live trading script (we'll create this next)

---

## 💰 Cost Analysis

| Tier | Cost | Time to 1M | Convenience |
|------|------|------------|-------------|
| **Free** | $0 | 24-30h | ⭐⭐ (resumes needed) |
| **Pro** | $10/mo | 12-18h | ⭐⭐⭐ (fewer resumes) |
| **Pro+** | $50/mo | 5-7h | ⭐⭐⭐⭐⭐ (one session) |

**Recommendation:**
- **Try Free first** (test if setup works)
- **Upgrade to Pro+** if you want it done in one day

---

## 🔄 Training Multiple Models (Ensemble)

To create a 5-model ensemble:

1. Train model 1 with seed 42
2. Download & backup
3. Change Cell 5: `SEED = 123`
4. Train model 2
5. Repeat for seeds: 777, 999, 1337

**Each model takes 5-30 hours depending on GPU.**

---

## ✅ Checklist

Before starting training:

- [ ] Google Drive has 2GB+ free space
- [ ] Project uploaded to Drive
- [ ] Colab notebook opened
- [ ] GPU enabled in runtime settings
- [ ] Cell 4 shows data loaded correctly
- [ ] Ready to commit 5-30 hours

---

## 🏆 Success Metrics

Training is successful if:

- ✅ Reaches 1,000,000 steps
- ✅ Final checkpoint saved (`dreamer_xauusd_final.pt`)
- ✅ Test set return > -10% (ideally positive)
- ✅ Model loss decreasing over time
- ✅ No crashes or errors

---

## 🚀 Ready to Start?

1. Upload project to Google Drive
2. Open `colab_train_dreamer.ipynb`
3. Enable GPU
4. Run cells 1-6
5. Wait for God Mode 🔥

**Time commitment:**
- **Active work**: 15-30 minutes (setup)
- **Passive waiting**: 5-30 hours (training)

---

**Questions? Check the troubleshooting section above.**

**Ready for live trading? Continue to the next step: Crisis Validation & Demo Trading Setup.**
