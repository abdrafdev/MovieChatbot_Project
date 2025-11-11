# Quick Start Guide

Get started with the Movie Chatbot project in 4 steps!

## 🚀 Quick Setup (30 minutes)

### Step 1: Install Dependencies (5 min)

```powershell
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
```

### Step 2: Download Dataset (10 min)

**Manual Download** (Easiest):
1. Go to: https://www.kaggle.com/datasets/phiitm/movie-posters
2. Download ZIP file
3. Extract to `dataset/raw/`

### Step 3: Prepare Dataset (5 min)

```powershell
python src/prepare_dataset.py
```

### Step 4: Test Generator (10 min)

**Test with Base SDXL (no training required):**

```powershell
python -c "from src.generate_posters import PosterGenerator; g = PosterGenerator(use_base_model=True); g.generate_for_genre('action', 2)"
```

This generates 2 action posters without training!

## 🎯 Full Workflow

### Training (8-12 hours)

```powershell
# Train LoRA model
python src/train_sdxl_lora.py
```

**Monitor training:**
- Watch loss decrease in console
- Checkpoints saved every 500 steps
- Final model in `models/`

### Generate & Evaluate

```powershell
# Generate 100 posters for evaluation
python -c "from src.generate_posters import generate_evaluation_set; generate_evaluation_set(100)"

# Calculate FID/CLIP scores
python src/evaluate_model.py
```

### Run Chatbot

```powershell
# Terminal 1: Start backend
python src/chatbot_api.py

# Terminal 2: Open frontend
start frontend/index.html
```

## 🎬 Using the Chatbot

1. **Open** `frontend/index.html` in browser
2. **Type** your movie preference (e.g., "futuristic sci-fi")
3. **Click** Generate
4. **View** generated poster + recommendations

## ⚡ Quick Tips

### Save Time
- Use base SDXL first to test everything works
- Train for fewer epochs (5 instead of 10) for faster testing
- Generate fewer evaluation samples (50 instead of 100)

### Save VRAM
```python
# In config.py:
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
IMAGE_SIZE = 512  # Don't increase this
```

### Faster Generation
```python
# In config.py:
NUM_INFERENCE_STEPS = 20  # Default is 30
```

## 📋 Checklist

- [ ] Python 3.8+ installed
- [ ] CUDA GPU available (check with `nvidia-smi`)
- [ ] Dependencies installed
- [ ] Dataset downloaded (1,150 images)
- [ ] Dataset prepared (images resized + captions)
- [ ] Base model tested (generates posters)
- [ ] LoRA training started
- [ ] Model evaluation completed
- [ ] Chatbot running

## 🐛 Common Issues

### "CUDA out of memory"
→ Reduce batch size to 1, close other GPU apps

### "No module named 'diffusers'"
→ Run: `pip install -r requirements.txt`

### "Dataset not found"
→ Make sure images are in `dataset/raw/`

### "Model loading slow"
→ First load downloads ~13GB, takes 5-10 minutes

## 📞 Need Help?

1. Check `README.md` for detailed docs
2. See troubleshooting section in README
3. Verify GPU with: `python -c "import torch; print(torch.cuda.is_available())"`

## 🎓 For College Submission

### Minimum Viable Demo
1. Dataset prepared ✓
2. Base model generates posters ✓
3. Chatbot interface works ✓
4. Screenshots of results ✓

### Full Project
1. Everything above +
2. LoRA model trained ✓
3. FID/CLIP evaluation ✓
4. 100 generated samples ✓
5. Comparison images ✓
6. Written report ✓

---

**Time Budget:**
- Setup: 30 min
- Training: 8-12 hours (can run overnight)
- Evaluation: 1 hour
- Demo preparation: 1 hour
- **Total: ~10-14 hours**

Good luck! 🚀
