# AI-Powered Movie Chatbot with SDXL Poster Generation

A college-level project that combines movie recommendations with AI-generated posters using Stable Diffusion XL (SDXL) fine-tuned with LoRA.

## 📋 Project Overview

This project creates an AI chatbot that:
- Recommends movies based on user queries and genre preferences
- Generates high-quality movie posters using fine-tuned SDXL
- Evaluates model performance using FID and CLIP metrics
- Provides an interactive web interface

**Key Features:**
- ✅ SDXL fine-tuning with LoRA (optimized for GTX 1090)
- ✅ 1,150 movie poster dataset with captions
- ✅ FID and CLIP evaluation metrics
- ✅ FastAPI backend with REST API
- ✅ Modern, responsive web interface
- ✅ Real-time poster generation

## 🗂️ Project Structure

```
MovieChatbot_Project/
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── README.md                 # This file
│
├── dataset/
│   ├── raw/                  # Original downloaded images
│   ├── images_512/           # Resized 512x512 images
│   └── captions/             # Text captions for each image
│
├── src/
│   ├── prepare_dataset.py    # Dataset preparation script
│   ├── train_sdxl_lora.py    # LoRA training script
│   ├── generate_posters.py   # Poster generation module
│   ├── evaluate_model.py     # FID/CLIP evaluation
│   └── chatbot_api.py        # FastAPI backend server
│
├── frontend/
│   └── index.html            # Web interface
│
├── models/                   # Trained LoRA models
├── checkpoints/              # Training checkpoints
└── generated/                # Generated posters
```

## 🚀 Setup Instructions

### 1. Prerequisites

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (GTX 1090 or better)
- **CUDA**: 11.8 or higher
- **RAM**: 16GB minimum
- **Disk Space**: ~50GB for models and dataset

### 2. Install Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project requirements
pip install -r requirements.txt
```

### 3. Download Dataset

**Option 1: Using Kaggle API**
```powershell
# Configure Kaggle API (requires kaggle.json in ~/.kaggle/)
kaggle datasets download -d phiitm/movie-posters
Expand-Archive movie-posters.zip -DestinationPath dataset/raw
```

**Option 2: Manual Download**
1. Visit: https://www.kaggle.com/datasets/phiitm/movie-posters
2. Download and extract to `dataset/raw/`

### 4. Prepare Dataset

```powershell
python src/prepare_dataset.py
```

This will:
- Resize images to 512×512
- Generate caption files
- Validate dataset structure

**Expected output:**
- `dataset/images_512/` → 1,150 resized images
- `dataset/captions/` → 1,150 text files

## 🎯 Usage Guide

### Step 1: Train SDXL with LoRA

```powershell
python src/train_sdxl_lora.py
```

**Training Configuration:**
- Batch size: 1 (optimized for GTX 1090)
- LoRA rank: 8
- LoRA alpha: 16
- Mixed precision: FP16
- Epochs: 10

**Training Time:** ~8-12 hours on GTX 1090

**Output:**
- Final model → `models/sdxl_lora_movie_posters/`
- Checkpoints → `checkpoints/checkpoint-*/`

### Step 2: Generate Evaluation Set

```powershell
python src/generate_posters.py
```

Or generate evaluation set directly:
```powershell
python -c "from src.generate_posters import generate_evaluation_set; generate_evaluation_set(100)"
```

**Output:** 100 generated posters in `generated/eval_set/`

### Step 3: Evaluate Model

```powershell
python src/evaluate_model.py
```

**Metrics Calculated:**
- **FID Score**: Measures distribution similarity (lower is better)
- **CLIP Similarity**: Measures text-image alignment (higher is better)

**Output:**
- `evaluation_report.txt` - Detailed metrics
- `comparison.png` - Visual comparison

### Step 4: Run Chatbot

**Start Backend Server:**
```powershell
python src/chatbot_api.py
```

Server starts at: http://127.0.0.1:8000

**API Endpoints:**
- `GET /` - Health check
- `POST /recommend` - Get recommendations + generate poster
- `GET /genres` - List available genres
- `POST /generate-poster` - Generate custom poster

**Open Frontend:**
Open `frontend/index.html` in your web browser.

## 🎬 Using the Chatbot

### Web Interface

1. **Enter Query**: Type what kind of movie you want
   - Example: "futuristic sci-fi adventure"
   - Example: "dark horror with monsters"

2. **Select Genre** (Optional): Click genre button for specific category

3. **Generate**: Click "Generate" button

4. **View Results**:
   - AI-generated movie poster
   - Movie recommendations for that genre

### Example Queries

| Query | Genre Detected | Result |
|-------|---------------|---------|
| "space adventure with aliens" | Sci-Fi | Sci-fi poster + recommendations |
| "scary ghost story" | Horror | Horror poster + recommendations |
| "romantic love story" | Romance | Romance poster + recommendations |
| "action hero saves the world" | Action | Action poster + recommendations |

## 📊 Evaluation Metrics

### FID (Fréchet Inception Distance)
- Measures how close generated images are to real posters
- **Lower is better**
- Good score: < 50
- Excellent: < 30

### CLIP Similarity
- Measures how well images match text descriptions
- **Higher is better**
- Good score: > 0.25
- Excellent: > 0.30

## 🛠️ Configuration

Edit `config.py` to customize:

```python
# Model settings
LORA_R = 8                    # LoRA rank
LORA_ALPHA = 16               # LoRA alpha
IMAGE_SIZE = 512              # Image dimensions
BATCH_SIZE = 1                # Training batch size

# Training settings
NUM_EPOCHS = 10               # Training epochs
LEARNING_RATE = 1e-4          # Learning rate
SAVE_STEPS = 500              # Checkpoint frequency

# Generation settings
NUM_INFERENCE_STEPS = 30      # Quality vs speed
```

## 🔧 Troubleshooting

### Out of Memory Errors
```python
# In config.py, reduce:
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

# Enable CPU offloading in generate_posters.py:
pipe.enable_sequential_cpu_offload()
```

### Slow Generation
```python
# Reduce inference steps:
NUM_INFERENCE_STEPS = 20  # Faster, lower quality

# Enable xformers (if installed):
pipe.enable_xformers_memory_efficient_attention()
```

### CUDA Out of Memory
- Close other GPU applications
- Restart Python kernel
- Reduce image size to 384×384

## 📝 Project Deliverables

### For College Project Submission

1. **Code & Models**
   - Complete source code
   - Trained LoRA model weights
   - Configuration files

2. **Dataset**
   - 1,150 movie posters (512×512)
   - Matching caption files
   - Dataset preparation scripts

3. **Generated Samples**
   - 50-100 generated posters
   - Visual comparison images
   - Sample outputs for each genre

4. **Evaluation Report**
   - FID scores with interpretation
   - CLIP similarity scores
   - Performance analysis
   - Visual comparisons

5. **Documentation**
   - This README
   - Setup instructions
   - Usage guide
   - API documentation

6. **Demo**
   - Working chatbot interface
   - Live poster generation
   - Screenshots/video

## 📚 Technical Details

### Model Architecture
- **Base Model**: Stable Diffusion XL 1.0
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Parameters**: 8M trainable (vs 6.6B total)
- **Memory**: ~11GB VRAM during training

### Dataset
- **Source**: Kaggle Movie Posters dataset
- **Size**: 1,150 images
- **Resolution**: 512×512 (resized)
- **Captions**: Auto-generated from metadata

### API Stack
- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript
- **Model Serving**: Diffusers pipeline
- **Image Processing**: PIL/OpenCV

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Fine-tuning large diffusion models
- ✅ Low-rank adaptation (LoRA) techniques
- ✅ Image generation evaluation (FID, CLIP)
- ✅ REST API development
- ✅ Full-stack web application
- ✅ GPU optimization for limited VRAM

## 🤝 Credits

- **SDXL**: Stability AI
- **Dataset**: Kaggle - Movie Posters by phiitm
- **Libraries**: Hugging Face Diffusers, PyTorch, FastAPI

## 📄 License

This is an educational project for college coursework.

## 🐛 Known Issues

1. First poster generation is slow (~30s) due to model loading
2. Large FID scores might require more training epochs
3. Windows may require `num_workers=0` in DataLoader

## 🔮 Future Enhancements

- [ ] Add more advanced recommendation algorithm
- [ ] Support multiple poster styles
- [ ] Add poster customization options
- [ ] Implement user rating system
- [ ] Deploy to cloud service

## 📞 Support

For issues or questions, refer to:
- Diffusers documentation: https://huggingface.co/docs/diffusers
- PEFT documentation: https://huggingface.co/docs/peft
- FastAPI documentation: https://fastapi.tiangolo.com

---

**Project by:** Abdul Rafay  
**Client:** Yassine Yagout  
**Date:** 2025
#   M o v i e C h a t b o t _ P r o j e c t  
 