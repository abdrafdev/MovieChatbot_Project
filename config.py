"""
Configuration file for Movie Chatbot with SDXL Poster Generation
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images_512"
CAPTIONS_DIR = DATASET_DIR / "captions"
GENERATED_DIR = BASE_DIR / "generated"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# Create directories if they don't exist
for dir_path in [DATASET_DIR, IMAGES_DIR, CAPTIONS_DIR, GENERATED_DIR, MODELS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configuration
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_SIZE = 512
BATCH_SIZE = 1  # Keep small for GTX 1090
NUM_INFERENCE_STEPS = 30

# LoRA configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

# Training configuration
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = "fp16"  # Use fp16 for GTX 1090
SAVE_STEPS = 500
VALIDATION_STEPS = 250

# Dataset configuration
DATASET_SIZE = 1150
TRAIN_SPLIT = 0.9  # 90% train, 10% validation

# Evaluation configuration
NUM_EVAL_SAMPLES = 100
FID_BATCH_SIZE = 4
FID_DIMS = 2048

# Chatbot configuration
CHATBOT_PORT = 8000
CHATBOT_HOST = "127.0.0.1"

# Movie database (simple rule-based recommendations)
MOVIE_DATABASE = {
    "action": ["The Dark Knight", "Mad Max: Fury Road", "John Wick", "Mission Impossible"],
    "sci-fi": ["Blade Runner 2049", "Inception", "Interstellar", "The Matrix"],
    "horror": ["The Shining", "Get Out", "A Quiet Place", "Hereditary"],
    "comedy": ["The Grand Budapest Hotel", "Superbad", "Knives Out", "Jojo Rabbit"],
    "drama": ["The Shawshank Redemption", "Parasite", "1917", "Marriage Story"],
    "fantasy": ["The Lord of the Rings", "Harry Potter", "Pan's Labyrinth", "Spirited Away"],
    "thriller": ["Gone Girl", "Prisoners", "No Country for Old Men", "Shutter Island"],
    "romance": ["La La Land", "The Notebook", "Before Sunrise", "Call Me By Your Name"],
}

# Poster generation prompts
PROMPT_TEMPLATES = {
    "action": "Epic action movie poster with explosions, intense colors, dramatic lighting",
    "sci-fi": "Futuristic sci-fi movie poster with spaceships, neon lights, cyberpunk aesthetic",
    "horror": "Dark horror movie poster with ominous shadows, red and black tones, eerie atmosphere",
    "comedy": "Bright comedy movie poster with vibrant colors, playful typography, cheerful mood",
    "drama": "Emotional drama movie poster with muted colors, powerful composition, artistic style",
    "fantasy": "Magical fantasy movie poster with mystical elements, rich colors, epic scale",
    "thriller": "Suspenseful thriller movie poster with dark tones, mysterious atmosphere, bold contrast",
    "romance": "Romantic movie poster with warm colors, soft lighting, intimate composition",
}
