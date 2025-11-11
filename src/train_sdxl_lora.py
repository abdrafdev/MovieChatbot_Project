"""
SDXL LoRA Training Script for Movie Poster Generation
Optimized for GTX 1090 with limited VRAM
"""
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    IMAGES_DIR, CAPTIONS_DIR, MODELS_DIR, CHECKPOINTS_DIR,
    SDXL_MODEL_ID, IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    SAVE_STEPS, GRADIENT_ACCUMULATION_STEPS, MIXED_PRECISION
)


class MoviePosterDataset(Dataset):
    """Dataset for movie posters with captions"""
    
    def __init__(self, images_dir, captions_dir, image_size=512):
        self.images_dir = Path(images_dir)
        self.captions_dir = Path(captions_dir)
        self.image_size = image_size
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        print(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Normalize to [-1, 1]
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Load caption
        caption_path = self.captions_dir / f"{img_path.stem}.txt"
        if caption_path.exists():
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        else:
            caption = "Movie poster with cinematic style"
        
        return {
            "pixel_values": image,
            "caption": caption
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    captions = [item["caption"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "captions": captions
    }


class SDXLLoRATrainer:
    """SDXL LoRA trainer optimized for low VRAM"""
    
    def __init__(self):
        self.accelerator = Accelerator(
            mixed_precision=MIXED_PRECISION,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
        )
        
        print("Initializing SDXL model...")
        self.device = self.accelerator.device
        
        # Load SDXL components
        self.tokenizer = CLIPTokenizer.from_pretrained(
            SDXL_MODEL_ID, subfolder="tokenizer"
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            SDXL_MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16
        )
        
        self.vae = AutoencoderKL.from_pretrained(
            SDXL_MODEL_ID, subfolder="vae", torch_dtype=torch.float16
        )
        
        self.unet = UNet2DConditionModel.from_pretrained(
            SDXL_MODEL_ID, subfolder="unet", torch_dtype=torch.float16
        )
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Apply LoRA to UNet
        print("Applying LoRA to UNet...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        # Move to device
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
        
        # Noise scheduler
        from diffusers import DDPMScheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            SDXL_MODEL_ID, subfolder="scheduler"
        )
    
    def encode_text(self, captions):
        """Encode text captions to embeddings"""
        inputs = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                inputs.input_ids.to(self.device)
            )[0]
        
        return text_embeddings
    
    def encode_images(self, images):
        """Encode images to latent space"""
        with torch.no_grad():
            latents = self.vae.encode(images.to(self.device, dtype=torch.float16)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def train_step(self, batch):
        """Single training step"""
        with self.accelerator.accumulate(self.unet):
            # Encode images to latents
            latents = self.encode_images(batch["pixel_values"])
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample timesteps
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                device=latents.device
            ).long()
            
            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Encode text
            text_embeddings = self.encode_text(batch["captions"])
            
            # Predict noise
            model_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
            
            # Calculate loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Backprop
            self.accelerator.backward(loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()
    
    def train(self, dataset, num_epochs=NUM_EPOCHS):
        """Main training loop"""
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Windows compatibility
        )
        
        # Prepare for training
        self.unet, self.optimizer, dataloader = self.accelerator.prepare(
            self.unet, self.optimizer, dataloader
        )
        
        global_step = 0
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Total steps: {len(dataloader) * num_epochs}")
        
        for epoch in range(num_epochs):
            self.unet.train()
            epoch_loss = 0
            
            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                epoch_loss += loss
                global_step += 1
                
                progress_bar.set_postfix({"loss": f"{loss:.4f}"})
                
                # Save checkpoint
                if global_step % SAVE_STEPS == 0:
                    self.save_checkpoint(global_step)
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save final model
        self.save_final_model()
        print("\n✅ Training complete!")
    
    def save_checkpoint(self, step):
        """Save training checkpoint"""
        if self.accelerator.is_main_process:
            checkpoint_path = CHECKPOINTS_DIR / f"checkpoint-{step}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            unwrapped_unet.save_pretrained(checkpoint_path)
            
            print(f"\n💾 Checkpoint saved at step {step}")
    
    def save_final_model(self):
        """Save final trained model"""
        if self.accelerator.is_main_process:
            model_path = MODELS_DIR / "sdxl_lora_movie_posters"
            model_path.mkdir(parents=True, exist_ok=True)
            
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            unwrapped_unet.save_pretrained(model_path)
            
            print(f"\n💾 Final model saved to {model_path}")


def main():
    """Main training script"""
    print("=" * 60)
    print("SDXL LoRA Training for Movie Posters")
    print("=" * 60)
    
    # Check if dataset exists
    if not IMAGES_DIR.exists() or not list(IMAGES_DIR.glob("*.jpg")):
        print("⚠️  Dataset not found! Please run prepare_dataset.py first.")
        return
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = MoviePosterDataset(IMAGES_DIR, CAPTIONS_DIR, IMAGE_SIZE)
    
    if len(dataset) == 0:
        print("⚠️  No images found in dataset!")
        return
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SDXLLoRATrainer()
    
    # Start training
    trainer.train(dataset)


if __name__ == "__main__":
    main()
