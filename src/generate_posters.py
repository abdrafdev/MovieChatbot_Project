"""
Movie Poster Generation using fine-tuned SDXL LoRA
"""
import os
import sys
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, GENERATED_DIR, SDXL_MODEL_ID,
    IMAGE_SIZE, NUM_INFERENCE_STEPS, PROMPT_TEMPLATES
)


class PosterGenerator:
    """Generate movie posters using fine-tuned SDXL LoRA"""
    
    def __init__(self, lora_model_path=None, use_base_model=False):
        """
        Initialize poster generator
        
        Args:
            lora_model_path: Path to fine-tuned LoRA weights
            use_base_model: If True, use base SDXL without LoRA (for comparison)
        """
        print("Loading SDXL model...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if use_base_model:
            # Use base SDXL without LoRA
            print("Using base SDXL model (no LoRA)")
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                SDXL_MODEL_ID,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None
            )
        else:
            # Load fine-tuned LoRA model
            if lora_model_path is None:
                lora_model_path = MODELS_DIR / "sdxl_lora_movie_posters"
            
            if not Path(lora_model_path).exists():
                print(f"⚠️  LoRA model not found at {lora_model_path}")
                print("Using base SDXL model instead...")
                use_base_model = True
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    SDXL_MODEL_ID,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if self.device == "cuda" else None
                )
            else:
                print(f"Loading LoRA weights from {lora_model_path}")
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    SDXL_MODEL_ID,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if self.device == "cuda" else None
                )
                # Load LoRA weights
                self.pipe.load_lora_weights(lora_model_path)
        
        # Use DPM++ scheduler for better quality
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            # self.pipe.enable_xformers_memory_efficient_attention()  # Uncomment if xformers installed
        
        print(f"✅ Model loaded on {self.device}")
    
    def generate(
        self,
        prompt,
        negative_prompt=None,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=7.5,
        seed=None,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE
    ):
        """
        Generate a movie poster from text prompt
        
        Args:
            prompt: Text description of the poster
            negative_prompt: Things to avoid in generation
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility
            width: Image width
            height: Image height
            
        Returns:
            PIL Image
        """
        if negative_prompt is None:
            negative_prompt = (
                "low quality, blurry, distorted, ugly, bad anatomy, "
                "watermark, text, signature, frame"
            )
        
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate image
        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                width=width,
                height=height
            ).images[0]
        
        return image
    
    def generate_batch(self, prompts, output_dir=None, **kwargs):
        """
        Generate multiple posters from a list of prompts
        
        Args:
            prompts: List of text prompts
            output_dir: Directory to save generated images
            **kwargs: Additional arguments for generate()
            
        Returns:
            List of PIL Images
        """
        if output_dir is None:
            output_dir = GENERATED_DIR
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images = []
        
        print(f"\nGenerating {len(prompts)} posters...")
        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] {prompt[:80]}...")
            
            image = self.generate(prompt, **kwargs)
            images.append(image)
            
            # Save image
            output_file = output_path / f"generated_poster_{i:04d}.jpg"
            image.save(output_file, quality=95)
            print(f"   Saved to {output_file}")
        
        print(f"\n✅ Generated {len(images)} posters")
        return images
    
    def generate_for_genre(self, genre, num_samples=5, output_dir=None, **kwargs):
        """
        Generate posters for a specific genre
        
        Args:
            genre: Movie genre (action, sci-fi, horror, etc.)
            num_samples: Number of posters to generate
            output_dir: Directory to save images
            **kwargs: Additional arguments for generate()
            
        Returns:
            List of PIL Images
        """
        genre = genre.lower()
        
        if genre in PROMPT_TEMPLATES:
            base_prompt = PROMPT_TEMPLATES[genre]
        else:
            base_prompt = f"{genre} movie poster with cinematic style"
        
        # Create variations of the prompt
        prompts = []
        variations = [
            "professional movie poster",
            "cinematic movie poster design",
            "high quality movie poster artwork",
            "theatrical release poster",
            "blockbuster movie poster"
        ]
        
        for i in range(num_samples):
            variation = variations[i % len(variations)]
            prompt = f"{base_prompt}, {variation}"
            prompts.append(prompt)
        
        return self.generate_batch(prompts, output_dir, **kwargs)


def generate_evaluation_set(num_samples=100):
    """
    Generate a set of posters for FID evaluation
    
    Args:
        num_samples: Number of posters to generate
    """
    print("=" * 60)
    print("Generating Evaluation Poster Set")
    print("=" * 60)
    
    generator = PosterGenerator()
    
    # Create diverse prompts across all genres
    all_prompts = []
    genres = list(PROMPT_TEMPLATES.keys())
    
    for i in range(num_samples):
        genre = genres[i % len(genres)]
        base_prompt = PROMPT_TEMPLATES[genre]
        
        # Add variety
        styles = [
            "professional design",
            "theatrical release",
            "award-winning poster",
            "blockbuster style",
            "artistic composition"
        ]
        style = styles[i % len(styles)]
        
        prompt = f"{base_prompt}, {style}, high quality"
        all_prompts.append(prompt)
    
    # Generate all posters
    eval_dir = GENERATED_DIR / "eval_set"
    images = generator.generate_batch(
        all_prompts,
        output_dir=eval_dir,
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    print(f"\n✅ Evaluation set complete: {eval_dir}")
    return images


def main():
    """Demo: Generate sample posters"""
    print("=" * 60)
    print("Movie Poster Generation Demo")
    print("=" * 60)
    
    # Initialize generator
    generator = PosterGenerator()
    
    # Generate sample posters for each genre
    print("\nGenerating sample posters for each genre...")
    
    for genre in ["action", "sci-fi", "horror", "comedy"]:
        print(f"\n--- {genre.upper()} ---")
        generator.generate_for_genre(genre, num_samples=2)
    
    print("\n✅ Demo complete! Check the 'generated' folder.")


if __name__ == "__main__":
    main()
