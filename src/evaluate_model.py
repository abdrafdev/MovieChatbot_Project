"""
Model Evaluation: FID and CLIP scores for generated posters
"""
import os
import sys
from pathlib import Path
import torch
from pytorch_fid import fid_score
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import IMAGES_DIR, GENERATED_DIR, FID_BATCH_SIZE, FID_DIMS


def calculate_fid(real_images_path, generated_images_path, batch_size=FID_BATCH_SIZE, dims=FID_DIMS):
    """
    Calculate Fréchet Inception Distance between real and generated images
    
    Args:
        real_images_path: Path to directory with real images
        generated_images_path: Path to directory with generated images
        batch_size: Batch size for processing
        dims: Dimensionality of Inception features
        
    Returns:
        FID score (lower is better)
    """
    print("Calculating FID score...")
    print(f"  Real images: {real_images_path}")
    print(f"  Generated images: {generated_images_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [str(real_images_path), str(generated_images_path)],
            batch_size=batch_size,
            device=device,
            dims=dims
        )
        
        print(f"\n✅ FID Score: {fid_value:.2f}")
        return fid_value
        
    except Exception as e:
        print(f"\n⚠️ Error calculating FID: {e}")
        return None


def calculate_clip_similarity(images_path, captions_path=None):
    """
    Calculate CLIP similarity between images and their captions
    
    Args:
        images_path: Path to directory with images
        captions_path: Path to directory with caption files (optional)
        
    Returns:
        Average CLIP similarity score
    """
    try:
        import clip
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
        
        print("\nCalculating CLIP similarity...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        images_path = Path(images_path)
        image_files = sorted(list(images_path.glob("*.jpg")) + list(images_path.glob("*.png")))
        
        if len(image_files) == 0:
            print("⚠️ No images found")
            return None
        
        similarities = []
        
        for img_file in image_files[:100]:  # Sample 100 images
            # Load and preprocess image
            image = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
            
            # Get caption if available
            if captions_path:
                caption_file = Path(captions_path) / f"{img_file.stem}.txt"
                if caption_file.exists():
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                else:
                    caption = "movie poster"
            else:
                caption = "movie poster"
            
            # Tokenize caption
            text = clip.tokenize([caption]).to(device)
            
            # Calculate similarity
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                similarity = (image_features @ text_features.T).item()
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        print(f"✅ Average CLIP Similarity: {avg_similarity:.4f}")
        
        return avg_similarity
        
    except ImportError:
        print("⚠️ CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
        return None
    except Exception as e:
        print(f"⚠️ Error calculating CLIP similarity: {e}")
        return None


def generate_evaluation_report(fid_score, clip_score=None, output_path="evaluation_report.txt"):
    """
    Generate evaluation report with metrics
    
    Args:
        fid_score: FID score
        clip_score: CLIP similarity score (optional)
        output_path: Path to save report
    """
    report = []
    report.append("=" * 60)
    report.append("Movie Poster Generation - Evaluation Report")
    report.append("=" * 60)
    report.append("")
    
    report.append("Model: SDXL with LoRA fine-tuning")
    report.append(f"Dataset size: {len(list(IMAGES_DIR.glob('*.jpg')))}")
    report.append("")
    
    report.append("--- Evaluation Metrics ---")
    report.append("")
    
    if fid_score is not None:
        report.append(f"FID Score: {fid_score:.2f}")
        report.append("  (Lower is better - measures distribution similarity)")
        report.append("")
        
        # Interpret FID score
        if fid_score < 30:
            interpretation = "Excellent - Very close to real posters"
        elif fid_score < 50:
            interpretation = "Good - Reasonable similarity to real posters"
        elif fid_score < 100:
            interpretation = "Fair - Noticeable differences from real posters"
        else:
            interpretation = "Poor - Significant differences from real posters"
        
        report.append(f"  Interpretation: {interpretation}")
        report.append("")
    
    if clip_score is not None:
        report.append(f"CLIP Similarity: {clip_score:.4f}")
        report.append("  (Higher is better - measures text-image alignment)")
        report.append("")
        
        # Interpret CLIP score
        if clip_score > 0.3:
            interpretation = "Excellent - Strong text-image alignment"
        elif clip_score > 0.25:
            interpretation = "Good - Reasonable text-image alignment"
        elif clip_score > 0.2:
            interpretation = "Fair - Moderate text-image alignment"
        else:
            interpretation = "Poor - Weak text-image alignment"
        
        report.append(f"  Interpretation: {interpretation}")
        report.append("")
    
    report.append("=" * 60)
    report.append("")
    
    report.append("Notes:")
    report.append("- FID compares the statistical distribution of generated vs real images")
    report.append("- CLIP measures how well images match their text descriptions")
    report.append("- Lower FID and higher CLIP scores indicate better model performance")
    report.append("")
    
    # Write report
    report_text = "\n".join(report)
    print(report_text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n📄 Report saved to {output_path}")


def visualize_samples(real_images_dir, generated_images_dir, num_samples=8, output_path="comparison.png"):
    """
    Create a visual comparison of real vs generated posters
    
    Args:
        real_images_dir: Directory with real posters
        generated_images_dir: Directory with generated posters
        num_samples: Number of samples to show
        output_path: Path to save comparison image
    """
    real_images = sorted(list(Path(real_images_dir).glob("*.jpg")))[:num_samples]
    generated_images = sorted(list(Path(generated_images_dir).glob("*.jpg")))[:num_samples]
    
    if len(real_images) == 0 or len(generated_images) == 0:
        print("⚠️ Not enough images for visualization")
        return
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))
    
    for i in range(num_samples):
        # Real images
        if i < len(real_images):
            img = Image.open(real_images[i])
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title("Real Posters", fontsize=12, fontweight='bold')
        
        # Generated images
        if i < len(generated_images):
            img = Image.open(generated_images[i])
            axes[1, i].imshow(img)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title("Generated Posters", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Comparison saved to {output_path}")


def main():
    """Run full evaluation"""
    print("=" * 60)
    print("Movie Poster Generation - Model Evaluation")
    print("=" * 60)
    
    # Check if directories exist
    if not IMAGES_DIR.exists():
        print("⚠️ Real images directory not found!")
        return
    
    eval_dir = GENERATED_DIR / "eval_set"
    if not eval_dir.exists():
        print("⚠️ Generated images not found!")
        print("Run generate_posters.py first to create evaluation set")
        return
    
    # Calculate FID
    print("\n[1/3] Calculating FID...")
    fid = calculate_fid(IMAGES_DIR, eval_dir)
    
    # Calculate CLIP similarity
    print("\n[2/3] Calculating CLIP similarity...")
    clip_sim = calculate_clip_similarity(eval_dir)
    
    # Generate report
    print("\n[3/3] Generating evaluation report...")
    generate_evaluation_report(fid, clip_sim)
    
    # Create visualization
    print("\nCreating visual comparison...")
    visualize_samples(IMAGES_DIR, eval_dir)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
