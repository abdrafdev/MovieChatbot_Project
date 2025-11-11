"""
Dataset preparation script for movie posters
Downloads from Kaggle, resizes to 512x512, generates captions
"""
import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import IMAGES_DIR, CAPTIONS_DIR, IMAGE_SIZE, DATASET_SIZE


def download_kaggle_dataset():
    """
    Download movie posters dataset from Kaggle
    Requires kaggle.json in ~/.kaggle/ with API credentials
    """
    print("Downloading dataset from Kaggle...")
    print("Make sure you have kaggle.json configured in ~/.kaggle/")
    print("\nRun: kaggle datasets download -d phiitm/movie-posters")
    print("Then extract to dataset/raw/")
    print("\nAlternatively, manually download from:")
    print("https://www.kaggle.com/datasets/phiitm/movie-posters")


def resize_images(input_dir, output_dir, target_size=512):
    """
    Resize all images to target_size x target_size
    
    Args:
        input_dir: Directory with original images
        output_dir: Directory to save resized images
        target_size: Target image dimension (default 512)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpeg"))
    
    print(f"\nResizing {len(image_files)} images to {target_size}x{target_size}...")
    
    resized_count = 0
    for img_file in tqdm(image_files[:DATASET_SIZE]):
        try:
            img = Image.open(img_file)
            
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize with high-quality Lanczos resampling
            img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Save with new name
            output_file = output_path / f"poster_{resized_count:05d}.jpg"
            img_resized.save(output_file, "JPEG", quality=95)
            
            resized_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_file}: {e}")
            continue
    
    print(f"\nSuccessfully resized {resized_count} images")
    return resized_count


def generate_captions(metadata_file, num_images, captions_dir):
    """
    Generate caption text files from metadata
    
    Args:
        metadata_file: Path to CSV with movie metadata
        num_images: Number of captions to generate
        captions_dir: Directory to save caption files
    """
    captions_path = Path(captions_dir)
    captions_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating captions...")
    
    # If metadata file exists, use it
    if Path(metadata_file).exists():
        df = pd.read_csv(metadata_file)
        print(f"Loaded metadata with {len(df)} entries")
        
        for i in tqdm(range(min(num_images, len(df)))):
            row = df.iloc[i]
            
            # Generate caption from available metadata
            caption_parts = []
            
            if 'title' in df.columns and pd.notna(row.get('title')):
                caption_parts.append(f"Movie poster for '{row['title']}'")
            
            if 'genres' in df.columns and pd.notna(row.get('genres')):
                genres = row['genres']
                caption_parts.append(f"Genre: {genres}")
            
            if 'overview' in df.columns and pd.notna(row.get('overview')):
                overview = row['overview'][:200]  # Limit length
                caption_parts.append(overview)
            
            # Default caption if no metadata
            if not caption_parts:
                caption_parts = [f"Movie poster with dramatic composition and cinematic style"]
            
            caption = ". ".join(caption_parts)
            
            # Save caption file
            caption_file = captions_path / f"poster_{i:05d}.txt"
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(caption)
    
    else:
        # Generate generic captions if no metadata available
        print("No metadata file found, generating generic captions...")
        
        genres = ["action", "sci-fi", "horror", "comedy", "drama", "fantasy", "thriller", "romance"]
        styles = ["dramatic lighting", "bold colors", "artistic composition", "cinematic style", 
                 "professional design", "compelling imagery", "striking visuals"]
        
        for i in tqdm(range(num_images)):
            genre = genres[i % len(genres)]
            style = styles[i % len(styles)]
            
            caption = f"Movie poster with {genre} theme, featuring {style}, high quality design"
            
            caption_file = captions_path / f"poster_{i:05d}.txt"
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(caption)
    
    print(f"Generated {num_images} caption files")


def main():
    """Main dataset preparation pipeline"""
    
    print("=" * 60)
    print("Movie Poster Dataset Preparation")
    print("=" * 60)
    
    # Step 1: Instructions for downloading
    print("\n[Step 1] Download Dataset")
    download_kaggle_dataset()
    
    # Check if raw data exists
    raw_data_dir = Path("dataset/raw")
    if not raw_data_dir.exists():
        print(f"\n⚠️  Please download and extract the dataset to {raw_data_dir}")
        print("After downloading, run this script again.")
        return
    
    # Check for poster_downloads subdirectory
    poster_dir = raw_data_dir / "poster_downloads"
    if poster_dir.exists():
        raw_data_dir = poster_dir
        print(f"Found images in: {raw_data_dir}")
    
    # Step 2: Resize images
    print("\n[Step 2] Resize Images")
    num_images = resize_images(raw_data_dir, IMAGES_DIR, IMAGE_SIZE)
    
    if num_images == 0:
        print("⚠️  No images found. Check your dataset directory.")
        return
    
    # Step 3: Generate captions
    print("\n[Step 3] Generate Captions")
    metadata_file = raw_data_dir / "metadata.csv"
    generate_captions(metadata_file, num_images, CAPTIONS_DIR)
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print(f"   Images: {IMAGES_DIR}")
    print(f"   Captions: {CAPTIONS_DIR}")
    print(f"   Total samples: {num_images}")
    print("=" * 60)


if __name__ == "__main__":
    main()
