"""
Download the movie posters dataset from Kaggle.
Requires Kaggle API credentials to be configured.
"""
import os
import zipfile
from pathlib import Path

def setup_kaggle_credentials():
    """Guide user through Kaggle API setup."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print("✅ Kaggle credentials found!")
        return True
    
    print("❌ Kaggle API credentials not found.")
    print("\nTo download the dataset, you need to:")
    print("1. Go to https://www.kaggle.com/settings/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New Token' (downloads kaggle.json)")
    print(f"4. Move kaggle.json to: {kaggle_dir}")
    print("\nAlternatively, you can:")
    print("- Download manually from: https://www.kaggle.com/datasets/phiitm/movie-posters")
    print("- Extract the zip file to: dataset/raw/")
    return False

def download_dataset():
    """Download and extract the movie posters dataset."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("Initializing Kaggle API...")
        api = KaggleApi()
        api.authenticate()
        
        print("Downloading movie-posters dataset...")
        dataset_name = "phiitm/movie-posters"
        download_path = "."
        
        api.dataset_download_files(dataset_name, path=download_path, unzip=False)
        
        zip_file = "movie-posters.zip"
        if os.path.exists(zip_file):
            print(f"Extracting {zip_file} to dataset/raw/...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall("dataset/raw")
            
            # Clean up zip file
            os.remove(zip_file)
            print("✅ Dataset downloaded and extracted successfully!")
            return True
        else:
            print("❌ Download failed - zip file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nIf you see authentication errors, follow these steps:")
        setup_kaggle_credentials()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("KAGGLE DATASET DOWNLOADER")
    print("=" * 60)
    
    if not setup_kaggle_credentials():
        print("\n⚠️  Cannot proceed without Kaggle credentials.")
        print("Please configure your API key and run this script again,")
        print("or download the dataset manually.")
    else:
        download_dataset()
