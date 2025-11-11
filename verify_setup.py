"""
Setup Verification Script
Run this to check if everything is configured correctly
"""
import sys
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("1. Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False

def check_cuda():
    """Check CUDA availability"""
    print("\n2. Checking CUDA/GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✅ CUDA available")
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {vram:.1f} GB")
            return True
        else:
            print("   ❌ CUDA not available (CPU mode)")
            print("   Training will be very slow without GPU!")
            return False
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def check_dependencies():
    """Check if key dependencies are installed"""
    print("\n3. Checking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'diffusers': 'Diffusers',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'accelerate': 'Accelerate',
        'fastapi': 'FastAPI',
        'PIL': 'Pillow',
    }
    
    all_installed = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} not installed")
            all_installed = False
    
    return all_installed

def check_directories():
    """Check if project directories exist"""
    print("\n4. Checking project structure...")
    
    dirs = {
        'dataset': 'Dataset directory',
        'dataset/images_512': 'Processed images',
        'dataset/captions': 'Captions',
        'src': 'Source code',
        'frontend': 'Frontend',
        'models': 'Models directory',
        'checkpoints': 'Checkpoints directory',
        'generated': 'Generated images',
    }
    
    all_exist = True
    for dir_path, name in dirs.items():
        if Path(dir_path).exists():
            print(f"   ✅ {name}")
        else:
            print(f"   ⚠️  {name} (will be created)")
            all_exist = False
    
    return all_exist

def check_dataset():
    """Check if dataset is ready"""
    print("\n5. Checking dataset...")
    
    raw_dir = Path("dataset/raw")
    images_dir = Path("dataset/images_512")
    captions_dir = Path("dataset/captions")
    
    # Check raw dataset
    if raw_dir.exists():
        raw_files = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png"))
        if raw_files:
            print(f"   ✅ Raw dataset: {len(raw_files)} images")
        else:
            print("   ⚠️  Raw dataset directory empty")
            print("      Download from: https://www.kaggle.com/datasets/phiitm/movie-posters")
            return False
    else:
        print("   ❌ Raw dataset not found")
        print("      Create 'dataset/raw/' and download images")
        return False
    
    # Check processed dataset
    if images_dir.exists():
        processed_files = list(images_dir.glob("*.jpg"))
        if len(processed_files) >= 100:
            print(f"   ✅ Processed images: {len(processed_files)}")
        else:
            print(f"   ⚠️  Only {len(processed_files)} processed images")
            print("      Run: python src/prepare_dataset.py")
            return False
    else:
        print("   ⚠️  Processed images not found")
        print("      Run: python src/prepare_dataset.py")
        return False
    
    # Check captions
    if captions_dir.exists():
        caption_files = list(captions_dir.glob("*.txt"))
        if len(caption_files) >= 100:
            print(f"   ✅ Captions: {len(caption_files)}")
        else:
            print(f"   ⚠️  Only {len(caption_files)} captions")
            return False
    else:
        print("   ⚠️  Captions not found")
        return False
    
    return True

def check_model():
    """Check if model is trained"""
    print("\n6. Checking trained model...")
    
    model_dir = Path("models/sdxl_lora_movie_posters")
    
    if model_dir.exists() and list(model_dir.glob("*")):
        print("   ✅ LoRA model found")
        return True
    else:
        print("   ⚠️  LoRA model not trained yet")
        print("      Run: python src/train_sdxl_lora.py")
        print("      (This takes 8-12 hours)")
        return False

def check_files():
    """Check if key files exist"""
    print("\n7. Checking key files...")
    
    files = {
        'config.py': 'Configuration',
        'requirements.txt': 'Requirements',
        'src/prepare_dataset.py': 'Dataset preparation',
        'src/train_sdxl_lora.py': 'Training script',
        'src/generate_posters.py': 'Generation script',
        'src/evaluate_model.py': 'Evaluation script',
        'src/chatbot_api.py': 'Chatbot API',
        'frontend/index.html': 'Web interface',
    }
    
    all_exist = True
    for file_path, name in files.items():
        if Path(file_path).exists():
            print(f"   ✅ {name}")
        else:
            print(f"   ❌ {name} missing!")
            all_exist = False
    
    return all_exist

def print_next_steps(results):
    """Print recommended next steps"""
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    
    if not results['dependencies']:
        print("\n1. Install dependencies:")
        print("   pip install -r requirements.txt")
    
    if not results['dataset']:
        print("\n2. Prepare dataset:")
        print("   - Download from Kaggle (see README)")
        print("   - Extract to dataset/raw/")
        print("   - Run: python src/prepare_dataset.py")
    
    if not results['model']:
        print("\n3. Train model:")
        print("   python src/train_sdxl_lora.py")
        print("   (Takes 8-12 hours)")
    
    if results['dataset'] and not results['model']:
        print("\n4. Or test with base model (no training):")
        print("   python -c \"from src.generate_posters import PosterGenerator; ")
        print("   g = PosterGenerator(use_base_model=True); ")
        print("   g.generate('action movie poster').save('test.jpg')\"")
    
    if results['model']:
        print("\n✅ Ready to use!")
        print("\nGenerate evaluation set:")
        print("   python -c \"from src.generate_posters import generate_evaluation_set; generate_evaluation_set(100)\"")
        print("\nRun evaluation:")
        print("   python src/evaluate_model.py")
        print("\nStart chatbot:")
        print("   python src/chatbot_api.py")
        print("   (Then open frontend/index.html)")

def main():
    """Run all checks"""
    print("="*60)
    print("MOVIE CHATBOT PROJECT - SETUP VERIFICATION")
    print("="*60)
    
    results = {
        'python': check_python_version(),
        'cuda': check_cuda(),
        'dependencies': check_dependencies(),
        'directories': check_directories(),
        'dataset': check_dataset(),
        'model': check_model(),
        'files': check_files(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    for key, value in results.items():
        status = "✅" if value else "⚠️"
        print(f"{status} {key.capitalize()}")
    
    # Overall status
    critical_checks = ['python', 'dependencies', 'files']
    if all(results[key] for key in critical_checks):
        print("\n✅ System ready for setup!")
    else:
        print("\n⚠️  Some critical components missing")
    
    print_next_steps(results)
    
    print("\n" + "="*60)
    print("For detailed instructions, see:")
    print("  - README.md (comprehensive guide)")
    print("  - QUICKSTART.md (quick setup)")
    print("  - PROJECT_SUMMARY.md (overview)")
    print("="*60)

if __name__ == "__main__":
    main()
