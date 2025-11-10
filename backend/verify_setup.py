"""
Verification script to check if all dependencies are installed correctly.
Run this after installing requirements.txt to verify your setup.
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name or module_name} is NOT installed")
        return False

def main():
    print("="*70)
    print("Verifying Project Setup")
    print("="*70)
    
    print("\nChecking Python version...")
    print(f"Python version: {sys.version}")
    
    print("\nChecking required packages...")
    packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    
    all_ok = True
    for module, name in packages:
        if not check_import(module, name):
            all_ok = False
    
    print("\nChecking optional packages...")
    optional = [
        ('flask', 'Flask (optional)'),
    ]
    
    for module, name in optional:
        check_import(module, name)
    
    print("\nChecking project modules...")
    try:
        from models.clv_model import get_model
        print("✓ models.clv_model can be imported")
    except ImportError as e:
        print(f"✗ Error importing models.clv_model: {e}")
        all_ok = False
    
    try:
        from data.data_loader import CLVDataProcessor
        print("✓ data.data_loader can be imported")
    except ImportError as e:
        print(f"✗ Error importing data.data_loader: {e}")
        all_ok = False
    
    try:
        from utils.trainer import CLVTrainer
        print("✓ utils.trainer can be imported")
    except ImportError as e:
        print(f"✗ Error importing utils.trainer: {e}")
        all_ok = False
    
    try:
        from utils.evaluator import CLVEvaluator
        print("✓ utils.evaluator can be imported")
    except ImportError as e:
        print(f"✗ Error importing utils.evaluator: {e}")
        all_ok = False
    
    print("\n" + "="*70)
    if all_ok:
        print("✓ All required components are installed and working!")
        print("You can now run the training and inference scripts.")
    else:
        print("✗ Some components are missing. Please install requirements:")
        print("  pip install -r requirements.txt")
    print("="*70)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())

