#!/usr/bin/env python3
"""
Machine Psychology Lab - Installation Script
============================================

Automated installation using conda-forge for maximum reliability.
This script handles the conda environment creation and package installation.

Usage:
    python install.py

Requirements:
    - conda or miniconda installed
    - Internet connection for package downloads
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors gracefully"""
    print(f"\n{'='*60}")
    if description:
        print(f"🔧 {description}")
    print(f"Running: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Error: {e}")
        return False

def check_conda():
    """Check if conda is available"""
    try:
        subprocess.run(["conda", "--version"], check=True, 
                      capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    print("🧠 Machine Psychology Lab - Installation Script")
    print("=" * 60)
    
    # Check if conda is available
    if not check_conda():
        print("❌ Error: conda not found!")
        print("\nPlease install conda or miniconda first:")
        print("https://docs.conda.io/en/latest/miniconda.html")
        sys.exit(1)
    
    print("✅ conda found")
    
    # Create environment
    if not run_command(
        "conda create -n machine-psychology python=3.11 -y",
        "Creating conda environment with Python 3.11"
    ):
        print("❌ Failed to create conda environment")
        sys.exit(1)
    
    # Install core scientific packages
    if not run_command(
        "conda install -n machine-psychology -c conda-forge spacy nltk pandas numpy scipy matplotlib scikit-learn -y",
        "Installing core scientific packages"
    ):
        print("❌ Failed to install core packages")
        sys.exit(1)
    
    # Install AI/ML packages
    if not run_command(
        "conda install -n machine-psychology -c conda-forge transformers sentence-transformers torch torchvision torchaudio -y",
        "Installing AI/ML packages (this may take a while...)"
    ):
        print("❌ Failed to install AI/ML packages")
        sys.exit(1)
    
    # Install additional NLP packages
    if not run_command(
        "conda install -n machine-psychology -c conda-forge textblob vadersentiment requests pyyaml tqdm -y",
        "Installing additional NLP packages"
    ):
        print("❌ Failed to install NLP packages")
        sys.exit(1)
    
    # Download spaCy model
    if not run_command(
        "conda run -n machine-psychology python -m spacy download en_core_web_sm",
        "Downloading spaCy English model"
    ):
        print("❌ Failed to download spaCy model")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 Installation completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Activate the environment:")
    print("   conda activate machine-psychology")
    print("\n2. Download AI models:")
    print("   python setup_models.py")
    print("\n3. Run the application:")
    print("   python machine_psychology_lab_1.py")
    print("\n4. Go to Settings → Analysis Engine Configuration to select your analysis engine")
    print("\nFor troubleshooting, see SETUP_GUIDE.md")

if __name__ == "__main__":
    main()
