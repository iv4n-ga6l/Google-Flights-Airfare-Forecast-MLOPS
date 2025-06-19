#!/usr/bin/env python3
"""
Setup script for the Google Flights Airfare Forecast system
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed: {e}")
        print(f"   📝 Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   📝 Please use Python 3.8 or higher")
        return False


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "models",
        "data",
        "logs",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created {directory}/ directory")
    
    return True


def check_data_file():
    """Check if the data file exists"""
    print("📊 Checking data file...")
    
    data_file = Path("data/google_flights_airfare_data.csv")
    if data_file.exists():
        print(f"   ✅ Data file found: {data_file}")
        
        # Check file size
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"   📏 File size: {size_mb:.1f} MB")
        
        return True
    else:
        print(f"   ⚠️  Data file not found: {data_file}")
        print(f"   📝 Please ensure the CSV file is in the data/ directory")
        return False


def train_initial_model():
    """Train the initial model"""
    print("🤖 Training initial model...")
    
    data_file = Path("data/google_flights_airfare_data.csv")
    if not data_file.exists():
        print("   ⚠️  Skipping model training - data file not found")
        return True
    
    # Train with basic settings first
    command = f"{sys.executable} train_model.py --model-name xgboost --log-level INFO"
    
    if run_command(command, "Training XGBoost model"):
        print("   🎉 Initial model training completed!")
        return True
    else:
        print("   ⚠️  Model training failed, but setup can continue")
        return True  # Don't fail setup if training fails


def run_tests():
    """Run basic tests"""
    print("🧪 Running basic tests...")
    
    # Test imports
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import fastapi
        import uvicorn
        print("   ✅ All required packages imported successfully")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    return True


def display_next_steps():
    """Display next steps for the user"""
    print("\n🎉 Setup completed successfully!")
    print("=" * 50)
    print("📋 Next steps:")
    print()
    print("1. 🤖 Train a model (if not already done):")
    print("   python train_model.py --model-name xgboost --tune-hyperparameters")
    print()
    print("2. 🚀 Start the API server:")
    print("   python start_server.py")
    print()
    print("3. 📚 View API documentation:")
    print("   http://localhost:8000/docs")
    print()
    print("4. 🧪 Run the demo:")
    print("   python demo.py")
    print()
    print("5. 🧪 Run tests:")
    print("   pytest tests/")
    print()
    print("📁 Important directories:")
    print("   - data/     : Training data")
    print("   - models/   : Trained models")
    print("   - logs/     : Application logs")
    print("   - tests/    : Test files")
    print()
    print("🔧 Configuration:")
    print("   - Edit requirements.txt to add/remove dependencies")
    print("   - Modify app/main.py to customize the API")
    print("   - Adjust pipelines/ for custom ML workflows")


def main():
    """Main setup function"""
    print("🛠️  Google Flights Airfare Forecast - Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_python_version():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check data
    data_available = check_data_file()
    
    # Run tests
    if not run_tests():
        return False
    
    # Train model if data is available
    if data_available:
        train_initial_model()
    
    # Display next steps
    display_next_steps()
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)
