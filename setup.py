#!/usr/bin/env python3
"""
Machine Learning Zoomcamp Setup Script
Automated setup for the complete learning environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import argparse

class MLZoomcampSetup:
    def __init__(self):
        """Initialize the setup process"""
        self.repo_path = Path.cwd()
        self.python_version = sys.version_info
        self.os_type = platform.system()
        
        print("🚀 Machine Learning Zoomcamp Setup")
        print("=" * 50)
        print(f"📁 Repository: {self.repo_path}")
        print(f"🐍 Python: {self.python_version.major}.{self.python_version.minor}")
        print(f"💻 OS: {self.os_type}")
        print()
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("🔍 Checking Python version...")
        
        if self.python_version.major < 3 or (self.python_version.major == 3 and self.python_version.minor < 8):
            print("❌ Python 3.8+ is required")
            print("   Please upgrade Python and try again")
            return False
        
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor} is compatible")
        return True
    
    def check_git(self):
        """Check if Git is available"""
        print("\n🔍 Checking Git installation...")
        
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Git is available: {result.stdout.strip()}")
                return True
            else:
                print("❌ Git is not working properly")
                return False
        except FileNotFoundError:
            print("❌ Git is not installed")
            print("   Please install Git and try again")
            return False
    
    def setup_virtual_environment(self):
        """Set up Python virtual environment"""
        print("\n🔧 Setting up virtual environment...")
        
        venv_path = self.repo_path / "venv"
        
        if venv_path.exists():
            print("ℹ️  Virtual environment already exists")
            return True
        
        try:
            # Create virtual environment
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
            print("✅ Virtual environment created")
            
            # Determine activation script path
            if self.os_type == "Windows":
                activate_script = venv_path / "Scripts" / "activate.bat"
                pip_path = venv_path / "Scripts" / "pip"
            else:
                activate_script = venv_path / "bin" / "activate"
                pip_path = venv_path / "bin" / "pip"
            
            print(f"📝 To activate the environment:")
            if self.os_type == "Windows":
                print(f"   {activate_script}")
            else:
                print(f"   source {activate_script}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def install_dependencies(self):
        """Install required Python packages"""
        print("\n📦 Installing dependencies...")
        
        # Check for requirements file
        requirements_file = self.repo_path / "comprehensive-guide" / "notebooks" / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ Requirements file not found")
            print("   Creating basic requirements...")
            self.create_requirements_file()
        
        try:
            # Install packages
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ], check=True)
            
            print("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("   You may need to install packages manually")
            return False
    
    def create_requirements_file(self):
        """Create a basic requirements.txt file"""
        requirements_content = """# Machine Learning Zoomcamp Requirements
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
jupyterlab>=3.0.0
xgboost>=1.5.0
tensorflow>=2.8.0
requests>=2.25.0
plotly>=5.0.0
"""
        
        requirements_file = self.repo_path / "comprehensive-guide" / "notebooks" / "requirements.txt"
        requirements_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        
        print(f"📝 Created requirements file: {requirements_file}")
    
    def download_datasets(self):
        """Download required datasets"""
        print("\n📊 Downloading datasets...")
        
        data_script = self.repo_path / "comprehensive-guide" / "data" / "download_datasets.py"
        
        if not data_script.exists():
            print("❌ Dataset download script not found")
            return False
        
        try:
            # Change to data directory and run script
            original_cwd = os.getcwd()
            os.chdir(data_script.parent)
            
            subprocess.run([sys.executable, 'download_datasets.py'], check=True)
            
            os.chdir(original_cwd)
            print("✅ Datasets downloaded successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download datasets: {e}")
            print("   You can download them manually later")
            return False
        except Exception as e:
            print(f"❌ Error during dataset download: {e}")
            return False
    
    def setup_jupyter(self):
        """Set up Jupyter Lab/Notebook"""
        print("\n📓 Setting up Jupyter...")
        
        try:
            # Check if Jupyter is installed
            result = subprocess.run([sys.executable, '-m', 'jupyter', '--version'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Jupyter is available")
                
                # Try to set up Jupyter Lab extensions (optional)
                try:
                    subprocess.run([sys.executable, '-m', 'jupyter', 'lab', 'build'], 
                                 capture_output=True, check=True)
                    print("✅ Jupyter Lab extensions built")
                except:
                    print("ℹ️  Jupyter Lab extensions not built (optional)")
                
                return True
            else:
                print("❌ Jupyter is not working properly")
                return False
                
        except FileNotFoundError:
            print("❌ Jupyter is not installed")
            return False
    
    def validate_setup(self):
        """Validate the setup by running validation script"""
        print("\n🔍 Validating setup...")
        
        validation_script = self.repo_path / "comprehensive-guide" / "validate_repository.py"
        
        if not validation_script.exists():
            print("❌ Validation script not found")
            return False
        
        try:
            result = subprocess.run([
                sys.executable, str(validation_script), '--quick'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Repository validation passed")
                return True
            else:
                print("⚠️  Repository validation found issues")
                print("   Check the validation report for details")
                return False
                
        except Exception as e:
            print(f"❌ Error during validation: {e}")
            return False
    
    def create_getting_started_guide(self):
        """Create a getting started guide"""
        print("\n📝 Creating getting started guide...")
        
        guide_content = """# 🚀 Getting Started with Machine Learning Zoomcamp

Welcome to your ML learning journey! Here's how to get started:

## 📋 Quick Start Checklist

- [ ] ✅ Environment setup completed
- [ ] 📦 Dependencies installed  
- [ ] 📊 Datasets downloaded
- [ ] 📓 Jupyter Lab working
- [ ] 🔍 Repository validated

## 🎯 Next Steps

### 1. Start Learning
```bash
# Navigate to the first module
cd comprehensive-guide/01-introduction/

# Read the introduction
cat README.md

# Open the first notebook
jupyter lab ../notebooks/01-python-ml-fundamentals.ipynb
```

### 2. Track Your Progress
```bash
# Set up progress tracking
cd comprehensive-guide/progress/
cp learning-checklist.md my-progress.md

# Use the progress dashboard
python progress-dashboard.py report
```

### 3. Practice with Exercises
```bash
# Start with the first exercise
cd comprehensive-guide/exercises/
cat exercise-01-eda-challenge.md
```

## 🆘 Need Help?

- **Technical Issues**: Check the validation report
- **Learning Questions**: Review the reference materials
- **Data Problems**: Re-run the data download script

## 📚 Learning Path

1. **Module 1**: Introduction to ML → `01-introduction/`
2. **Module 2**: Regression → `02-regression/`
3. **Module 3**: Classification → `03-classification/`
4. **Continue**: Follow the module sequence

## 🎉 You're Ready!

Your ML Zoomcamp environment is set up and ready to go!

Happy learning! 🚀
"""
        
        guide_file = self.repo_path / "GETTING_STARTED.md"
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        
        print(f"✅ Getting started guide created: {guide_file}")
    
    def run_setup(self, skip_venv=False, skip_datasets=False):
        """Run the complete setup process"""
        print("🔧 Starting setup process...\n")
        
        success = True
        
        # Check prerequisites
        if not self.check_python_version():
            return False
        
        if not self.check_git():
            print("⚠️  Git not available - some features may not work")
        
        # Set up environment
        if not skip_venv:
            if not self.setup_virtual_environment():
                success = False
        
        # Install dependencies
        if not self.install_dependencies():
            success = False
        
        # Download datasets
        if not skip_datasets:
            if not self.download_datasets():
                success = False
        
        # Set up Jupyter
        if not self.setup_jupyter():
            success = False
        
        # Validate setup
        if not self.validate_setup():
            success = False
        
        # Create getting started guide
        self.create_getting_started_guide()
        
        # Final summary
        print("\n" + "="*50)
        if success:
            print("🎉 SETUP COMPLETED SUCCESSFULLY!")
            print("\n📖 Next steps:")
            print("   1. Read GETTING_STARTED.md")
            print("   2. Start with comprehensive-guide/01-introduction/")
            print("   3. Open your first notebook in Jupyter Lab")
            print("\n🚀 Happy learning!")
        else:
            print("⚠️  SETUP COMPLETED WITH ISSUES")
            print("\n🔧 Some components may need manual setup")
            print("   Check the error messages above")
            print("   You can still proceed with learning")
        
        return success

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Set up ML Zoomcamp environment')
    parser.add_argument('--skip-venv', action='store_true',
                       help='Skip virtual environment setup')
    parser.add_argument('--skip-datasets', action='store_true',
                       help='Skip dataset download')
    parser.add_argument('--quick', action='store_true',
                       help='Quick setup (skip venv and datasets)')
    
    args = parser.parse_args()
    
    setup = MLZoomcampSetup()
    
    if args.quick:
        success = setup.run_setup(skip_venv=True, skip_datasets=True)
    else:
        success = setup.run_setup(skip_venv=args.skip_venv, skip_datasets=args.skip_datasets)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
