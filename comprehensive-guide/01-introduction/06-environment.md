# 🛠️ Environment Setup

> **Set up a professional machine learning development environment**

A well-configured development environment is crucial for productive machine learning work. This guide will help you set up everything you need to follow along with the course and build real ML projects.

## 🎯 Learning Objectives

By the end of this guide, you will have:
- **Python 3.8+** installed and configured
- **Virtual environment** set up for isolation
- **Essential ML libraries** installed and tested
- **Jupyter Lab** running for interactive development
- **Git** configured for version control
- **IDE/Editor** set up for coding

## 🐍 Python Installation

### **Option 1: Anaconda (Recommended for Beginners)**

Anaconda includes Python, Jupyter, and many ML libraries pre-installed.

```bash
# Download Anaconda from https://www.anaconda.com/products/distribution
# Follow the installation instructions for your operating system

# Verify installation
conda --version
python --version
```

### **Option 2: Miniconda (Lightweight)**

Miniconda provides just Python and conda package manager.

```bash
# Download Miniconda from https://docs.conda.io/en/latest/miniconda.html
# Install following the instructions

# Verify installation
conda --version
python --version
```

### **Option 3: System Python (Advanced Users)**

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# macOS (using Homebrew)
brew install python

# Windows
# Download from https://www.python.org/downloads/
```

## 🏠 Virtual Environment Setup

Virtual environments isolate your project dependencies and prevent conflicts.

### **Using Conda (Recommended)**

```bash
# Create new environment
conda create -n ml-zoomcamp python=3.9

# Activate environment
conda activate ml-zoomcamp

# Verify activation
which python
python --version
```

### **Using venv (Built-in)**

```bash
# Create virtual environment
python -m venv ml-env

# Activate environment
# On Linux/macOS:
source ml-env/bin/activate
# On Windows:
ml-env\Scripts\activate

# Verify activation
which python
python --version
```

### **Environment Management Best Practices**

```bash
# Always activate your environment before working
conda activate ml-zoomcamp  # or source ml-env/bin/activate

# Deactivate when done
conda deactivate  # or deactivate

# List environments
conda env list

# Remove environment (if needed)
conda env remove -n ml-zoomcamp
```

## 📦 Package Installation

### **Core ML Libraries**

```bash
# Essential packages
pip install numpy pandas scikit-learn matplotlib seaborn

# Jupyter environment
pip install jupyter jupyterlab ipywidgets

# Deep learning (choose one)
pip install tensorflow  # TensorFlow
# OR
pip install torch torchvision  # PyTorch

# Additional useful packages
pip install plotly xgboost lightgbm
```

### **Using Requirements File**

```bash
# Install from requirements.txt (if available)
pip install -r requirements.txt

# Create your own requirements file
pip freeze > my-requirements.txt
```

### **Package Versions for Compatibility**

```bash
# Specific versions for reproducibility
pip install numpy==1.21.0 pandas==1.3.0 scikit-learn==1.0.0
```

## 📓 Jupyter Lab Setup

### **Installation and Basic Usage**

```bash
# Install Jupyter Lab
pip install jupyterlab

# Start Jupyter Lab
jupyter lab

# Access at http://localhost:8888
```

### **Useful Jupyter Extensions**

```bash
# Install extensions
pip install jupyter-contrib-nbextensions
jupyter contrib nbextension install --user

# Enable useful extensions
jupyter nbextension enable --py widgetsnbextension
```

### **Jupyter Configuration**

```python
# Create Jupyter config
jupyter lab --generate-config

# Common configurations in ~/.jupyter/jupyter_lab_config.py
c.ServerApp.open_browser = False  # Don't auto-open browser
c.ServerApp.port = 8888  # Default port
c.ServerApp.notebook_dir = '/path/to/your/projects'  # Default directory
```

## 🔧 IDE/Editor Setup

### **VS Code (Recommended)**

```bash
# Install VS Code from https://code.visualstudio.com/

# Essential extensions:
# - Python
# - Jupyter
# - Python Docstring Generator
# - GitLens
```

**VS Code Configuration for ML:**

```json
// settings.json
{
    "python.defaultInterpreterPath": "/path/to/your/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "jupyter.askForKernelRestart": false,
    "files.autoSave": "afterDelay"
}
```

### **PyCharm (Alternative)**

```bash
# Download PyCharm from https://www.jetbrains.com/pycharm/
# Configure Python interpreter to use your virtual environment
```

## 🔄 Git Setup

### **Installation**

```bash
# Ubuntu/Debian
sudo apt install git

# macOS
brew install git

# Windows
# Download from https://git-scm.com/
```

### **Configuration**

```bash
# Set up your identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Useful aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
```

### **Project Setup**

```bash
# Initialize repository
git init

# Create .gitignore for Python/ML projects
echo "*.pyc
__pycache__/
.env
.venv/
.DS_Store
.jupyter/
*.log
data/raw/
models/
.ipynb_checkpoints/" > .gitignore

# First commit
git add .
git commit -m "Initial commit"
```

## 🧪 Environment Testing

### **Test Script**

Create a test script to verify your setup:

```python
# test_environment.py
import sys
print(f"Python version: {sys.version}")

# Test core libraries
try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except ImportError:
    print("❌ NumPy not installed")

try:
    import pandas as pd
    print(f"✅ Pandas {pd.__version__}")
except ImportError:
    print("❌ Pandas not installed")

try:
    import sklearn
    print(f"✅ Scikit-learn {sklearn.__version__}")
except ImportError:
    print("❌ Scikit-learn not installed")

try:
    import matplotlib
    print(f"✅ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("❌ Matplotlib not installed")

try:
    import seaborn as sns
    print(f"✅ Seaborn {sns.__version__}")
except ImportError:
    print("❌ Seaborn not installed")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__}")
except ImportError:
    print("⚠️ TensorFlow not installed (optional)")

# Test basic functionality
print("\n🧪 Testing basic functionality:")
arr = np.array([1, 2, 3, 4, 5])
print(f"NumPy array: {arr}")
print(f"Mean: {np.mean(arr)}")

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(f"Pandas DataFrame:\n{df}")

print("\n🎉 Environment setup successful!")
```

Run the test:

```bash
python test_environment.py
```

## 📁 Project Structure

### **Recommended Directory Structure**

```
ml-project/
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and processed data
│   └── external/     # External datasets
├── notebooks/        # Jupyter notebooks
│   ├── exploratory/  # EDA and experiments
│   └── final/        # Final analysis notebooks
├── src/             # Source code
│   ├── data/        # Data processing scripts
│   ├── features/    # Feature engineering
│   ├── models/      # Model training and prediction
│   └── visualization/ # Plotting scripts
├── models/          # Trained models
├── reports/         # Generated reports and figures
├── requirements.txt # Package dependencies
├── README.md       # Project description
└── .gitignore      # Git ignore file
```

### **Create Project Template**

```bash
# Create project structure
mkdir -p ml-project/{data/{raw,processed,external},notebooks/{exploratory,final},src/{data,features,models,visualization},models,reports}

cd ml-project

# Create basic files
touch README.md requirements.txt .gitignore
touch src/__init__.py src/data/__init__.py src/features/__init__.py src/models/__init__.py src/visualization/__init__.py
```

## 🔧 Troubleshooting

### **Common Issues**

#### **Import Errors**
```bash
# Check if package is installed
pip list | grep numpy

# Reinstall if needed
pip uninstall numpy
pip install numpy
```

#### **Jupyter Kernel Issues**
```bash
# Install kernel for your environment
python -m ipykernel install --user --name=ml-zoomcamp

# List available kernels
jupyter kernelspec list

# Remove kernel if needed
jupyter kernelspec uninstall ml-zoomcamp
```

#### **Permission Errors**
```bash
# Use --user flag for user-level installation
pip install --user package_name

# Or fix permissions (Linux/macOS)
sudo chown -R $USER /path/to/python/site-packages
```

### **Performance Optimization**

```bash
# Install optimized BLAS libraries
conda install mkl

# For faster pandas operations
pip install numexpr bottleneck

# For parallel processing
pip install joblib
```

## ✅ Environment Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Core ML libraries installed (numpy, pandas, scikit-learn)
- [ ] Jupyter Lab working
- [ ] Git configured
- [ ] IDE/Editor set up
- [ ] Test script runs successfully
- [ ] Project structure created

## 🚀 Next Steps

With your environment set up, you're ready to:
1. **Start with NumPy fundamentals** - [NumPy Guide](07-numpy.md)
2. **Explore the first notebook** - [Python ML Fundamentals](../notebooks/01-python-ml-fundamentals.ipynb)
3. **Begin your first project** - [Regression Module](../02-regression/)

## 📚 Additional Resources

- **Anaconda Documentation**: [Getting Started](https://docs.anaconda.com/anaconda/user-guide/getting-started/)
- **Virtual Environments**: [Python Guide](https://docs.python.org/3/tutorial/venv.html)
- **Jupyter Lab**: [User Guide](https://jupyterlab.readthedocs.io/en/stable/user/interface.html)

---

**Navigation:**
- **Previous**: [Model Selection Process](05-model-selection.md)
- **Next**: [NumPy Fundamentals](07-numpy.md)
- **Module Home**: [Introduction](README.md)

*Last Updated: 2025-01-27*
