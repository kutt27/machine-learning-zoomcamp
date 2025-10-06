#!/usr/bin/env python3
"""
Dataset Download Script for Machine Learning Zoomcamp
Downloads all datasets used in the course materials and exercises.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_boston, load_iris, load_wine, load_breast_cancer,
    fetch_california_housing, make_classification, make_regression
)
import requests
from pathlib import Path
import zipfile
import warnings
warnings.filterwarnings('ignore')

class DatasetDownloader:
    def __init__(self, data_dir="./"):
        """Initialize the dataset downloader"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "external").mkdir(exist_ok=True)
        
        print(f"Data directory: {self.data_dir.absolute()}")
    
    def download_sklearn_datasets(self):
        """Download built-in sklearn datasets"""
        print("\n=== DOWNLOADING SKLEARN DATASETS ===")
        
        datasets = {
            'boston_housing': load_boston,
            'iris': load_iris,
            'wine': load_wine,
            'breast_cancer': load_breast_cancer,
            'california_housing': fetch_california_housing
        }
        
        for name, loader in datasets.items():
            try:
                print(f"Downloading {name}...")
                
                if name == 'california_housing':
                    data = loader()
                else:
                    data = loader()
                
                # Create DataFrame
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                
                # Save to CSV
                filepath = self.data_dir / "raw" / f"{name}.csv"
                df.to_csv(filepath, index=False)
                
                # Save metadata
                metadata = {
                    'description': data.DESCR,
                    'shape': df.shape,
                    'features': list(data.feature_names),
                    'target_names': getattr(data, 'target_names', None)
                }
                
                with open(self.data_dir / "raw" / f"{name}_metadata.txt", 'w') as f:
                    f.write(f"Dataset: {name}\n")
                    f.write(f"Shape: {metadata['shape']}\n")
                    f.write(f"Features: {metadata['features']}\n")
                    f.write(f"Target names: {metadata['target_names']}\n\n")
                    f.write("Description:\n")
                    f.write(metadata['description'])
                
                print(f"  ✓ Saved {name}.csv ({df.shape[0]} rows, {df.shape[1]} columns)")
                
            except Exception as e:
                print(f"  ✗ Error downloading {name}: {e}")
    
    def download_external_datasets(self):
        """Download external datasets from URLs"""
        print("\n=== DOWNLOADING EXTERNAL DATASETS ===")
        
        external_datasets = {
            'car_data': {
                'url': 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv',
                'description': 'Car price dataset for regression exercises'
            },
            'telco_churn': {
                'url': 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv',
                'description': 'Telco customer churn dataset for classification'
            },
            'titanic': {
                'url': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
                'description': 'Titanic dataset for classification exercises'
            }
        }
        
        for name, info in external_datasets.items():
            try:
                print(f"Downloading {name}...")
                
                response = requests.get(info['url'], timeout=30)
                response.raise_for_status()
                
                # Save raw data
                filepath = self.data_dir / "external" / f"{name}.csv"
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                # Load and analyze
                df = pd.read_csv(filepath)
                
                # Save metadata
                with open(self.data_dir / "external" / f"{name}_metadata.txt", 'w') as f:
                    f.write(f"Dataset: {name}\n")
                    f.write(f"Source: {info['url']}\n")
                    f.write(f"Description: {info['description']}\n")
                    f.write(f"Shape: {df.shape}\n")
                    f.write(f"Columns: {list(df.columns)}\n")
                    f.write(f"Data types:\n{df.dtypes.to_string()}\n")
                    f.write(f"Missing values:\n{df.isnull().sum().to_string()}\n")
                
                print(f"  ✓ Saved {name}.csv ({df.shape[0]} rows, {df.shape[1]} columns)")
                
            except Exception as e:
                print(f"  ✗ Error downloading {name}: {e}")
    
    def create_synthetic_datasets(self):
        """Create synthetic datasets for exercises"""
        print("\n=== CREATING SYNTHETIC DATASETS ===")
        
        # Classification dataset
        try:
            print("Creating synthetic classification dataset...")
            X_class, y_class = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=10,
                n_redundant=5,
                n_clusters_per_class=1,
                random_state=42
            )
            
            df_class = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(X_class.shape[1])])
            df_class['target'] = y_class
            
            filepath = self.data_dir / "raw" / "synthetic_classification.csv"
            df_class.to_csv(filepath, index=False)
            print(f"  ✓ Saved synthetic_classification.csv ({df_class.shape[0]} rows, {df_class.shape[1]} columns)")
            
        except Exception as e:
            print(f"  ✗ Error creating classification dataset: {e}")
        
        # Regression dataset
        try:
            print("Creating synthetic regression dataset...")
            X_reg, y_reg = make_regression(
                n_samples=1000,
                n_features=15,
                n_informative=10,
                noise=0.1,
                random_state=42
            )
            
            df_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(X_reg.shape[1])])
            df_reg['target'] = y_reg
            
            filepath = self.data_dir / "raw" / "synthetic_regression.csv"
            df_reg.to_csv(filepath, index=False)
            print(f"  ✓ Saved synthetic_regression.csv ({df_reg.shape[0]} rows, {df_reg.shape[1]} columns)")
            
        except Exception as e:
            print(f"  ✗ Error creating regression dataset: {e}")
    
    def create_sample_datasets(self):
        """Create small sample datasets for quick testing"""
        print("\n=== CREATING SAMPLE DATASETS ===")
        
        # Simple linear regression data
        np.random.seed(42)
        n_samples = 100
        X_simple = np.random.randn(n_samples, 1)
        y_simple = 3 * X_simple.flatten() + 2 + 0.1 * np.random.randn(n_samples)
        
        df_simple = pd.DataFrame({
            'feature': X_simple.flatten(),
            'target': y_simple
        })
        
        filepath = self.data_dir / "raw" / "simple_linear.csv"
        df_simple.to_csv(filepath, index=False)
        print(f"  ✓ Saved simple_linear.csv ({df_simple.shape[0]} rows, {df_simple.shape[1]} columns)")
        
        # Time series data
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        trend = np.linspace(100, 200, 365)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)
        noise = np.random.normal(0, 5, 365)
        values = trend + seasonal + noise
        
        df_ts = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        filepath = self.data_dir / "raw" / "time_series_sample.csv"
        df_ts.to_csv(filepath, index=False)
        print(f"  ✓ Saved time_series_sample.csv ({df_ts.shape[0]} rows, {df_ts.shape[1]} columns)")
    
    def create_data_summary(self):
        """Create a summary of all available datasets"""
        print("\n=== CREATING DATA SUMMARY ===")
        
        summary_data = []
        
        # Check all CSV files
        for subdir in ['raw', 'external']:
            data_path = self.data_dir / subdir
            if data_path.exists():
                for csv_file in data_path.glob('*.csv'):
                    try:
                        df = pd.read_csv(csv_file)
                        summary_data.append({
                            'Dataset': csv_file.stem,
                            'Location': subdir,
                            'Rows': df.shape[0],
                            'Columns': df.shape[1],
                            'Size_MB': round(csv_file.stat().st_size / 1024 / 1024, 2),
                            'Has_Target': 'target' in df.columns,
                            'Missing_Values': df.isnull().sum().sum(),
                            'Numeric_Columns': len(df.select_dtypes(include=[np.number]).columns),
                            'Categorical_Columns': len(df.select_dtypes(include=['object']).columns)
                        })
                    except Exception as e:
                        print(f"  Warning: Could not analyze {csv_file}: {e}")
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values(['Location', 'Dataset'])
            
            # Save summary
            summary_path = self.data_dir / "dataset_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            
            print(f"  ✓ Created dataset summary with {len(summary_df)} datasets")
            print("\nDataset Summary:")
            print(summary_df.to_string(index=False))
            
            # Create README
            readme_content = self.create_data_readme(summary_df)
            with open(self.data_dir / "README.md", 'w') as f:
                f.write(readme_content)
            print(f"  ✓ Created data README.md")
        
        else:
            print("  ✗ No datasets found to summarize")
    
    def create_data_readme(self, summary_df):
        """Create README content for the data directory"""
        readme = """# 📊 Machine Learning Datasets

This directory contains all datasets used in the Machine Learning Zoomcamp course.

## 📁 Directory Structure

- **`raw/`**: Built-in sklearn datasets and synthetic data
- **`external/`**: Downloaded datasets from external sources
- **`processed/`**: Cleaned and preprocessed datasets (created during exercises)

## 📋 Available Datasets

"""
        
        # Add dataset table
        readme += summary_df.to_markdown(index=False)
        
        readme += """

## 🚀 Quick Start

### Download All Datasets
```bash
python download_datasets.py
```

### Load a Dataset
```python
import pandas as pd

# Load any dataset
df = pd.read_csv('raw/boston_housing.csv')
print(df.head())
```

## 📊 Dataset Descriptions

### Built-in Datasets (raw/)
- **boston_housing**: Boston house prices (regression)
- **california_housing**: California house prices (regression)
- **iris**: Iris flower classification
- **wine**: Wine classification
- **breast_cancer**: Breast cancer classification
- **synthetic_classification**: Generated classification data
- **synthetic_regression**: Generated regression data
- **simple_linear**: Simple linear relationship data
- **time_series_sample**: Sample time series data

### External Datasets (external/)
- **car_data**: Car price prediction dataset
- **telco_churn**: Customer churn prediction
- **titanic**: Titanic survival prediction

## 🔧 Data Processing

Each dataset includes:
- **CSV file**: The actual data
- **Metadata file**: Description, shape, and column information
- **Summary statistics**: Basic data analysis

## 📚 Usage in Exercises

- **Exercise 1 (EDA)**: Use `boston_housing.csv`
- **Exercise 5 (Linear Regression)**: Use `california_housing.csv`
- **Project 1 (Churn)**: Use `telco_churn.csv`
- **General Practice**: Use any synthetic datasets

## 🔄 Data Updates

To refresh all datasets:
```bash
python download_datasets.py
```

This will re-download external datasets and recreate synthetic data.

---

*Generated automatically by download_datasets.py*
"""
        
        return readme
    
    def run_all(self):
        """Download all datasets and create summary"""
        print("🚀 Starting dataset download process...")
        
        self.download_sklearn_datasets()
        self.download_external_datasets()
        self.create_synthetic_datasets()
        self.create_sample_datasets()
        self.create_data_summary()
        
        print("\n✅ Dataset download process completed!")
        print(f"📁 All datasets saved to: {self.data_dir.absolute()}")
        print("📖 Check README.md for dataset descriptions and usage instructions")

def main():
    """Main function to run the dataset downloader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download ML Zoomcamp datasets')
    parser.add_argument('--data-dir', default='./', help='Directory to save datasets')
    parser.add_argument('--sklearn-only', action='store_true', help='Download only sklearn datasets')
    parser.add_argument('--external-only', action='store_true', help='Download only external datasets')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.sklearn_only:
        downloader.download_sklearn_datasets()
        downloader.create_data_summary()
    elif args.external_only:
        downloader.download_external_datasets()
        downloader.create_data_summary()
    else:
        downloader.run_all()

if __name__ == "__main__":
    main()
