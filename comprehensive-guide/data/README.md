# 📊 Machine Learning Datasets

> **Comprehensive collection of datasets for the Machine Learning Zoomcamp course**

This directory contains all datasets used throughout the course, from basic exercises to advanced projects. Each dataset is carefully selected to demonstrate specific ML concepts and techniques.

## 📁 Directory Structure

```
data/
├── raw/                    # Built-in sklearn datasets and synthetic data
├── external/               # Downloaded datasets from external sources  
├── processed/              # Cleaned and preprocessed datasets (created during exercises)
├── download_datasets.py    # Automated download script
├── dataset_summary.csv     # Summary of all available datasets
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Download All Datasets
```bash
# Navigate to the data directory
cd comprehensive-guide/data/

# Run the download script
python download_datasets.py

# Or with specific options
python download_datasets.py --data-dir ./datasets/
python download_datasets.py --sklearn-only
python download_datasets.py --external-only
```

### 2. Load and Explore
```python
import pandas as pd
import numpy as np

# Load any dataset
df = pd.read_csv('raw/boston_housing.csv')

# Quick exploration
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())
print(df.describe())
```

### 3. Use in Exercises
```python
# For Exercise 1 (EDA Challenge)
df = pd.read_csv('raw/boston_housing.csv')

# For Exercise 5 (Linear Regression from Scratch)
df = pd.read_csv('raw/california_housing.csv')

# For Project 1 (Customer Churn)
df = pd.read_csv('external/telco_churn.csv')
```

## 📊 Dataset Catalog

### **Built-in Datasets (raw/)**

#### **Regression Datasets**
| Dataset | Samples | Features | Target | Description |
|---------|---------|----------|---------|-------------|
| `boston_housing.csv` | 506 | 13 | House prices | Classic Boston housing dataset |
| `california_housing.csv` | 20,640 | 8 | House values | California housing prices |
| `synthetic_regression.csv` | 1,000 | 15 | Continuous | Generated regression data |
| `simple_linear.csv` | 100 | 1 | Linear target | Simple linear relationship |

#### **Classification Datasets**
| Dataset | Samples | Features | Classes | Description |
|---------|---------|----------|---------|-------------|
| `iris.csv` | 150 | 4 | 3 | Iris flower species |
| `wine.csv` | 178 | 13 | 3 | Wine classification |
| `breast_cancer.csv` | 569 | 30 | 2 | Breast cancer diagnosis |
| `synthetic_classification.csv` | 1,000 | 20 | 2 | Generated classification data |

#### **Special Purpose Datasets**
| Dataset | Samples | Features | Type | Description |
|---------|---------|----------|------|-------------|
| `time_series_sample.csv` | 365 | 2 | Time series | Sample time series with trend and seasonality |

### **External Datasets (external/)**

#### **Real-world Projects**
| Dataset | Samples | Features | Type | Description |
|---------|---------|----------|------|-------------|
| `car_data.csv` | 11,914 | 16 | Regression | Car price prediction |
| `telco_churn.csv` | 7,043 | 21 | Classification | Customer churn prediction |
| `titanic.csv` | 891 | 12 | Classification | Titanic survival prediction |

## 🎯 Dataset Usage Guide

### **For Beginners**
Start with these datasets to learn fundamentals:
1. **`simple_linear.csv`** - Understanding linear relationships
2. **`iris.csv`** - Basic classification concepts
3. **`boston_housing.csv`** - Regression with real data

### **For Intermediate Learners**
Progress to more complex datasets:
1. **`california_housing.csv`** - Larger regression dataset
2. **`wine.csv`** - Multi-class classification
3. **`synthetic_classification.csv`** - Controlled complexity

### **For Advanced Projects**
Tackle real-world challenges:
1. **`telco_churn.csv`** - Business problem with imbalanced data
2. **`car_data.csv`** - Feature engineering opportunities
3. **`titanic.csv`** - Classic ML competition dataset

## 🔧 Data Processing Examples

### **Basic Data Loading**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('raw/california_housing.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### **Handling Missing Data**
```python
# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Handle missing values
df_clean = df.copy()

# For numerical columns
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numerical_cols] = df_clean[numerical_cols].fillna(df_clean[numerical_cols].median())

# For categorical columns
categorical_cols = df_clean.select_dtypes(include=['object']).columns
df_clean[categorical_cols] = df_clean[categorical_cols].fillna(df_clean[categorical_cols].mode().iloc[0])
```

### **Feature Engineering**
```python
# Create new features
df['feature_ratio'] = df['feature_1'] / (df['feature_2'] + 1e-8)
df['feature_interaction'] = df['feature_1'] * df['feature_2']
df['feature_binned'] = pd.cut(df['feature_1'], bins=5, labels=False)

# Log transformation for skewed features
df['feature_log'] = np.log1p(df['feature_1'])

# One-hot encoding for categorical features
df_encoded = pd.get_dummies(df, columns=['categorical_feature'], drop_first=True)
```

## 📈 Dataset Characteristics

### **Data Quality**
- **Clean datasets**: `iris`, `wine`, `breast_cancer`
- **Missing values**: `car_data`, `telco_churn`
- **Outliers present**: `boston_housing`, `california_housing`

### **Complexity Levels**
- **Simple**: `simple_linear`, `iris`
- **Moderate**: `boston_housing`, `wine`
- **Complex**: `california_housing`, `telco_churn`

### **Problem Types**
- **Binary classification**: `breast_cancer`, `telco_churn`, `titanic`
- **Multi-class classification**: `iris`, `wine`
- **Regression**: `boston_housing`, `california_housing`, `car_data`

## 🔄 Data Updates and Maintenance

### **Automatic Updates**
The download script automatically:
- Downloads latest versions of external datasets
- Recreates synthetic datasets with consistent random seeds
- Updates metadata and summary files
- Validates data integrity

### **Manual Updates**
To add new datasets:
1. Add URL to `external_datasets` in `download_datasets.py`
2. Run the download script
3. Update this README if needed

### **Data Validation**
Each dataset includes:
- **Metadata file**: Description, source, and column information
- **Summary statistics**: Basic data analysis
- **Quality checks**: Missing values, data types, outliers

## 🚨 Important Notes

### **Data Ethics and Privacy**
- All datasets are publicly available or synthetic
- No personal or sensitive information is included
- External datasets are used for educational purposes only

### **Reproducibility**
- All synthetic datasets use fixed random seeds
- Download script ensures consistent data versions
- Preprocessing steps are documented and reproducible

### **Performance Considerations**
- Large datasets (>100MB) are noted in descriptions
- Consider memory usage when loading multiple datasets
- Use chunking for very large datasets if needed

## 🆘 Troubleshooting

### **Common Issues**

#### **Download Failures**
```bash
# Check internet connection
ping google.com

# Try downloading individual datasets
python download_datasets.py --sklearn-only
python download_datasets.py --external-only
```

#### **Memory Issues**
```python
# Load data in chunks for large datasets
chunk_size = 10000
chunks = []
for chunk in pd.read_csv('large_dataset.csv', chunksize=chunk_size):
    # Process chunk
    processed_chunk = process_data(chunk)
    chunks.append(processed_chunk)

df = pd.concat(chunks, ignore_index=True)
```

#### **Missing Dependencies**
```bash
# Install required packages
pip install pandas numpy scikit-learn requests

# Or install from requirements.txt
pip install -r ../notebooks/requirements.txt
```

## 📚 Related Resources

- **[Exercises](../exercises/README.md)**: Hands-on practice with these datasets
- **[Notebooks](../notebooks/README.md)**: Detailed analysis examples
- **[Reference](../reference/README.md)**: Quick reference guides

---

**Last Updated**: 2025-01-27  
**Total Datasets**: 12+ datasets covering all major ML tasks  
**Total Size**: ~50MB (after download)

*Happy learning! 🚀*
