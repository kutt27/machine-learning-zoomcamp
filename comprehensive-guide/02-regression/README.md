# Module 2: Machine Learning for Regression

> **Master regression techniques through a complete car price prediction project**

This module provides a comprehensive introduction to regression through a hands-on car price prediction project. You'll learn the complete ML workflow from data preparation to model deployment.

## 📚 Learning Objectives

By the end of this module, you will:
- **Build** an end-to-end regression project from scratch
- **Master** linear regression theory and implementation
- **Apply** proper validation frameworks and evaluation metrics
- **Implement** feature engineering and data preprocessing techniques
- **Understand** regularization and overfitting prevention
- **Use** the model for real-world predictions

## Project Overview: Car Price Prediction

**Dataset**: [Kaggle Car Dataset](https://www.kaggle.com/CooperUnion/cardataset)  
**Goal**: Predict car prices based on features like make, model, year, mileage, etc.  
**Approach**: Complete ML pipeline using linear regression

### Project Workflow
1. **Data Preparation** → Clean and explore the dataset
2. **Exploratory Data Analysis** → Understand patterns and relationships
3. **Validation Framework** → Set up proper train/validation/test splits
4. **Linear Regression** → Implement and understand the algorithm
5. **Model Evaluation** → Use RMSE and other metrics
6. **Feature Engineering** → Create and transform features
7. **Regularization** → Prevent overfitting with Ridge regression
8. **Model Deployment** → Use the model for predictions

## 🗂️ Module Contents

### **2.1 Car Price Prediction Project Introduction**
**File**: [`01-car-price-intro.md`](01-car-price-intro.md)

**Key Concepts:**
- Project overview and objectives
- Dataset introduction and exploration
- ML project planning methodology

### **2.2 Data Preparation**
**File**: [`02-data-preparation.md`](02-data-preparation.md)

**Key Concepts:**
- Data loading and initial inspection
- Handling missing values and outliers
- Data type conversions and cleaning
- Column name standardization

**Enhanced Techniques:**
```python
import pandas as pd
import numpy as np

# Advanced data cleaning pipeline
def clean_car_data(df):
    """Comprehensive data cleaning for car dataset"""
    
    # 1. Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # 2. Handle missing values strategically
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # 3. Remove outliers using IQR method
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df
```

### **2.3 Exploratory Data Analysis (EDA)**
**File**: [`03-eda.md`](03-eda.md)

**Key Concepts:**
- Distribution analysis of target variable
- Feature correlation analysis
- Categorical variable exploration
- Data visualization techniques

**Enhanced Analysis:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_eda(df, target_col='price'):
    """Complete EDA pipeline for car price data"""
    
    # 1. Target variable analysis
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(df[target_col], bins=50, alpha=0.7)
    plt.title('Price Distribution')
    
    plt.subplot(1, 3, 2)
    plt.hist(np.log1p(df[target_col]), bins=50, alpha=0.7)
    plt.title('Log Price Distribution')
    
    plt.subplot(1, 3, 3)
    plt.boxplot(df[target_col])
    plt.title('Price Boxplot')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.show()
    
    # 3. Feature importance analysis
    correlations_with_target = correlation_matrix[target_col].abs().sort_values(ascending=False)
    print("Features most correlated with price:")
    print(correlations_with_target.head(10))
```

### **2.4 Validation Framework Setup**
**File**: [`04-validation-framework.md`](04-validation-framework.md)

**Key Concepts:**
- Train/validation/test split strategy
- Random seed for reproducibility
- Stratified sampling considerations
- Cross-validation alternatives

**Implementation:**
```python
from sklearn.model_selection import train_test_split

def setup_validation_framework(df, target_col='price', test_size=0.2, val_size=0.2):
    """Set up proper validation framework"""
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
```

### **2.5 Linear Regression Theory**
**File**: [`05-linear-regression-simple.md`](05-linear-regression-simple.md)

**Key Concepts:**
- Linear regression mathematical foundation
- Bias term and feature weights
- Prediction formula and interpretation

**Mathematical Foundation:**
```
Linear Regression Formula:
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

Matrix Form:
ŷ = X @ w

Where:
- ŷ = predictions
- X = feature matrix (with bias column)
- w = weight vector (including bias)
```

### **2.6 Linear Regression: Vector Form**
**File**: [`06-linear-regression-vector.md`](06-linear-regression-vector.md)

**Key Concepts:**
- Matrix representation of linear regression
- Vectorized operations for efficiency
- NumPy implementation

**Implementation:**
```python
import numpy as np

class LinearRegressionFromScratch:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Train linear regression using normal equation"""
        # Add bias column
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        
        # Normal equation: w = (X^T X)^(-1) X^T y
        XTX = X_with_bias.T @ X_with_bias
        XTy = X_with_bias.T @ y
        
        weights = np.linalg.solve(XTX, XTy)
        
        self.bias = weights[0]
        self.weights = weights[1:]
    
    def predict(self, X):
        """Make predictions"""
        return self.bias + X @ self.weights
```

### **2.7 Training Linear Regression: Normal Equation**
**File**: [`07-linear-regression-training.md`](07-linear-regression-training.md)

**Key Concepts:**
- Normal equation derivation
- Matrix operations for optimization
- Computational considerations

### **2.8 Baseline Model**
**File**: [`08-baseline-model.md`](08-baseline-model.md)

**Key Concepts:**
- Creating simple baseline models
- Mean and median baselines
- Baseline performance evaluation

### **2.9 Root Mean Squared Error (RMSE)**
**File**: [`09-rmse.md`](09-rmse.md)

**Key Concepts:**
- RMSE calculation and interpretation
- Comparison with other metrics
- RMSE in context of car prices

**Implementation:**
```python
def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def evaluate_model(model, X, y, dataset_name=""):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X)
    
    rmse = calculate_rmse(y, y_pred)
    mae = np.mean(np.abs(y - y_pred))
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    print(f"{dataset_name} Performance:")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R²: {r2:.3f}")
    print(f"Mean price: ${np.mean(y):,.2f}")
    print(f"RMSE as % of mean: {(rmse/np.mean(y)*100):.1f}%")
```

### **2.10 Validation Data Evaluation**
**File**: [`10-car-price-validation.md`](10-car-price-validation.md)

**Key Concepts:**
- Model evaluation on validation set
- Overfitting detection
- Performance comparison strategies

### **2.11 Feature Engineering**
**File**: [`11-feature-engineering.md`](11-feature-engineering.md)

**Key Concepts:**
- Creating new features from existing ones
- Polynomial features and interactions
- Domain-specific feature creation

**Advanced Techniques:**
```python
def engineer_car_features(df):
    """Create advanced features for car price prediction"""
    
    # 1. Age-related features
    current_year = 2024
    df['car_age'] = current_year - df['year']
    df['age_squared'] = df['car_age'] ** 2
    
    # 2. Mileage-related features
    df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 1)
    df['high_mileage'] = (df['mileage'] > 100000).astype(int)
    
    # 3. Engine-related features
    df['power_to_weight'] = df['engine_hp'] / df['curb_weight']
    df['efficiency'] = df['city_mpg'] * df['highway_mpg']
    
    # 4. Luxury indicators
    luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus']
    df['is_luxury'] = df['make'].isin(luxury_brands).astype(int)
    
    # 5. Interaction features
    df['age_mileage_interaction'] = df['car_age'] * df['mileage']
    
    return df
```

### **2.12 Categorical Variables**
**File**: [`12-categorical-variables.md`](12-categorical-variables.md)

**Key Concepts:**
- One-hot encoding for categorical features
- Handling high-cardinality categories
- Target encoding alternatives

**Implementation:**
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

def encode_categorical_features(df, categorical_cols, method='onehot'):
    """Encode categorical variables for ML models"""
    
    df_encoded = df.copy()
    
    if method == 'onehot':
        # One-hot encoding
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    
    elif method == 'label':
        # Label encoding
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
    
    elif method == 'target':
        # Target encoding (requires target variable)
        for col in categorical_cols:
            target_mean = df.groupby(col)['price'].mean()
            df_encoded[col] = df_encoded[col].map(target_mean)
    
    return df_encoded
```

### **2.13 Regularization**
**File**: [`13-regularization.md`](13-regularization.md)

**Key Concepts:**
- Ridge regression for overfitting prevention
- Regularization parameter tuning
- Mathematical foundation of regularization

**Implementation:**
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def tune_ridge_regression(X_train, y_train, X_val, y_val):
    """Tune Ridge regression hyperparameters"""
    
    # Define parameter grid
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100, 1000]}
    
    # Grid search with cross-validation
    ridge = Ridge()
    grid_search = GridSearchCV(
        ridge, param_grid, cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_ridge = grid_search.best_estimator_
    
    # Evaluate on validation set
    val_score = best_ridge.score(X_val, y_val)
    
    print(f"Best alpha: {grid_search.best_params_['alpha']}")
    print(f"Validation R²: {val_score:.3f}")
    
    return best_ridge
```

### **2.14 Model Tuning**
**File**: [`14-tuning-model.md`](14-tuning-model.md)

**Key Concepts:**
- Hyperparameter optimization
- Cross-validation for model selection
- Performance monitoring

### **2.15 Using the Model**
**File**: [`15-using-model.md`](15-using-model.md)

**Key Concepts:**
- Model deployment and inference
- Prediction pipeline creation
- Real-world usage considerations

## 🛠️ Complete Project Implementation

### End-to-End Pipeline
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class CarPricePredictionPipeline:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def preprocess_data(self, df):
        """Complete data preprocessing pipeline"""
        
        # 1. Clean data
        df = self.clean_data(df)
        
        # 2. Engineer features
        df = self.engineer_features(df)
        
        # 3. Encode categorical variables
        df = pd.get_dummies(df, drop_first=True)
        
        return df
    
    def train(self, df, target_col='price'):
        """Train the complete pipeline"""
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Separate features and target
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
        
        # Store feature columns
        self.feature_columns = X.columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = Ridge(alpha=10)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Model trained successfully!")
        print(f"Test RMSE: ${rmse:,.2f}")
        
        return self
    
    def predict(self, df):
        """Make predictions on new data"""
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Ensure same columns as training
        df_processed = df_processed.reindex(columns=self.feature_columns, fill_value=0)
        
        # Scale features
        X_scaled = self.scaler.transform(df_processed)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions

# Usage example
pipeline = CarPricePredictionPipeline()
pipeline.train(car_data)

# Predict on new cars
new_cars = pd.DataFrame({
    'make': ['BMW'],
    'model': ['3 Series'],
    'year': [2020],
    'mileage': [25000],
    'engine_hp': [255]
})

predicted_price = pipeline.predict(new_cars)
print(f"Predicted price: ${predicted_price[0]:,.2f}")
```

## 📊 Key Performance Metrics

### Model Evaluation Framework
- **RMSE**: Primary metric for regression performance
- **MAE**: Mean Absolute Error for interpretability
- **R²**: Coefficient of determination for variance explained
- **MAPE**: Mean Absolute Percentage Error for relative performance

### Expected Performance Benchmarks
- **Baseline RMSE**: ~$8,000-10,000
- **Linear Regression RMSE**: ~$5,000-7,000
- **Ridge Regression RMSE**: ~$4,000-6,000
- **R² Score**: 0.7-0.85 (good performance)

## 🎯 Module Completion Checklist

- [ ] Understand the complete ML project workflow
- [ ] Can implement linear regression from scratch
- [ ] Master data preprocessing and feature engineering
- [ ] Understand regularization and overfitting prevention
- [ ] Can evaluate models using appropriate metrics
- [ ] Built a complete car price prediction system
- [ ] Understand when and how to use regression models

## 🔗 Additional Resources

### **Video Lectures**
- [Complete Playlist](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)
- [Project Notebook](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-02-car-price)

### **Community Notes**
- [Notes from Kwang Yang](https://www.kaggle.com/kwangyangchia/notebook-for-lesson-2-mle)
- [Notes from Sebastián Ayala Ruano](https://github.com/sayalaruano/100DaysOfMLCode)
- [Notes from Alvaro Navas](https://github.com/ziritrion/ml-zoomcamp)

## 🎯 Next Steps

After completing this module, you're ready for **Module 3: Machine Learning for Classification**, where you'll learn to solve classification problems using logistic regression.

---

**Navigation:**
- **Previous**: [Module 1: Introduction](../01-introduction/README.md)
- **Next**: [Module 3: Classification](../03-classification/README.md)
- **Course Home**: [Main Guide](../README.md)

*Last Updated: 2025-01-27*
