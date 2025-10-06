# 🧠 ML Algorithms Cheat Sheet

> **Quick reference for choosing and implementing ML algorithms**

## 🎯 Algorithm Selection Guide

### **Problem Type Decision Tree**

```
What type of problem are you solving?
├── Supervised Learning
│   ├── Regression (Continuous Target)
│   │   ├── Linear Relationship? → Linear Regression
│   │   ├── Non-linear? → Polynomial Regression, Random Forest
│   │   ├── High Dimensions? → Ridge/Lasso Regression
│   │   └── Complex Patterns? → XGBoost, Neural Networks
│   └── Classification (Categorical Target)
│       ├── Binary Classification
│       │   ├── Linear Boundary? → Logistic Regression
│       │   ├── Non-linear? → SVM, Random Forest
│       │   └── Complex? → XGBoost, Neural Networks
│       └── Multi-class Classification
│           ├── Small Dataset? → Naive Bayes, KNN
│           ├── Medium Dataset? → Random Forest, SVM
│           └── Large Dataset? → XGBoost, Neural Networks
└── Unsupervised Learning
    ├── Clustering → K-Means, DBSCAN, Hierarchical
    ├── Dimensionality Reduction → PCA, t-SNE, UMAP
    └── Association Rules → Apriori, FP-Growth
```

## 📊 Regression Algorithms

### **Linear Regression**
```python
from sklearn.linear_model import LinearRegression

# Basic usage
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Key parameters: None (simple linear regression)
# Best for: Linear relationships, interpretability needed
# Pros: Fast, interpretable, no hyperparameters
# Cons: Assumes linear relationship, sensitive to outliers
```

### **Ridge Regression (L2 Regularization)**
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # Regularization strength
model.fit(X_train, y_train)

# Key parameters:
# - alpha: Regularization strength (higher = more regularization)
# Best for: Multicollinearity, overfitting prevention
# Pros: Handles multicollinearity, reduces overfitting
# Cons: Doesn't perform feature selection
```

### **Lasso Regression (L1 Regularization)**
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0)
model.fit(X_train, y_train)

# Key parameters:
# - alpha: Regularization strength
# Best for: Feature selection, sparse solutions
# Pros: Automatic feature selection, interpretable
# Cons: Can be unstable with correlated features
```

### **Random Forest Regressor**
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,    # Number of trees
    max_depth=None,      # Maximum depth of trees
    random_state=42
)
model.fit(X_train, y_train)

# Best for: Non-linear relationships, feature importance
# Pros: Handles non-linearity, robust to outliers
# Cons: Less interpretable, can overfit with small datasets
```

## 🎯 Classification Algorithms

### **Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,              # Inverse regularization strength
    max_iter=1000,      # Maximum iterations
    random_state=42
)
model.fit(X_train, y_train)

# Best for: Binary classification, probability estimates
# Pros: Fast, interpretable, probability outputs
# Cons: Assumes linear decision boundary
```

### **Random Forest Classifier**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
model.fit(X_train, y_train)

# Best for: Non-linear classification, feature importance
# Pros: Handles non-linearity, robust, feature importance
# Cons: Can overfit, less interpretable
```

### **Support Vector Machine (SVM)**
```python
from sklearn.svm import SVC

model = SVC(
    C=1.0,              # Regularization parameter
    kernel='rbf',       # Kernel type
    gamma='scale',      # Kernel coefficient
    probability=True    # Enable probability estimates
)
model.fit(X_train, y_train)

# Best for: High-dimensional data, non-linear boundaries
# Pros: Effective in high dimensions, memory efficient
# Cons: Slow on large datasets, requires feature scaling
```

### **XGBoost Classifier**
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Best for: Structured data, competitions, high performance
# Pros: High performance, handles missing values
# Cons: Many hyperparameters, can overfit
```

## 🔍 Unsupervised Learning

### **K-Means Clustering**
```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=3,       # Number of clusters
    random_state=42,
    n_init=10
)
clusters = model.fit_predict(X)

# Best for: Spherical clusters, known number of clusters
# Pros: Simple, fast, works well with spherical clusters
# Cons: Need to specify k, sensitive to initialization
```

### **DBSCAN Clustering**
```python
from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=0.5,           # Maximum distance between samples
    min_samples=5      # Minimum samples in neighborhood
)
clusters = model.fit_predict(X)

# Best for: Arbitrary shaped clusters, outlier detection
# Pros: Finds arbitrary shapes, identifies outliers
# Cons: Sensitive to hyperparameters, struggles with varying densities
```

### **Principal Component Analysis (PCA)**
```python
from sklearn.decomposition import PCA

model = PCA(
    n_components=2,     # Number of components
    random_state=42
)
X_reduced = model.fit_transform(X)

# Best for: Dimensionality reduction, visualization
# Pros: Reduces dimensions, removes correlation
# Cons: Components not interpretable, linear transformation
```

## ⚙️ Hyperparameter Tuning

### **Grid Search**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### **Random Search**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [3, 5, 7, None],
    'min_samples_split': randint(2, 11)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
```

## 📊 Model Evaluation

### **Regression Metrics**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Mean Squared Error
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# Mean Absolute Error
mae = mean_absolute_error(y_true, y_pred)

# R-squared Score
r2 = r2_score(y_true, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
```

### **Classification Metrics**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# ROC AUC (for binary classification)
auc = roc_auc_score(y_true, y_pred_proba[:, 1])

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"ROC AUC: {auc:.3f}")
```

## 🔧 Data Preprocessing

### **Feature Scaling**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard Scaling (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Min-Max Scaling (range 0-1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

# Robust Scaling (median and IQR)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

### **Encoding Categorical Variables**
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Label Encoding (ordinal)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['category_column'], drop_first=True)

# Or using sklearn
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = encoder.fit_transform(X_categorical)
```

### **Handling Missing Values**
```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)

# KNN imputation
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

## 🎯 Algorithm Selection Criteria

### **Dataset Size**
- **Small (<1K samples)**: Simple models (Linear/Logistic Regression, Naive Bayes)
- **Medium (1K-100K)**: Tree-based models (Random Forest, XGBoost)
- **Large (>100K)**: Scalable algorithms (SGD, Neural Networks)

### **Feature Count**
- **Few features (<10)**: Any algorithm
- **Many features (10-1000)**: Regularized models (Ridge, Lasso)
- **High dimensions (>1000)**: Dimensionality reduction + simple models

### **Interpretability Requirements**
- **High**: Linear/Logistic Regression, Decision Trees
- **Medium**: Random Forest (feature importance)
- **Low**: XGBoost, Neural Networks, SVM

### **Training Time Constraints**
- **Fast**: Linear models, Naive Bayes
- **Medium**: Random Forest, SVM
- **Slow**: XGBoost, Neural Networks

### **Prediction Speed Requirements**
- **Real-time**: Linear models, simple trees
- **Batch**: Any algorithm
- **Memory constrained**: Linear models, simple trees

## 🚀 Quick Implementation Template

```python
# Standard ML pipeline template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load and prepare data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features (if needed)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

## 📚 When to Use Each Algorithm

| Algorithm | Best For | Avoid When |
|-----------|----------|------------|
| **Linear Regression** | Linear relationships, interpretability | Non-linear data, many features |
| **Logistic Regression** | Binary classification, probability estimates | Non-linear boundaries |
| **Random Forest** | Non-linear data, feature importance | Need fast predictions |
| **SVM** | High-dimensional data, small datasets | Large datasets, need probabilities |
| **XGBoost** | Structured data, competitions | Need interpretability |
| **Neural Networks** | Complex patterns, large datasets | Small datasets, need interpretability |
| **K-Means** | Spherical clusters, known k | Unknown clusters, arbitrary shapes |
| **DBSCAN** | Arbitrary shapes, outlier detection | Spherical clusters, varying densities |

---

**Navigation:**
- **Related**: [Code Templates](code-templates.md)
- **Reference Home**: [Quick Reference](README.md)
- **Course Home**: [Main Guide](../README.md)

*Keep this handy for quick algorithm selection and implementation!* 🚀
