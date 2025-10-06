# 🚀 ML Zoomcamp Quick Reference Guide

> **Essential algorithms, code snippets, and formulas for quick lookup**

## 📊 Algorithm Quick Reference

### **Linear Regression**
```python
from sklearn.linear_model import LinearRegression

# Basic usage
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# From scratch
def linear_regression(X, y):
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    weights = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
    return weights
```

### **Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression

# Basic usage
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### **Decision Trees**
```python
from sklearn.tree import DecisionTreeClassifier

# Basic usage
model = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### **Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

# Basic usage
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### **XGBoost**
```python
import xgboost as xgb

# Basic usage
model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## 📈 Evaluation Metrics

### **Regression Metrics**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MAE
mae = mean_absolute_error(y_true, y_pred)

# R²
r2 = r2_score(y_true, y_pred)

# MAPE
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

### **Classification Metrics**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# ROC AUC
auc = roc_auc_score(y_true, y_pred_proba)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

## 🔧 Data Preprocessing

### **Data Cleaning**
```python
# Handle missing values
df.fillna(df.mean())  # Numerical
df.fillna(df.mode().iloc[0])  # Categorical
df.dropna()  # Remove missing

# Remove outliers (IQR method)
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]
```

### **Feature Engineering**
```python
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['categorical_column'], drop_first=True)

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded_column'] = le.fit_transform(df['categorical_column'])

# Feature scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### **Train-Test Split**
```python
from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Three-way split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
```

## 🎯 Model Selection & Validation

### **Cross-Validation**
```python
from sklearn.model_selection import cross_val_score, KFold

# K-Fold CV
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Custom CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
```

### **Hyperparameter Tuning**
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Random Search
from scipy.stats import randint
param_dist = {'max_depth': randint(3, 10), 'n_estimators': randint(50, 200)}
random_search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=5)
```

## 🚀 Model Deployment

### **Save/Load Models**
```python
import pickle
import joblib

# Using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Using joblib (better for sklearn)
joblib.dump(model, 'model.joblib')
loaded_model = joblib.load('model.joblib')
```

### **Flask API**
```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
```

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

## 📊 Common Code Patterns

### **Complete ML Pipeline**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Preprocessing
X = df.drop('target', axis=1)
y = df['target']

# Handle missing values
X = X.fillna(X.mean())

# Encode categorical variables
X = pd.get_dummies(X)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

### **Feature Importance Analysis**
```python
# For tree-based models
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.title('Top 10 Feature Importances')
plt.show()
```

## 🔍 Debugging & Troubleshooting

### **Common Issues**
```python
# Check for data leakage
print("Training accuracy:", model.score(X_train, y_train))
print("Validation accuracy:", model.score(X_val, y_val))
# If training >> validation, likely overfitting

# Check feature distributions
X_train.describe()
X_test.describe()
# Should be similar distributions

# Check for missing values
print("Missing values in training:", X_train.isnull().sum().sum())
print("Missing values in test:", X_test.isnull().sum().sum())

# Check target distribution
print("Training target distribution:")
print(y_train.value_counts(normalize=True))
print("Test target distribution:")
print(y_test.value_counts(normalize=True))
```

### **Performance Optimization**
```python
# Reduce memory usage
def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
    return df

# Parallel processing
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1)  # Use all cores
```

## 📚 Mathematical Formulas

### **Linear Regression**
```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
Matrix form: ŷ = Xw
Normal equation: w = (XᵀX)⁻¹Xᵀy
```

### **Logistic Regression**
```
p = 1 / (1 + e^(-z))
where z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

### **Evaluation Metrics**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### **Regularization**
```
Ridge: Cost = MSE + α∑wᵢ²
Lasso: Cost = MSE + α∑|wᵢ|
```

## 🎯 Quick Decision Guide

### **Algorithm Selection**
- **Small dataset (<1K)**: Logistic Regression, Naive Bayes
- **Medium dataset (1K-100K)**: Random Forest, SVM
- **Large dataset (>100K)**: XGBoost, Neural Networks
- **High interpretability needed**: Linear/Logistic Regression, Decision Trees
- **High accuracy needed**: Ensemble methods, Deep Learning

### **When to Use Each Metric**
- **Accuracy**: Balanced datasets, general performance
- **Precision**: When false positives are costly
- **Recall**: When false negatives are costly
- **F1-Score**: Imbalanced datasets
- **ROC-AUC**: Binary classification, probability ranking

---

*This quick reference covers the most commonly used patterns and techniques from the ML Zoomcamp course. Keep this handy for rapid development and debugging!*
