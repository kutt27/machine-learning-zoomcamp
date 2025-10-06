# 🎯 Module 3: Machine Learning for Classification

> **Master binary classification through customer churn prediction**

This module introduces classification problems through a comprehensive customer churn prediction project. You'll learn logistic regression, feature engineering for categorical data, and evaluation techniques specific to classification tasks.

## 📚 Learning Objectives

By the end of this module, you will:
- **Understand** binary classification problems and their business applications
- **Master** logistic regression theory and implementation
- **Apply** advanced feature engineering for categorical variables
- **Implement** proper validation strategies for classification
- **Use** mutual information and correlation for feature selection
- **Build** a complete churn prediction system

## 📊 Project Overview: Customer Churn Prediction

**Dataset**: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)  
**Goal**: Predict which customers are likely to cancel their service  
**Business Impact**: Target retention campaigns to high-risk customers  
**Approach**: Binary classification using logistic regression

### Business Context
Customer churn prediction helps businesses:
- **Reduce customer acquisition costs** by retaining existing customers
- **Increase revenue** through targeted retention campaigns
- **Improve customer satisfaction** by proactive intervention
- **Optimize marketing spend** by focusing on high-risk customers

## 🗂️ Module Contents

### **3.1 Churn Prediction Project Introduction**
**Key Concepts:**
- Binary classification problem formulation
- Business value of churn prediction
- Dataset overview and exploration

**Mathematical Foundation:**
```
Binary Classification: g(xi) = yi where yi ∈ {0, 1}
- 0: Customer stays (negative class)
- 1: Customer churns (positive class)
- Output: Probability of churning P(churn = 1|features)
```

### **3.2 Data Preparation for Classification**
**Enhanced Preprocessing Pipeline:**
```python
import pandas as pd
import numpy as np

def prepare_churn_data(df):
    """Comprehensive data preparation for churn prediction"""
    
    # 1. Clean column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # 2. Handle categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # 3. Convert target to binary
    df['churn'] = (df['churn'] == 'Yes').astype(int)
    
    # 4. Handle missing values
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    # 5. Convert numerical strings to numbers
    numerical_string_cols = ['total_charges']
    for col in numerical_string_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    return df
```

### **3.3 Validation Framework for Classification**
**Stratified Sampling:**
```python
from sklearn.model_selection import train_test_split

def setup_classification_validation(df, target_col='churn'):
    """Set up validation with stratified sampling"""
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    print("Class distribution:")
    print(f"Training: {y_train.value_counts(normalize=True)}")
    print(f"Validation: {y_val.value_counts(normalize=True)}")
    print(f"Test: {y_test.value_counts(normalize=True)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
```

### **3.4 Exploratory Data Analysis for Classification**
**Classification-Specific EDA:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

def classification_eda(df, target_col='churn'):
    """EDA focused on classification patterns"""
    
    # 1. Target distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    df[target_col].value_counts().plot(kind='bar')
    plt.title('Churn Distribution')
    plt.xticks(rotation=0)
    
    # 2. Numerical features by target
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = numerical_cols.drop(target_col)
    
    plt.subplot(1, 3, 2)
    for col in numerical_cols[:3]:  # Top 3 numerical features
        sns.boxplot(data=df, x=target_col, y=col, alpha=0.7)
    plt.title('Numerical Features by Churn')
    
    # 3. Categorical features by target
    plt.subplot(1, 3, 3)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:2]:  # Top 2 categorical features
        pd.crosstab(df[col], df[target_col], normalize='index').plot(kind='bar', stacked=True)
    plt.title('Categorical Features by Churn')
    
    plt.tight_layout()
    plt.show()
```

### **3.5 Risk Assessment and Business Metrics**
**Churn Risk Scoring:**
```python
def calculate_churn_risk_metrics(df, target_col='churn'):
    """Calculate business-relevant churn metrics"""
    
    # Overall churn rate
    churn_rate = df[target_col].mean()
    
    # Churn by customer segments
    segment_analysis = {}
    
    # By contract type
    if 'contract' in df.columns:
        contract_churn = df.groupby('contract')[target_col].agg(['count', 'sum', 'mean'])
        segment_analysis['contract'] = contract_churn
    
    # By tenure
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 100], 
                                   labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])
        tenure_churn = df.groupby('tenure_group')[target_col].agg(['count', 'sum', 'mean'])
        segment_analysis['tenure'] = tenure_churn
    
    # Revenue impact
    if 'monthly_charges' in df.columns:
        churned_revenue = df[df[target_col] == 1]['monthly_charges'].sum()
        total_revenue = df['monthly_charges'].sum()
        revenue_at_risk = churned_revenue / total_revenue
        
        print(f"Overall churn rate: {churn_rate:.2%}")
        print(f"Revenue at risk: {revenue_at_risk:.2%}")
        print(f"Average monthly charges - Churned: ${df[df[target_col] == 1]['monthly_charges'].mean():.2f}")
        print(f"Average monthly charges - Retained: ${df[df[target_col] == 0]['monthly_charges'].mean():.2f}")
    
    return segment_analysis
```

### **3.6 Mutual Information for Feature Selection**
**Advanced Feature Selection:**
```python
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def mutual_information_analysis(X, y):
    """Calculate mutual information for feature selection"""
    
    # Encode categorical variables for mutual information
    X_encoded = X.copy()
    label_encoders = {}
    
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'][:15], feature_importance['mutual_info'][:15])
    plt.title('Top 15 Features by Mutual Information')
    plt.xlabel('Mutual Information Score')
    plt.tight_layout()
    plt.show()
    
    return feature_importance, label_encoders
```

### **3.7 Correlation Analysis**
**Feature Correlation for Classification:**
```python
def correlation_analysis_classification(df, target_col='churn'):
    """Analyze correlations in classification context"""
    
    # Encode categorical variables
    df_encoded = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col:
            df_encoded[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # Calculate correlation matrix
    correlation_matrix = df_encoded.corr()
    
    # Focus on correlations with target
    target_correlations = correlation_matrix[target_col].abs().sort_values(ascending=False)
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    print("Features most correlated with churn:")
    print(target_correlations.head(10))
    
    return target_correlations
```

### **3.8 One-Hot Encoding**
**Advanced Categorical Encoding:**
```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def advanced_categorical_encoding(df, categorical_cols, method='onehot', max_categories=10):
    """Advanced categorical variable encoding"""
    
    df_encoded = df.copy()
    
    if method == 'onehot':
        # One-hot encoding with category limit
        for col in categorical_cols:
            # Handle high cardinality by keeping top categories
            top_categories = df[col].value_counts().head(max_categories).index
            df_encoded[col] = df[col].where(df[col].isin(top_categories), 'Other')
        
        # Apply one-hot encoding
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    
    elif method == 'target':
        # Target encoding (mean encoding)
        for col in categorical_cols:
            target_mean = df.groupby(col)['churn'].mean()
            df_encoded[col + '_target_encoded'] = df[col].map(target_mean)
            df_encoded = df_encoded.drop(col, axis=1)
    
    elif method == 'frequency':
        # Frequency encoding
        for col in categorical_cols:
            freq_map = df[col].value_counts().to_dict()
            df_encoded[col + '_frequency'] = df[col].map(freq_map)
            df_encoded = df_encoded.drop(col, axis=1)
    
    return df_encoded
```

### **3.9 Logistic Regression Theory**
**Mathematical Foundation:**
```python
import numpy as np

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train logistic regression using gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = X @ self.weights + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute cost (log-likelihood)
            cost = self.compute_cost(y, predictions)
            
            # Compute gradients
            dw = (1/n_samples) * X.T @ (predictions - y)
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print cost every 100 iterations
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
    
    def compute_cost(self, y_true, y_pred):
        """Compute logistic regression cost function"""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        linear_pred = X @ self.weights + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
```

### **3.10 Training Logistic Regression**
**Scikit-learn Implementation:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train and evaluate logistic regression"""
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    train_score = pipeline.score(X_train, y_train)
    val_score = pipeline.score(X_val, y_val)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Validation accuracy: {val_score:.3f}")
    
    # Get feature importance (coefficients)
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
    coefficients = pipeline.named_steps['classifier'].coef_[0]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    return pipeline, feature_importance
```

### **3.11 Model Interpretation**
**Understanding Logistic Regression Results:**
```python
def interpret_logistic_regression(model, feature_names, X_sample):
    """Interpret logistic regression predictions"""
    
    # Get model coefficients
    coefficients = model.named_steps['classifier'].coef_[0]
    intercept = model.named_steps['classifier'].intercept_[0]
    
    # Scale the sample data
    X_scaled = model.named_steps['scaler'].transform(X_sample.reshape(1, -1))
    
    # Calculate linear combination
    linear_combination = np.sum(coefficients * X_scaled[0]) + intercept
    
    # Calculate probability
    probability = 1 / (1 + np.exp(-linear_combination))
    
    # Feature contributions
    contributions = coefficients * X_scaled[0]
    
    # Create interpretation dataframe
    interpretation = pd.DataFrame({
        'feature': feature_names,
        'value': X_sample,
        'scaled_value': X_scaled[0],
        'coefficient': coefficients,
        'contribution': contributions
    }).sort_values('contribution', key=abs, ascending=False)
    
    print(f"Prediction probability: {probability:.3f}")
    print(f"Linear combination: {linear_combination:.3f}")
    print("\nTop feature contributions:")
    print(interpretation.head(10))
    
    return interpretation
```

## 🛠️ Complete Classification Pipeline

```python
class ChurnPredictionPipeline:
    def __init__(self):
        self.model = None
        self.feature_encoders = {}
        self.feature_columns = None
    
    def preprocess_data(self, df):
        """Complete preprocessing pipeline"""
        
        # 1. Clean data
        df = df.copy()
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # 2. Handle target variable
        if 'churn' in df.columns:
            df['churn'] = (df['churn'] == 'Yes').astype(int)
        
        # 3. Handle missing values
        df = df.fillna('Unknown')
        
        # 4. Convert numerical strings
        if 'total_charges' in df.columns:
            df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
            df['total_charges'] = df['total_charges'].fillna(df['total_charges'].median())
        
        # 5. Feature engineering
        df = self.engineer_features(df)
        
        # 6. Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols.drop('churn', errors='ignore')
        
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        return df_encoded
    
    def engineer_features(self, df):
        """Create additional features"""
        
        # Customer lifetime value
        if 'tenure' in df.columns and 'monthly_charges' in df.columns:
            df['customer_lifetime_value'] = df['tenure'] * df['monthly_charges']
        
        # Service usage intensity
        service_cols = [col for col in df.columns if 'service' in col.lower()]
        if service_cols:
            df['total_services'] = df[service_cols].sum(axis=1)
        
        # Contract risk factors
        if 'contract' in df.columns:
            df['is_month_to_month'] = (df['contract'] == 'Month-to-month').astype(int)
        
        return df
    
    def train(self, df):
        """Train the complete pipeline"""
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Separate features and target
        X = df_processed.drop('churn', axis=1)
        y = df_processed['churn']
        
        # Store feature columns
        self.feature_columns = X.columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        return self
    
    def predict_churn_probability(self, df):
        """Predict churn probability for new customers"""
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Ensure same columns as training
        df_processed = df_processed.reindex(columns=self.feature_columns, fill_value=0)
        
        # Make predictions
        probabilities = self.model.predict_proba(df_processed)[:, 1]
        
        return probabilities
    
    def identify_high_risk_customers(self, df, threshold=0.7):
        """Identify customers at high risk of churning"""
        
        probabilities = self.predict_churn_probability(df)
        
        # Add probabilities to dataframe
        df_with_risk = df.copy()
        df_with_risk['churn_probability'] = probabilities
        df_with_risk['risk_level'] = pd.cut(
            probabilities, 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Return high-risk customers
        high_risk_customers = df_with_risk[df_with_risk['churn_probability'] >= threshold]
        
        return high_risk_customers.sort_values('churn_probability', ascending=False)
```

## 📊 Business Impact Analysis

```python
def calculate_business_impact(df, model, monthly_charges_col='monthly_charges'):
    """Calculate business impact of churn prediction model"""
    
    # Predict churn probabilities
    churn_probs = model.predict_churn_probability(df)
    
    # Calculate potential revenue at risk
    df_analysis = df.copy()
    df_analysis['churn_probability'] = churn_probs
    df_analysis['revenue_at_risk'] = df_analysis[monthly_charges_col] * df_analysis['churn_probability']
    
    # Segment customers by risk
    df_analysis['risk_segment'] = pd.cut(
        churn_probs, 
        bins=[0, 0.3, 0.7, 1.0], 
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Business metrics
    total_customers = len(df_analysis)
    total_monthly_revenue = df_analysis[monthly_charges_col].sum()
    total_revenue_at_risk = df_analysis['revenue_at_risk'].sum()
    
    # Risk segment analysis
    segment_analysis = df_analysis.groupby('risk_segment').agg({
        monthly_charges_col: ['count', 'sum', 'mean'],
        'churn_probability': 'mean',
        'revenue_at_risk': 'sum'
    }).round(2)
    
    print("=== BUSINESS IMPACT ANALYSIS ===")
    print(f"Total customers: {total_customers:,}")
    print(f"Total monthly revenue: ${total_monthly_revenue:,.2f}")
    print(f"Revenue at risk: ${total_revenue_at_risk:,.2f} ({total_revenue_at_risk/total_monthly_revenue:.1%})")
    print("\nRisk Segment Analysis:")
    print(segment_analysis)
    
    return segment_analysis
```

## 🎯 Module Completion Checklist

- [ ] Understand binary classification and business applications
- [ ] Can implement logistic regression from scratch
- [ ] Master categorical variable encoding techniques
- [ ] Understand mutual information and feature selection
- [ ] Can interpret logistic regression coefficients
- [ ] Built a complete churn prediction system
- [ ] Understand business impact of classification models

## 🔗 Additional Resources

### **Video Lectures**
- [Complete Classification Playlist](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)
- [Project Notebook](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-03-churn-prediction)

### **Community Notes**
- [Notes from Peter Ernicke](https://knowmledge.com/category/courses/ml-zoomcamp/classification/)
- [Notes from Sebastián Ayala Ruano](https://github.com/sayalaruano/100DaysOfMLCode)

## 🎯 Next Steps

After completing this module, you're ready for **Module 4: Evaluation Metrics**, where you'll learn comprehensive techniques for evaluating classification models.

---

**Navigation:**
- **Previous**: [Module 2: Regression](../02-regression/README.md)
- **Next**: [Module 4: Evaluation](../04-evaluation/README.md)
- **Course Home**: [Main Guide](../README.md)

*Last Updated: 2025-01-27*
