# 💻 Code Templates and Snippets

> **Ready-to-use code templates for common ML tasks**

## 🚀 Complete ML Pipeline Template

### **Standard Classification Pipeline**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False
    
    def load_data(self, filepath, target_column):
        """Load and prepare data"""
        self.df = pd.read_csv(filepath)
        self.X = self.df.drop(target_column, axis=1)
        self.y = self.df[target_column]
        print(f"Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        return self
    
    def explore_data(self):
        """Basic data exploration"""
        print("=== DATA OVERVIEW ===")
        print(f"Shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        print(f"Target distribution:\n{self.y.value_counts()}")
        
        # Correlation with target
        if self.y.dtype in ['int64', 'float64']:
            correlations = self.X.corrwith(self.y).abs().sort_values(ascending=False)
            print(f"\nTop 5 correlations with target:\n{correlations.head()}")
        
        return self
    
    def preprocess(self):
        """Data preprocessing"""
        # Handle missing values
        self.X = self.X.fillna(self.X.median())
        
        # Encode categorical variables
        categorical_cols = self.X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col].astype(str))
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data preprocessing completed")
        return self
    
    def train_model(self, model_type='rf'):
        """Train model"""
        if model_type == 'rf':
            self.model = RandomForestClassifier(random_state=self.random_state)
        elif model_type == 'lr':
            self.model = LogisticRegression(random_state=self.random_state)
        
        self.model.fit(self.X_train_scaled, self.y_train)
        self.is_fitted = True
        print(f"Model trained: {type(self.model).__name__}")
        return self
    
    def evaluate(self):
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        print("=== MODEL PERFORMANCE ===")
        print(classification_report(self.y_test, y_pred))
        print(f"ROC AUC: {roc_auc_score(self.y_test, y_pred_proba):.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        return self
    
    def predict(self, new_data):
        """Make predictions on new data"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        new_data_scaled = self.scaler.transform(new_data)
        predictions = self.model.predict(new_data_scaled)
        probabilities = self.model.predict_proba(new_data_scaled)
        
        return predictions, probabilities

# Usage example
# pipeline = MLPipeline()
# pipeline.load_data('data.csv', 'target').explore_data().preprocess().train_model().evaluate()
```

## 📊 Data Exploration Templates

### **Comprehensive EDA Function**
```python
def comprehensive_eda(df, target_col=None):
    """Perform comprehensive exploratory data analysis"""
    
    print("=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nData types:\n{df.dtypes.value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    # Numerical features analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    if numerical_cols:
        print(f"\n=== NUMERICAL FEATURES ({len(numerical_cols)}) ===")
        print(df[numerical_cols].describe())
        
        # Distribution plots
        n_cols = min(4, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Remove empty subplots
        for i in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
    
    # Categorical features analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        print(f"\n=== CATEGORICAL FEATURES ({len(categorical_cols)}) ===")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"{col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"  Values: {df[col].value_counts().to_dict()}")
            print()
    
    # Target analysis
    if target_col and target_col in df.columns:
        print(f"\n=== TARGET VARIABLE: {target_col} ===")
        if df[target_col].dtype in ['object', 'category']:
            print(df[target_col].value_counts())
            df[target_col].value_counts().plot(kind='bar')
            plt.title(f'{target_col} Distribution')
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(df[target_col].describe())
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            df[target_col].hist(bins=30, alpha=0.7)
            plt.title(f'{target_col} Distribution')
            plt.xlabel(target_col)
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            df[target_col].plot(kind='box')
            plt.title(f'{target_col} Box Plot')
            plt.ylabel(target_col)
            
            plt.tight_layout()
            plt.show()
    
    # Correlation analysis
    if len(numerical_cols) > 1:
        print("\n=== CORRELATION ANALYSIS ===")
        corr_matrix = df[numerical_cols + ([target_col] if target_col and df[target_col].dtype in ['int64', 'float64'] else [])].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        if target_col and df[target_col].dtype in ['int64', 'float64']:
            target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
            print(f"Features most correlated with {target_col}:")
            print(target_corr.head(10))

# Usage
# comprehensive_eda(df, target_col='target')
```

### **Missing Data Analysis**
```python
def analyze_missing_data(df):
    """Analyze missing data patterns"""
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    })
    
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if len(missing_data) > 0:
        print("=== MISSING DATA ANALYSIS ===")
        print(missing_data)
        
        # Visualize missing data pattern
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        plt.title('Missing Data Pattern')
        plt.show()
        
        # Missing data bar plot
        plt.figure(figsize=(10, 6))
        missing_data.plot(x='Column', y='Missing_Percentage', kind='bar')
        plt.title('Missing Data Percentage by Column')
        plt.ylabel('Missing Percentage (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No missing data found!")
    
    return missing_data

# Usage
# missing_analysis = analyze_missing_data(df)
```

## 🔧 Feature Engineering Templates

### **Automated Feature Engineering**
```python
def create_features(df, target_col=None):
    """Create new features automatically"""
    df_features = df.copy()
    
    # Numerical feature engineering
    numerical_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    for col in numerical_cols:
        # Log transformation (for positive values)
        if (df_features[col] > 0).all():
            df_features[f'{col}_log'] = np.log1p(df_features[col])
        
        # Square root transformation
        if (df_features[col] >= 0).all():
            df_features[f'{col}_sqrt'] = np.sqrt(df_features[col])
        
        # Binning
        df_features[f'{col}_binned'] = pd.cut(df_features[col], bins=5, labels=False)
        
        # Z-score
        df_features[f'{col}_zscore'] = (df_features[col] - df_features[col].mean()) / df_features[col].std()
    
    # Interaction features (for first 5 numerical columns)
    for i, col1 in enumerate(numerical_cols[:5]):
        for col2 in numerical_cols[i+1:6]:
            df_features[f'{col1}_x_{col2}'] = df_features[col1] * df_features[col2]
            df_features[f'{col1}_div_{col2}'] = df_features[col1] / (df_features[col2] + 1e-8)
    
    # Categorical feature engineering
    categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        # Frequency encoding
        freq_map = df_features[col].value_counts().to_dict()
        df_features[f'{col}_frequency'] = df_features[col].map(freq_map)
        
        # Length of string
        df_features[f'{col}_length'] = df_features[col].astype(str).str.len()
    
    print(f"Original features: {df.shape[1]}")
    print(f"New features: {df_features.shape[1]}")
    print(f"Added: {df_features.shape[1] - df.shape[1]} features")
    
    return df_features

# Usage
# df_engineered = create_features(df, target_col='target')
```

## 🎯 Model Training Templates

### **Model Comparison Framework**
```python
def compare_models(X_train, X_test, y_train, y_test, models=None):
    """Compare multiple models"""
    if models is None:
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42)
        }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Training_Time': training_time
        }
        
        if y_pred_proba is not None:
            results[name]['ROC_AUC'] = roc_auc_score(y_test, y_pred_proba)
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    print("\n=== MODEL COMPARISON ===")
    print(results_df)
    
    # Visualize results
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    if 'ROC_AUC' in results_df.columns:
        metrics_to_plot.append('ROC_AUC')
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 4))
    
    for i, metric in enumerate(metrics_to_plot):
        results_df[metric].plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# Usage
# results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
```

### **Hyperparameter Tuning Template**
```python
def tune_hyperparameters(model, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
    """Comprehensive hyperparameter tuning"""
    
    print(f"Tuning {type(model).__name__}...")
    print(f"Parameter grid: {param_grid}")
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Analyze results
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Plot parameter importance (for single parameter)
    if len(param_grid) == 1:
        param_name = list(param_grid.keys())[0]
        param_values = param_grid[param_name]
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, results_df['mean_test_score'], 'bo-')
        plt.fill_between(param_values, 
                        results_df['mean_test_score'] - results_df['std_test_score'],
                        results_df['mean_test_score'] + results_df['std_test_score'],
                        alpha=0.3)
        plt.xlabel(param_name)
        plt.ylabel(f'CV {scoring}')
        plt.title(f'{param_name} vs {scoring}')
        plt.grid(True)
        plt.show()
    
    return grid_search.best_estimator_, grid_search.best_params_

# Usage
# param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
# best_model, best_params = tune_hyperparameters(RandomForestClassifier(), param_grid, X_train, y_train)
```

## 📊 Visualization Templates

### **Model Performance Visualization**
```python
def plot_model_performance(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """Comprehensive model performance visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title(f'{model_name} - Confusion Matrix')
    axes[0,0].set_ylabel('Actual')
    axes[0,0].set_xlabel('Predicted')
    
    # Classification Report Heatmap
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :].T
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', ax=axes[0,1])
    axes[0,1].set_title(f'{model_name} - Classification Report')
    
    if y_pred_proba is not None:
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        axes[1,0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[1,0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1,0].set_xlabel('False Positive Rate')
        axes[1,0].set_ylabel('True Positive Rate')
        axes[1,0].set_title(f'{model_name} - ROC Curve')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        axes[1,1].plot(recall, precision, label=f'PR Curve')
        axes[1,1].set_xlabel('Recall')
        axes[1,1].set_ylabel('Precision')
        axes[1,1].set_title(f'{model_name} - Precision-Recall Curve')
        axes[1,1].legend()
        axes[1,1].grid(True)
    else:
        # Remove empty subplots
        fig.delaxes(axes[1,0])
        fig.delaxes(axes[1,1])
    
    plt.tight_layout()
    plt.show()

# Usage
# plot_model_performance(y_test, y_pred, y_pred_proba, "Random Forest")
```

---

**Navigation:**
- **Related**: [ML Algorithms Cheat Sheet](ml-algorithms-cheatsheet.md)
- **Reference Home**: [Quick Reference](README.md)
- **Course Home**: [Main Guide](../README.md)

*Copy and customize these templates for your ML projects!* 🚀
