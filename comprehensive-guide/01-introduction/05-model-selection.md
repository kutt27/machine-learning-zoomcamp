# 🎯 Model Selection Process

> **Learn to choose the best model while avoiding overfitting and ensuring generalization**

Model selection is one of the most critical aspects of machine learning. It's not just about finding the model with the highest accuracy on your training data—it's about finding the model that will perform best on new, unseen data.

## 🧠 Why Model Selection Matters

### **The Fundamental Challenge**
- **Goal**: Build models that generalize well to new data
- **Problem**: We only have access to historical data for training
- **Solution**: Proper validation strategies that simulate real-world performance

### **Common Mistakes**
- **Data Leakage**: Using future information to predict the past
- **Overfitting**: Models that memorize training data but fail on new data
- **Multiple Comparisons**: Testing too many models without proper validation
- **Biased Evaluation**: Using the same data for both selection and evaluation

## 📊 Data Splitting Strategies

### **1. Three-Way Split (Hold-Out Method)**

The most common and intuitive approach for model selection.

```python
from sklearn.model_selection import train_test_split

# First split: separate test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: separate train and validation (60% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
```

#### **Purpose of Each Set**
- **Training Set (60%)**: Used to train models
- **Validation Set (20%)**: Used to select best model and tune hyperparameters
- **Test Set (20%)**: Used for final, unbiased evaluation

#### **When to Use**
- Large datasets (>10,000 samples)
- When you have enough data for reliable estimates
- Simple and fast implementation needed

### **2. Cross-Validation**

More robust approach that uses data more efficiently.

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model_cv(model, X, y):
    """Evaluate model using cross-validation"""
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'scores': scores
    }

# Example usage
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
results = evaluate_model_cv(model, X_train, y_train)

print(f"CV Score: {results['mean_score']:.3f} (+/- {results['std_score']*2:.3f})")
```

#### **Advantages**
- **Better Data Utilization**: Uses all data for both training and validation
- **More Robust Estimates**: Reduces variance in performance estimates
- **Confidence Intervals**: Provides uncertainty estimates

#### **When to Use**
- Smaller datasets (<10,000 samples)
- When you need robust performance estimates
- For hyperparameter tuning

### **3. Time Series Split**

Special case for temporal data where order matters.

```python
from sklearn.model_selection import TimeSeriesSplit

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

def evaluate_time_series_model(model, X, y):
    """Evaluate model respecting temporal order"""
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
    
    return np.array(scores)
```

#### **When to Use**
- Time series data
- When temporal order is important
- Forecasting problems

## 🔍 Model Selection Workflow

### **Step 1: Define Evaluation Metric**

Choose metrics that align with your business objectives.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def comprehensive_evaluation(y_true, y_pred, y_pred_proba=None):
    """Calculate multiple evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics
```

### **Step 2: Create Model Candidates**

Start with simple baselines and gradually increase complexity.

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Define model candidates
models = {
    'Dummy (Baseline)': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}
```

### **Step 3: Evaluate Models**

Compare models using your validation strategy.

```python
def compare_models(models, X_train, y_train, X_val, y_val):
    """Compare multiple models on validation set"""
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Evaluate
        metrics = comprehensive_evaluation(y_val, y_pred, y_pred_proba)
        results[name] = metrics
        
        print(f"{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
        print()
    
    return results

# Compare models
model_results = compare_models(models, X_train, y_train, X_val, y_val)
```

### **Step 4: Hyperparameter Tuning**

Optimize the best performing models.

```python
from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(model, param_grid, X_train, y_train, cv=5):
    """Tune hyperparameters using grid search"""
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, 
        scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }

# Example: Tune Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf_tuning_results = tune_hyperparameters(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    X_train, y_train
)

print(f"Best parameters: {rf_tuning_results['best_params']}")
print(f"Best CV score: {rf_tuning_results['best_score']:.3f}")
```

### **Step 5: Final Model Selection**

Choose the best model based on validation performance.

```python
def select_best_model(results, metric='roc_auc'):
    """Select best model based on specified metric"""
    best_model = max(results.items(), key=lambda x: x[1][metric])
    return best_model[0], best_model[1][metric]

best_model_name, best_score = select_best_model(model_results, 'roc_auc')
print(f"Best model: {best_model_name} (ROC-AUC: {best_score:.3f})")
```

### **Step 6: Final Evaluation**

Evaluate the selected model on the test set.

```python
def final_evaluation(best_model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Final evaluation on test set"""
    # Retrain on combined train + validation data
    X_full_train = np.vstack([X_train, X_val])
    y_full_train = np.hstack([y_train, y_val])
    
    best_model.fit(X_full_train, y_full_train)
    
    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_metrics = comprehensive_evaluation(y_test, y_test_pred, y_test_proba)
    
    print("Final Test Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    return test_metrics

# Get the best model and evaluate
best_model = models[best_model_name]
final_metrics = final_evaluation(best_model, X_train, y_train, X_val, y_val, X_test, y_test)
```

## ⚠️ Common Pitfalls and How to Avoid Them

### **1. Data Leakage**

**Problem**: Using future information to predict the past.

```python
# ❌ WRONG: Feature engineering before splitting
X_scaled = StandardScaler().fit_transform(X)  # Uses information from entire dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# ✅ CORRECT: Feature engineering after splitting
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
X_test_scaled = scaler.transform(X_test)  # Transform test data using training statistics
```

### **2. Multiple Comparisons Problem**

**Problem**: Testing many models increases chance of finding spuriously good results.

**Solutions**:
- Use nested cross-validation for unbiased estimates
- Apply Bonferroni correction for multiple comparisons
- Hold out a separate test set that's never used for model selection

### **3. Overfitting to Validation Set**

**Problem**: Repeatedly evaluating on the same validation set can lead to overfitting.

**Solutions**:
- Use cross-validation instead of single validation set
- Limit the number of model iterations
- Use a separate test set for final evaluation

### **4. Insufficient Data Splitting**

**Problem**: Validation/test sets too small to provide reliable estimates.

**Rule of Thumb**:
- Minimum 1000 samples for test set
- At least 30 positive examples in each class for classification
- Use cross-validation for smaller datasets

## 🎯 Best Practices

### **Model Selection Checklist**
- [ ] **Clear Objective**: Define what "best" means for your problem
- [ ] **Proper Splitting**: Ensure no data leakage between sets
- [ ] **Baseline Models**: Start with simple baselines
- [ ] **Multiple Metrics**: Don't rely on a single metric
- [ ] **Cross-Validation**: Use CV for robust estimates
- [ ] **Hyperparameter Tuning**: Optimize promising models
- [ ] **Final Evaluation**: Test on held-out data
- [ ] **Documentation**: Record all decisions and rationale

### **Practical Tips**
1. **Start Simple**: Begin with logistic regression or decision trees
2. **Understand Your Data**: Know your features and target distribution
3. **Consider Business Constraints**: Factor in interpretability, speed, etc.
4. **Monitor Performance**: Track metrics over time in production
5. **Iterate**: Model selection is an iterative process

## 📊 Example: Complete Model Selection Pipeline

```python
class ModelSelector:
    def __init__(self, models, cv=5, test_size=0.2, random_state=42):
        self.models = models
        self.cv = cv
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}
        self.best_model = None
    
    def split_data(self, X, y):
        """Split data into train/validation/test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def evaluate_models(self, X_train, y_train):
        """Evaluate all models using cross-validation"""
        for name, model in self.models.items():
            scores = cross_val_score(model, X_train, y_train, cv=self.cv, scoring='roc_auc')
            self.results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            print(f"{name}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
    
    def select_best_model(self):
        """Select the best performing model"""
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['mean_score'])
        self.best_model = self.models[best_name]
        return best_name, self.results[best_name]['mean_score']
    
    def final_evaluation(self, X_train, y_train, X_test, y_test):
        """Final evaluation on test set"""
        self.best_model.fit(X_train, y_train)
        test_score = self.best_model.score(X_test, y_test)
        return test_score

# Usage example
selector = ModelSelector(models)
X_train, X_test, y_train, y_test = selector.split_data(X, y)
selector.evaluate_models(X_train, y_train)
best_name, best_score = selector.select_best_model()
final_score = selector.final_evaluation(X_train, y_train, X_test, y_test)

print(f"\nBest model: {best_name}")
print(f"CV score: {best_score:.3f}")
print(f"Test score: {final_score:.3f}")
```

## 📚 Additional Resources

- **Scikit-learn Guide**: [Model Selection](https://scikit-learn.org/stable/model_selection.html)
- **Cross-Validation**: [Detailed Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- **Hyperparameter Tuning**: [Grid Search and Random Search](https://scikit-learn.org/stable/modules/grid_search.html)

---

**Navigation:**
- **Previous**: [CRISP-DM Methodology](04-crisp-dm.md)
- **Next**: [Environment Setup](06-environment.md)
- **Module Home**: [Introduction](README.md)

*Last Updated: 2025-01-27*
