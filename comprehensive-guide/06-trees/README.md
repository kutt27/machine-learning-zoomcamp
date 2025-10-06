# 🎯 Module 6: Decision Trees and Ensemble Learning

> **Master tree-based algorithms and ensemble methods for superior performance**

This module covers decision trees and powerful ensemble methods including Random Forest and XGBoost. You'll learn when and how to use these algorithms for both classification and regression tasks.

## 📚 Learning Objectives

By the end of this module, you will:
- **Understand** decision tree algorithms and their mechanics
- **Master** ensemble methods: Random Forest and Gradient Boosting
- **Implement** XGBoost for high-performance machine learning
- **Tune** hyperparameters for optimal performance
- **Handle** feature importance and model interpretation
- **Apply** tree-based methods to real-world problems

## 🌳 Why Tree-Based Methods?

### Advantages
- **Interpretability**: Easy to understand and visualize
- **No assumptions**: Don't require data distribution assumptions
- **Handle mixed data**: Work with both numerical and categorical features
- **Feature selection**: Automatically identify important features
- **Non-linear patterns**: Capture complex relationships

### Ensemble Power
- **Reduced overfitting**: Combine multiple models for better generalization
- **Improved accuracy**: Often achieve state-of-the-art performance
- **Robustness**: Less sensitive to outliers and noise

## 🗂️ Module Contents

### **6.1 Credit Risk Assessment Project**
**Business Context:**
```python
# Credit risk prediction framework
class CreditRiskPredictor:
    def __init__(self):
        self.risk_categories = {
            'low_risk': 0,
            'medium_risk': 1, 
            'high_risk': 2
        }
        self.business_rules = {
            'max_loan_amount': 100000,
            'min_credit_score': 300,
            'max_debt_to_income': 0.5
        }
    
    def calculate_business_impact(self, predictions, loan_amounts):
        """Calculate business impact of risk predictions"""
        
        # Define costs and benefits
        cost_default = 0.8  # 80% loss on default
        profit_good_loan = 0.05  # 5% profit on good loans
        cost_rejected_good = 0.02  # Opportunity cost
        
        total_impact = 0
        
        for pred, amount in zip(predictions, loan_amounts):
            if pred == 0:  # Low risk - approve
                total_impact += amount * profit_good_loan
            elif pred == 1:  # Medium risk - approve with conditions
                total_impact += amount * (profit_good_loan * 0.7)
            else:  # High risk - reject
                total_impact -= amount * cost_rejected_good
        
        return total_impact
```

### **6.2 Data Preparation for Tree Models**
**Tree-Specific Preprocessing:**
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prepare_tree_data(df):
    """Prepare data specifically for tree-based models"""
    
    # Trees handle missing values well, but let's be explicit
    df_processed = df.copy()
    
    # 1. Handle categorical variables
    # Trees can handle label encoding better than one-hot for high cardinality
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # 2. Create interaction features (trees will find these anyway, but helps)
    if 'income' in df.columns and 'debt' in df.columns:
        df_processed['debt_to_income_ratio'] = df['debt'] / (df['income'] + 1)
    
    # 3. Binning continuous variables (optional for trees)
    if 'age' in df.columns:
        df_processed['age_group'] = pd.cut(df['age'], 
                                         bins=[0, 25, 35, 50, 65, 100], 
                                         labels=['young', 'adult', 'middle', 'senior', 'elderly'])
        df_processed['age_group'] = LabelEncoder().fit_transform(df_processed['age_group'])
    
    # 4. Feature engineering for credit risk
    if 'credit_history_length' in df.columns and 'num_accounts' in df.columns:
        df_processed['avg_account_age'] = df['credit_history_length'] / (df['num_accounts'] + 1)
    
    return df_processed, label_encoders
```

### **6.3 Decision Trees Deep Dive**
**From Scratch Implementation:**
```python
import numpy as np
from collections import Counter

class DecisionTreeFromScratch:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def gini_impurity(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        impurity = 1.0
        
        for count in counts.values():
            prob = count / len(y)
            impurity -= prob ** 2
        
        return impurity
    
    def entropy(self, y):
        """Calculate entropy"""
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        entropy = 0.0
        
        for count in counts.values():
            prob = count / len(y)
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def information_gain(self, X_column, y, threshold):
        """Calculate information gain for a split"""
        
        # Parent entropy
        parent_entropy = self.entropy(y)
        
        # Split data
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        # Calculate weighted entropy of children
        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        
        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])
        
        weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        
        return parent_entropy - weighted_entropy
    
    def find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                gain = self.information_gain(feature_values, y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1):
            
            # Return leaf node
            return {
                'type': 'leaf',
                'prediction': Counter(y).most_common(1)[0][0],
                'samples': len(y),
                'distribution': dict(Counter(y))
            }
        
        # Find best split
        best_feature, best_threshold, best_gain = self.find_best_split(X, y)
        
        if best_gain == 0:
            # No good split found
            return {
                'type': 'leaf',
                'prediction': Counter(y).most_common(1)[0][0],
                'samples': len(y),
                'distribution': dict(Counter(y))
            }
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Check minimum samples in leaf
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return {
                'type': 'leaf',
                'prediction': Counter(y).most_common(1)[0][0],
                'samples': len(y),
                'distribution': dict(Counter(y))
            }
        
        # Recursively build subtrees
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'type': 'split',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree,
            'samples': len(y),
            'gain': best_gain
        }
    
    def fit(self, X, y):
        """Train the decision tree"""
        self.tree = self.build_tree(X, y)
        return self
    
    def predict_sample(self, x, tree):
        """Predict a single sample"""
        if tree['type'] == 'leaf':
            return tree['prediction']
        
        if x[tree['feature']] <= tree['threshold']:
            return self.predict_sample(x, tree['left'])
        else:
            return self.predict_sample(x, tree['right'])
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_sample(x, self.tree) for x in X])
    
    def print_tree(self, tree=None, depth=0):
        """Print the tree structure"""
        if tree is None:
            tree = self.tree
        
        indent = "  " * depth
        
        if tree['type'] == 'leaf':
            print(f"{indent}Leaf: {tree['prediction']} (samples: {tree['samples']})")
        else:
            print(f"{indent}Feature {tree['feature']} <= {tree['threshold']:.2f} (gain: {tree['gain']:.3f})")
            print(f"{indent}├─ True:")
            self.print_tree(tree['left'], depth + 1)
            print(f"{indent}└─ False:")
            self.print_tree(tree['right'], depth + 1)
```

### **6.4 Random Forest Implementation**
**Advanced Random Forest with Feature Importance:**
```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class AdvancedRandomForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None
        self.feature_importance_df = None
    
    def fit(self, X, y, feature_names=None):
        """Train Random Forest with feature importance analysis"""
        
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Train model
        self.model.fit(X, y)
        
        # Calculate feature importance
        self.analyze_feature_importance()
        
        return self
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create feature importance dataframe
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Calculate standard deviation of importance across trees
        tree_importances = [tree.feature_importances_ for tree in self.model.estimators_]
        importance_std = np.std(tree_importances, axis=0)
        
        self.feature_importance_df['importance_std'] = importance_std[
            self.feature_importance_df.index
        ]
        
        return self.feature_importance_df
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance with error bars"""
        
        top_features = self.feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'], 
                xerr=top_features['importance_std'], alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def get_tree_diversity(self):
        """Analyze diversity among trees"""
        
        # Get predictions from each tree
        tree_predictions = np.array([
            tree.predict(self.X_train) for tree in self.model.estimators_
        ])
        
        # Calculate pairwise agreement
        n_trees = len(self.model.estimators_)
        agreements = []
        
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                agreement = np.mean(tree_predictions[i] == tree_predictions[j])
                agreements.append(agreement)
        
        avg_agreement = np.mean(agreements)
        
        print(f"Average pairwise agreement between trees: {avg_agreement:.3f}")
        print(f"Tree diversity (1 - agreement): {1 - avg_agreement:.3f}")
        
        return avg_agreement
    
    def predict_with_uncertainty(self, X):
        """Predict with uncertainty estimation"""
        
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict_proba(X) for tree in self.model.estimators_
        ])
        
        # Calculate mean and standard deviation
        mean_proba = np.mean(tree_predictions, axis=0)
        std_proba = np.std(tree_predictions, axis=0)
        
        # Final predictions
        predictions = np.argmax(mean_proba, axis=1)
        
        # Uncertainty (entropy of predictions)
        uncertainty = -np.sum(mean_proba * np.log(mean_proba + 1e-10), axis=1)
        
        return predictions, mean_proba, uncertainty
```

### **6.5 XGBoost Mastery**
**Advanced XGBoost Implementation:**
```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna

class XGBoostOptimizer:
    def __init__(self, objective='binary:logistic', eval_metric='logloss'):
        self.objective = objective
        self.eval_metric = eval_metric
        self.best_model = None
        self.best_params = None
        self.optimization_history = []
    
    def basic_hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """Basic hyperparameter tuning with GridSearch"""
        
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective=self.objective,
            eval_metric=self.eval_metric,
            random_state=42
        )
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, 
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_model = grid_search.best_estimator_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return self.best_model
    
    def advanced_optuna_tuning(self, X_train, y_train, X_val, y_val, n_trials=100):
        """Advanced hyperparameter tuning with Optuna"""
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5)
            }
            
            model = xgb.XGBClassifier(
                **params,
                objective=self.objective,
                eval_metric=self.eval_metric,
                random_state=42
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        
        # Train final model with best parameters
        self.best_model = xgb.XGBClassifier(
            **self.best_params,
            objective=self.objective,
            eval_metric=self.eval_metric,
            random_state=42
        )
        
        self.best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best score: {study.best_value:.4f}")
        
        return self.best_model
    
    def plot_learning_curves(self, X_train, y_train, X_val, y_val):
        """Plot learning curves to analyze training progress"""
        
        if self.best_model is None:
            print("No model trained yet. Run hyperparameter tuning first.")
            return
        
        # Get evaluation results
        eval_results = self.best_model.evals_result()
        
        plt.figure(figsize=(12, 4))
        
        # Training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(eval_results['validation_0'][self.eval_metric], label='Validation')
        plt.xlabel('Boosting Round')
        plt.ylabel(self.eval_metric)
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        
        # Feature importance
        plt.subplot(1, 2, 2)
        xgb.plot_importance(self.best_model, max_num_features=20, height=0.8)
        plt.title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_model_complexity(self, X_train, y_train, X_val, y_val):
        """Analyze model complexity and overfitting"""
        
        n_estimators_range = range(50, 501, 50)
        train_scores = []
        val_scores = []
        
        for n_est in n_estimators_range:
            model = xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=self.best_params.get('max_depth', 6),
                learning_rate=self.best_params.get('learning_rate', 0.1),
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            train_pred = model.predict_proba(X_train)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]
            
            train_score = roc_auc_score(y_train, train_pred)
            val_score = roc_auc_score(y_val, val_pred)
            
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        plt.figure(figsize=(10, 6))
        plt.plot(n_estimators_range, train_scores, 'o-', label='Training AUC')
        plt.plot(n_estimators_range, val_scores, 'o-', label='Validation AUC')
        plt.xlabel('Number of Estimators')
        plt.ylabel('AUC Score')
        plt.title('Model Complexity Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Find optimal number of estimators
        optimal_idx = np.argmax(val_scores)
        optimal_n_estimators = n_estimators_range[optimal_idx]
        
        print(f"Optimal number of estimators: {optimal_n_estimators}")
        print(f"Best validation AUC: {val_scores[optimal_idx]:.4f}")
        
        return optimal_n_estimators
```

### **6.6 Model Interpretation and SHAP**
**Advanced Model Interpretation:**
```python
import shap
import matplotlib.pyplot as plt

class TreeModelInterpreter:
    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.explainer = None
        self.shap_values = None
    
    def setup_shap_explainer(self):
        """Setup SHAP explainer for tree models"""
        
        if hasattr(self.model, 'estimators_'):  # Random Forest
            self.explainer = shap.TreeExplainer(self.model)
        elif hasattr(self.model, 'get_booster'):  # XGBoost
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Fallback to general explainer
            self.explainer = shap.Explainer(self.model, self.X_train)
    
    def calculate_shap_values(self, X_explain):
        """Calculate SHAP values for explanation"""
        
        if self.explainer is None:
            self.setup_shap_explainer()
        
        self.shap_values = self.explainer.shap_values(X_explain)
        
        # For binary classification, take positive class
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        return self.shap_values
    
    def plot_shap_summary(self, X_explain, max_display=20):
        """Plot SHAP summary plot"""
        
        if self.shap_values is None:
            self.calculate_shap_values(X_explain)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, X_explain, 
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.show()
    
    def plot_shap_waterfall(self, X_explain, sample_idx=0):
        """Plot SHAP waterfall for single prediction"""
        
        if self.shap_values is None:
            self.calculate_shap_values(X_explain)
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=X_explain.iloc[sample_idx] if hasattr(X_explain, 'iloc') else X_explain[sample_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_interactions(self, X_explain, feature1_idx, feature2_idx):
        """Analyze interaction between two features"""
        
        if self.explainer is None:
            self.setup_shap_explainer()
        
        # Calculate interaction values
        interaction_values = self.explainer.shap_interaction_values(X_explain)
        
        # Plot interaction
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            (feature1_idx, feature2_idx),
            interaction_values, X_explain,
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f'Interaction: {self.feature_names[feature1_idx]} vs {self.feature_names[feature2_idx]}')
        plt.tight_layout()
        plt.show()
    
    def global_feature_importance(self, X_explain):
        """Calculate global feature importance using SHAP"""
        
        if self.shap_values is None:
            self.calculate_shap_values(X_explain)
        
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Global Feature Importance (SHAP)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df
```

## 🛠️ Complete Tree-Based Pipeline

```python
class ComprehensiveTreePipeline:
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.feature_names = None
    
    def train_all_models(self, X_train, y_train, X_val, y_val, feature_names=None):
        """Train all tree-based models and compare"""
        
        self.feature_names = feature_names
        
        # 1. Decision Tree
        print("Training Decision Tree...")
        dt = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)
        dt.fit(X_train, y_train)
        dt_score = roc_auc_score(y_val, dt.predict_proba(X_val)[:, 1])
        self.models['decision_tree'] = {'model': dt, 'score': dt_score}
        
        # 2. Random Forest
        print("Training Random Forest...")
        rf = AdvancedRandomForest(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train, feature_names)
        rf_score = roc_auc_score(y_val, rf.model.predict_proba(X_val)[:, 1])
        self.models['random_forest'] = {'model': rf, 'score': rf_score}
        
        # 3. XGBoost
        print("Training XGBoost...")
        xgb_optimizer = XGBoostOptimizer()
        xgb_model = xgb_optimizer.basic_hyperparameter_tuning(X_train, y_train, X_val, y_val)
        xgb_score = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])
        self.models['xgboost'] = {'model': xgb_model, 'score': xgb_score}
        
        # Find best model
        for name, model_info in self.models.items():
            if model_info['score'] > self.best_score:
                self.best_score = model_info['score']
                self.best_model = model_info['model']
        
        # Print results
        print("\n=== MODEL COMPARISON ===")
        for name, model_info in self.models.items():
            print(f"{name}: {model_info['score']:.4f}")
        
        print(f"\nBest model: {self.get_best_model_name()} (AUC: {self.best_score:.4f})")
    
    def get_best_model_name(self):
        """Get name of best performing model"""
        for name, model_info in self.models.items():
            if model_info['score'] == self.best_score:
                return name
        return None
    
    def comprehensive_evaluation(self, X_test, y_test):
        """Comprehensive evaluation of all models"""
        
        results = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Get model predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.model.predict_proba(X_test)[:, 1]
            
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate metrics
            results[name] = {
                'auc': roc_auc_score(y_test, y_pred_proba),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
        
        # Create comparison dataframe
        results_df = pd.DataFrame(results).T
        
        print("=== COMPREHENSIVE EVALUATION ===")
        print(results_df.round(4))
        
        return results_df
    
    def interpret_best_model(self, X_explain):
        """Interpret the best performing model"""
        
        if self.best_model is None:
            print("No model trained yet.")
            return
        
        interpreter = TreeModelInterpreter(self.best_model, X_explain, self.feature_names)
        
        # SHAP analysis
        interpreter.plot_shap_summary(X_explain)
        interpreter.global_feature_importance(X_explain)
        
        return interpreter
```

## 🎯 Module Completion Checklist

- [ ] Understand decision tree algorithms and splitting criteria
- [ ] Can implement Random Forest and understand ensemble benefits
- [ ] Master XGBoost hyperparameter tuning and optimization
- [ ] Know how to interpret tree-based models using SHAP
- [ ] Can compare different tree-based algorithms effectively
- [ ] Understand when to use each tree-based method

## 🔗 Additional Resources

### **Video Lectures**
- [Trees and Ensemble Learning Playlist](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)

### **Research Papers**
- [Random Forests (Breiman, 2001)](https://link.springer.com/article/10.1023/A:1010933404324)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)

## 🎯 Next Steps

After completing this module, you're ready for **Module 8: Deep Learning**, where you'll learn neural networks and modern deep learning techniques.

---

**Navigation:**
- **Previous**: [Module 5: Deployment](../05-deployment/README.md)
- **Next**: [Module 8: Deep Learning](../08-deep-learning/README.md)
- **Course Home**: [Main Guide](../README.md)

*Last Updated: 2025-01-27*
