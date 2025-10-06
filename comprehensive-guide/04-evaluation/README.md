# 🎯 Module 4: Evaluation Metrics for Classification

> **Master comprehensive evaluation techniques for classification models**

This module provides a deep dive into evaluation metrics for classification problems. You'll learn when and how to use different metrics, understand their trade-offs, and implement proper evaluation frameworks.

## 📚 Learning Objectives

By the end of this module, you will:
- **Master** all major classification metrics and their interpretations
- **Understand** when to use each metric based on business context
- **Implement** proper cross-validation and evaluation frameworks
- **Analyze** model performance using ROC curves and AUC
- **Handle** imbalanced datasets and metric selection
- **Build** comprehensive evaluation pipelines

## 🎯 Why Evaluation Matters

Proper evaluation is crucial because:
- **Business Impact**: Wrong metrics can lead to poor business decisions
- **Model Selection**: Choose the best model for your specific problem
- **Performance Monitoring**: Track model degradation in production
- **Stakeholder Communication**: Explain model performance clearly

## 🗂️ Module Contents

### **4.1 Evaluation Overview**
**Key Concepts:**
- Classification vs regression evaluation
- Business context in metric selection
- Evaluation framework design

### **4.2 Accuracy and Its Limitations**
**Understanding Accuracy:**
```python
def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy with detailed analysis"""
    correct_predictions = (y_true == y_pred).sum()
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    
    print(f"Correct predictions: {correct_predictions}")
    print(f"Total predictions: {total_predictions}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return accuracy

# When accuracy is misleading
def accuracy_paradox_example():
    """Demonstrate when accuracy can be misleading"""
    
    # Imbalanced dataset: 95% negative, 5% positive
    y_true = np.array([0]*950 + [1]*50)
    
    # Naive classifier: always predict negative
    y_pred_naive = np.array([0]*1000)
    
    # Smart classifier: some correct positive predictions
    y_pred_smart = np.array([0]*920 + [1]*30 + [0]*20 + [1]*30)
    
    acc_naive = calculate_accuracy(y_true, y_pred_naive)
    acc_smart = calculate_accuracy(y_true, y_pred_smart)
    
    print(f"\nNaive classifier accuracy: {acc_naive:.3f}")
    print(f"Smart classifier accuracy: {acc_smart:.3f}")
    print("Accuracy alone doesn't tell the full story!")
```

### **4.3 Confusion Matrix Deep Dive**
**Complete Confusion Matrix Analysis:**
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class ConfusionMatrixAnalyzer:
    def __init__(self, y_true, y_pred, class_names=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.class_names = class_names or ['Negative', 'Positive']
        self.cm = confusion_matrix(y_true, y_pred)
        
    def plot_confusion_matrix(self, normalize=False):
        """Plot confusion matrix with detailed annotations"""
        
        if normalize:
            cm_plot = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            cm_plot = self.cm
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add counts in corners
        if not normalize:
            tn, fp, fn, tp = self.cm.ravel()
            plt.text(0.5, 0.1, f'TN: {tn}', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
            plt.text(1.5, 0.1, f'FP: {fp}', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
            plt.text(0.5, 1.1, f'FN: {fn}', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
            plt.text(1.5, 1.1, f'TP: {tp}', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_all_metrics(self):
        """Calculate all metrics from confusion matrix"""
        tn, fp, fn, tp = self.cm.ravel()
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Advanced metrics
        balanced_accuracy = (recall + specificity) / 2
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
        
        metrics = {
            'True Positives (TP)': tp,
            'True Negatives (TN)': tn,
            'False Positives (FP)': fp,
            'False Negatives (FN)': fn,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall (Sensitivity)': recall,
            'Specificity': specificity,
            'F1-Score': f1_score,
            'Balanced Accuracy': balanced_accuracy,
            'Matthews Correlation Coefficient': mcc
        }
        
        return metrics
    
    def print_detailed_analysis(self):
        """Print comprehensive analysis"""
        metrics = self.calculate_all_metrics()
        
        print("=== CONFUSION MATRIX ANALYSIS ===")
        print(f"Total Samples: {len(self.y_true)}")
        print(f"Positive Samples: {sum(self.y_true)} ({sum(self.y_true)/len(self.y_true):.1%})")
        print(f"Negative Samples: {len(self.y_true) - sum(self.y_true)} ({(len(self.y_true) - sum(self.y_true))/len(self.y_true):.1%})")
        print()
        
        print("=== CONFUSION MATRIX VALUES ===")
        print(f"True Positives (TP):  {metrics['True Positives (TP)']}")
        print(f"True Negatives (TN):  {metrics['True Negatives (TN)']}")
        print(f"False Positives (FP): {metrics['False Positives (FP)']}")
        print(f"False Negatives (FN): {metrics['False Negatives (FN)']}")
        print()
        
        print("=== PERFORMANCE METRICS ===")
        for metric, value in metrics.items():
            if metric not in ['True Positives (TP)', 'True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)']:
                print(f"{metric}: {value:.3f}")
```

### **4.4 Precision and Recall Trade-offs**
**Understanding the Trade-off:**
```python
def precision_recall_analysis(y_true, y_pred_proba, thresholds=None):
    """Analyze precision-recall trade-offs across thresholds"""
    
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
    
    results_df = pd.DataFrame(results)
    
    # Plot precision-recall trade-off
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(results_df['threshold'], results_df['precision'], 'b-', label='Precision', marker='o')
    plt.plot(results_df['threshold'], results_df['recall'], 'r-', label='Recall', marker='s')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision vs Recall by Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(results_df['recall'], results_df['precision'], 'g-', marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(results_df['threshold'], results_df['f1_score'], 'purple', marker='d')
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.title('F1-Score by Threshold')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal threshold
    optimal_idx = results_df['f1_score'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    optimal_f1 = results_df.loc[optimal_idx, 'f1_score']
    
    print(f"Optimal threshold for F1-score: {optimal_threshold:.2f}")
    print(f"Optimal F1-score: {optimal_f1:.3f}")
    
    return results_df
```

### **4.5 ROC Curves and AUC**
**Comprehensive ROC Analysis:**
```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

class ROCAnalyzer:
    def __init__(self, y_true, y_pred_proba):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.fpr, self.tpr, self.thresholds = roc_curve(y_true, y_pred_proba)
        self.auc_score = auc(self.fpr, self.tpr)
    
    def plot_roc_curve(self, title="ROC Curve"):
        """Plot ROC curve with detailed annotations"""
        
        plt.figure(figsize=(10, 8))
        
        # Main ROC curve
        plt.subplot(2, 2, 1)
        plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {self.auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Threshold analysis
        plt.subplot(2, 2, 2)
        plt.plot(self.thresholds, self.fpr[:-1], 'b-', label='False Positive Rate')
        plt.plot(self.thresholds, self.tpr[:-1], 'r-', label='True Positive Rate')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('TPR and FPR vs Threshold')
        plt.legend()
        plt.grid(True)
        
        # Youden's J statistic (optimal threshold)
        plt.subplot(2, 2, 3)
        j_scores = self.tpr - self.fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = self.thresholds[optimal_idx]
        
        plt.plot(self.thresholds, j_scores[:-1])
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal threshold: {optimal_threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel("Youden's J (TPR - FPR)")
        plt.title("Optimal Threshold Selection")
        plt.legend()
        plt.grid(True)
        
        # AUC interpretation
        plt.subplot(2, 2, 4)
        auc_interpretation = self.interpret_auc()
        plt.text(0.1, 0.5, auc_interpretation, fontsize=12, 
                verticalalignment='center', transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title('AUC Interpretation')
        
        plt.tight_layout()
        plt.show()
        
        return optimal_threshold
    
    def interpret_auc(self):
        """Provide interpretation of AUC score"""
        if self.auc_score >= 0.9:
            interpretation = f"Excellent classifier (AUC = {self.auc_score:.3f})\nThe model has outstanding discriminative ability."
        elif self.auc_score >= 0.8:
            interpretation = f"Good classifier (AUC = {self.auc_score:.3f})\nThe model has good discriminative ability."
        elif self.auc_score >= 0.7:
            interpretation = f"Fair classifier (AUC = {self.auc_score:.3f})\nThe model has acceptable discriminative ability."
        elif self.auc_score >= 0.6:
            interpretation = f"Poor classifier (AUC = {self.auc_score:.3f})\nThe model has poor discriminative ability."
        else:
            interpretation = f"Very poor classifier (AUC = {self.auc_score:.3f})\nThe model performs worse than random guessing."
        
        return interpretation
    
    def calculate_optimal_threshold(self, method='youden'):
        """Calculate optimal threshold using different methods"""
        
        if method == 'youden':
            # Youden's J statistic: maximize (TPR - FPR)
            j_scores = self.tpr - self.fpr
            optimal_idx = np.argmax(j_scores)
        elif method == 'closest_to_topleft':
            # Point closest to top-left corner (0, 1)
            distances = np.sqrt((self.fpr - 0)**2 + (self.tpr - 1)**2)
            optimal_idx = np.argmin(distances)
        elif method == 'f1_optimal':
            # Threshold that maximizes F1-score
            f1_scores = []
            for threshold in self.thresholds:
                y_pred = (self.y_pred_proba >= threshold).astype(int)
                f1 = f1_score(self.y_true, y_pred)
                f1_scores.append(f1)
            optimal_idx = np.argmax(f1_scores)
        
        optimal_threshold = self.thresholds[optimal_idx]
        optimal_tpr = self.tpr[optimal_idx]
        optimal_fpr = self.fpr[optimal_idx]
        
        return optimal_threshold, optimal_tpr, optimal_fpr
```

### **4.6 Cross-Validation for Robust Evaluation**
**Advanced Cross-Validation Strategies:**
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer

class CrossValidationEvaluator:
    def __init__(self, model, X, y, cv_folds=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    def comprehensive_cv_evaluation(self):
        """Perform comprehensive cross-validation evaluation"""
        
        # Define multiple scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'balanced_accuracy': 'balanced_accuracy'
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            self.model, self.X, self.y, 
            cv=self.cv, scoring=scoring, 
            return_train_score=True, n_jobs=-1
        )
        
        # Calculate statistics
        results_summary = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results_summary[metric] = {
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores),
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),
                'overfitting': np.mean(train_scores) - np.mean(test_scores)
            }
        
        # Print results
        print("=== CROSS-VALIDATION RESULTS ===")
        print(f"CV Folds: {self.cv_folds}")
        print(f"Total Samples: {len(self.y)}")
        print(f"Positive Class: {sum(self.y)} ({sum(self.y)/len(self.y):.1%})")
        print()
        
        for metric, stats in results_summary.items():
            print(f"{metric.upper()}:")
            print(f"  Test:  {stats['test_mean']:.3f} ± {stats['test_std']:.3f}")
            print(f"  Train: {stats['train_mean']:.3f} ± {stats['train_std']:.3f}")
            print(f"  Overfitting: {stats['overfitting']:.3f}")
            print()
        
        return results_summary
    
    def plot_cv_results(self, results_summary):
        """Plot cross-validation results"""
        
        metrics = list(results_summary.keys())
        test_means = [results_summary[m]['test_mean'] for m in metrics]
        test_stds = [results_summary[m]['test_std'] for m in metrics]
        train_means = [results_summary[m]['train_mean'] for m in metrics]
        train_stds = [results_summary[m]['train_std'] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(x - width/2, test_means, width, yerr=test_stds, 
               label='Test', alpha=0.8, capsize=5)
        plt.bar(x + width/2, train_means, width, yerr=train_stds, 
               label='Train', alpha=0.8, capsize=5)
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Cross-Validation Results')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Overfitting analysis
        plt.subplot(1, 2, 2)
        overfitting = [results_summary[m]['overfitting'] for m in metrics]
        colors = ['red' if x > 0.05 else 'green' for x in overfitting]
        plt.bar(metrics, overfitting, color=colors, alpha=0.7)
        plt.axhline(y=0.05, color='red', linestyle='--', label='Overfitting threshold')
        plt.xlabel('Metrics')
        plt.ylabel('Train - Test Score')
        plt.title('Overfitting Analysis')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
```

### **4.7 Handling Imbalanced Datasets**
**Evaluation Strategies for Imbalanced Data:**
```python
from sklearn.metrics import precision_recall_curve, average_precision_score
from imblearn.metrics import classification_report_imbalanced

class ImbalancedDatasetEvaluator:
    def __init__(self, y_true, y_pred, y_pred_proba):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.class_distribution = np.bincount(y_true) / len(y_true)
    
    def comprehensive_imbalanced_evaluation(self):
        """Comprehensive evaluation for imbalanced datasets"""
        
        print("=== IMBALANCED DATASET EVALUATION ===")
        print(f"Class distribution: {self.class_distribution}")
        print(f"Imbalance ratio: {self.class_distribution[0]/self.class_distribution[1]:.1f}:1")
        print()
        
        # Standard metrics
        print("=== STANDARD METRICS ===")
        print(classification_report(self.y_true, self.y_pred))
        
        # Imbalanced-learn metrics
        print("=== IMBALANCED-SPECIFIC METRICS ===")
        print(classification_report_imbalanced(self.y_true, self.y_pred))
        
        # Precision-Recall AUC (better for imbalanced data)
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_true, self.y_pred_proba)
        
        print(f"Precision-Recall AUC: {pr_auc:.3f}")
        
        # Plot PR curve vs ROC curve
        self.plot_pr_vs_roc()
        
        return pr_auc
    
    def plot_pr_vs_roc(self):
        """Compare Precision-Recall curve with ROC curve"""
        
        # Calculate curves
        precision, recall, pr_thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        fpr, tpr, roc_thresholds = roc_curve(self.y_true, self.y_pred_proba)
        
        pr_auc = average_precision_score(self.y_true, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_true, self.y_pred_proba)
        
        plt.figure(figsize=(12, 5))
        
        # Precision-Recall curve
        plt.subplot(1, 2, 1)
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.axhline(y=self.class_distribution[1], color='red', linestyle='--', 
                   label=f'Random classifier (AP = {self.class_distribution[1]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve\n(Better for imbalanced data)')
        plt.legend()
        plt.grid(True)
        
        # ROC curve
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier (AUC = 0.5)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve\n(Can be optimistic for imbalanced data)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Recommendation
        if self.class_distribution[1] < 0.1:  # Less than 10% positive class
            print("RECOMMENDATION: For this imbalanced dataset, focus on Precision-Recall AUC rather than ROC AUC")
        else:
            print("Dataset is moderately balanced. Both ROC AUC and PR AUC are informative.")
```

## 🛠️ Complete Evaluation Pipeline

```python
class ComprehensiveModelEvaluator:
    def __init__(self, model, X_test, y_test, class_names=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names or ['Negative', 'Positive']
        
        # Generate predictions
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    def full_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        
        print("=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("=" * 60)
        
        # 1. Basic Information
        print(f"Test set size: {len(self.y_test)}")
        print(f"Positive class prevalence: {self.y_test.mean():.1%}")
        print()
        
        # 2. Confusion Matrix Analysis
        cm_analyzer = ConfusionMatrixAnalyzer(self.y_test, self.y_pred, self.class_names)
        cm_analyzer.print_detailed_analysis()
        cm_analyzer.plot_confusion_matrix()
        
        # 3. ROC Analysis
        roc_analyzer = ROCAnalyzer(self.y_test, self.y_pred_proba)
        optimal_threshold = roc_analyzer.plot_roc_curve()
        
        # 4. Precision-Recall Analysis
        pr_results = precision_recall_analysis(self.y_test, self.y_pred_proba)
        
        # 5. Threshold Optimization
        print("\n=== THRESHOLD OPTIMIZATION ===")
        thresholds_to_test = [0.3, 0.5, 0.7, optimal_threshold]
        
        for threshold in thresholds_to_test:
            y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
            precision = precision_score(self.y_test, y_pred_thresh)
            recall = recall_score(self.y_test, y_pred_thresh)
            f1 = f1_score(self.y_test, y_pred_thresh)
            
            print(f"Threshold {threshold:.2f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # 6. Business Impact Analysis (if applicable)
        self.business_impact_analysis()
        
        return {
            'confusion_matrix': cm_analyzer.cm,
            'roc_auc': roc_analyzer.auc_score,
            'optimal_threshold': optimal_threshold,
            'precision_recall_results': pr_results
        }
    
    def business_impact_analysis(self):
        """Analyze business impact of model predictions"""
        
        print("\n=== BUSINESS IMPACT ANALYSIS ===")
        
        # Cost-benefit analysis framework
        # These would be customized based on actual business costs
        cost_false_positive = 10  # Cost of unnecessary action
        cost_false_negative = 100  # Cost of missed opportunity
        benefit_true_positive = 50  # Benefit of correct action
        
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()
        
        total_cost = (fp * cost_false_positive + fn * cost_false_negative)
        total_benefit = tp * benefit_true_positive
        net_benefit = total_benefit - total_cost
        
        print(f"True Positives: {tp} (Benefit: ${tp * benefit_true_positive})")
        print(f"False Positives: {fp} (Cost: ${fp * cost_false_positive})")
        print(f"False Negatives: {fn} (Cost: ${fn * cost_false_negative})")
        print(f"Net Business Impact: ${net_benefit}")
        
        # ROI calculation
        baseline_cost = len(self.y_test) * cost_false_negative * self.y_test.mean()  # Cost of doing nothing
        roi = ((baseline_cost - total_cost) / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        print(f"ROI compared to no model: {roi:.1f}%")
```

## 📊 Metric Selection Guide

```python
def metric_selection_guide(problem_context):
    """Guide for selecting appropriate metrics based on problem context"""
    
    guides = {
        'balanced_dataset': {
            'primary_metrics': ['Accuracy', 'F1-Score', 'ROC AUC'],
            'secondary_metrics': ['Precision', 'Recall'],
            'explanation': 'For balanced datasets, accuracy and F1-score are reliable. ROC AUC provides good discrimination assessment.'
        },
        'imbalanced_dataset': {
            'primary_metrics': ['Precision-Recall AUC', 'F1-Score', 'Balanced Accuracy'],
            'secondary_metrics': ['Precision', 'Recall', 'Matthews Correlation Coefficient'],
            'explanation': 'For imbalanced datasets, focus on precision-recall metrics. ROC AUC can be misleading.'
        },
        'cost_sensitive': {
            'primary_metrics': ['Custom Cost Function', 'Precision (if FP costly)', 'Recall (if FN costly)'],
            'secondary_metrics': ['F1-Score', 'ROC AUC'],
            'explanation': 'When different errors have different costs, use custom metrics or focus on the relevant precision/recall.'
        },
        'ranking_problem': {
            'primary_metrics': ['ROC AUC', 'Precision-Recall AUC'],
            'secondary_metrics': ['Top-K Precision', 'NDCG'],
            'explanation': 'For ranking problems, focus on metrics that evaluate the quality of the ranking order.'
        }
    }
    
    if problem_context in guides:
        guide = guides[problem_context]
        print(f"=== METRIC SELECTION GUIDE: {problem_context.upper()} ===")
        print(f"Primary metrics: {', '.join(guide['primary_metrics'])}")
        print(f"Secondary metrics: {', '.join(guide['secondary_metrics'])}")
        print(f"Explanation: {guide['explanation']}")
    else:
        print("Available contexts: balanced_dataset, imbalanced_dataset, cost_sensitive, ranking_problem")
```

## 🎯 Module Completion Checklist

- [ ] Understand all major classification metrics and their use cases
- [ ] Can interpret confusion matrices and calculate metrics manually
- [ ] Master ROC curves, AUC, and optimal threshold selection
- [ ] Understand precision-recall trade-offs and when to use each
- [ ] Can implement proper cross-validation evaluation
- [ ] Know how to evaluate imbalanced datasets appropriately
- [ ] Can select appropriate metrics based on business context

## 🔗 Additional Resources

### **Video Lectures**
- [Evaluation Metrics Playlist](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)
- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)

### **Community Notes**
- [Notes from Peter Ernicke](https://knowmledge.com/category/courses/ml-zoomcamp/evaluation/)

## 🎯 Next Steps

After completing this module, you're ready for **Module 5: Model Deployment**, where you'll learn to deploy your evaluated models to production.

---

**Navigation:**
- **Previous**: [Module 3: Classification](../03-classification/README.md)
- **Next**: [Module 5: Deployment](../05-deployment/README.md)
- **Course Home**: [Main Guide](../README.md)

*Last Updated: 2025-01-27*
