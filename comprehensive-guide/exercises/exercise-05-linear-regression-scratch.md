# 🧮 Exercise 5: Linear Regression from Scratch

> **🟡 Intermediate Level | Estimated Time: 3-4 hours**

## 🎯 Problem Statement

You're a machine learning engineer at a tech startup. Your team needs to understand how linear regression works under the hood before implementing more complex models. Your task is to implement linear regression from scratch using only NumPy, then compare your implementation with scikit-learn's version on a real dataset.

## 📚 Learning Objectives

By completing this exercise, you will:
- **Understand** the mathematical foundation of linear regression
- **Implement** gradient descent optimization from scratch
- **Master** vectorized operations with NumPy
- **Compare** different optimization approaches
- **Validate** your implementation against established libraries
- **Analyze** model performance and convergence

## 🧮 Mathematical Background

### **Linear Regression Model**
```
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θᵀx
```

### **Cost Function (Mean Squared Error)**
```
J(θ) = (1/2m) Σᵢ₌₁ᵐ (h(xⁱ) - yⁱ)²
```

### **Gradient Descent Update Rule**
```
θⱼ := θⱼ - α * (∂J(θ)/∂θⱼ)
```

### **Normal Equation (Analytical Solution)**
```
θ = (XᵀX)⁻¹Xᵀy
```

## 📊 Dataset

**California Housing Dataset**
- **Size**: 20,640 samples
- **Features**: 8 numerical features
- **Target**: Median house value
- **Challenge**: Real-world data with varying scales

### **Features**
- **MedInc**: Median income in block group
- **HouseAge**: Median house age in block group
- **AveRooms**: Average number of rooms per household
- **AveBedrms**: Average number of bedrooms per household
- **Population**: Block group population
- **AveOccup**: Average number of household members
- **Latitude**: Block group latitude
- **Longitude**: Block group longitude

## 🎯 Requirements

### **Part 1: Implementation (40 points)**

#### **1.1 Basic Linear Regression Class (15 points)**
```python
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        # Initialize parameters
        pass
    
    def fit(self, X, y):
        # Implement training logic
        pass
    
    def predict(self, X):
        # Implement prediction logic
        pass
    
    def cost_function(self, X, y):
        # Implement cost calculation
        pass
```

#### **1.2 Gradient Descent Implementation (15 points)**
- Implement batch gradient descent
- Track cost history for convergence analysis
- Handle feature scaling automatically
- Include early stopping based on tolerance

#### **1.3 Normal Equation Implementation (10 points)**
- Implement analytical solution
- Handle matrix inversion safely
- Compare with gradient descent results

### **Part 2: Advanced Features (30 points)**

#### **2.1 Multiple Optimization Methods (15 points)**
Implement at least two additional optimization methods:
- **Stochastic Gradient Descent (SGD)**
- **Mini-batch Gradient Descent**
- **Momentum-based Gradient Descent** (bonus)

#### **2.2 Regularization (15 points)**
Add regularization capabilities:
- **Ridge Regression (L2 regularization)**
- **Lasso Regression (L1 regularization)** (bonus)

### **Part 3: Validation and Comparison (20 points)**

#### **3.1 Performance Comparison (10 points)**
Compare your implementation with scikit-learn:
- Training time
- Prediction accuracy
- Convergence behavior

#### **3.2 Hyperparameter Analysis (10 points)**
Analyze the effect of:
- Learning rate values
- Number of iterations
- Regularization strength

### **Part 4: Visualization and Analysis (10 points)**

#### **4.1 Convergence Analysis (5 points)**
- Plot cost function over iterations
- Compare convergence of different methods
- Identify optimal hyperparameters

#### **4.2 Model Interpretation (5 points)**
- Analyze feature importance (coefficients)
- Visualize predictions vs actual values
- Identify model limitations

## 🛠️ Starter Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6, method='gradient_descent'):
        """
        Initialize Linear Regression model
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        method : str
            Optimization method ('gradient_descent', 'normal_equation', 'sgd', 'mini_batch')
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.method = method
        
        # Model parameters
        self.weights = None
        self.bias = None
        
        # Training history
        self.cost_history = []
        self.converged = False
        self.iterations_run = 0
    
    def _add_bias_column(self, X):
        """Add bias column to feature matrix"""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _compute_cost(self, X, y):
        """Compute mean squared error cost"""
        # TODO: Implement cost function
        pass
    
    def _gradient_descent(self, X, y):
        """Implement batch gradient descent"""
        # TODO: Implement gradient descent
        pass
    
    def _normal_equation(self, X, y):
        """Implement normal equation solution"""
        # TODO: Implement normal equation
        pass
    
    def fit(self, X, y):
        """Train the linear regression model"""
        # TODO: Implement training logic
        pass
    
    def predict(self, X):
        """Make predictions on new data"""
        # TODO: Implement prediction logic
        pass

# Load and prepare data
california = fetch_california_housing()
X, y = california.data, california.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data loaded and preprocessed!")
print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# Your implementation starts here...
```

## 📋 Implementation Guidelines

### **Step 1: Basic Implementation**
1. Start with the cost function - make sure it works correctly
2. Implement basic gradient descent
3. Test on simple synthetic data first
4. Add bias term handling

### **Step 2: Optimization**
1. Vectorize all operations for efficiency
2. Add convergence checking
3. Implement feature scaling
4. Add early stopping

### **Step 3: Advanced Features**
1. Implement alternative optimization methods
2. Add regularization terms
3. Create comprehensive comparison framework

### **Step 4: Validation**
1. Compare with scikit-learn extensively
2. Test on multiple datasets
3. Analyze edge cases and limitations

## 🎯 Success Criteria

### **Excellent (90-100 points)**
- All optimization methods implemented correctly
- Regularization working properly
- Comprehensive comparison and analysis
- Clean, well-documented code
- Creative insights and visualizations

### **Good (80-89 points)**
- Basic gradient descent and normal equation working
- Good comparison with scikit-learn
- Clear visualizations and analysis
- Well-structured code

### **Satisfactory (70-79 points)**
- Basic implementation working
- Some comparison with scikit-learn
- Adequate documentation
- Functional code

### **Needs Improvement (<70 points)**
- Implementation issues or errors
- Limited comparison or analysis
- Poor code structure
- Missing key components

## 🧪 Test Cases

Your implementation should pass these tests:

### **Test 1: Simple Linear Data**
```python
# Generate simple linear data
np.random.seed(42)
X_simple = np.random.randn(100, 1)
y_simple = 3 * X_simple.flatten() + 2 + 0.1 * np.random.randn(100)

# Your model should recover weights close to [2, 3]
```

### **Test 2: Convergence**
```python
# Your model should converge within reasonable iterations
# Cost should decrease monotonically
# Final cost should be close to sklearn's result
```

### **Test 3: Prediction Accuracy**
```python
# R² score should be > 0.6 on California housing dataset
# RMSE should be reasonable
# Predictions should correlate well with actual values
```

## 💡 Hints and Tips

### **Implementation Tips**
1. **Start simple**: Get basic version working before adding complexity
2. **Vectorize operations**: Use NumPy broadcasting for efficiency
3. **Handle edge cases**: Check for singular matrices, zero gradients
4. **Debug systematically**: Test each component separately

### **Mathematical Tips**
1. **Feature scaling**: Essential for gradient descent convergence
2. **Learning rate**: Start with 0.01, adjust based on convergence
3. **Regularization**: Add small epsilon to prevent overfitting
4. **Matrix operations**: Use np.linalg.solve() instead of matrix inverse

### **Performance Tips**
1. **Batch operations**: Process all samples at once
2. **Memory efficiency**: Avoid creating unnecessary copies
3. **Early stopping**: Stop when improvement is minimal
4. **Convergence criteria**: Use relative change in cost

## 🔍 Advanced Challenges (Bonus Points)

### **Challenge 1: Polynomial Features (5 points)**
Extend your implementation to handle polynomial features automatically.

### **Challenge 2: Online Learning (5 points)**
Implement online/streaming version that can update with new data points.

### **Challenge 3: Robust Regression (10 points)**
Implement Huber loss or other robust loss functions.

## 📚 Resources

### **Mathematical Background**
- [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning)
- [Linear Algebra Review](https://www.khanacademy.org/math/linear-algebra)

### **Implementation References**
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Scikit-learn Source Code](https://github.com/scikit-learn/scikit-learn)

## ✅ Submission Checklist

- [ ] All required methods implemented
- [ ] Code runs without errors
- [ ] Comprehensive comparison with scikit-learn
- [ ] Convergence analysis included
- [ ] Performance metrics calculated
- [ ] Code well-documented with comments
- [ ] Visualizations clear and informative
- [ ] Test cases pass
- [ ] Analysis and insights provided

## 🎉 Next Steps

After completing this exercise:
1. **Experiment** with different datasets
2. **Optimize** your implementation further
3. **Move to** Exercise 6: Logistic Regression Implementation
4. **Apply** your understanding to real projects

---

**Navigation:**
- **Previous Exercise**: [Feature Engineering Workshop](exercise-04-feature-engineering.md)
- **Next Exercise**: [Logistic Regression Implementation](exercise-06-logistic-regression.md)
- **Exercise Home**: [Practice Exercises](README.md)

*Remember: Understanding the fundamentals deeply will make you a better ML practitioner!* 🚀
