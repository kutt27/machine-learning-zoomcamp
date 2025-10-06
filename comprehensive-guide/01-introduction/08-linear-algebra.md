# 🧮 Linear Algebra Refresher

> **Essential mathematical foundations for machine learning**

Linear algebra is the mathematical foundation of machine learning. Understanding vectors, matrices, and their operations is crucial for grasping how ML algorithms work under the hood.

## 🎯 Why Linear Algebra Matters in ML

### **Core ML Concepts Built on Linear Algebra**
- **Linear Regression**: Solving systems of linear equations
- **Neural Networks**: Matrix multiplications and transformations
- **Principal Component Analysis**: Eigenvalues and eigenvectors
- **Support Vector Machines**: Vector operations and projections
- **Recommendation Systems**: Matrix factorization

### **Computational Efficiency**
- **Vectorization**: Express operations on entire datasets
- **Parallel Processing**: GPUs excel at matrix operations
- **Memory Efficiency**: Compact representation of data

## 📊 Vectors

### **What is a Vector?**

A vector is an ordered list of numbers that represents a point in space or a direction.

```python
import numpy as np

# Vector as a 1D array
v = np.array([3, 4])  # 2D vector
w = np.array([1, 2, 3])  # 3D vector

# In ML context: a data point with features
customer = np.array([25, 50000, 3])  # [age, income, purchases]
```

### **Vector Operations**

#### **Vector Addition and Subtraction**
```python
v1 = np.array([1, 2])
v2 = np.array([3, 4])

# Addition
v_sum = v1 + v2  # [4, 6]

# Subtraction  
v_diff = v2 - v1  # [2, 2]

# Geometric interpretation: tip-to-tail rule
```

#### **Scalar Multiplication**
```python
v = np.array([2, 3])
scalar = 3

# Scalar multiplication
scaled_v = scalar * v  # [6, 9]

# Changes magnitude, preserves direction
```

#### **Dot Product (Inner Product)**
```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product
dot_product = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32

# Alternative notation
dot_product = v1 @ v2  # Same result

# Geometric interpretation: measures similarity
```

#### **Vector Magnitude (Norm)**
```python
v = np.array([3, 4])

# L2 norm (Euclidean norm)
magnitude = np.linalg.norm(v)  # 5.0
# Or manually: sqrt(3² + 4²) = 5

# L1 norm (Manhattan norm)
l1_norm = np.sum(np.abs(v))  # 7

# Unit vector (normalized)
unit_v = v / np.linalg.norm(v)  # [0.6, 0.8]
```

### **ML Applications of Vectors**

#### **Feature Vectors**
```python
# Each data point is a vector
house1 = np.array([1500, 3, 2, 10])  # [sqft, bedrooms, bathrooms, age]
house2 = np.array([2000, 4, 3, 5])

# Dataset as collection of vectors
houses = np.array([house1, house2])
```

#### **Similarity Measurement**
```python
def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot_product / norms

# Example: Document similarity
doc1 = np.array([1, 2, 0, 1])  # Word frequencies
doc2 = np.array([2, 1, 1, 0])
similarity = cosine_similarity(doc1, doc2)
```

## 🔲 Matrices

### **What is a Matrix?**

A matrix is a 2D array of numbers arranged in rows and columns.

```python
# Matrix as 2D array
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3 matrix

# In ML: dataset with samples as rows, features as columns
data = np.array([[25, 50000],   # Customer 1: [age, income]
                 [30, 60000],   # Customer 2
                 [35, 70000]])  # Customer 3
```

### **Matrix Properties**

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(f"Shape: {A.shape}")      # (2, 3) - 2 rows, 3 columns
print(f"Size: {A.size}")        # 6 - total elements
print(f"Transpose:\n{A.T}")     # 3x2 matrix
```

### **Matrix Operations**

#### **Matrix Addition and Subtraction**
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise operations (same shape required)
C = A + B  # [[6, 8], [10, 12]]
D = B - A  # [[4, 4], [4, 4]]
```

#### **Matrix Multiplication**
```python
A = np.array([[1, 2],
              [3, 4]])  # 2x2

B = np.array([[5, 6],
              [7, 8]])  # 2x2

# Matrix multiplication
C = np.dot(A, B)  # or A @ B
# Result: [[19, 22], [43, 50]]

# Rule: (m×n) × (n×p) = (m×p)
```

#### **Matrix-Vector Multiplication**
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3 matrix

x = np.array([1, 2, 3])   # 3x1 vector

# Matrix-vector multiplication
result = A @ x  # [14, 32] (2x1 vector)

# This is how linear models make predictions!
```

### **Special Matrices**

#### **Identity Matrix**
```python
I = np.eye(3)  # 3x3 identity matrix
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]

# Property: A @ I = I @ A = A
```

#### **Diagonal Matrix**
```python
D = np.diag([1, 2, 3])  # Diagonal matrix
# [[1, 0, 0],
#  [0, 2, 0],
#  [0, 0, 3]]
```

#### **Zero Matrix**
```python
Z = np.zeros((2, 3))  # 2x3 zero matrix
```

## 🔧 Advanced Operations

### **Matrix Inverse**

```python
A = np.array([[1, 2],
              [3, 4]])

# Matrix inverse (if it exists)
A_inv = np.linalg.inv(A)

# Verification: A @ A_inv ≈ I
identity_check = A @ A_inv
print(np.allclose(identity_check, np.eye(2)))  # True

# Used in: Normal equation for linear regression
```

### **Determinant**

```python
A = np.array([[1, 2],
              [3, 4]])

det_A = np.linalg.det(A)  # -2.0

# If det = 0, matrix is singular (no inverse)
# If det ≠ 0, matrix is invertible
```

### **Eigenvalues and Eigenvectors**

```python
A = np.array([[4, 2],
              [1, 3]])

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Property: A @ v = λ @ v (for eigenvalue λ and eigenvector v)
```

## 🎯 ML Applications

### **Linear Regression**

```python
def linear_regression_matrix(X, y):
    """Linear regression using matrix operations"""
    # Add bias column
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    # Normal equation: w = (X^T X)^(-1) X^T y
    XTX = X_with_bias.T @ X_with_bias
    XTy = X_with_bias.T @ y
    
    # Solve for weights
    weights = np.linalg.solve(XTX, XTy)  # More stable than inverse
    
    return weights

# Example
X = np.array([[1], [2], [3], [4]])  # Features
y = np.array([2, 4, 6, 8])          # Targets
weights = linear_regression_matrix(X, y)
```

### **Principal Component Analysis (PCA)**

```python
def simple_pca(X, n_components=2):
    """Simple PCA implementation"""
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components
    components = eigenvectors[:, :n_components]
    
    # Transform data
    X_transformed = X_centered @ components
    
    return X_transformed, components
```

### **Neural Network Forward Pass**

```python
def neural_network_forward(X, W1, b1, W2, b2):
    """Simple 2-layer neural network forward pass"""
    # Layer 1
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)  # ReLU activation
    
    # Layer 2
    z2 = a1 @ W2 + b2
    a2 = 1 / (1 + np.exp(-z2))  # Sigmoid activation
    
    return a2

# Example dimensions
# X: (batch_size, input_features)
# W1: (input_features, hidden_units)
# W2: (hidden_units, output_units)
```

## 🧮 Computational Considerations

### **Matrix Multiplication Complexity**

```python
# Time complexity: O(n³) for n×n matrices
# Space complexity: O(n²)

# For large matrices, consider:
# 1. Block matrix multiplication
# 2. Sparse matrices
# 3. GPU acceleration
```

### **Numerical Stability**

```python
# ❌ Avoid direct matrix inverse for solving Ax = b
A_inv = np.linalg.inv(A)
x = A_inv @ b

# ✅ Use solve() instead (more stable)
x = np.linalg.solve(A, b)

# ✅ For least squares, use lstsq()
x = np.linalg.lstsq(A, b, rcond=None)[0]
```

### **Memory Efficiency**

```python
# For large datasets, consider:
# 1. Batch processing
# 2. In-place operations
# 3. Appropriate data types

# Example: In-place operations
A += B  # Instead of A = A + B
A *= 2  # Instead of A = A * 2
```

## 🎯 Geometric Intuition

### **Linear Transformations**

```python
# Rotation matrix (45 degrees)
theta = np.pi / 4
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])

# Apply transformation
point = np.array([1, 0])
rotated_point = rotation_matrix @ point

# Scaling matrix
scaling_matrix = np.array([[2, 0],
                          [0, 3]])
scaled_point = scaling_matrix @ point
```

### **Hyperplanes and Decision Boundaries**

```python
# Linear classifier: w^T x + b = 0
# w: normal vector to hyperplane
# b: bias term

def classify_point(x, w, b):
    """Classify point using linear decision boundary"""
    score = w @ x + b
    return 1 if score > 0 else -1

# Example
w = np.array([1, -1])  # Normal vector
b = 0                  # Bias
point = np.array([2, 1])
prediction = classify_point(point, w, b)
```

## ✅ Linear Algebra Checklist

- [ ] Understand vectors and vector operations
- [ ] Know matrix operations and properties
- [ ] Can perform matrix multiplication
- [ ] Understand matrix inverse and determinant
- [ ] Know eigenvalues and eigenvectors
- [ ] Can apply linear algebra to ML problems
- [ ] Understand geometric interpretations
- [ ] Know computational considerations

## 🚀 Next Steps

With linear algebra fundamentals mastered:
1. **Practice with code** - [NumPy/Pandas Notebook](../notebooks/02-numpy-pandas-mastery.ipynb)
2. **Learn Pandas** - [Pandas Guide](09-pandas.md)
3. **Apply to regression** - [Linear Regression Module](../02-regression/)

## 📚 Additional Resources

- **Khan Academy**: [Linear Algebra Course](https://www.khanacademy.org/math/linear-algebra)
- **3Blue1Brown**: [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- **MIT OpenCourseWare**: [Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)

---

**Navigation:**
- **Previous**: [NumPy Fundamentals](07-numpy.md)
- **Next**: [Pandas for Data Manipulation](09-pandas.md)
- **Module Home**: [Introduction](README.md)

*Last Updated: 2025-01-27*
