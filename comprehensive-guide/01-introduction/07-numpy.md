# 🔢 NumPy Fundamentals

> **Master the foundation of numerical computing in Python**

NumPy (Numerical Python) is the cornerstone of the Python data science ecosystem. It provides efficient operations on large arrays and matrices, which are essential for machine learning computations.

## 🎯 Why NumPy Matters for ML

### **Performance Benefits**
- **Speed**: 10-100x faster than pure Python for numerical operations
- **Memory Efficiency**: Compact storage of homogeneous data
- **Vectorization**: Apply operations to entire arrays without loops

### **Foundation for ML**
- **Scikit-learn**: Built on NumPy arrays
- **TensorFlow/PyTorch**: Use NumPy-compatible arrays
- **Pandas**: Uses NumPy arrays internally
- **Matplotlib**: Expects NumPy arrays for plotting

## 📊 NumPy Arrays (ndarray)

### **Creating Arrays**

```python
import numpy as np

# From Python lists
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# Using built-in functions
zeros = np.zeros((3, 4))          # 3x4 array of zeros
ones = np.ones((2, 3))            # 2x3 array of ones
identity = np.eye(3)              # 3x3 identity matrix
range_arr = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)   # [0, 0.25, 0.5, 0.75, 1]

# Random arrays
random_arr = np.random.random((3, 3))      # Random values [0, 1)
normal_arr = np.random.normal(0, 1, (3, 3)) # Normal distribution
```

### **Array Properties**

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(f"Shape: {arr.shape}")        # (2, 3)
print(f"Size: {arr.size}")          # 6
print(f"Dimensions: {arr.ndim}")     # 2
print(f"Data type: {arr.dtype}")     # int64
print(f"Item size: {arr.itemsize}")  # 8 bytes
```

## 🔧 Array Operations

### **Basic Arithmetic**

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Element-wise operations
print(a + b)    # [6, 8, 10, 12]
print(a - b)    # [-4, -4, -4, -4]
print(a * b)    # [5, 12, 21, 32]
print(a / b)    # [0.2, 0.33, 0.43, 0.5]
print(a ** 2)   # [1, 4, 9, 16]

# Scalar operations
print(a + 10)   # [11, 12, 13, 14]
print(a * 2)    # [2, 4, 6, 8]
```

### **Mathematical Functions**

```python
arr = np.array([1, 4, 9, 16])

# Common functions
print(np.sqrt(arr))     # [1, 2, 3, 4]
print(np.log(arr))      # Natural logarithm
print(np.exp(arr))      # Exponential
print(np.sin(arr))      # Sine
print(np.abs(arr))      # Absolute value

# Statistical functions
data = np.array([1, 2, 3, 4, 5])
print(f"Mean: {np.mean(data)}")         # 3.0
print(f"Median: {np.median(data)}")     # 3.0
print(f"Std: {np.std(data)}")           # 1.41
print(f"Min: {np.min(data)}")           # 1
print(f"Max: {np.max(data)}")           # 5
print(f"Sum: {np.sum(data)}")           # 15
```

## 🎯 Array Indexing and Slicing

### **1D Arrays**

```python
arr = np.array([10, 20, 30, 40, 50])

# Basic indexing
print(arr[0])      # 10 (first element)
print(arr[-1])     # 50 (last element)
print(arr[1:4])    # [20, 30, 40] (slice)
print(arr[::2])    # [10, 30, 50] (every 2nd element)
```

### **2D Arrays**

```python
arr2d = np.array([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])

# Indexing
print(arr2d[0, 1])     # 2 (row 0, column 1)
print(arr2d[1])        # [4, 5, 6] (entire row 1)
print(arr2d[:, 2])     # [3, 6, 9] (entire column 2)

# Slicing
print(arr2d[0:2, 1:3]) # [[2, 3], [5, 6]] (subarray)
```

### **Boolean Indexing**

```python
arr = np.array([1, 2, 3, 4, 5])

# Boolean mask
mask = arr > 3
print(mask)           # [False, False, False, True, True]
print(arr[mask])      # [4, 5]

# Direct boolean indexing
print(arr[arr > 3])   # [4, 5]
print(arr[(arr > 2) & (arr < 5)])  # [3, 4]
```

## 🔄 Array Manipulation

### **Reshaping**

```python
arr = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape to 2D
arr2d = arr.reshape(3, 4)
print(arr2d.shape)   # (3, 4)

# Reshape to 3D
arr3d = arr.reshape(2, 2, 3)
print(arr3d.shape)   # (2, 2, 3)

# Flatten back to 1D
flat = arr3d.flatten()
print(flat.shape)    # (12,)
```

### **Concatenation and Splitting**

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Concatenate
h_concat = np.hstack([a, b])  # Horizontal: [[1,2,5,6], [3,4,7,8]]
v_concat = np.vstack([a, b])  # Vertical: [[1,2], [3,4], [5,6], [7,8]]

# Split
arr = np.arange(8)
split_arr = np.split(arr, 4)  # [array([0,1]), array([2,3]), ...]
```

## 🧮 Linear Algebra Operations

### **Matrix Multiplication**

```python
# Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # 32 (1*4 + 2*5 + 3*6)

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)  # or A @ B
print(C)  # [[19, 22], [43, 50]]
```

### **Linear Algebra Functions**

```python
# Matrix operations
A = np.array([[1, 2], [3, 4]])

# Transpose
print(A.T)  # [[1, 3], [2, 4]]

# Determinant
det = np.linalg.det(A)  # -2.0

# Inverse
inv = np.linalg.inv(A)
print(np.dot(A, inv))  # Identity matrix (approximately)

# Eigenvalues and eigenvectors
eigenvals, eigenvecs = np.linalg.eig(A)
```

## 📊 Broadcasting

Broadcasting allows operations between arrays of different shapes.

```python
# Scalar and array
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr + 10  # Adds 10 to each element

# Array and vector
vector = np.array([10, 20, 30])
result = arr + vector  # Adds vector to each row

# Broadcasting rules example
a = np.array([[1], [2], [3]])  # Shape: (3, 1)
b = np.array([10, 20, 30])     # Shape: (3,)
result = a + b                 # Shape: (3, 3)
```

## 🎯 ML-Specific Operations

### **Feature Scaling**

```python
# Sample data
data = np.array([[1, 2], [3, 4], [5, 6]])

# Min-Max scaling
min_vals = np.min(data, axis=0)
max_vals = np.max(data, axis=0)
scaled = (data - min_vals) / (max_vals - min_vals)

# Standardization (Z-score)
mean_vals = np.mean(data, axis=0)
std_vals = np.std(data, axis=0)
standardized = (data - mean_vals) / std_vals
```

### **Distance Calculations**

```python
# Euclidean distance
point1 = np.array([1, 2])
point2 = np.array([4, 6])
distance = np.sqrt(np.sum((point1 - point2) ** 2))

# Or using linalg.norm
distance = np.linalg.norm(point1 - point2)

# Distance matrix
points = np.array([[1, 2], [3, 4], [5, 6]])
n = len(points)
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
```

## 🚀 Performance Tips

### **Vectorization vs Loops**

```python
# ❌ Slow: Using loops
def slow_sum_of_squares(arr):
    result = 0
    for x in arr:
        result += x ** 2
    return result

# ✅ Fast: Vectorized
def fast_sum_of_squares(arr):
    return np.sum(arr ** 2)

# Timing comparison
arr = np.random.random(1000000)
%timeit slow_sum_of_squares(arr)  # ~100ms
%timeit fast_sum_of_squares(arr)  # ~1ms
```

### **Memory Efficiency**

```python
# Use appropriate data types
small_ints = np.array([1, 2, 3], dtype=np.int8)    # 1 byte per element
large_ints = np.array([1, 2, 3], dtype=np.int64)   # 8 bytes per element

# In-place operations
arr = np.random.random(1000)
arr += 1        # In-place addition
arr *= 2        # In-place multiplication
```

## 🧪 Practical Examples

### **Linear Regression Implementation**

```python
def linear_regression_numpy(X, y):
    """Implement linear regression using NumPy"""
    # Add bias column
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    # Normal equation: w = (X^T X)^(-1) X^T y
    XTX = X_with_bias.T @ X_with_bias
    XTy = X_with_bias.T @ y
    weights = np.linalg.solve(XTX, XTy)
    
    return weights

# Example usage
X = np.array([[1], [2], [3], [4], [5]])  # Features
y = np.array([2, 4, 6, 8, 10])           # Target
weights = linear_regression_numpy(X, y)
print(f"Weights: {weights}")  # [0, 2] (intercept=0, slope=2)
```

### **K-Means Clustering**

```python
def kmeans_numpy(X, k, max_iters=100):
    """Simple K-means implementation using NumPy"""
    # Initialize centroids randomly
    centroids = X[np.random.choice(len(X), k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to closest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels
```

## ✅ NumPy Checklist

- [ ] Understand array creation and properties
- [ ] Master indexing and slicing
- [ ] Know basic mathematical operations
- [ ] Understand broadcasting rules
- [ ] Can perform matrix operations
- [ ] Know when to use vectorization
- [ ] Understand memory and performance considerations

## 🚀 Next Steps

Now that you understand NumPy fundamentals:
1. **Practice with real data** - [NumPy/Pandas Notebook](../notebooks/02-numpy-pandas-mastery.ipynb)
2. **Learn linear algebra** - [Linear Algebra Guide](08-linear-algebra.md)
3. **Move to Pandas** - [Pandas Guide](09-pandas.md)

## 📚 Additional Resources

- **NumPy Documentation**: [Official Guide](https://numpy.org/doc/stable/)
- **NumPy Tutorial**: [Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- **Performance Tips**: [NumPy Performance](https://numpy.org/doc/stable/user/c-info.beyond-basics.html)

---

**Navigation:**
- **Previous**: [Environment Setup](06-environment.md)
- **Next**: [Linear Algebra Refresher](08-linear-algebra.md)
- **Module Home**: [Introduction](README.md)

*Last Updated: 2025-01-27*
