# 🧹 Data Preparation

> **Clean and prepare your data for machine learning success**

Data preparation is often the most time-consuming part of any ML project, typically taking 50-80% of the total project time. However, it's also one of the most important steps - clean, well-prepared data is essential for building effective models.

## 🎯 Learning Objectives

By the end of this section, you will:
- **Load and inspect** the car price dataset
- **Identify and handle** missing values effectively
- **Detect and treat** outliers appropriately
- **Clean and standardize** data formats
- **Prepare data** for machine learning algorithms

## 📊 Loading the Dataset

### **Dataset Download**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
# Note: You can download the dataset from Kaggle or use the provided sample
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
df = pd.read_csv(url)

print(f"Dataset shape: {df.shape}")
print(f"Dataset size: {df.size:,} total values")
```

### **Initial Data Inspection**

```python
# Basic information about the dataset
print("=== DATASET OVERVIEW ===")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display first few rows
print("\n=== FIRST 5 ROWS ===")
print(df.head())

# Column information
print("\n=== COLUMN INFORMATION ===")
print(df.info())

# Basic statistics
print("\n=== NUMERICAL STATISTICS ===")
print(df.describe())
```

### **Column Analysis**

```python
# Analyze each column type
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Check unique values in categorical columns
print("\n=== CATEGORICAL COLUMN ANALYSIS ===")
for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique values")
    if unique_count <= 10:
        print(f"  Values: {df[col].unique()}")
    print()
```

## 🔍 Missing Data Analysis

### **Identify Missing Values**

```python
# Calculate missing values
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
})

missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

print("=== MISSING DATA ANALYSIS ===")
print(missing_data)

# Visualize missing data pattern
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Data Pattern')
plt.tight_layout()
plt.show()
```

### **Missing Data Strategies**

```python
def analyze_missing_patterns(df):
    """Analyze patterns in missing data to inform handling strategy"""
    
    # Check if missing values are random or systematic
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    for col in missing_cols:
        print(f"\n=== MISSING DATA ANALYSIS: {col} ===")
        
        # Check correlation with other missing values
        for other_col in missing_cols:
            if col != other_col:
                correlation = df[col].isnull().corr(df[other_col].isnull())
                if abs(correlation) > 0.1:
                    print(f"Correlation with {other_col} missing: {correlation:.3f}")
        
        # Analyze missing values by categorical variables
        if col in numerical_cols:
            for cat_col in categorical_cols[:3]:  # Check first 3 categorical columns
                missing_by_category = df.groupby(cat_col)[col].apply(lambda x: x.isnull().sum())
                if missing_by_category.max() > 0:
                    print(f"Missing values by {cat_col}:")
                    print(missing_by_category[missing_by_category > 0])

analyze_missing_patterns(df)
```

### **Handle Missing Values**

```python
def handle_missing_values(df):
    """Comprehensive missing value handling strategy"""
    df_clean = df.copy()
    
    # Strategy 1: Remove columns with >50% missing values
    high_missing_cols = df_clean.columns[df_clean.isnull().mean() > 0.5].tolist()
    if high_missing_cols:
        print(f"Removing columns with >50% missing: {high_missing_cols}")
        df_clean = df_clean.drop(columns=high_missing_cols)
    
    # Strategy 2: Handle numerical columns
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().any():
            if col == 'engine_hp':
                # Fill with median by make
                df_clean[col] = df_clean.groupby('make')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Fill remaining with overall median
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
            elif col == 'engine_cylinders':
                # Fill with mode by engine_hp range
                df_clean[col] = df_clean.groupby(pd.cut(df_clean['engine_hp'], bins=5))[col].transform(
                    lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x.median())
                )
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
            else:
                # Default: fill with median
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Strategy 3: Handle categorical columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            if col == 'market_category':
                # Fill with 'Unknown' for market category
                df_clean[col].fillna('Unknown', inplace=True)
            else:
                # Fill with mode (most frequent value)
                mode_value = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_value, inplace=True)
    
    return df_clean

# Apply missing value handling
df_clean = handle_missing_values(df)

# Verify no missing values remain
print("=== MISSING VALUES AFTER CLEANING ===")
print(df_clean.isnull().sum().sum())
```

## 🎯 Outlier Detection and Treatment

### **Identify Outliers**

```python
def detect_outliers(df, columns=None):
    """Detect outliers using multiple methods"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outlier_summary = {}
    
    for col in columns:
        data = df[col].dropna()
        
        # Method 1: IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        
        # Method 2: Z-score method
        z_scores = np.abs((data - data.mean()) / data.std())
        z_outliers = (z_scores > 3).sum()
        
        # Method 3: Modified Z-score (using median)
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        modified_z_outliers = (np.abs(modified_z_scores) > 3.5).sum()
        
        outlier_summary[col] = {
            'IQR_outliers': iqr_outliers,
            'Z_score_outliers': z_outliers,
            'Modified_Z_outliers': modified_z_outliers,
            'Total_values': len(data)
        }
    
    return pd.DataFrame(outlier_summary).T

# Detect outliers
outlier_analysis = detect_outliers(df_clean)
print("=== OUTLIER ANALYSIS ===")
print(outlier_analysis)
```

### **Visualize Outliers**

```python
def visualize_outliers(df, columns=None, max_cols=4):
    """Visualize outliers using box plots and histograms"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns[:max_cols]
    
    fig, axes = plt.subplots(2, len(columns), figsize=(5*len(columns), 10))
    
    for i, col in enumerate(columns):
        # Box plot
        axes[0, i].boxplot(df[col].dropna())
        axes[0, i].set_title(f'{col} - Box Plot')
        axes[0, i].set_ylabel('Value')
        
        # Histogram
        axes[1, i].hist(df[col].dropna(), bins=50, alpha=0.7)
        axes[1, i].set_title(f'{col} - Distribution')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Visualize key numerical columns
key_columns = ['msrp', 'engine_hp', 'highway_mpg', 'city_mpg']
visualize_outliers(df_clean, key_columns)
```

### **Handle Outliers**

```python
def handle_outliers(df, method='iqr', factor=1.5):
    """Handle outliers using specified method"""
    df_no_outliers = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    outlier_info = {}
    
    for col in numerical_cols:
        original_count = len(df_no_outliers)
        
        if method == 'iqr':
            Q1 = df_no_outliers[col].quantile(0.25)
            Q3 = df_no_outliers[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Cap outliers instead of removing them
            df_no_outliers[col] = df_no_outliers[col].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df_no_outliers[col] - df_no_outliers[col].mean()) / df_no_outliers[col].std())
            # Cap values beyond 3 standard deviations
            threshold = 3
            mean_val = df_no_outliers[col].mean()
            std_val = df_no_outliers[col].std()
            lower_bound = mean_val - threshold * std_val
            upper_bound = mean_val + threshold * std_val
            df_no_outliers[col] = df_no_outliers[col].clip(lower=lower_bound, upper=upper_bound)
        
        outliers_handled = original_count - len(df_no_outliers)
        outlier_info[col] = outliers_handled
    
    return df_no_outliers, outlier_info

# Handle outliers (using capping instead of removal to preserve data)
df_clean, outlier_info = handle_outliers(df_clean, method='iqr', factor=2.0)  # More conservative

print("=== OUTLIER HANDLING SUMMARY ===")
for col, count in outlier_info.items():
    if count > 0:
        print(f"{col}: {count} outliers handled")
```

## 🔧 Data Cleaning and Standardization

### **Clean Column Names**

```python
def clean_column_names(df):
    """Standardize column names"""
    df_clean = df.copy()
    
    # Convert to lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Remove special characters
    df_clean.columns = df_clean.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
    
    # Ensure no duplicate column names
    cols = pd.Series(df_clean.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df_clean.columns = cols
    
    return df_clean

df_clean = clean_column_names(df_clean)
print("=== CLEANED COLUMN NAMES ===")
print(df_clean.columns.tolist())
```

### **Standardize Categorical Values**

```python
def standardize_categorical_values(df):
    """Clean and standardize categorical values"""
    df_clean = df.copy()
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        # Remove leading/trailing whitespace
        df_clean[col] = df_clean[col].astype(str).str.strip()
        
        # Standardize case
        if col in ['make', 'model']:
            df_clean[col] = df_clean[col].str.title()  # Title case for names
        else:
            df_clean[col] = df_clean[col].str.lower()  # Lowercase for others
        
        # Handle common inconsistencies
        if col == 'transmission_type':
            df_clean[col] = df_clean[col].replace({
                'manual': 'manual',
                'automatic': 'automatic',
                'automated_manual': 'automated_manual',
                'cvt': 'cvt'
            })
        
        # Replace empty strings with NaN
        df_clean[col] = df_clean[col].replace('', np.nan)
    
    return df_clean

df_clean = standardize_categorical_values(df_clean)
```

### **Data Type Optimization**

```python
def optimize_data_types(df):
    """Optimize data types to reduce memory usage"""
    df_optimized = df.copy()
    
    # Convert object columns with low cardinality to category
    for col in df_optimized.select_dtypes(include=['object']).columns:
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:
            df_optimized[col] = df_optimized[col].astype('category')
    
    # Downcast numerical columns
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
    
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    return df_optimized

# Check memory usage before and after optimization
print("=== MEMORY USAGE COMPARISON ===")
print(f"Before optimization: {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

df_clean = optimize_data_types(df_clean)

print(f"After optimization: {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

## ✅ Data Quality Validation

### **Final Data Quality Check**

```python
def validate_data_quality(df):
    """Perform final data quality validation"""
    print("=== FINAL DATA QUALITY REPORT ===")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    print(f"Missing values: {missing_count}")
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")
    
    # Check data types
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    # Check target variable
    if 'msrp' in df.columns:
        print(f"\nTarget variable (MSRP) statistics:")
        print(f"Min: ${df['msrp'].min():,.2f}")
        print(f"Max: ${df['msrp'].max():,.2f}")
        print(f"Mean: ${df['msrp'].mean():,.2f}")
        print(f"Median: ${df['msrp'].median():,.2f}")
        print(f"Std: ${df['msrp'].std():,.2f}")
    
    # Check for any remaining issues
    issues = []
    
    # Check for infinite values
    if np.isinf(df.select_dtypes(include=[np.number])).any().any():
        issues.append("Infinite values found")
    
    # Check for negative values where they shouldn't be
    negative_cols = ['msrp', 'engine_hp', 'highway_mpg', 'city_mpg']
    for col in negative_cols:
        if col in df.columns and (df[col] < 0).any():
            issues.append(f"Negative values in {col}")
    
    if issues:
        print(f"\n⚠️ Issues found: {issues}")
    else:
        print(f"\n✅ Data quality validation passed!")
    
    return len(issues) == 0

# Validate final data quality
is_valid = validate_data_quality(df_clean)

# Save cleaned dataset
df_clean.to_csv('car_data_cleaned.csv', index=False)
print(f"\n💾 Cleaned dataset saved as 'car_data_cleaned.csv'")
print(f"Final dataset shape: {df_clean.shape}")
```

## 🎯 Key Takeaways

### **Data Preparation Best Practices**
1. **Understand Your Data**: Always start with thorough exploration
2. **Handle Missing Values Thoughtfully**: Consider the reason for missingness
3. **Be Conservative with Outliers**: Cap rather than remove when possible
4. **Standardize Consistently**: Apply consistent formatting rules
5. **Validate Your Work**: Always check the final result

### **Common Pitfalls to Avoid**
- **Data Leakage**: Don't use future information
- **Over-cleaning**: Don't remove too much valuable information
- **Inconsistent Handling**: Apply the same logic across similar features
- **Ignoring Domain Knowledge**: Consider what makes sense in the real world

## 🚀 Next Steps

With clean, prepared data, you're ready to:
1. **Explore the data** - [Exploratory Data Analysis](03-eda.md)
2. **Set up validation** - [Validation Framework](04-validation-framework.md)
3. **Start modeling** - [Linear Regression Theory](05-linear-regression-simple.md)

---

**Navigation:**
- **Previous**: [Car Price Prediction Introduction](01-car-price-intro.md)
- **Next**: [Exploratory Data Analysis](03-eda.md)
- **Module Home**: [Regression](README.md)

*Last Updated: 2025-01-27*
