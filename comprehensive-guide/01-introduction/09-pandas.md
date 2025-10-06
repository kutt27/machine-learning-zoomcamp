# 🐼 Pandas for Data Manipulation

> **Master the essential tool for data analysis and preprocessing**

Pandas is the go-to library for data manipulation and analysis in Python. It provides powerful, flexible data structures and tools that make working with structured data intuitive and efficient.

## 🎯 Why Pandas is Essential for ML

### **Data Preprocessing**
- **Loading Data**: Read from various formats (CSV, JSON, SQL, Excel)
- **Cleaning Data**: Handle missing values, duplicates, outliers
- **Transforming Data**: Reshape, merge, group, and aggregate data
- **Feature Engineering**: Create new features from existing data

### **Exploratory Data Analysis**
- **Data Inspection**: Understand structure and quality
- **Statistical Analysis**: Calculate descriptive statistics
- **Data Visualization**: Integration with matplotlib/seaborn
- **Pattern Discovery**: Find relationships and trends

## 📊 Core Data Structures

### **Series (1D labeled array)**

```python
import pandas as pd
import numpy as np

# Create Series
prices = pd.Series([100, 150, 200, 175], 
                  index=['Apple', 'Google', 'Microsoft', 'Amazon'])

print(prices)
# Apple       100
# Google      150
# Microsoft   200
# Amazon      175

# Access elements
print(prices['Apple'])     # 100
print(prices.iloc[0])      # 100 (by position)
```

### **DataFrame (2D labeled array)**

```python
# Create DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000],
    'Department': ['IT', 'Finance', 'IT', 'HR']
}

df = pd.DataFrame(data)
print(df)
#      Name  Age  Salary Department
# 0   Alice   25   50000         IT
# 1     Bob   30   60000    Finance
# 2 Charlie   35   70000         IT
# 3   Diana   28   55000         HR
```

## 📁 Data Loading and Saving

### **Reading Data**

```python
# CSV files (most common)
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', index_col=0)  # Use first column as index
df = pd.read_csv('data.csv', parse_dates=['date_column'])  # Parse dates

# Other formats
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_json('data.json')
df = pd.read_sql('SELECT * FROM table', connection)

# With specific parameters
df = pd.read_csv('data.csv', 
                 sep=';',           # Different separator
                 encoding='utf-8',  # Encoding
                 na_values=['N/A', 'NULL'],  # Custom missing values
                 dtype={'column': 'category'})  # Specify data types
```

### **Saving Data**

```python
# Save to different formats
df.to_csv('output.csv', index=False)  # Don't save index
df.to_excel('output.xlsx', sheet_name='Data')
df.to_json('output.json')
df.to_pickle('output.pkl')  # Preserves all pandas data types
```

## 🔍 Data Exploration

### **Basic Information**

```python
# Dataset overview
print(df.shape)        # (rows, columns)
print(df.info())       # Data types and memory usage
print(df.describe())   # Statistical summary
print(df.head())       # First 5 rows
print(df.tail())       # Last 5 rows

# Column information
print(df.columns.tolist())  # Column names
print(df.dtypes)           # Data types
print(df.nunique())        # Unique values per column
```

### **Missing Data Analysis**

```python
# Check for missing values
print(df.isnull().sum())           # Count missing values
print(df.isnull().sum() / len(df)) # Percentage missing

# Visualize missing data pattern
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
plt.title('Missing Data Pattern')
plt.show()
```

### **Unique Values and Duplicates**

```python
# Unique values
print(df['Department'].unique())      # Unique values
print(df['Department'].value_counts()) # Count of each value

# Duplicates
print(df.duplicated().sum())          # Count duplicates
duplicates = df[df.duplicated()]      # Show duplicate rows
df_clean = df.drop_duplicates()       # Remove duplicates
```

## 🎯 Data Selection and Indexing

### **Column Selection**

```python
# Single column
ages = df['Age']              # Returns Series
ages = df.Age                 # Alternative notation

# Multiple columns
subset = df[['Name', 'Age']]  # Returns DataFrame

# Column operations
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Senior')
```

### **Row Selection**

```python
# By index
first_row = df.iloc[0]        # First row by position
last_row = df.iloc[-1]        # Last row by position
subset = df.iloc[1:4]         # Rows 1-3

# By label (if index is set)
df_indexed = df.set_index('Name')
alice_data = df_indexed.loc['Alice']

# Boolean indexing
young_employees = df[df['Age'] < 30]
it_employees = df[df['Department'] == 'IT']
high_earners = df[df['Salary'] > 60000]

# Multiple conditions
young_it = df[(df['Age'] < 30) & (df['Department'] == 'IT')]
```

### **Advanced Selection**

```python
# Query method (more readable)
result = df.query('Age < 30 and Department == "IT"')
result = df.query('Salary > @threshold')  # Use variable

# isin() for multiple values
departments = ['IT', 'Finance']
subset = df[df['Department'].isin(departments)]

# String operations
names_with_a = df[df['Name'].str.contains('a', case=False)]
```

## 🔧 Data Cleaning

### **Handling Missing Values**

```python
# Remove missing values
df_clean = df.dropna()                    # Drop rows with any NaN
df_clean = df.dropna(subset=['Age'])      # Drop rows with NaN in specific column
df_clean = df.dropna(axis=1)              # Drop columns with any NaN

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill with median
df['Department'].fillna('Unknown', inplace=True)    # Fill with constant

# Forward/backward fill
df['Price'].fillna(method='ffill', inplace=True)    # Forward fill
df['Price'].fillna(method='bfill', inplace=True)    # Backward fill

# Interpolation
df['Price'].interpolate(method='linear', inplace=True)
```

### **Data Type Conversion**

```python
# Convert data types
df['Age'] = df['Age'].astype(int)
df['Department'] = df['Department'].astype('category')
df['Date'] = pd.to_datetime(df['Date'])

# Numeric conversion with error handling
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')  # Invalid -> NaN
```

### **String Cleaning**

```python
# String operations
df['Name'] = df['Name'].str.strip()           # Remove whitespace
df['Name'] = df['Name'].str.lower()           # Convert to lowercase
df['Name'] = df['Name'].str.replace(' ', '_') # Replace spaces

# Extract information
df['First_Name'] = df['Name'].str.split().str[0]
df['Email_Domain'] = df['Email'].str.split('@').str[1]
```

## 📊 Data Transformation

### **Grouping and Aggregation**

```python
# Group by single column
dept_stats = df.groupby('Department').agg({
    'Age': ['mean', 'min', 'max'],
    'Salary': ['mean', 'sum', 'count']
})

# Group by multiple columns
complex_grouping = df.groupby(['Department', 'Age_Group']).mean()

# Custom aggregation functions
def salary_range(series):
    return series.max() - series.min()

custom_agg = df.groupby('Department')['Salary'].agg([
    'mean', 'std', salary_range
])
```

### **Pivot Tables**

```python
# Create pivot table
pivot = df.pivot_table(
    values='Salary',
    index='Department',
    columns='Age_Group',
    aggfunc='mean',
    fill_value=0
)

# Cross-tabulation
crosstab = pd.crosstab(df['Department'], df['Age_Group'])
```

### **Merging and Joining**

```python
# Sample DataFrames
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'Score': [85, 90, 95]})

# Different types of joins
inner_join = pd.merge(df1, df2, on='ID', how='inner')  # Only matching
left_join = pd.merge(df1, df2, on='ID', how='left')    # All from left
outer_join = pd.merge(df1, df2, on='ID', how='outer')  # All records

# Concatenation
combined = pd.concat([df1, df2], axis=0)  # Vertical
combined = pd.concat([df1, df2], axis=1)  # Horizontal
```

## 🎯 Feature Engineering

### **Creating New Features**

```python
# Mathematical operations
df['Salary_per_Age'] = df['Salary'] / df['Age']
df['Log_Salary'] = np.log(df['Salary'])

# Binning continuous variables
df['Age_Bins'] = pd.cut(df['Age'], bins=[0, 25, 35, 50], labels=['Young', 'Middle', 'Senior'])
df['Salary_Quartiles'] = pd.qcut(df['Salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Date features
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6])
```

### **Encoding Categorical Variables**

```python
# One-hot encoding
dummies = pd.get_dummies(df['Department'], prefix='Dept')
df = pd.concat([df, dummies], axis=1)

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Department_Encoded'] = le.fit_transform(df['Department'])

# Ordinal encoding for ordered categories
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
df['Education_Encoded'] = df['Education'].map({edu: i for i, edu in enumerate(education_order)})
```

## 📈 Data Analysis Patterns

### **Time Series Analysis**

```python
# Set datetime index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Resampling
monthly_avg = df.resample('M').mean()    # Monthly average
daily_sum = df.resample('D').sum()       # Daily sum

# Rolling statistics
df['Rolling_Mean'] = df['Value'].rolling(window=7).mean()
df['Rolling_Std'] = df['Value'].rolling(window=7).std()

# Lag features
df['Value_Lag1'] = df['Value'].shift(1)
df['Value_Lag7'] = df['Value'].shift(7)
```

### **Statistical Analysis**

```python
# Correlation analysis
correlation_matrix = df.corr()
high_corr = correlation_matrix[correlation_matrix > 0.8]

# Outlier detection
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Salary'] < Q1 - 1.5*IQR) | (df['Salary'] > Q3 + 1.5*IQR)]

# Z-score outliers
from scipy import stats
z_scores = np.abs(stats.zscore(df['Salary']))
outliers = df[z_scores > 3]
```

## 🚀 Performance Optimization

### **Memory Optimization**

```python
# Check memory usage
print(df.memory_usage(deep=True))

# Optimize data types
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to category
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif df[col].dtype == 'int64':
            # Downcast integers
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            # Downcast floats
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df

df_optimized = optimize_dtypes(df.copy())
```

### **Efficient Operations**

```python
# Use vectorized operations instead of loops
# ❌ Slow
result = []
for index, row in df.iterrows():
    result.append(row['A'] * row['B'])

# ✅ Fast
result = df['A'] * df['B']

# Use query() for complex filtering
# ✅ Fast and readable
result = df.query('Age > 25 and Salary < 60000')

# Use .loc for setting values
# ✅ Efficient
df.loc[df['Age'] > 30, 'Category'] = 'Senior'
```

## 🎯 ML Integration

### **Preparing Data for Scikit-learn**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df[['Age', 'Salary']]  # Features
y = df['Department']       # Target

# Handle categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### **Feature Selection**

```python
# Correlation-based feature selection
def select_features_by_correlation(df, target_col, threshold=0.1):
    correlations = df.corr()[target_col].abs()
    selected_features = correlations[correlations > threshold].index.tolist()
    selected_features.remove(target_col)  # Remove target itself
    return selected_features

# Variance-based feature selection
def remove_low_variance_features(df, threshold=0.01):
    variances = df.var()
    return df.loc[:, variances > threshold]
```

## ✅ Pandas Checklist

- [ ] Can load and save data in various formats
- [ ] Master data exploration and inspection
- [ ] Know how to handle missing values
- [ ] Understand indexing and selection
- [ ] Can perform grouping and aggregation
- [ ] Know how to merge and join DataFrames
- [ ] Can create new features
- [ ] Understand performance optimization
- [ ] Can prepare data for ML models

## 🚀 Next Steps

With Pandas mastery achieved:
1. **Practice with real datasets** - [Data Visualization Notebook](../notebooks/03-data-visualization.ipynb)
2. **Start your first ML project** - [Regression Module](../02-regression/)
3. **Learn advanced techniques** - [Feature Engineering Notebook](../notebooks/07-feature-engineering-masterclass.ipynb)

## 📚 Additional Resources

- **Pandas Documentation**: [Official Guide](https://pandas.pydata.org/docs/)
- **Pandas Cookbook**: [Practical Recipes](https://pandas.pydata.org/docs/user_guide/cookbook.html)
- **10 Minutes to Pandas**: [Quick Tutorial](https://pandas.pydata.org/docs/user_guide/10min.html)

---

**Navigation:**
- **Previous**: [Linear Algebra Refresher](08-linear-algebra.md)
- **Next**: [Module 2: Regression](../02-regression/README.md)
- **Module Home**: [Introduction](README.md)

*Last Updated: 2025-01-27*
