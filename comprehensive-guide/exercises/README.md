# 🏋️ Practice Exercises and Challenges

> **Structured exercises to test and reinforce your ML knowledge**

This directory contains practice exercises, coding challenges, and mini-projects designed to test your understanding of machine learning concepts. Each exercise builds upon the knowledge from the comprehensive guide modules.

## 🎯 Exercise Categories

### **📊 Data Analysis Exercises**
- **Exercise 1**: Exploratory Data Analysis Challenge
- **Exercise 2**: Data Cleaning and Preprocessing
- **Exercise 3**: Feature Engineering Workshop
- **Exercise 4**: Statistical Analysis Practice

### **🤖 Algorithm Implementation**
- **Exercise 5**: Linear Regression from Scratch
- **Exercise 6**: Logistic Regression Implementation
- **Exercise 7**: Decision Tree Building
- **Exercise 8**: K-Means Clustering

### **📈 Model Evaluation**
- **Exercise 9**: Cross-Validation Mastery
- **Exercise 10**: Hyperparameter Tuning
- **Exercise 11**: Model Selection Challenge
- **Exercise 12**: Bias-Variance Analysis

### **🚀 End-to-End Projects**
- **Project 1**: Customer Churn Prediction
- **Project 2**: House Price Prediction
- **Project 3**: Image Classification
- **Project 4**: Recommendation System

## 📋 Exercise Structure

Each exercise follows this format:

### **Problem Statement**
Clear description of what you need to accomplish

### **Learning Objectives**
What skills and concepts you'll practice

### **Dataset Description**
Information about the data you'll work with

### **Requirements**
Specific deliverables and success criteria

### **Starter Code**
Basic structure to get you started

### **Solution Guidelines**
Hints and approaches (without giving away the answer)

### **Evaluation Rubric**
How your solution will be assessed

## 🏆 Difficulty Levels

### **🟢 Beginner (Level 1)**
- Basic data manipulation
- Simple algorithm implementation
- Guided step-by-step exercises
- Clear success criteria

### **🟡 Intermediate (Level 2)**
- Complex data preprocessing
- Algorithm optimization
- Multiple solution approaches
- Performance comparison

### **🔴 Advanced (Level 3)**
- Research-level challenges
- Novel algorithm implementation
- Open-ended problems
- Real-world complexity

## 📊 Sample Exercise: Customer Churn Prediction

### **Problem Statement**
Build a machine learning model to predict customer churn for a telecommunications company. You need to achieve at least 85% accuracy while maintaining interpretability for business stakeholders.

### **Learning Objectives**
- Practice end-to-end ML workflow
- Handle imbalanced datasets
- Feature engineering for business problems
- Model interpretation and explanation

### **Dataset**
- **Size**: 7,043 customers
- **Features**: 21 (demographics, services, account info)
- **Target**: Binary (churned/not churned)
- **Challenge**: 26.5% churn rate (imbalanced)

### **Requirements**
1. **Data Analysis** (25 points)
   - Comprehensive EDA with visualizations
   - Identify key patterns and insights
   - Handle missing values appropriately

2. **Feature Engineering** (25 points)
   - Create meaningful derived features
   - Handle categorical variables
   - Feature selection and importance analysis

3. **Model Development** (30 points)
   - Try at least 3 different algorithms
   - Proper train/validation/test splits
   - Handle class imbalance

4. **Evaluation** (20 points)
   - Use appropriate metrics for imbalanced data
   - Cross-validation for robust estimates
   - Business impact analysis

### **Starter Code**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load data
df = pd.read_csv('customer_churn.csv')

# Your code here
# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Feature Engineering
# 4. Model Training
# 5. Evaluation

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def preprocess_data(self, df):
        """Implement your preprocessing logic"""
        pass
    
    def train(self, X, y):
        """Train your model"""
        pass
    
    def predict(self, X):
        """Make predictions"""
        pass
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        pass
```

### **Evaluation Rubric**

| Criteria | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|----------|---------------|----------|------------------|----------------------|
| **Data Analysis** | Comprehensive insights, beautiful visualizations | Good analysis, clear visualizations | Basic analysis, adequate visualizations | Minimal analysis, poor visualizations |
| **Feature Engineering** | Creative, well-justified features | Good feature creation | Basic feature engineering | Minimal feature work |
| **Model Performance** | >90% AUC, well-tuned | 85-90% AUC, good tuning | 80-85% AUC, basic tuning | <80% AUC, poor tuning |
| **Code Quality** | Clean, documented, modular | Well-structured, mostly documented | Functional, some documentation | Messy, undocumented |
| **Business Impact** | Clear ROI analysis, actionable insights | Good business understanding | Basic business context | No business consideration |

## 🎯 Exercise Progression

### **Week 1-2: Foundations**
- Exercise 1: EDA Challenge
- Exercise 2: Data Cleaning
- Exercise 5: Linear Regression from Scratch

### **Week 3-4: Core Algorithms**
- Exercise 6: Logistic Regression
- Exercise 7: Decision Trees
- Exercise 9: Cross-Validation

### **Week 5-6: Advanced Techniques**
- Exercise 10: Hyperparameter Tuning
- Exercise 11: Model Selection
- Project 1: Customer Churn

### **Week 7-8: Specialization**
- Project 2: House Prices
- Project 3: Image Classification
- Project 4: Recommendation System

## 💡 Success Tips

### **Before Starting**
- Read the problem statement carefully
- Understand the evaluation criteria
- Set up your development environment
- Plan your approach

### **During Development**
- Start simple, then iterate
- Document your thought process
- Test your code frequently
- Seek feedback early and often

### **Code Quality Standards**
- Use meaningful variable names
- Add comments for complex logic
- Follow PEP 8 style guidelines
- Include docstrings for functions

### **Common Pitfalls to Avoid**
- Data leakage in preprocessing
- Overfitting to validation set
- Ignoring class imbalance
- Poor feature engineering
- Inadequate model evaluation

## 🔗 Resources and Support

### **Getting Help**
1. **Documentation**: Check official docs first
2. **Community**: Ask in ML Zoomcamp Slack
3. **Stack Overflow**: Search for similar problems
4. **Office Hours**: Attend weekly Q&A sessions

### **Additional Practice**
- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [DrivenData Challenges](https://www.drivendata.org/)
- [Analytics Vidhya](https://www.analyticsvidhya.com/)

### **Code Review**
- Share your solutions for peer review
- Review others' code to learn different approaches
- Participate in code review sessions

## 📈 Progress Tracking

### **Beginner Exercises**
- [ ] Exercise 1: EDA Challenge
- [ ] Exercise 2: Data Cleaning
- [ ] Exercise 3: Feature Engineering
- [ ] Exercise 5: Linear Regression

### **Intermediate Exercises**
- [ ] Exercise 6: Logistic Regression
- [ ] Exercise 7: Decision Trees
- [ ] Exercise 9: Cross-Validation
- [ ] Exercise 10: Hyperparameter Tuning

### **Advanced Projects**
- [ ] Project 1: Customer Churn
- [ ] Project 2: House Prices
- [ ] Project 3: Image Classification
- [ ] Project 4: Recommendation System

## 🏅 Certification Path

Complete exercises to earn certificates:

### **Foundation Certificate**
- Complete 4 beginner exercises
- Score average of 3+ on rubric
- Submit reflection essay

### **Practitioner Certificate**
- Complete 2 intermediate exercises
- Complete 1 end-to-end project
- Present solution to peers

### **Expert Certificate**
- Complete 2 advanced projects
- Contribute to community (help others)
- Create original exercise/tutorial

---

**Ready to Practice?** 🚀

Start with Exercise 1 and work your way through the progression. Remember, the goal is deep understanding, not just completion!

---

**Navigation:**
- **Course Home**: [Main Guide](../README.md)
- **Notebooks**: [Practice Notebooks](../notebooks/README.md)

*Last Updated: 2025-01-27*
