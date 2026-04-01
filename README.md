# 👥 Advanced Customer Segmentation & Churn Modeling

This repository focuses on breaking down an unstructured consumer dataset into actionable business demographics. The goal is to drive precision marketing and drastically reduce customer churn. I accomplished this through deep exploratory data analysis, powerful clustering models, and predictive classification algorithms.

## 🧠 Modeling & Deep Analysis

### 1. K-Means Clustering & PCA
Before predicting behavior, I applied Principal Component Analysis (PCA) to discover variance features, followed by **K-Means Clustering** to divide the customer base into distinct "Buyer Personas" based on recency, frequency, and monetary (RFM) metrics. 

### 2. Decision Tree Classifiers
I implemented robust **Decision Trees** to map out the exact behavioral pathways that lead a customer to purchase or churn. The tree splits were evaluated using Gini Impurity and Entropy to maximize business clarity on the critical drivers of retention.

### 3. Logistic Regression for Churn Prediction
To give the marketing team actionable data, I deployed a **Logistic Regression** model predicting the continuous probability of a customer lapsing within 90 days. I optimized the model's coefficients using cross-validation, ensuring the dashboard outputs were extremely precise.

## 🚀 Business Impact & Dashboards
The final output of these models fed directly into a creative, dynamic dashboard, allowing stakeholders to dynamically select segments and deploy tailored email campaigns based on the output of my predictive models.

---
**Tech Stack:** `Python`, `Scikit-Learn`, `Pandas`, `Matplotlib`, `Logistic Regression`, `Clustering`