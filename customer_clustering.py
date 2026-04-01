# ==============================================================================
# 👥 Customer Segmentation & Predictive Churn Modeling
# Author: Paras Dhand
#
# OVERVIEW:
# This project simulates the entire unstructured raw data to actionable business
# pipeline for an Enterprise e-commerce retailer. It leverages Recency, Frequency,
# and Monetary (RFM) modeling, reduces high-dimensionality using Principal
# Component Analysis (PCA), clusters users via K-Means (unsupervised), and
# predicts trailing 90-day churn probabilities using Logistic Regression (supervised).
# ==============================================================================

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, silhouette_score
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

print("\n--- Initializing Customer Profiling & Churn Pipeline ---\n")

# 1. Synthesize 50,000 Big Data Customers (RFM Model formulation)
print("[1/5] Synthesizing complex raw retail interactions for 50,000 customers...")
n_customers = 50000

# Generating RFM underlying behaviors
recency = np.random.exponential(scale=60, size=n_customers).astype(int) + 1 # Days since last purchase
frequency = np.random.poisson(lam=10, size=n_customers) + 1 # Total lifetime purchases
monetary = (frequency * np.random.normal(loc=120, scale=40, size=n_customers)).clip(min=10).round(2)
tenure = np.random.randint(1, 3650, size=n_customers) # Account age in days
csat_score = np.random.randint(1, 6, size=n_customers) # 1-5 Customer Sat score

# Simulate 'Churn' (1 = Yes, 0 = No) based on mathematical formulation
# High recency, low freq, low CSAT dramatically increases churn probability
churn_formula = (recency * 0.4) - (frequency * 5) - (csat_score * 15)
churn_prob = 1 / (1 + np.exp(-churn_formula / 100)) # Sigmoid squashing
churn = np.random.binomial(1, churn_prob)

df = pd.DataFrame({
    'Customer_ID': range(1001, 1001 + n_customers),
    'Recency_Days': recency,
    'Frequency_LT': frequency,
    'Monetary_Val_USD': monetary,
    'Tenure_Days': tenure,
    'CSAT': csat_score,
    'Will_Churn_90D': churn
})

df.to_csv("rfm_customer_data.csv", index=False)
print(f"Data constructed. Churn Class distribution: {df['Will_Churn_90D'].value_counts().to_dict()}")

#%%
# 2. Advanced Preprocessing & Unsupervised PCA
print("\n[2/5] Running Preprocessing and Principal Component Analysis (PCA)...")
# We remove 'Customer_ID' and 'Churn' before clustering since clustering is Unsupervised
features = ['Recency_Days', 'Frequency_LT', 'Monetary_Val_USD', 'Tenure_Days', 'CSAT']
X_raw = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
print(f"PCA Variance Explained: {pca.explained_variance_ratio_.sum()*100:.1f}% by 2 components.")

df['PCA_1'] = principal_components[:, 0]
df['PCA_2'] = principal_components[:, 1]

#%%
# 3. K-Means Clustering for Buyer Personas
print("\n[3/5] Computing WCSS (Elbow Method) and deploying K-Means Clustering...")
# In a real environment, you run WCSS 1-10 to find K. We assert K=4 for High-Value, Core, New, & Risk
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Evaluating clustering quality taking a sample (huge matrix otherwise)
sample_idx = np.random.choice(len(X_scaled), 5000, replace=False)
sil_score = silhouette_score(X_scaled[sample_idx], df['Cluster'].iloc[sample_idx])
print(f"Cluster Separation Density (Silhouette Index): {sil_score:.3f}")

# Plotting the 2D PCA representation of our 4 Clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA_1', y='PCA_2', hue='Cluster', palette='magma', data=df, alpha=0.5, edgecolor=None)
plt.title('2D PCA Unsupervised K-Means Customer Segments', fontsize=16)
plt.xlabel('Principal Component 1 (Spending Power & Freq)')
plt.ylabel('Principal Component 2 (Tenure & Recency)')
plt.legend(title='Buyer Persona Cluster', loc='upper right')
plt.tight_layout()
plt.savefig('Customer_Segments_PCA.png', dpi=300)
plt.close()
print("✓ Customer segmentation PCA scatter plot saved.")


#%%
# 4. Supervised ML: Logistic Regression vs Churn
print("\n[4/5] Training Logistic Regression model to predict 90-day Churn across segments...")

# Using the clusters as explicit Categorical dummy variables for the predictive model
df_modeling = pd.get_dummies(df, columns=['Cluster'], drop_first=True)
predictive_features = ['Recency_Days', 'Frequency_LT', 'Monetary_Val_USD', 'Tenure_Days', 'CSAT', 'Cluster_1', 'Cluster_2', 'Cluster_3']

X_clf = df_modeling[predictive_features]
y_clf = df_modeling['Will_Churn_90D']

X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

print("\n--- Predictive Churn Classification Report ---")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"Operational Churn ROC-AUC: {roc_auc:.4f}")

#%%
# 5. Extracting Business Value (Feature Importance)
print("\n[5/5] Extracting Actionable Insights via Logistic Coefficients...")

coefficients = pd.DataFrame({'Feature': X_clf.columns, 'Importance_Weight': log_reg.coef_[0]})
coefficients = coefficients.sort_values(by='Importance_Weight', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance_Weight', y='Feature', data=coefficients, palette='coolwarm')
plt.title('Business Impact: Drivers of Customer Lapsing (Churn)', fontsize=14)
plt.xlabel('Logistic Beta Coefficient (Impact Magnitude)')
plt.ylabel('Predictive Metric')
plt.tight_layout()
plt.savefig('Churn_Drivers_Feature_Importance.png', dpi=300)
plt.close()

print("✓ Churn Drivers output generated.")
print("\nEnd-to-end pipeline completed. Outputs generated: Customer_Segments_PCA.png, Churn_Drivers_Feature_Importance.png")
