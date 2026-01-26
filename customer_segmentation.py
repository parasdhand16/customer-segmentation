#!/usr/bin/env python
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
customers = pd.read_csv('customers.csv')

# Select features for clustering
X = customers[['AnnualIncome', 'SpendingScore']]

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customers['Cluster'] = kmeans.fit_predict(X)

# Save cluster assignments
customers.to_csv('customers_with_clusters.csv', index=False)

# Plot clusters
plt.figure(figsize=(8, 6))
for cluster in customers['Cluster'].unique():
    cluster_data = customers[customers['Cluster'] == cluster]
    plt.scatter(cluster_data['AnnualIncome'], cluster_data['SpendingScore'], label=f'Cluster {cluster}')

plt.title('Customer Segments')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.savefig('clusters.png')
print('Clustering complete. Results saved to customers_with_clusters.csv and clusters.png')
