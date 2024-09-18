# Step 1: Set Up the Environment
# Make sure you have the necessary libraries installed
# pip install pandas numpy matplotlib seaborn scikit-learn

# Step 2: Load the Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
url = r'C:\Users\Rutuja\Documents\Prodigy\Prodigy_ML_02\Mall_Customers (1).csv' 
data = pd.read_csv(url)

# Display the first few rows
print(data.head())

# Step 3: Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Selecting relevant features (assumed columns)
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardizing the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Determine the Optimal Number of Clusters
wcss = []  # Within-cluster sums of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Step 5: Apply K-means Clustering
# Fit K-means with the chosen number of clusters (assume 5)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the original data
data['Cluster'] = clusters

# Step 6: Visualize the Clusters
plt.figure(figsize=(10, 6))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=data['Cluster'], cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('Annual Income (standardized)')
plt.ylabel('Spending Score (standardized)')
plt.colorbar(label='Cluster')
plt.show()

# Step 7: Analyze the Clusters
# Group by cluster to analyze
cluster_analysis = data.groupby('Cluster').mean()
print(cluster_analysis)
