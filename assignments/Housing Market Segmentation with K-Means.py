import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://raw.githubusercontent.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/master/USA_Housing.csv"
df = pd.read_csv(url)

# Select features for clustering
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
        'Avg. Area Number of Bedrooms', 'Area Population', 'Price']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters (using two features for 2D plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='avg._area_income', y='price', hue='Cluster', palette='viridis')
plt.title('Housing Market Segments by Income and Price')
plt.xlabel('Average Area Income')
plt.ylabel('Price')
plt.savefig('housing_clusters.png')
plt.close()

# Print cluster centers (in original scale)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers (in original scale):")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i}: {center}")