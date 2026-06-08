import pandas as pd
import os
os.makedirs("screenshots", exist_ok=True)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("retail_customer_segmentation_dataset.csv")

features = [
    'Age',
    'AnnualIncome',
    'TotalSpend',
    'OrderCount',
    'AvgBasket',
    'NumCategories'
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1, 11):
    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.savefig("screenshots/elbow_method.png")
plt.show()

k = int(input("Enter optimal number of clusters: "))

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=df["Cluster"],
    cmap="viridis",
    s=60
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.savefig("screenshots/cluster_visualization.png")
plt.show()

cluster_summary = df.groupby("Cluster")[features].mean()

print("\nCluster Summary:")
print(cluster_summary)

df.to_csv("customer_segments.csv", index=False)

print("\nClustered dataset saved as customer_segments.csv")
