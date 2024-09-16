import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

cust_data = pd.read_csv('/Users/bgracias/MAIN/Python/Data-Sci/Customer-Segmentation/Mall_Customers.csv')

# print(cust_data.head())
# print(cust_data.shape)
# print(cust_data.info())
# print(cust_data.isnull().sum())

## Annual Income & Spending Score columns
X = cust_data.iloc[:,[3,4]].values

## Choosing number of clusters: WCSS - Within Cluster Sum of Squares
## Finding WCSS for different number of clusters
# wcss = []
# for i in range(1,11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)

## Plot elbow graph
# sns.set()
# plt.plot(range(1,11), wcss)
# plt.title('Elbow Point Graph')
# plt.xlabel('No. Clusters')
# plt.ylabel('WCSS')
# plt.show()

## Optimum number of Clusters = 5 because no further sudden drops
## Train k-Means Clustering Model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

## Return label for each data point based on each cluster i.e. cluster 1 through 5
Y = kmeans.fit_predict(X)
# print(Y)

## Visualize clusters: plotting cluster and centroids
## First value is cluster number, second value is income vs spending score
plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')
## Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Groups')
plt.show()