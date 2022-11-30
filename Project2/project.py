import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1

df = pd.read_csv("Car_sales.csv")
print("Number of samples before dropping NaN values:",len(df))
df = df.dropna().reset_index(drop=True)
print("Number of samples after dropping NaN values:",len(df))

# 2

sse = {}
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, max_iter=1000)
    label = kmeans.fit_predict(df[['Price_in_thousands','Engine_size']])
    u_labels = np.unique(label)
    df["clusters"] = kmeans.labels_
    centroids = kmeans.cluster_centers_
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    
    num_elements = df['clusters'].value_counts().to_numpy()
    clusters = df['clusters'].value_counts().index.to_numpy()
    
    print("\nK =", k)
    for i in range(len(num_elements)):
        print("Number of samples in cluster", clusters[i]+1, "is:", num_elements[i])
    
    plt.figure(figsize=(10,5))
    plt.title("K = " + str(k))
    for i in u_labels:
        plt.scatter(df[df["clusters"] == i]['Price_in_thousands'], 
                    df[df["clusters"] == i]['Engine_size'],
                    label="cluster: "+ str(i+1))
    plt.xlabel("Price in thousands")
    plt.ylabel("Engine size")
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k',label='centroid')
    plt.legend()
    plt.show()
    
    df.drop(columns=['clusters'],inplace=True)
    
plt.figure(figsize=(10,5))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sums of squares (WCSS)')
plt.title('Elbow method to determine optimum number of clusters')
plt.show()

# 3

# Need to drop manufacturer column as well
df.drop(columns=["Manufacturer","Model", "Vehicle_type", "Latest_Launch"], inplace=True)
# Standardize data
data_norm = StandardScaler().fit_transform(df.values)

pca = PCA(n_components=2)
pcaComponents = pca.fit_transform(data_norm)

df_pca = pd.DataFrame(data = pcaComponents,
             columns = ['1_PC', '2_PC'])

sse = {}
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, max_iter=1000)
    label = kmeans.fit_predict(df_pca[['1_PC','2_PC']])
    u_labels = np.unique(label)
    df_pca["clusters"] = kmeans.labels_
    centroids = kmeans.cluster_centers_
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    
    num_elements = df_pca['clusters'].value_counts().to_numpy()
    clusters = df_pca['clusters'].value_counts().index.to_numpy()
    
    print("\nK =", k)
    for i in range(len(num_elements)):
        print("Number of samples in cluster", clusters[i]+1, "is:", num_elements[i])
    
    plt.figure(figsize=(10,5))
    plt.title("K = " + str(k))
    for i in u_labels:
        plt.scatter(df_pca[df_pca["clusters"] == i]['1_PC'], 
                    df_pca[df_pca["clusters"] == i]['2_PC'],
                    label="cluster: "+ str(i+1))
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k',label='centroid')
    plt.legend()
    plt.show()
    
    df_pca.drop(columns=['clusters'],inplace=True)
    
plt.figure(figsize=(10,5))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sums of squares (WCSS)')
plt.title('Elbow method to determine optimum number of clusters')
plt.show()
