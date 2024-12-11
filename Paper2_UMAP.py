import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.cluster import rand_score, adjusted_rand_score, adjusted_mutual_info_score, v_measure_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def kmeans8(umap1,umap2,cells, num_clusters):
    start_time = timeit.default_timer() # Time algorithm
    centroids, z = kmeans2(cells, num_clusters)
    elapsed = timeit.default_timer() - start_time
    print(f"Kmeans Elapsed time: {elapsed*1000} ms") # Print duration

    plt.figure(figsize=(10, 10))

    colors = plt.cm.Spectral(np.linspace(0, 1, num_clusters))
    for x, z_val in zip(range(len(umap1)), z):
        color = colors[z_val] if z_val < len(colors) else 'violet'
        plt.scatter(umap1[x], umap2[x], s=50, color=color, edgecolor='k')

    #plt.scatter(centroids[:,0], centroids[:,1], s=10, c='black')
    plt.title("K-Means Clustering")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.show()
    return z

def DBSC(cells, umap1, umap2, set_eps):
    start_time = timeit.default_timer() # Time algorithm
    dbscan = DBSCAN(eps=set_eps, min_samples=5)
    dbscan_labels = dbscan.fit_predict(cells)
    elapsed = timeit.default_timer() - start_time
    print(f"DBSCAN Elapsed time: {elapsed*1000} ms") # Print duration

    plt.figure(figsize=(10, 10))

    # Plot all clusters (ignore label -1 for now)
    unique_labels = np.unique(dbscan_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        if label == -1:
            continue  # Skip noise for now
        # Plot each cluster as a separate color
        cluster_point1 = umap1[dbscan_labels == label]
        cluster_point2 = umap2[dbscan_labels == label]
        plt.scatter(cluster_point1, cluster_point2, 
                    s=50, color=color, label=f'Cluster {label}', edgecolor='k')

    # Plot noise points (label = -1) in black
    noise_points1 = umap1[dbscan_labels == -1]
    noise_points2 = umap2[dbscan_labels == -1]
    plt.scatter(noise_points1, noise_points2, 
                s=30, color='k', label='Noise', edgecolor='k')

    plt.title("DBSCAN Clustering")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.show()
    return dbscan_labels

def AggCluster(cells,umap1,umap2,n_clusters):
    start_time = timeit.default_timer() # Time algorithm
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg_cluster.fit_predict(cells)
    elapsed = timeit.default_timer() - start_time
    print(f"Agglomerative Clustering Elapsed time: {elapsed*1000} ms") # Print duration

    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(agg_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        cluster_points1 = umap1[agg_labels == label]
        cluster_points2 = umap2[agg_labels == label]

        plt.scatter(cluster_points1, cluster_points2, 
                    s=30, color=color, label=f'Cluster {label}', edgecolor='k')

    plt.title("Agglomerative Clustering")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.show()
    return agg_labels

if __name__ == "__main__":
    # Initialize data
    data = pd.read_csv('..\\Project\\paper2_data\\Cells.csv') 
    data.dropna(subset=["umap1"], inplace=True)
    data = data.to_numpy()[:,:]
    plotx = np.asarray(data[2:,5], dtype=float) # TSNE X-coordinate
    ploty = np.asarray(data[2:,6], dtype=float) # TSNE Y-coordinate
    original_labels = np.asarray(data[2:,2]) 
    num_clusters = 8
    eps = 11

    plt.figure(figsize=(8, 4))
    plt.scatter(plotx, ploty, s=30, edgecolor='k')
    plt.title("UMAP Clustering")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.show()

    print("Cells in dataset:", len(plotx))

    # Enumerate original labels for comparison
    d = {ni: indi for indi, ni in enumerate(set(original_labels))}
    numbered_labels = [d[ni] for ni in original_labels]

    print(d)

    cells = []
    for x in range(len(plotx)):
        cells.append([plotx[x], ploty[x]])

    # Agglomerative Clustering
    ac_labels = AggCluster(cells,plotx,ploty,num_clusters)

    # DBSCAN
    dbscan_labels = DBSC(cells,plotx,ploty,eps)

    # kmeans clustering
    kmeans_labels = kmeans8(plotx,ploty,cells,num_clusters)

    # RAND SCORES
    print("         Rand Score Agglomerative:", rand_score(numbered_labels, ac_labels)) #numbered labels and clustering label outputs
    print("Adjusted Rand Score Agglomerative:", adjusted_rand_score(numbered_labels, ac_labels))
    print("         Rand Score DBSCAN:", rand_score(numbered_labels, dbscan_labels)) #numbered labels and clustering label outputs
    print("Adjusted Rand Score DBSCAN:", adjusted_rand_score(numbered_labels, dbscan_labels))
    print("         Rand Score kmeans:", rand_score(numbered_labels, kmeans_labels)) #numbered labels and clustering label outputs
    print("Adjusted Rand Score kmeans:", adjusted_rand_score(numbered_labels, kmeans_labels)) #numbered labels and clustering label outputs

    # SILHOUETTE SCORE
    print("Original Silhouette Score:", silhouette_score(cells, numbered_labels))
    print("Agglom Silhouette Score:", silhouette_score(cells, ac_labels))
    print("DBSCAN Silhouette Score:", silhouette_score(cells, dbscan_labels))
    print("kmeans Silhouette Score:", silhouette_score(cells, kmeans_labels))

    # adjusted_mutual_info_score, v_measure_score, normalized_mutual_info_score
    # NORMALIZED MUTUAL INFO SCORE
    print("Normalized Mutual Info Score Agglomerative:", normalized_mutual_info_score(numbered_labels, ac_labels))
    print("Normalized Mutual Info Score DBSCAN:", normalized_mutual_info_score(numbered_labels, dbscan_labels))
    print("Normalized Mutual Info Score kmeans:", normalized_mutual_info_score(numbered_labels, kmeans_labels))

    # ADJUSTED MUTUAL INFO SCORE
    print("Adjusted Mutual Info Score Agglomerative:", adjusted_mutual_info_score(numbered_labels, ac_labels))
    print("Adjusted Mutual Info Score DBSCAN:", adjusted_mutual_info_score(numbered_labels, dbscan_labels))
    print("Adjusted Mutual Info Score kmeans:", adjusted_mutual_info_score(numbered_labels, kmeans_labels))

    # V_SCORE
    print("V-Measure Score Agglomerative:", v_measure_score(numbered_labels, ac_labels))
    print("V-Measure Score DBSCAN:", v_measure_score(numbered_labels, dbscan_labels))
    print("V-Measure Score kmeans:", v_measure_score(numbered_labels, kmeans_labels))
