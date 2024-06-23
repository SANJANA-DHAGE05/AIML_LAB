import numpy as np

# Example data - you can replace this with your actual data
data = np.array([
    [1.0, 1.0],
    [1.5, 2.0],
    [3.0, 4.0],
    [5.0, 7.0],
    [3.5, 5.0],
    [4.5, 5.0],
    [3.5, 4.5]
])

# Initialize two centroids randomly from the data points
initial_centroids = data[np.random.choice(data.shape[0], 2, replace=False)]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def k_means_clustering(data, centroids, max_iterations=100):
    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        clusters = {}
        for i in range(len(centroids)):
            clusters[i] = []

        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)

        # Update the centroids
        new_centroids = []
        for i in range(len(centroids)):
            if clusters[i]:  # Check if there are points assigned to the centroid
                new_centroid = np.mean(clusters[i], axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[i])  # Keep the old centroid if no points assigned

        new_centroids = np.array(new_centroids)

        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters

# Print the initial centroids
print("Initial centroids:")
for idx, centroid in enumerate(initial_centroids):
    print(f"Centroid {idx + 1}: {centroid}")
print()

# Perform k-means clustering
final_centroids, clusters = k_means_clustering(data, initial_centroids)

# Print the cluster assignments
for idx, cluster_points in clusters.items():
    print(f"Cluster {idx + 1}:")
    for point in cluster_points:
        print(point)
    print()

# Print the final centroids
print("Final centroids:")
for idx, centroid in enumerate(final_centroids):
    print(f"Centroid {idx + 1}: {centroid}")
