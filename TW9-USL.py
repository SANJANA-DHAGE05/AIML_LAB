import numpy as np
from collections import Counter
from tabulate import tabulate  # Import tabulate for table formatting

def euclidean_distance(x1, x2):
  return np.sqrt(np.sum((x1-x2)**2))

class KNN:
  def __init__(self, k=3):
    self.k = k

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X_test):
    for test_point in X_test:
      # Calculate distances between test point and all training points
      distances = [euclidean_distance(test_point, train_point) for train_point in self.X_train]

      # Find k-nearest neighbors
      k_nearest_indices = np.argsort(distances)[:self.k]

      # Get labels of k-nearest neighbors
      k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

      # Count occurrences of each label
      label_counts = Counter(k_nearest_labels)

      # Find the majority class label
      majority_label, majority_count = label_counts.most_common(1)[0]

      # Get the nearest neighbors data points
      nearest_neighbors = self.X_train[k_nearest_indices]

      # Print results
      print(f"Test point: {test_point}")
      print(f"Distances:")

      # Create a list of distances and labels for table formatting
      data = []
      for i, neighbor in enumerate(nearest_neighbors):
        data.append([i+1, neighbor[0], neighbor[1], distances[k_nearest_indices[i]]])

      # Print distances in a table format using tabulate
      print(tabulate(data, headers=["Index", "Feature 1", "Feature 2", "Distance"]))

      print(f"Nearest neighbors:")
      for i, neighbor in enumerate(nearest_neighbors):
        print(f"  - Index: {k_nearest_indices[i]}, Label: {self.y_train[k_nearest_indices[i]]}")
      print(f"Majority class: {majority_label} (Count: {majority_count})")

# Load the data from the table
X = np.array([[40, 20], [50, 50], [60, 90], [10, 25], [70, 70], [60, 10], [25, 80]])
y = np.array(['Red', 'Blue', 'Blue', 'Red', 'Blue', 'Red', 'Blue'])
new_data = np.array([[55, 80]])  # Include only [55, 80] in the test data

# Create a KNN model with k=3
knn = KNN(k=5)

# Train the model
knn.fit(X, y)

# Predict the class for new data points
knn.predict(new_data)
