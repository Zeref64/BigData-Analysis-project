import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Φόρτωση δεδομένων από το καθαρισμένο dataset
data = pd.read_csv("Cleaned_Fire_Incidents_Data.csv")

# Επιλέγουμε δύο στήλες για clustering (π.χ., Latitude και Longitude)
X = data[['Latitude', 'Longitude']].dropna()

# Ορισμός αριθμού clusters
k = 5

# Εφαρμογή K-Means Clustering
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Οπτικοποίηση των αποτελεσμάτων
plt.figure(figsize=(10, 8))
plt.scatter(X['Latitude'], X['Longitude'], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')
plt.title('K-Means Clustering of Fire Incidents')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()
