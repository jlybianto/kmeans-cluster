# ----------------
# IMPORT PACKAGES
# ----------------

# The pandas package is used to fetch and store data in a DataFrame.
# Whitening to rescale or normalize features through dividing by its standard deviation.
import pandas as pd
from scipy.cluster.vq import whiten, kmeans
from scipy.spatial.distance import cdist

# ----------------
# OBTAIN DATA
# ----------------

# Data imported from a Comma Separated Value document.
un = pd.read_csv("un.csv")

# ----------------
# PROFILE DATA
# ----------------

# Determine the number of observations (rows) and number of features (columns) in the dataset.
num_observations = len(un)
num_features = len(un.columns)
features = un.count()
print("Number of Data Points (Rows): " + str(num_observations))
print("")
print("Number of Features (Columns): " + str(num_features))
print("")
print("Number of Data Points per Feature: ")
print(features)
print("")
print("Data Type of each Feature: ")
print(un.dtypes)
print("")

# Filter data to columns 6, 7, 8, 9; lifeMale, lifeFemale, infantMortality and GDPperCapita.
data = un.ix[0:, :10].dropna()
new_list = []
for i, j in data.iterrows():
	new_list.append([j["infantMortality"], j["GDPperCapita"]])

# ----------------
# MODEL DATA
# ----------------

# Before running k-means, it is beneficial to rescale each feature dimension of the observation set with whitening.
# Each feature is divided by its standard deviation across all observations to give it unit variance.
whiten_list = whiten(new_list)

# Determine the coordinates of centers for a range of k-number clusters.
k_list = range(1, 11)
centers = [kmeans(whiten_list, k) for k in k_list]
# Result: ("A k x N array of k-centroids, Distortion")
# Distortion is defined as the sume of the squared differences between the observations and the corresponding centroid.

# Append the center coordinates to a new centroid list without the distortion.
centroids = [center for (center, distortion) in centers]