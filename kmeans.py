# ----------------
# IMPORT PACKAGES
# ----------------

# The pandas package is used to fetch and store data in a DataFrame.
# Whitening to rescale or normalize features through dividing by its standard deviation.
import pandas as pd
from scipy.cluster.vq import whiten, kmeans, vq
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

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

Male_GDP_list = []
for m, n in data.iterrows():
	Male_GDP_list.append([n["GDPperCapita"], n["lifeMale"]])

Female_GDP_list = []
for x, y in data.iterrows():
	Female_GDP_list.append([y["GDPperCapita"], y["lifeFemale"]])

IM_GDP_list = []
for i, j in data.iterrows():
	IM_GDP_list.append([j["GDPperCapita"], j["infantMortality"]])

# ----------------
# MODEL DATA (Infant Mortality vs. GDP per Capita)
# ----------------

# Convert list into a NumPy array type.
IM_GDP_numpy = np.array(IM_GDP_list)

# Determine the coordinates of centers for a range of k-number clusters.
k_list = range(1, 11)
centers = [kmeans(IM_GDP_numpy, k) for k in k_list]
# Result: ("A k x N array of k-centroids, Distortion")
# Distortion is defined as the sum of the squared differences between the observations and the corresponding centroid.

# Append the center coordinates to a new centroid list without the distortion.
centroids = [center for (center, distortion) in centers]

# Calculation of distances from each data point in whiten_list to centroid centers.
distances = [cdist(IM_GDP_numpy, center, "euclidean") for center in centroids]
distances = [np.min(d, axis=1) for d in distances] # Shift individual distances as its own list
sum_squares = [sum(d) / len(d) for d in distances]
 
print("Sum of Squares Within Each Number of Clusters (k): ")
for i, j in zip(k_list, sum_squares):
	print i, j

# Visualization of different k-means clustering to determine the best number of clusters to be used.
plt.figure()
plt.plot(k_list, sum_squares)
plt.gca().grid(True)
plt.xlabel("Number of Clusters (k)", fontsize=14)
plt.ylabel("Within-Cluster Sum of Squares", fontsize=14)
plt.title("Number of Clusters vs. Sum of Squares Within Each Cluster", fontsize=16)
plt.show()

# Infant Mortality vs. GDP per Capita
print("")
infant_k_input = int(raw_input("Select the number of clusters (1 - 10) for Infant Mortality: "))
if infant_k_input == 1:
	IM_GDP,_ = vq(IM_GDP_numpy, centroids[0])
	plt.plot(IM_GDP_numpy[IM_GDP == 0, 0], IM_GDP_numpy[IM_GDP == 0, 1], "or")
elif infant_k_input == 2:
	IM_GDP,_ = vq(IM_GDP_numpy, centroids[1])
	plt.plot(IM_GDP_numpy[IM_GDP == 0, 0], IM_GDP_numpy[IM_GDP == 0, 1], "or",
		IM_GDP_numpy[IM_GDP == 1, 0], IM_GDP_numpy[IM_GDP == 1, 1], "ob")
elif infant_k_input == 3:
	IM_GDP,_ = vq(IM_GDP_numpy, centroids[2])
	plt.plot(IM_GDP_numpy[IM_GDP == 0, 0], IM_GDP_numpy[IM_GDP == 0, 1], "or",
		IM_GDP_numpy[IM_GDP == 1, 0], IM_GDP_numpy[IM_GDP == 1, 1], "ob",
		IM_GDP_numpy[IM_GDP == 2, 0], IM_GDP_numpy[IM_GDP == 2, 1], "og")
plt.xlabel("GDP per Capita (USD)", fontsize=14)
plt.ylabel("Infant Mortality (Per 1000)", fontsize=14)
plt.title("Clustering Infant Mortality and GDP per Capita", fontsize=16)
plt.show()