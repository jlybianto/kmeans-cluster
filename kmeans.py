# ----------------
# IMPORT PACKAGES
# ----------------

# The pandas package is used to fetch and store data in a DataFrame.
import pandas as pd

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