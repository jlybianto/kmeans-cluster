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
observations = len(un)
features = un.count()
