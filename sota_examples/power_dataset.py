from ucimlrepo import fetch_ucirepo 

# Fetch dataset
household_power = fetch_ucirepo(id=235) 

# Load features as a pandas DataFrame
X = household_power.data.features 
print(X.head())
