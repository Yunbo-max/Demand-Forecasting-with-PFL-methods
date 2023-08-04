import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Input data
data = np.array([
    [69373],
    [29223],
    [11327],
    [9103],
    [7631],
    [5456],
    [4095],
    [3824],
    [3804],
    [3191],
    [3026],
    [2503],
    [2415],
    [1992],
    [1914],
    [1849],
    [1668],
    [1607],
    [1582],
    [1440],
    [1354],
    [1248],
    [1094],
    [968],
    [949],
    [948],
    [920],
    [850],
    [672],
    [665],
    [581],
    [579],
    [487],
    [458],
    [269],
    [243],
    [241],
    [232],
    [215],
    [167],
    [164],
    [87],
    [67],
    [35]
])

# Standardize the data
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

# Apply k-means

kmeans = KMeans(n_clusters=5, max_iter=100)
kmeans.fit(data_std)
centroids = kmeans.cluster_centers_

# Print the centroids
print(centroids)

# Assign each data point to a cluster
labels = kmeans.predict(data_std)

# Print the labels
print(labels)

import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans

# read the data into a pandas dataframe
data = pd.DataFrame({
    'state': ['PR', 'CA', 'NY', 'TX', 'IL', 'FL', 'OH', 'PA', 'MI', 'NJ', 'AZ', 'GA', 'MD', 'NC', 'CO', 'VA', 'OR', 'MA', 'TN', 'NV', 'MO', 'HI', 'CT', 'UT', 'NM', 'LA', 'WA', 'WI', 'MN', 'SC', 'IN', 'DC', 'KY', 'KS', 'DE', 'RI', 'WV', 'OK', 'ND', 'ID', 'AR', 'MT', 'IA', 'AL'],
    'count': [69373, 29223, 11327, 9103, 7631, 5456, 4095, 3824, 3804, 3191, 3026, 2503, 2415, 1992, 1914, 1849, 1668, 1607, 1582, 1440, 1354, 1248, 1094, 968, 949, 948, 920, 850, 672, 665, 581, 579, 487, 458, 269, 243, 241, 232, 215, 167, 164, 87, 67, 35]
})

# initialize geolocator
geolocator = Nominatim(user_agent="my_app")

# function to get latitude and longitude for a given state code
def get_lat_long(state_code):
    location = geolocator.geocode(state_code + ', USA')
    return (location.latitude, location.longitude)

# add latitude and longitude columns to the dataframe
data['lat_long'] = data['state'].apply(get_lat_long)

# split the latitude and longitude into separate columns
data['lat'] = data['lat_long'].apply(lambda x: x[0])
data['long'] = data['lat_long'].apply(lambda x: x[1])

# drop the lat_long column
data = data.drop('lat_long', axis=1)

# perform k-means clustering
kmeans = KMeans(n_clusters=6, random_state=0).fit(data[['lat', 'long']])

# add the cluster labels to the dataframe
data['cluster'] = kmeans.labels_

# print the results
print(data.groupby('cluster')['count'].sum())






