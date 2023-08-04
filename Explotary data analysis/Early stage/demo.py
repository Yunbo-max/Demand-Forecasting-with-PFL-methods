# import pandas as pd
# import numpy as np
# from geopy.geocoders import Nominatim
# from sklearn.cluster import KMeans
#
#
# def get_latitude(x):
#   if hasattr(x,'latitude') and (x.latitude is not None):
#      return x.latitude
#
#
# def get_longitude(x):
#   if hasattr(x,'longitude') and (x.longitude is not None):
#      return x.longitude
#
#
# # create a dataframe with state abbreviations and case numbers
# data = {'State': ['PR', 'CA', 'NY', 'TX', 'IL', 'FL', 'OH', 'PA', 'MI', 'NJ', 'AZ', 'GA', 'MD', 'NC', 'CO', 'VA', 'OR', 'MA', 'TN', 'NV', 'MO', 'HI', 'CT', 'UT', 'NM', 'LA', 'WA', 'WI', 'MN', 'SC', 'IN', 'DC', 'KY', 'KS', 'DE', 'RI', 'WV', 'OK', 'ND', 'ID', 'AR', 'MT', 'IA', 'AL'],
#         'Cases': [69373, 29223, 11327, 9103, 7631, 5456, 4095, 3824, 3804, 3191, 3026, 2503, 2415, 1992, 1914, 1849, 1668, 1607, 1582, 1440, 1354, 1248, 1094, 968, 949, 948, 920, 850, 672, 665, 581, 579, 487, 458, 269, 243, 241, 232, 215, 167, 164, 87, 67, 35]}
# df = pd.DataFrame(data)
#
# # get the latitude and longitude for each state
# geolocator = Nominatim(user_agent='my_app')
# df['Latitude'] = df['State'].apply(lambda x: get_latitude(geolocator.geocode(x + ', USA')))
# df['Longitude'] = df['State'].apply(lambda x: get_longitude(geolocator.geocode(x + ', USA')))
#
# # create a matrix with latitude and longitude
# X = df[['Latitude', 'Longitude']].to_numpy()
#
# # cluster the states into five groups
# kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
# df['Group'] = kmeans.labels_
#
# # add up the case numbers for each group
# grouped = df.groupby('Group').sum()['Cases']
#
# print(grouped)


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read in the data
data = pd.read_csv('data.csv', header=None, names=['area', 'money'])
print(data)
# Normalize the data
scaler = StandardScaler()
data['money'] = scaler.fit_transform(data[['money']])

# Set the number of clusters
num_clusters = 3

# Initialize the model
kmeans = KMeans(n_clusters=num_clusters, init='random', random_state=42)

# Fit the model to the data
kmeans.fit(data[['money']])

# Get the cluster assignments
data['cluster'] = kmeans.labels_

# Print the cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(centers):
    print(f'Cluster {i}: {center[0]:,.0f}')

# Print the areas in each cluster
for i in range(num_clusters):
    print(f'Cluster {i}:')
    print(data.loc[data['cluster'] == i, 'area'].values)