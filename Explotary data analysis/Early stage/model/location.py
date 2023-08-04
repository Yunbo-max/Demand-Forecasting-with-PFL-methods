# -*- coding = utf-8 -*-
# @time:18/03/2023 14:41
# Author:Yunbo Long
# @File:location.py
# @Software:PyCharm

# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut
# from time import sleep
#
# geolocator = Nominatim(user_agent="myGeocoder")
#
# coordinates = [
#     (21.30562019, -157.8386993),
#     (21.31185532,-158.0046844),
#     (37.29223251, -121.881279),
#     # Add the rest of the coordinates here
# ]
#
# def get_city(coordinate):
#     try:
#         location = geolocator.reverse(coordinate, timeout=10)
#         city = location.raw["address"].get("city", "Unknown")
#         return city
#     except GeocoderTimedOut:
#         sleep(1)
#         return get_city(coordinate)
#
# cities = [get_city(coord) for coord in coordinates]
#
# for coord, city in zip(coordinates, cities):
#     print(f"Coordinate: {coord}, City: {city}")












# import pandas as pd
# from sklearn.cluster import KMeans
# from openpyxl import Workbook
#
# # Read latitude and longitude data from the input Excel file
# input_file = 'E:\Cambridge\Dissertation\Dataset\DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS\location.xlsx'
# df = pd.read_excel(input_file, header=None, names=['Latitude', 'Longitude'])
#
# # Determine the number of clusters (you can adjust this value)
# num_clusters = 150
#
# df = df.iloc[1:,:]
#
# # Cluster the data using K-means
# kmeans = KMeans(n_clusters=num_clusters,random_state=0)
# df['cluster'] = kmeans.fit_predict(df)
#
#
#
# # Calculate the cluster centers
# cluster_centers = kmeans.cluster_centers_
# cluster_centers_df = pd.DataFrame(cluster_centers, columns=["Latitude", "Longitude"])
# cluster_centers_df["Cluster"] = range(num_clusters)
# cluster_centers_df["Cluster"] = "Center_" + cluster_centers_df["Cluster"].astype(str)
# count = pd.DataFrame()
# count[''] =df['cluster'].value_counts()
#
# # Save the clustered data and the cluster centers to separate sheets in a new Excel file
# with pd.ExcelWriter("clustering_results.xlsx") as writer:
#     df.to_excel(writer, sheet_name="Clustered_Data", index=False)
#     cluster_centers_df.to_excel(writer, sheet_name="Cluster_Centers", index=False)
#     count.to_excel(writer, sheet_name="Cluster_Count", index=False)



import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# read data from excel file
df = pd.read_excel('E:\Cambridge\Dissertation\Dataset\DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS\location.xlsx')

# extract latitude and longitude columns
latitude = df['Latitude'].values
longitude = df['Longitude'].values

# create a 2D array of latitude and longitude
X = df.values

# initialize an empty list to store WCSS (Within-Cluster-Sum-of-Squared) values
wcss = []

# fit k-means to the data for different numbers of clusters
for i in range(1,150):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# plot WCSS values to determine the best number of clusters
plt.plot(range(1,150), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

