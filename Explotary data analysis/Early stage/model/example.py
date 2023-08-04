# -*- coding = utf-8 -*-
# @time:17/03/2023 01:35
# Author:Yunbo Long
# @File:example.py
# @Software:PyCharm
import pandas as pd
from sklearn.cluster import KMeans
from openpyxl import Workbook

# Read latitude and longitude data from the input Excel file
input_file = 'E:\Cambridge\Dissertation\Dataset\DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS\location.xlsx'
df = pd.read_excel(input_file, header=None, names=['Latitude', 'Longitude'])

# Determine the number of clusters (you can adjust this value)
num_clusters = 260

df = df.iloc[1:,:]

# Cluster the data using K-means
kmeans = KMeans(n_clusters=num_clusters,random_state=0)
df['cluster'] = kmeans.fit_predict(df)



# Calculate the cluster centers
cluster_centers = kmeans.cluster_centers_
cluster_centers_df = pd.DataFrame(cluster_centers, columns=["Latitude", "Longitude"])
cluster_centers_df["Cluster"] = range(num_clusters)
cluster_centers_df["Cluster"] = cluster_centers_df["Cluster"]
count = pd.DataFrame()

count['Center'] =df['cluster'].value_counts().index
count['value'] =df['cluster'].value_counts()
print(count)

# # Save the clustered data and the cluster centers to separate sheets in a new Excel file
# with pd.ExcelWriter("clustering_260.xlsx") as writer:
#     df.to_excel(writer, sheet_name="Clustered_Data", index=False)
#     cluster_centers_df.to_excel(writer, sheet_name="Cluster_Centers", index=False)
#     count.to_excel(writer, sheet_name="Cluster_Count", index=False)
