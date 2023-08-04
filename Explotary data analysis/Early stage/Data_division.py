# -*- coding = utf-8 -*-
# @time:18/03/2023 14:10
# Author:Yunbo Long
# @File:Data_division.py
# @Software:PyCharm
import pandas as pd

dataset = pd.read_csv('E:\Cambridge\Dissertation\Dataset\DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS\DataCoSupplyChainDataset.csv',encoding= 'utf-8')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(dataset.head(30))


import pandas as pd

# Read the Excel file
data = dataset.copy()

# Extract the Latitude and Longitude columns
latitude = data['Latitude']
longitude = data['Longitude']

# Create a set of unique pairs of Latitude and Longitude values
pairs = set(zip(latitude, longitude))

# Print the number of unique pairs
num_pairs = len(pairs)
print(f"Number of unique pairs of Latitude and Longitude: {num_pairs}")
#
# data_count1 = pd.DataFrame()
# data_count1['Latitude'] = dataset['Latitude']
# data_count2 = pd.DataFrame()
# data_count2['Longitude'] = dataset['Longitude']
#
#
#
#
# print(data_count1)
# print(data_count1.shape)
# print(data_count2)
# print(data_count2.shape)
# print(dataset.loc[:,['Latitude','Longitude']])

import pandas as pd
import plotly.express as px
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="my-app")

data = pd.DataFrame({
    'state': ['PR', 'CA', 'NY', 'TX', 'IL', 'FL', 'OH', 'PA', 'MI', 'NJ', 'AZ', 'GA', 'MD', 'NC', 'CO', 'VA', 'OR', 'MA', 'TN', 'NV', 'MO', 'HI', 'CT', 'UT', 'NM', 'LA', 'WA', 'WI', 'MN', 'SC', 'IN', 'DC', 'KY', 'KS', 'DE', 'RI', 'WV', 'OK', 'ND', 'ID', 'AR', 'MT', 'IA', 'AL'],
    'count': [69373, 29223, 11327, 9103, 7631, 5456, 4095, 3824, 3804, 3191, 3026, 2503, 2415, 1992, 1914, 1849, 1668, 1607, 1582, 1440, 1354, 1248, 1094, 968, 949, 948, 920, 850, 672, 665, 581, 579, 487, 458, 269, 243, 241, 232, 215, 167, 164, 87, 67, 35]
})

# Calculate latitude and longitude for each state
latitude = []
longitude = []
for state in data['state']:
    location = geolocator.geocode(state + ', USA')
    if location is not None:
        latitude.append(location.latitude)
        longitude.append(location.longitude)
    else:
        latitude.append(None)
        longitude.append(None)

# Add latitude and longitude to the DataFrame
data['latitude'] = latitude
data['longitude'] = longitude

fig = px.scatter_geo(data, lat='latitude', lon='longitude', color='count', hover_name='state',
                     size='count', projection='natural earth')

fig.show()