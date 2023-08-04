import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

# initialize Nominatim API
geolocator = Nominatim(user_agent="geoapiExercises")



Data = pd.read_csv('DataCoSupplyChainDataset.csv',encoding= 'unicode_escape')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(Data.head(30))
# A = Data['Customer Zipcode'].value_counts()
A = Data['Customer State'].value_counts()
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 1000)

print(A.head(1000))
print(A.shape)


Total_address = pd.DataFrame()

Latitude = Data['Latitude']
Longitude = Data['Longitude']

Latitude=np.array(Latitude)
Latitude = Latitude.tolist()

Longitude=np.array(Longitude)
Longitude = Longitude.tolist()



j =0
# print(Latitude)
total_value = []
city=[]
for i in range(len(Latitude)):
    location = []
    location = geolocator.reverse(str(Latitude[i]) + "," + str(Longitude[i]))
    address = location.raw['address']
    # print(address)
    Countyname = address.get('county','')
    cityname = address.get('city', '')
    townname = address.get('town', '')

    if townname!='':
        city.append(townname)
    elif cityname!='':
        city.append(cityname)
    elif Countyname!='':
        city.append(Countyname)

    print(city)

print(city.shape)
Data['Location']=city
import csv

import csv
csvFile = open("./data.csv",'w',newline='',encoding='utf-8')
writer = csv.writer(csvFile)
csvRow = []

f = open("data.txt",'r',encoding='GB2312')
for line in f:
    csvRow = line.split(',')
    writer.writerow(csvRow)

f.close()
csvFile.close()



