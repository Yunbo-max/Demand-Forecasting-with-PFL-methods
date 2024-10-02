# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-01-24 10:28:46
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-07-04 16:10:40
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

Data = pd.read_csv('DataCoSupplyChainDataset.csv',encoding= 'unicode_escape')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(Data.head())

print('\n')

print(Data.columns)
print('\n')
print(Data.shape)
print('\n')
print(Data.apply(lambda x: sum(x.isnull())))


fig, ax = plt.subplots(figsize=(24,12))         # figsize
sns.heatmap(Data.corr(),annot=True,linewidths=.5,fmt='.1g',cmap= 'coolwarm') # Heatmap for correlation matrix
plt.show()


Data1 = Data.copy()

data_delivery_status = Data1.groupby(['Delivery Status'])['Order Id'].count().reset_index(name='Number of Orders').sort_values(by= 'Number of Orders', ascending= False)

fig1 = px.bar(x=data_delivery_status['Delivery Status'] , y=data_delivery_status['Number of Orders']  , color=data_delivery_status['Number of Orders'],
      labels = { 'Delivery Status': 'Delivery Status', 'Number of Orders': 'Number of Orders'})

fig1.show()

data_delivery_status_region=Data1.groupby(['Delivery Status', 'Order Region'])['Order Id'].count().reset_index(name='Number of Orders').sort_values(by= 'Number of Orders', ascending= False)
fig2 = px.bar(data_delivery_status_region, x='Delivery Status', y='Number of Orders'  , color='Order Region')

fig2.show()

