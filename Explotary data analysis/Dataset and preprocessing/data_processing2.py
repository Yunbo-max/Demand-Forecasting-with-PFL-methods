# -*- coding = utf-8 -*-
# @time:08/07/2023 00:49
# Author:Yunbo Long
# @File:data_processing.py
# @Software:PyCharm
# -*- coding = utf-8 -*-
# @time:03/07/2023 13:11
# Author:Yunbo Long
# @File:CNN_local.py
# @Software:PyCharm
# %% 
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn import svm,metrics,tree,preprocessing,linear_model
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import Ridge,LinearRegression,LogisticRegression,ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, GradientBoostingRegressor,BaggingClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score,mean_squared_error,recall_score,confusion_matrix,f1_score,roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# Hiding the warnings
warnings.filterwarnings('ignore')

# # Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# Load dataset
# dataset = pd.read_csv('/content/drive/My Drive/DataCoSupplyChainDataset.csv')


dataset =pd.read_csv('/Users/yunbo-max/Desktop/Personalised_FL/Demand-Forecasting-with-PFL-methods/Explotary data analysis/Original Data/DataCoSupplyChainDataset.csv',encoding='latin-1')
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


dataset['Customer Full Name'] = dataset['Customer Fname'].astype(str) + dataset['Customer Lname'].astype(str)
dataset['TotalPrice'] = dataset['Order Item Quantity'] * dataset[
    'Sales per customer']  # Multiplying item price * Order quantity



data = dataset.drop(
    ['Customer Email', 'Customer Id', 'Customer Password', 'Customer Fname', 'Customer Lname',
      'Product Description', 'Product Image', 'Order Zipcode','Product Status','Order Profit Per Order','Product Price'], axis=1)

data['Customer Zipcode'] = data['Customer Zipcode'].fillna(0)  # Filling NaN columns with zero



data['order_year'] = pd.DatetimeIndex(data['order date (DateOrders)']).year
data['order_month'] = pd.DatetimeIndex(data['order date (DateOrders)']).month
data['order_week_day'] = pd.DatetimeIndex(data['order date (DateOrders)']).day_name()
data['order_hour'] = pd.DatetimeIndex(data['order date (DateOrders)']).hour
# data['order_second'] = pd.DatetimeIndex(data['order date (DateOrders)'])

data['shipping_year'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).year
data['shipping_month'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).month
data['shipping_week_day'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).day_name()
data['shipping_hour'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).hour


label_data = data[['Order Region','shipping_week_day','order_week_day','Customer Full Name','Type','Delivery Status','Category Name','Customer City','Customer Country','Customer Segment','Customer State','Customer Street','Department Name','Market','Order City','Order Country','order date (DateOrders)','Order State','Order Status','Product Name','shipping date (DateOrders)','Shipping Mode']]

data['index'] = data['Order Region']

target=data['Sales']

data=data.drop(columns=['shipping_week_day','order_week_day','Customer Full Name','Sales','Type','Delivery Status','Category Name','Customer City','Customer Country','Customer Segment','Customer State','Customer Street','Department Name','Market','Order City','Order Country','order date (DateOrders)','Order Region','Order State','Order Status','Product Name','shipping date (DateOrders)','Shipping Mode'])




def Labelencoder_feature(x):
    le=LabelEncoder()
    x=le.fit_transform(x)
    return x

# Exclude datetime columns from label encoding
# datetime_columns = ['order date (DateOrders)', 'shipping date (DateOrders)']
# data_encoded = data.drop(datetime_columns, axis=1)

# Apply label encoding to remaining columns
data_encoded = label_data.apply(Labelencoder_feature)

data = pd.concat([data_encoded, data], axis=1)


# Calculate correlation matrix
correlation_matrix = data.corr()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(data.head(100))
print(data.shape)

# Create heatmap
plt.figure(figsize=(100, 50))
sns.heatmap(correlation_matrix, annot=True, fmt=".3f", cmap="BuPu")
plt.show()




#Feature Selection based on importance
from sklearn.feature_selection import f_regression
F_values, p_values  = f_regression(data, target)

import itertools
f_reg_results = [(i, v, z) for i, v, z in itertools.zip_longest(data.columns, F_values,  ['%.3f' % p for p in p_values])]
f_reg_results=pd.DataFrame(f_reg_results, columns=['Variable','F_Value', 'P_Value'])

f_reg_results=pd.DataFrame(f_reg_results, columns=['Variable','F_Value', 'P_Value'])
f_reg_results = f_reg_results.sort_values(by=['P_Value'])
f_reg_results.P_Value= f_reg_results.P_Value.astype(float)
f_reg_results=f_reg_results[f_reg_results.P_Value<0.06]
print(f_reg_results)

f_reg_list=f_reg_results.Variable.values
print(f_reg_list)
#
data_final=data[['Order Id', 'Order Customer Id', 'Order Item Id',
 'Order Item Product Price' ,'Department Id',
 'Category Id' ,'shipping_month' ,'Benefit per order' ,
 'Product Card Id', 'Product Name', 'Order Item Cardprod Id' ,
 'Order State', 'Product Category Id', 'order_week_day', 'shipping_year',
 'Category Name', 'order_month', 'order_year' ,'Order Item Discount',
 'Department Name', 'Market', 'Order City',
 'Days for shipment (scheduled)' ,'Customer Segment', 'Customer Full Name','index']]
#
# train_data = data_final
# xs=train_data.loc[:, train_data.columns != 'Sales']
# ys=target
# xs_train, xs_test,ys_train,ys_test = train_test_split(xs,ys,test_size = 0.3, random_state = 42)
#
# # Concatenate train_data and ys
# train_data['Sales'] = ys.values
#
# # Save the integrated DataFrame to CSV
# train_data.to_csv('integrated_train_data_ISMM.csv', index=False)
#
#
# # dataset =pd.read_csv('E:\Federated_learning_flower\experiments\Presentation\integrated_train_data_ISMM.csv')
# # pd.set_option('display.max_rows', 500)
# # pd.set_option('display.max_columns', 500)
# # pd.set_option('display.width', 1000)
# # print(dataset.head(30))
# #
# # train_data = dataset
# # xs=train_data.loc[:, train_data.columns != 'Sales']
# # ys=dataset['Sales']
# # xs_train, xs_test,ys_train,ys_test = train_test_split(xs,ys,test_size = 0.3, random_state = 42)
# #
# # # xq=train_data.loc[:, train_data.columns != 'Order Item Quantity']
# # # yq=train_data['Order Item Quantity']
# # # xq_train, xq_test,yq_train,yq_test = train_test_split(xq,yq,test_size = 0.3, random_state = 42)
# #
# # scaler=MinMaxScaler()
# # xs_train=scaler.fit_transform(xs_train)
# # xs_test=scaler.transform(xs_test)
# # # xq_train=scaler.fit_transform(xq_train)
# # # xq_test=scaler.transform(xq_test)
# #


# # train_inputs = torch.tensor(xs_train, dtype=torch.float32)
# # train_targets = torch.tensor(ys_train.values, dtype=torch.float32)
# # test_inputs = torch.tensor(xs_test, dtype=torch.float32)
# # test_targets = torch.tensor(ys_test.values, dtype=torch.float32)


