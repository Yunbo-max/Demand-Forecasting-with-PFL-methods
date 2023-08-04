import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import datetime as dt
import calendar,warnings
import tensorflow as tf2
import statsmodels.api as sm
from datetime import datetime
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn import svm,metrics,tree,preprocessing,linear_model
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge,LinearRegression,LogisticRegression,ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, GradientBoostingRegressor,BaggingClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score,mean_squared_error,recall_score,confusion_matrix,f1_score,roc_curve, auc
from sklearn.datasets import load_iris,make_regression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_ridge import KernelRidge
from tensorflow import keras
from IPython.core import display as ICD
from keras import Sequential
from keras.layers import Dense

#Importing Dataset using pandas
dataset=pd.read_csv("DataCoSupplyChainDataset.csv",header= 0,encoding= 'unicode_escape')
dataset.head(5)# Checking 5 rows in dataset


# Adding first name and last name together to create new column
dataset['Customer Full Name'] = dataset['Customer Fname'].astype(str)+dataset['Customer Lname'].astype(str)

data=dataset.drop(['Customer Email','Product Status','Customer Password','Customer Street','Customer Fname','Customer Lname',
           'Latitude','Longitude','Product Description','Product Image','Order Zipcode'],axis=1)

# Data_west = data
# a  = 0
#
# for i in Data_west['Customer State']:
#     if i not in ['CA','OR','WA','HI']:
#         Data_west.drop(a, axis=0)
#     a=a+1
#
# Data_west.head(30)

df = data.set_index(['Customer State'])

#或者，要包含多个值，可以使用df.index.isin：
data_west = df.loc[df.index.isin(['CA','OR','WA','HI'])]
data_northeast= df.loc[df.index.isin(['NY','NJ','MA','CT','PA','RI','MD','DE','DC'])]
data_Midwestwest = df.loc[df.index.isin(['IL','MI','OH','IN','WI','MN','IA','MO','KS','NE','ND','SD'])]
data_Southeast= df.loc[df.index.isin(['FL','GA','NC','VA','SC','KY','TN','AL','MS','AR','LA'])]
data_Southwest= df.loc[df.index.isin(['TX','AZ','NM','CO','UT','NV','OK'])]
data_island= df.loc[df.index.isin(['PR'])]
