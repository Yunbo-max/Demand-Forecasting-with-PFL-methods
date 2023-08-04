import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
# import lightgbm as lgb
import datetime as dt
import calendar,warnings,itertools,matplotlib,keras,shutil
# import tensorflow as tf
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
# from keras import Sequential
# from keras.layers import Dense
import torch
from torch import nn
import torch.nn.functional as F
from IPython.core import display as ICD

#Hiding the warnings
warnings.filterwarnings('ignore')

Data = pd.read_csv('DataCoSupplyChainDataset.csv',encoding= 'unicode_escape')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(Data.head(30))


print('\n')
print(Data.shape)

#  check the missing data
print('\n')
print(Data.apply(lambda x: sum(x.isnull())))

# Add and delete new feature
Data["Customer full name"] = Data['Customer Fname'].astype(str)+Data['Customer Lname'].astype(str)
Data = Data.drop(['Customer Fname','Customer Lname','Order Zipcode','Product Description','Customer Password','Customer Street','Product Image','Product Status','Customer Email'],axis=1)

print('\n')
print(Data.shape)
print('\n')
print(Data.apply(lambda x: sum(x.isnull())))

# Country name from spain to english
# country_map = {"República Dominicana":"Dominican Republic",
#                "Estados Unidos": "USA",
#                "México":"Mexico",
#                "Alemania" : "Germany",
#                "Reino Unido":"UK",
#                "España":"Spain",
#                "Turquía" : "Turkey",
#                "Brasil":"Brazil",
#                "perú" :"Peru",
#                "filipinas" : "Philippines" ,
#                "Egipto" : "Egypt",
#                "Irak" : "Iraq",
#                "SudAfrica": "South Africa",
#                "Nueva Zelanda" : "New Zealand",
#                "Tailandia" : "Thailand",
#                "Panamá" : "Panama",
#                "Rumania" : "Romania",
#                "Marruecos" : "Morocco",
#                "Rusia" : "Russia",
#                "arabia saudí":"Saudi Arabia",
#                "bélgica":"Belgium",
#                "Francia":"France",
#                "guayana francesa":"France",
#                "suiza":"Switzerland",
#                "japón":"Japan",
#                "república democrática del congo" : "Democratic Republic of the Congo",
#                "martinica":"Martinique",
#                "argelia":"Algeria",
#                "trinidad y tobago":"Trinidad and Tobago",
#                "corea del sur" : "South Korea",
#                "irlanda":"Ireland",
#                "hong kong":"china",
#                "camerún":"Cameroon",
#                "myanmar (birmania)":"Myanmar",
#                "kenia":"Kenya",
#                "polonia":"Poland",
#                "ucrania":"Ukraine",
#                "sudán":"Sudan",
#                "kazajistán":"Kazakhstan",
#                "afganistán":"Afghanistan",
#                "bielorrusia":"Belarus",
#                "etiopía":"Ethiopia",
#                "singapur":"Singapore",
#                "dinamarca":"Denmark",
#                "malasia":"Malaysia",
#                "pakistán":"Pakistan",
#                "Camboya":"Cambodia",
#                "España":"Spain",
#                "taiwán":"taiwan",
#                "bosnia y herzegovina":"Bosnia and Herzegovina",
#                "ruanda":"Rwanda",
#                 }
#
# country_maps = {k.lower(): v.lower() for k, v in country_map.items()}
#
#
# def replace_contry_name(x):
#     x = x.lower()
#     y = country_maps.get(x)
#     if y is not None:
#         return y
#     else:
#         return x
#
#
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# # import seaborn as sns
# import plotly.express as px
#
# data_copy = Data.copy()
# data_copy["Order Country"] = data_copy["Order Country"].apply(replace_contry_name)
# geo_df = data_copy.groupby(['Order Country'])['Benefit per order'].count().reset_index(name='total order').sort_values(by='total order', ascending=False).reset_index()
# del (data_copy)
# fig = px.choropleth(geo_df, locationmode='country names', locations='Order Country',
#                     color='total order',  # lifeExp is a column of data
#                     hover_name='Order Country',
#                     # hover_data ='Order City',
#                     color_continuous_scale=px.colors.sequential.Reds,
#                     )
# del(geo_df)
# fig.show()







# preprocessing

import requests
import urllib.parse

def getLatLong(address):
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
    response = requests.get(url).json()
    if len(response) > 0:
        return response[0]["lat"], response[0]["lon"]
    return None

def get_address(x):
    if x['Order City'].strip() == x['Order State'].strip():
        addr =f"{x['Order State'].strip()},{x['Order Country'].strip()}"
    else:
        addr =f"{x['Order City'].strip()},{x['Order State'].strip()},{x['Order Country'].strip()}"
    return addr


data = Data

address_df = data.groupby(["Order City","Order State","Order Country"]).agg({"Customer Zipcode":"count"}).drop(columns=["Customer Zipcode"]).reset_index()
address_df["Order address"] = address_df.apply(get_address,axis = 1)

address_list = address_df["Order address"].to_list()

print(address_df)

print('\n')

print(address_list[0:20])

#
# import pickle as pk
# import time
# from tqdm.notebook import tqdm
#
# address_dict = {}
# with open("latlong.pkl", "wb") as f:
#     tk = tqdm(address_list)
#     sucess, fail = 0, 0
#     for addr in tk:
#         res = getLatLong(addr)
#         if res is not None:
#             lat, long = res
#             address_dict[addr] = (lat, long)
#             pk.dump(address_dict, f)
#             sucess += 1
#         else:
#             fail += 1
#         tk.set_postfix(sucess=sucess, fail=fail)
#         time.sleep(1)

import pickle as pk

# with open("latlong.pkl", "wb") as f:
#     pk.dump(address_dict, f)
#
# with open("address.pkl", "wb") as f:
#     pk.dump(address_list, f)

# with open("latlong.pkl", "rb") as f:
#     address_dict = pk.load(f)
#
# with open("address.pkl", "rb") as f:
#     address_list = pk.load(f)
#
# data["Order address"] = data.apply(get_address,axis = 1)
#
# from geopy import distance
#
#
# def getDistance(location1, location2):
#     return distance.distance(location1, location2).kilometers
#
#
# def cal_distance(x):
#     lat_long = address_dict.get(x["Order address"])
#     if lat_long is not None:
#         ord_lat, ord_long = lat_long
#         order_loc = (float(ord_lat), float(ord_long))
#         store_loc = (x["Latitude"], x["Longitude"])
#         distance = getDistance(store_loc, order_loc)
#         return distance
#     else:
#         return -100
#
#
# data_tmp = data.copy()
# data_tmp["Shipping Distance"] = data_tmp.apply(cal_distance, axis=1)
# data_tmp = data_tmp[data_tmp["Shipping Distance"] != -100]
#
# data_tmp.to_csv("cleaned_dataco.csv", encoding="utf8", index=False)

data_tmp = pd.read_csv("cleaned_dataco.csv")



data_tmp = data_tmp.drop(columns=["Latitude","Longitude",'Order City','Order Country','Order State',
                                  "Product Card Id", "Product Category Id",'Order Item Cardprod Id',
                                  'Order Item Id',
                                  'Order Customer Id',
                                  'Order address', "Order Profit Per Order",
                                  "Sales per customer", "Customer Id", "Order Id",
                                  "Order Item Profit Ratio",
                                  'Customer full name', 'Type',
                                  'Days for shipping (real)', "Delivery Status",
                                  "Category Name", 'Department Name','Product Name',
#                                   'Market',
                                  'Category Id',
                                  'Customer City', 'Customer Country',
                                  'Customer Segment', 'Customer State','order date (DateOrders)',
                                  'Order Status'
                                 ])

print(data_tmp.head(10))
print(data_tmp.shape)
print(data_tmp.columns)


data2 = Data.loc[:,['Latitude','Longitude']]
data2 = data2.round()
print(data2.apply(lambda x: sum(x.isnull())))
print(data2)

print(data2.value_counts().shape)







# MLP



continuos_fields = ['Order Item Discount', 'Order Item Discount Rate', "Order Item Product Price", "Days for shipping (real)","Days for shipment (scheduled)","Benefit per order","Department Id","Order Item Quantity","Sales","Order Item Total","Product Price","Shipping Distance"]
lables_fields = ["Late_delivery_risk"]
odering_fields = []
categorical_fields = list((set(data_tmp.columns) - set(continuos_fields)) - set(lables_fields) - set(odering_fields))

data_tmp2 = data_tmp.copy()




for col in categorical_fields:
    dummy = pd.get_dummies(data_tmp2[col], drop_first=False)
    data_tmp2 = pd.concat([data_tmp2, dummy], axis=1)
data_tmp2 = data_tmp2.drop(columns=categorical_fields)


pd.set_option('display.max_columns',None)

pd.set_option('display.max_rows',None)

print(data_tmp2.head(10))

# data_tmp2 = np.float32(data_tmp2)

# import matplotlib.pyplot as plt
#
# corr = data_tmp2.corr()
# corr.style.background_gradient(cmap='coolwarm')

data_tmp2.groupby(["Late_delivery_risk"])["Order Item Quantity"].count()

xl=data_tmp2.loc[:, data_tmp2.columns != 'Late_delivery_risk']
# print(xl)
#Only fraud column
yl=data_tmp2['Late_delivery_risk']
#Splitting the data into two parts in which 80% data will be used for training the model and 20% for testing
xl_train, xl_test,yl_train,yl_test = train_test_split(xl,yl,test_size = 0.2, random_state = 42)

sc = StandardScaler()

xl_train=sc.fit_transform(xl_train)
xl_test=sc.transform(xl_test)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

class LateDataset(Dataset):
    def __init__(self, x, y):
        self.X = np.array(x)
        self.y = np.array(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Net(nn.Module):

    def __init__(self, D_in):
        super().__init__()
        self.fc1 = nn.Linear(D_in, 2048)
        self.out = nn.Linear(2048, 1)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        return F.sigmoid(x).squeeze()


from tqdm import tqdm


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(2022)

n_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

train_data = LateDataset(xl_train, yl_train)
test_data = LateDataset(xl_test, yl_test)

trainloader = DataLoader(train_data, batch_size=512, shuffle=True)
testloader = DataLoader(test_data, batch_size=512, shuffle=False)

# Define the model
D_in = xl_train.shape[-1]
net = Net(D_in).to(device)

# Loss function
criterion = nn.BCEWithLogitsLoss().to(device)

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=0.00001)


# Train the net
loss_per_iter = []
loss_per_batch = [-100]
tk = tqdm(list(range(n_epochs)))
for epoch in tk:
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        net.train()
        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor).to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Save loss to plot
        running_loss += loss.item()
        tk.set_postfix(path_loss = loss_per_batch[-1], iter_loss=loss.item(), running_loss=running_loss/ (i + 1))
    loss_per_batch.append(running_loss / (i + 1))
    running_loss = 0.0

# Comparing training to test
net.eval()
dataiter = iter(testloader)
inputs, labels = dataiter.next()
inputs = inputs.to(device)
labels = labels.to(device)
outputs = net(inputs.float())
print("Training:", loss_per_batch[-1])
print("Test", np.sqrt(criterion(labels.float(), outputs).detach().cpu().numpy()))


from sklearn.metrics import classification_report

print(f"Acc:{torch.sum(labels==outputs)/labels.shape[0]}\n\n")
print(classification_report(labels.cpu().numpy(), outputs.type(torch.LongTensor).detach().cpu().numpy()))


