# -*- coding = utf-8 -*-
# @time:08/07/2023 01:50
# Author:Yunbo Long
# @File:split.py
# @Software:PyCharm

import pandas as pd
import h5py

import pandas as pd

# Read the integrated_train_data.csv file
dataset = pd.read_csv('E:\Federated_learning_flower\experiments\Presentation\integrated_train_data_ISMM.csv')

# Get unique market values
markets = dataset['index'].unique()

# Add a new column 'Market Index' to represent the index strings with numbers
region_mapping = {region: i for i, region in enumerate(markets)}
dataset['Region_Number'] = dataset['index'].map(region_mapping)


# Print the modified dataset
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(dataset.head(30))


with h5py.File('market_data.h5', 'w') as f:
    for market in markets:
        market_str = str(market)
        market_data = dataset[dataset['Region_Number'] == market]
        # Store the data as a dataset
        f.create_dataset(market_str, data=market_data.to_numpy())
        # Store the column names as an attribute
        f[market_str].attrs['columns'] = market_data.columns.tolist()
